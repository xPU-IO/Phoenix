// test_io.cu
//
// Tests for phxfs read/write correctness and performance:
//   Part A — Correctness:
//     1. Write known pattern GPU->file, read back file->GPU, verify
//     2. Partial reads at various offsets
//     3. Multiple I/O rounds on same registration
//   Part B — Performance:
//     4. Read throughput at different transfer sizes
//     5. Write throughput at different transfer sizes
//     6. Registration latency
//
// Build: via CMake (make test_io)
// Run:   ./test_io [gpu_id]
//        Default file: /mnt/nvme0/phxfs_test.bin (override with -DTEST_FILE=...)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>

#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include "phoenix.h"

#define GPU_PAGE_SIZE (64 * 1024)
#define KiB (1024ULL)
#define MiB (1024ULL * 1024)
#define GiB (1024ULL * 1024 * 1024)

#ifndef TEST_FILE
#define TEST_FILE "/mnt/nvme0/phxfs_test.bin"
#endif

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg, ...) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  [PASS] " msg "\n", ##__VA_ARGS__); \
    } else { \
        tests_failed++; \
        printf("  [FAIL] " msg "\n", ##__VA_ARGS__); \
    } \
} while (0)

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_pattern(uint64_t *buf, size_t size, uint64_t base_offset) {
    for (size_t i = 0; i < size / 8; i++)
        buf[i] = base_offset + (uint64_t)(i * 8);
}

static bool verify_pattern(const uint64_t *buf, size_t size, uint64_t base_offset,
                           size_t *err_count, size_t max_report = 5) {
    size_t errs = 0;
    for (size_t i = 0; i < size / 8; i++) {
        uint64_t expected = base_offset + (uint64_t)(i * 8);
        if (buf[i] != expected) {
            if (errs < max_report)
                printf("    MISMATCH off=0x%zx expected=0x%lx got=0x%lx\n",
                       base_offset + i * 8, expected, buf[i]);
            errs++;
        }
    }
    *err_count = errs;
    return errs == 0;
}

static int ensure_test_file(size_t size) {
    struct stat st;
    if (stat(TEST_FILE, &st) == 0 && (size_t)st.st_size >= size) {
        printf("[info] test file exists (%lld bytes), reusing\n", (long long)st.st_size);
        return 0;
    }

    printf("[info] creating test file %s (%zu bytes)...\n", TEST_FILE, size);
    uint64_t *buf = (uint64_t *)malloc(size);
    if (!buf) return -1;
    fill_pattern(buf, size, 0);

    int fd = open(TEST_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { free(buf); return -1; }

    size_t done = 0;
    while (done < size) {
        ssize_t w = write(fd, (char *)buf + done, size - done);
        if (w <= 0) { close(fd); free(buf); return -1; }
        done += (size_t)w;
    }
    fsync(fd);
    close(fd);
    free(buf);

    posix_fadvise(open(TEST_FILE, O_RDONLY), 0, 0, POSIX_FADV_DONTNEED);
    printf("[info] test file created\n");
    return 0;
}

// ===========================================================================
// Part A: Correctness Tests
// ===========================================================================

static void test_write_read_verify(int dev_id, size_t size) {
    printf("\n=== Test 1: Write + Read + Verify (%zuMiB) ===\n", size / MiB);

    void *gpu_buf = nullptr, *target = nullptr;
    uint64_t *host_buf = nullptr;

    cudaMalloc(&gpu_buf, size);
    cudaMallocHost(&host_buf, size);

    fill_pattern(host_buf, size, 0);
    cudaMemcpy(gpu_buf, host_buf, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
    CHECK(ret == 0, "regmem");
    if (ret != 0) { cudaFree(gpu_buf); cudaFreeHost(host_buf); return; }

    // Write GPU -> file
    int wfd = open(TEST_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(wfd >= 0, "open file for write");
    if (wfd >= 0) {
        ssize_t wr = phxfs_write(wfd, dev_id, gpu_buf, 0, size, 0);
        CHECK(wr == (ssize_t)size, "phxfs_write %zu bytes (got %zd)", size, wr);
        fsync(wfd);
        close(wfd);
    }

    // Clear GPU buffer
    cudaMemset(gpu_buf, 0xFF, size);
    cudaDeviceSynchronize();

    // Read file -> GPU
    int rfd = open(TEST_FILE, O_RDONLY | O_DIRECT);
    CHECK(rfd >= 0, "open file for read (O_DIRECT)");
    if (rfd >= 0) {
        ssize_t rd = phxfs_read(rfd, dev_id, gpu_buf, 0, size, 0);
        CHECK(rd == (ssize_t)size, "phxfs_read %zu bytes (got %zd)", size, rd);
        close(rfd);
    }

    // Verify: GPU -> host
    memset(host_buf, 0, size);
    cudaMemcpy(host_buf, gpu_buf, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    size_t errs = 0;
    bool ok = verify_pattern(host_buf, size, 0, &errs);
    CHECK(ok, "data integrity: %zu mismatches", errs);

    phxfs_deregmem(dev_id, gpu_buf, size);
    cudaFree(gpu_buf);
    cudaFreeHost(host_buf);
}

static void test_partial_reads(int dev_id) {
    printf("\n=== Test 2: Partial reads at various offsets ===\n");

    size_t total = 4 * MiB;
    size_t chunk = 256 * KiB;

    void *gpu_buf = nullptr, *target = nullptr;
    uint64_t *host_buf = nullptr;

    cudaMalloc(&gpu_buf, total);
    cudaMallocHost(&host_buf, total);
    cudaMemset(gpu_buf, 0, total);
    cudaDeviceSynchronize();

    int ret = phxfs_regmem(dev_id, gpu_buf, total, &target);
    CHECK(ret == 0, "regmem 4MiB");
    if (ret != 0) { cudaFree(gpu_buf); cudaFreeHost(host_buf); return; }

    if (ensure_test_file(total) != 0) {
        CHECK(false, "failed to create test file");
        phxfs_deregmem(dev_id, gpu_buf, total);
        cudaFree(gpu_buf);
        cudaFreeHost(host_buf);
        return;
    }

    int rfd = open(TEST_FILE, O_RDONLY | O_DIRECT);
    CHECK(rfd >= 0, "open test file");
    if (rfd < 0) {
        phxfs_deregmem(dev_id, gpu_buf, total);
        cudaFree(gpu_buf);
        cudaFreeHost(host_buf);
        return;
    }

    {
        bool all_ok = true;

        for (size_t off = 0; off < total; off += chunk) {
            size_t sz = (total - off < chunk) ? (total - off) : chunk;
            ssize_t rd = phxfs_read(rfd, dev_id, gpu_buf, off, sz, off);
            if (rd != (ssize_t)sz) {
                printf("    read at offset %zu returned %zd (expected %zu)\n",
                       off, rd, sz);
                all_ok = false;
                break;
            }
        }
        CHECK(all_ok, "all partial reads completed");

        if (all_ok) {
            cudaMemcpy(host_buf, gpu_buf, total, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            size_t total_errs = 0;
            for (size_t off = 0; off < total; off += chunk) {
                size_t sz = (total - off < chunk) ? (total - off) : chunk;
                size_t errs = 0;
                verify_pattern((uint64_t *)((char *)host_buf + off), sz, off, &errs, 0);
                total_errs += errs;
            }
            CHECK(total_errs == 0, "partial read data integrity (%zu errs)", total_errs);
        }
    }

    close(rfd);
    phxfs_deregmem(dev_id, gpu_buf, total);
    cudaFree(gpu_buf);
    cudaFreeHost(host_buf);
}

static void test_multiple_io_rounds(int dev_id) {
    printf("\n=== Test 3: Multiple I/O rounds on same registration ===\n");

    size_t size = 1 * MiB;
    int rounds = 5;

    void *gpu_buf = nullptr, *target = nullptr;
    uint64_t *host_buf = nullptr;

    cudaMalloc(&gpu_buf, size);
    cudaMallocHost(&host_buf, size);

    int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
    CHECK(ret == 0, "regmem (single registration for multiple I/O)");
    if (ret != 0) { cudaFree(gpu_buf); cudaFreeHost(host_buf); return; }

    if (ensure_test_file(size * rounds) != 0) {
        CHECK(false, "failed to create test file");
        phxfs_deregmem(dev_id, gpu_buf, size);
        cudaFree(gpu_buf);
        cudaFreeHost(host_buf);
        return;
    }

    int rfd = open(TEST_FILE, O_RDONLY | O_DIRECT);
    CHECK(rfd >= 0, "open test file");
    if (rfd < 0) {
        phxfs_deregmem(dev_id, gpu_buf, size);
        cudaFree(gpu_buf);
        cudaFreeHost(host_buf);
        return;
    }

    {
        bool all_ok = true;

        for (int r = 0; r < rounds && all_ok; r++) {
            size_t foff = (size_t)r * size;
            cudaMemset(gpu_buf, 0xFF, size);
            cudaDeviceSynchronize();

            ssize_t rd = phxfs_read(rfd, dev_id, gpu_buf, 0, size, foff);
            if (rd != (ssize_t)size) {
                CHECK(false, "round %d: read returned %zd", r, rd);
                all_ok = false;
                break;
            }

            cudaMemcpy(host_buf, gpu_buf, size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            size_t errs = 0;
            verify_pattern(host_buf, size, foff, &errs, 0);
            if (errs > 0) {
                CHECK(false, "round %d: %zu data errors", r, errs);
                all_ok = false;
            }
        }
        if (all_ok)
            CHECK(true, "all %d rounds verified", rounds);
    }

    close(rfd);
    phxfs_deregmem(dev_id, gpu_buf, size);
    cudaFree(gpu_buf);
    cudaFreeHost(host_buf);
}

// ===========================================================================
// Part B: Performance Tests
// ===========================================================================

static void test_read_perf(int dev_id) {
    printf("\n=== Test 4: Read throughput ===\n");

    size_t test_sizes[] = {
        64 * KiB,    256 * KiB,   1 * MiB,
        16 * MiB,    64 * MiB,    256 * MiB,
    };
    int n = sizeof(test_sizes) / sizeof(test_sizes[0]);

    size_t max_size = 256 * MiB;
    if (ensure_test_file(max_size) != 0) {
        printf("  [SKIP] cannot create test file\n");
        return;
    }

    printf("  %-12s  %10s  %10s  %10s\n",
           "Size", "Time(ms)", "GB/s", "GiB/s");
    printf("  ------------  ----------  ----------  ----------\n");

    for (int i = 0; i < n; i++) {
        size_t size = test_sizes[i];

        void *gpu_buf = nullptr, *target = nullptr;
        cudaMalloc(&gpu_buf, size);
        cudaMemset(gpu_buf, 0, size);
        cudaDeviceSynchronize();

        int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
        if (ret != 0) {
            printf("  %-12zu  SKIP (regmem failed)\n", size);
            cudaFree(gpu_buf);
            continue;
        }

        int rfd = open(TEST_FILE, O_RDONLY | O_DIRECT);
        if (rfd < 0) {
            phxfs_deregmem(dev_id, gpu_buf, size);
            cudaFree(gpu_buf);
            continue;
        }

        // Warm up
        phxfs_read(rfd, dev_id, gpu_buf, 0, size, 0);
        posix_fadvise(rfd, 0, 0, POSIX_FADV_DONTNEED);

        // Timed run
        double t0 = now_sec();
        ssize_t rd = phxfs_read(rfd, dev_id, gpu_buf, 0, size, 0);
        double t1 = now_sec();
        close(rfd);

        if (rd == (ssize_t)size) {
            double secs = t1 - t0;
            double gbps = (double)size / secs / 1e9;
            double gibps = (double)size / secs / (double)GiB;
            printf("  %-12zu  %10.3f  %10.2f  %10.2f\n",
                   size, secs * 1e3, gbps, gibps);
        } else {
            printf("  %-12zu  READ FAILED (got %zd)\n", size, rd);
        }

        phxfs_deregmem(dev_id, gpu_buf, size);
        cudaFree(gpu_buf);
    }
}

static void test_write_perf(int dev_id) {
    printf("\n=== Test 5: Write throughput ===\n");

    size_t test_sizes[] = {
        64 * KiB,    256 * KiB,   1 * MiB,
        16 * MiB,    64 * MiB,    256 * MiB,
    };
    int n = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("  %-12s  %10s  %10s  %10s\n",
           "Size", "Time(ms)", "GB/s", "GiB/s");
    printf("  ------------  ----------  ----------  ----------\n");

    for (int i = 0; i < n; i++) {
        size_t size = test_sizes[i];

        void *gpu_buf = nullptr, *target = nullptr;
        cudaMalloc(&gpu_buf, size);
        cudaMemset(gpu_buf, 0xAB, size);
        cudaDeviceSynchronize();

        int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
        if (ret != 0) {
            printf("  %-12zu  SKIP (regmem failed)\n", size);
            cudaFree(gpu_buf);
            continue;
        }

        std::string wfile = std::string(TEST_FILE) + ".write";
        int wfd = open(wfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (wfd < 0) {
            wfd = open(wfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        }
        if (wfd < 0) {
            phxfs_deregmem(dev_id, gpu_buf, size);
            cudaFree(gpu_buf);
            continue;
        }

        // Warm up
        phxfs_write(wfd, dev_id, gpu_buf, 0, size, 0);

        // Timed run
        double t0 = now_sec();
        ssize_t wr = phxfs_write(wfd, dev_id, gpu_buf, 0, size, 0);
        double t1 = now_sec();
        fsync(wfd);
        close(wfd);
        unlink(wfile.c_str());

        if (wr == (ssize_t)size) {
            double secs = t1 - t0;
            double gbps = (double)size / secs / 1e9;
            double gibps = (double)size / secs / (double)GiB;
            printf("  %-12zu  %10.3f  %10.2f  %10.2f\n",
                   size, secs * 1e3, gbps, gibps);
        } else {
            printf("  %-12zu  WRITE FAILED (got %zd)\n", size, wr);
        }

        phxfs_deregmem(dev_id, gpu_buf, size);
        cudaFree(gpu_buf);
    }
}

static void test_regmem_latency(int dev_id) {
    printf("\n=== Test 6: Registration latency ===\n");

    size_t test_sizes[] = {
        64 * KiB,    1 * MiB,     16 * MiB,
        64 * MiB,    256 * MiB,
    };
    int n = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("  %-12s  %10s\n", "Size", "Time(ms)");
    printf("  ------------  ----------\n");

    for (int i = 0; i < n; i++) {
        size_t size = test_sizes[i];
        void *gpu_buf = nullptr, *target = nullptr;

        cudaMalloc(&gpu_buf, size);
        cudaMemset(gpu_buf, 0, size);
        cudaDeviceSynchronize();

        double total_time = 0;
        int runs = 3;
        bool ok = true;

        for (int r = 0; r < runs; r++) {
            double t0 = now_sec();
            int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
            double t1 = now_sec();

            if (ret != 0) { ok = false; break; }
            total_time += (t1 - t0);

            phxfs_deregmem(dev_id, gpu_buf, size);
        }

        if (ok) {
            double avg_ms = (total_time / runs) * 1e3;
            printf("  %-12zu  %10.3f\n", size, avg_ms);
        } else {
            printf("  %-12zu  FAILED\n", size);
        }

        cudaFree(gpu_buf);
    }
}

// ===========================================================================

int main(int argc, char **argv) {
    int cuda_gpu_id = (argc > 1) ? atoi(argv[1]) : 0;

    printf("=== Phoenix I/O Correctness + Performance Tests ===\n");
    printf("CUDA GPU ID: %d\n", cuda_gpu_id);
    printf("Test file: %s\n", TEST_FILE);

    if (cudaSetDevice(cuda_gpu_id) != cudaSuccess) {
        printf("FATAL: cudaSetDevice(%d) failed\n", cuda_gpu_id);
        return 1;
    }

    int dev_id = phxfs_find_dev_for_cuda_gpu(cuda_gpu_id);
    if (dev_id < 0) {
        printf("FATAL: no phxfs device for CUDA GPU %d\n", cuda_gpu_id);
        return 1;
    }
    printf("phxfs device ID: %d\n", dev_id);

    if (phxfs_open(dev_id) != 0) {
        printf("FATAL: phxfs_open(%d) failed\n", dev_id);
        return 1;
    }

    // Part A: Correctness
    printf("\n========== Part A: Correctness ==========\n");
    test_write_read_verify(dev_id, 1 * MiB);
    test_write_read_verify(dev_id, 16 * MiB);
    test_partial_reads(dev_id);
    test_multiple_io_rounds(dev_id);

    // Part B: Performance
    printf("\n========== Part B: Performance ==========\n");
    test_read_perf(dev_id);
    test_write_perf(dev_id);
    test_regmem_latency(dev_id);

    phxfs_close(dev_id);

    printf("\n=== Correctness Summary ===\n");
    printf("Total: %d, Passed: %d, Failed: %d\n",
           tests_run, tests_passed, tests_failed);
    printf("Result: %s\n", tests_failed == 0 ? "ALL PASSED" : "SOME FAILED");
    return tests_failed ? 1 : 0;
}
