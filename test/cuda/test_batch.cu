// test_batch.cu
//
// phxfs_read_batch / phxfs_write_batch (io_uring) bandwidth.
//   Concurrent bandwidth: NGPU worker threads, each on its own
//   GPU with its own io_uring ring, submit BATCHES batches of
//   REQS x CHUNK requests. Reports aggregate throughput.
//
// Build: via CMake (make test_batch)
// Run:   ./test_batch <base_file> [ngpu] [gpu_start] [mode]
//          base_file - test file path (required; on a direct-I/O-capable fs)
//          ngpu      - worker count (default: all visible GPUs, capped 8)
//          gpu_start - first GPU index (default 0)
//          mode      - read (default) | write | both


#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include "phoenix.h"

#define KiB (1024ULL)
#define MiB (1024ULL * 1024)
#define GiB (1024ULL * 1024 * 1024)

#define REQS      512          // requests per batch
#define CHUNK     (128 * KiB)  // bytes per request
#define BATCHES   32           // batches per GPU worker

static int tests_run = 0, tests_passed = 0, tests_failed = 0, tests_skipped = 0;
#define CHECK(cond, msg, ...) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [PASS] " msg "\n", ##__VA_ARGS__); } \
    else { tests_failed++; printf("  [FAIL] " msg "\n", ##__VA_ARGS__); } \
} while (0)
/* Environment-not-available (no GPU / module / O_DIRECT): counted, not a pass
 * or fail — distinct from a genuine setup error, which uses CHECK(false). */
#define SKIP(msg, ...) do { \
    tests_skipped++; \
    printf("  [SKIP] " msg "\n", ##__VA_ARGS__); \
} while (0)

static double now_sec() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Unique per-chunk pattern: every 8-byte word at file offset `o` holds a
// value that encodes BOTH the chunk index and the word's position:
//     word(o) = (chunk_index << 40) | (o & 0xffffffffff)
// where chunk_index = o / CHUNK. This makes swapped, duplicated, missing, or
// mis-routed chunks detectable (unlike a uniform byte pattern that repeats
// every 256 bytes).
static inline uint64_t pattern_word(uint64_t off) {
    uint64_t chunk = off / CHUNK;
    return (chunk << 40) | (off & 0xffffffffffULL);
}

// Write the pattern over [start, size) of `path` (created if absent), fully
// O_DIRECT so file prep does not populate the page cache ahead of the read
// benchmark. The pattern encodes the absolute file offset, so extending an
// existing pattern file from its current length stays consistent with the
// part already there. `size` must be 4096-aligned (it is: BATCHES*REQS*CHUNK).
static int ensure_pattern_file(const char *path, size_t size, size_t start) {
    int fd = open(path, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) { printf("open(write) %s: %s\n", path, strerror(errno)); return -1; }
    const size_t BUF = 8 * MiB;
    void *tmp = NULL;
    if (posix_memalign(&tmp, 4096, BUF) != 0) { close(fd); return -1; }
    /* Round the start down to the O_DIRECT alignment; bytes in between are
     * rewritten with the identical pattern (offset-derived), so no harm. */
    size_t done = start & ~(size_t)4095;
    while (done < size) {
        size_t w = (size - done < BUF) ? (size - done) : BUF;
        uint64_t *wp = (uint64_t *)tmp;
        for (size_t j = 0; j < w / 8; j++) wp[j] = pattern_word(done + j * 8);
        if (pwrite(fd, tmp, w, done) != (ssize_t)w) {
            printf("pwrite: %s\n", strerror(errno)); free(tmp); close(fd); return -1;
        }
        done += w;
    }
    free(tmp); fsync(fd);
    close(fd);
    return 0;
}

// Verify GPU buffer region [buf_base, +len) equals the file pattern starting
// at file offset f_base. Returns mismatch count.
static size_t verify_region(const uint8_t *hbuf, size_t buf_base, size_t len,
                            uint64_t f_base) {
    const uint64_t *wp = (const uint64_t *)(hbuf + buf_base);
    size_t bad = 0;
    for (size_t j = 0; j < len / 8; j++)
        if (wp[j] != pattern_word(f_base + j * 8)) bad++;
    return bad;
}

// Verify [0, size) of `path` holds the pattern, reading O_DIRECT so the check
// sees what actually reached the device rather than a cached copy. Returns the
// mismatch count, or SIZE_MAX if the file could not be read.
static size_t verify_file_pattern(const char *path, size_t size) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { printf("open(verify) %s: %s\n", path, strerror(errno)); return SIZE_MAX; }
    const size_t BUF = 8 * MiB;
    void *tmp = NULL;
    if (posix_memalign(&tmp, 4096, BUF) != 0) { close(fd); return SIZE_MAX; }
    size_t bad = 0, done = 0;
    while (done < size) {
        size_t r = (size - done < BUF) ? (size - done) : BUF;
        if (pread(fd, tmp, r, done) != (ssize_t)r) {
            printf("pread(verify) at %zu: %s\n", done, strerror(errno));
            bad = SIZE_MAX;
            break;
        }
        const uint64_t *wp = (const uint64_t *)tmp;
        for (size_t j = 0; j < r / 8; j++)
            if (wp[j] != pattern_word(done + j * 8)) bad++;
        done += r;
    }
    free(tmp);
    close(fd);
    return bad;
}


// ---- per-GPU worker: own GPU, own file, own ring (thread_local) ----
struct worker_result { double secs; size_t bytes; bool ok; };

// Prepare each GPU's file up front (NOT timed).
static bool prepare_files(int ngpu, int gpu_start, const char *base_file) {
    const size_t file_bytes = (size_t)BATCHES * (size_t)REQS * CHUNK;
    for (int g = 0; g < ngpu; g++) {
        int gpu = gpu_start + g;
        std::string fp = std::string(base_file) + ".batch." + std::to_string(gpu);
        struct stat st;
        size_t have = (stat(fp.c_str(), &st) == 0) ? (size_t)st.st_size : 0;
        if (have >= file_bytes) {
            printf("  gpu%d: reuse %s (%zu MiB)\n", gpu, fp.c_str(), have / MiB);
            continue;
        }
        printf("  gpu%d: %s %s from %zu MiB to %zu MiB\n", gpu,
               have ? "extend" : "create", fp.c_str(), have / MiB, file_bytes / MiB);
        if (ensure_pattern_file(fp.c_str(), file_bytes, have) != 0) {
            printf("prepare file for gpu%d failed\n", gpu);
            return false;
        }
    }
    return true;
}

static void gpu_worker(int gpu, const char *base_file, int do_write,
                       worker_result *out) {
    out->ok = false; out->bytes = 0; out->secs = 0;

    if (cudaSetDevice(gpu) != cudaSuccess) { printf("gpu%d: cudaSetDevice failed\n", gpu); return; }
    int dev = phxfs_find_dev(gpu);
    if (dev < 0) { printf("gpu%d: phxfs_find_dev=%d\n", gpu, dev); return; }
    if (phxfs_open(dev) != 0) { printf("gpu%d: phxfs_open(%d) failed\n", gpu, dev); return; }

    const int    NREQ = REQS * BATCHES;
    const size_t file_bytes = (size_t)NREQ * CHUNK;
    // One distinct GPU target region per in-flight request (no overlap), so
    // the result can be verified and mis-routing/duplication is detectable.
    const size_t buf_bytes = file_bytes;

    std::string fp = std::string(base_file) +
                     (do_write ? ".wbatch." : ".batch.") + std::to_string(gpu);

    void *gbuf = nullptr, *target = nullptr;
    if (cudaMalloc(&gbuf, buf_bytes) != cudaSuccess) { printf("gpu%d: cudaMalloc(%zu) failed\n", gpu, buf_bytes); return; }
    if (do_write) {
        // Seed the GPU buffer with the same offset-encoded pattern the read
        // side verifies, so the file written out can be checked the same way.
        // Not timed.
        uint8_t *hbuf = (uint8_t *)malloc(buf_bytes);
        if (!hbuf) { printf("gpu%d: malloc(%zu) failed\n", gpu, buf_bytes); cudaFree(gbuf); return; }
        uint64_t *wp = (uint64_t *)hbuf;
        for (size_t j = 0; j < buf_bytes / 8; j++) wp[j] = pattern_word(j * 8);
        cudaMemcpy(gbuf, hbuf, buf_bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(0);
        free(hbuf);
    } else {
        cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    }
    if (phxfs_regmem(dev, gbuf, buf_bytes, &target) != 0) {
        printf("gpu%d: phxfs_regmem(dev=%d) failed\n", gpu, dev); cudaFree(gbuf); return;
    }

    int dfd = do_write ? open(fp.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644)
                       : open(fp.c_str(), O_RDONLY | O_DIRECT);
    if (dfd < 0) { printf("gpu%d: open %s: %s\n", gpu, fp.c_str(), strerror(errno));
                   phxfs_deregmem(dev, gbuf, buf_bytes); cudaFree(gbuf); return; }

    // One large batch: request i moves file chunk i to/from GPU chunk i —
    // every target region is distinct. A single blocking call should saturate
    // the array while exercising the internal worker pool.
    std::vector<phxfs_io_req_t> reqs(NREQ);
    for (int i = 0; i < NREQ; i++) {
        reqs[i] = phxfs_io_req_t{};
        reqs[i].fd = dfd; reqs[i].device_id = dev; reqs[i].buf = gbuf;
        reqs[i].nbytes = CHUNK;
        reqs[i].buf_offset = (off_t)i * CHUNK;   // distinct GPU target
        reqs[i].f_offset   = (off_t)i * CHUNK;   // distinct file region
    }

    posix_fadvise(dfd, 0, file_bytes, POSIX_FADV_DONTNEED);
    double t0 = now_sec();
    int rc = do_write ? phxfs_write_batch(reqs.data(), NREQ)
                      : phxfs_read_batch(reqs.data(), NREQ);
    out->secs = now_sec() - t0;
    out->bytes = (size_t)NREQ * CHUNK;

    bool ok = (rc == 0);
    if (rc != 0) printf("gpu%d big batch failed rc=%d\n", gpu, rc);
    // Verify against the pattern (not timed).
    if (ok && do_write) {
        fsync(dfd);
        size_t bad = verify_file_pattern(fp.c_str(), file_bytes);
        if (bad) { printf("gpu%d: FILE MISMATCH %zu words\n", gpu, bad); ok = false; }
    } else if (ok) {
        uint8_t *hbuf = (uint8_t *)malloc(buf_bytes);
        cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
        size_t bad = verify_region(hbuf, 0, buf_bytes, 0);
        if (bad) { printf("gpu%d: DATA MISMATCH %zu words\n", gpu, bad); ok = false; }
        free(hbuf);
    }
    out->ok = ok;

    close(dfd); phxfs_deregmem(dev, gbuf, buf_bytes); phxfs_close(dev); cudaFree(gbuf);
}

// Run one direction across all workers and report aggregate bandwidth.
static bool run_phase(const char *label, int do_write, int ngpu, int gpu_start,
                      const char *base_file) {
    printf("\n--- %s (%d GPU worker(s) from gpu%d, independent rings) ---\n",
           label, ngpu, gpu_start);
    // The write phase creates its own files; only the read phase needs a
    // pre-seeded pattern file.
    if (!do_write && !prepare_files(ngpu, gpu_start, base_file)) {
        printf("file prep failed\n");
        CHECK(false, "%s: file prep", label);
        return false;
    }

    std::vector<std::thread> ts;
    std::vector<worker_result> res(ngpu);
    for (int g = 0; g < ngpu; g++)
        ts.emplace_back(gpu_worker, gpu_start + g, base_file, do_write, &res[g]);
    for (auto &t : ts) t.join();

    size_t total_bytes = 0;
    double max_batch_ms = 0;
    bool all_ok = true;
    for (int g = 0; g < ngpu; g++) {
        if (!res[g].ok) { all_ok = false; printf("  gpu%d: FAILED\n", gpu_start + g); continue; }
        double gib = (double)res[g].bytes / GiB;
        printf("  gpu%d: %6.2f GiB in %7.3f ms -> %6.2f GiB/s\n",
               gpu_start + g, gib, res[g].secs * 1e3, gib / res[g].secs);
        total_bytes += res[g].bytes;
        if (res[g].secs * 1e3 > max_batch_ms)
            max_batch_ms = res[g].secs * 1e3;
    }

    double tgib = (double)total_bytes / GiB;
    printf("  ----\n");
    printf("  aggregate: %.2f GiB / %.3f ms -> %.2f GiB/s (%.2f GB/s)\n",
           tgib, max_batch_ms, tgib / (max_batch_ms / 1e3),
           (double)total_bytes / 1e9 / (max_batch_ms / 1e3));
    CHECK(all_ok, "%s: all workers completed", label);
    return all_ok;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <base_file> [ngpu] [gpu_start] [mode]\n"
                "  base_file  test file path (required; on a direct-I/O-capable fs)\n"
                "  ngpu       worker count (default: all visible GPUs, capped 8)\n"
                "  gpu_start  first GPU index (default 0)\n"
                "  mode       read (default) | write | both\n",
                argv[0]);
        return 1;
    }
    const char *base_file = argv[1];

    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int ngpu      = (argc > 2) ? atoi(argv[2]) : ndev;   // Part B worker count
    int gpu_start = (argc > 3) ? atoi(argv[3]) : 0;      // first GPU index
    const char *mode = (argc > 4) ? argv[4] : "read";

    bool do_read  = (strcmp(mode, "write") != 0);
    bool do_write = (strcmp(mode, "write") == 0 || strcmp(mode, "both") == 0);
    if (!do_read && !do_write) {
        fprintf(stderr, "unknown mode '%s' (read | write | both)\n", mode);
        return 1;
    }

    if (gpu_start < 0) gpu_start = 0;
    if (gpu_start >= ndev) gpu_start = 0;
    if (ngpu > ndev - gpu_start) ngpu = ndev - gpu_start;
    if (ngpu > 8) ngpu = 8;
    if (ngpu < 1) ngpu = 1;

    printf("=== test_batch ===\n");
    printf("file: %s | gpu_start: %d | ngpu: %d | mode: %s\n",
           base_file, gpu_start, ngpu, mode);
    printf("I/O engine: %s | shape: %d batches x %d reqs x %zu KiB\n",
           phxfs_io_engine_name(), BATCHES, REQS, CHUNK / KiB);

    if (do_read)
        run_phase("concurrent read bandwidth", 0, ngpu, gpu_start, base_file);
    if (do_write)
        run_phase("concurrent write bandwidth", 1, ngpu, gpu_start, base_file);

    printf("\n=== %d run, %d passed, %d failed, %d skipped ===\n",
           tests_run, tests_passed, tests_failed, tests_skipped);
    return tests_failed ? 1 : 0;
}
