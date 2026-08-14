// test_regmem.cu
//
// Tests for phxfs memory registration lifecycle:
//   1. Basic register/deregister
//   2. Multiple concurrent registrations
//   3. Various sizes (small, 64KiB-aligned, large)
//   4. Deregister in reverse order
//   5. Double-deregister error handling
//   6. Reuse after deregister
//
// Build: via CMake (make test_regmem)
// Run:   ./test_regmem [gpu_id]

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include "phoenix.h"

#define GPU_PAGE_SIZE (64 * 1024)
#define MiB (1024ULL * 1024)
#define GiB (1024ULL * 1024 * 1024)

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

// ---------------------------------------------------------------------------
// Test 1: Basic register/deregister
// ---------------------------------------------------------------------------
static void test_basic_regmem(int dev_id) {
    printf("\n=== Test 1: Basic register/deregister ===\n");

    void *gpu_buf = nullptr, *target = nullptr;
    size_t size = 1 * MiB;

    cudaMalloc(&gpu_buf, size);
    cudaMemset(gpu_buf, 0, size);
    cudaDeviceSynchronize();

    int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
    CHECK(ret == 0, "phxfs_regmem 1MiB");
    CHECK(target != nullptr, "target_addr is non-null");

    ret = phxfs_deregmem(dev_id, gpu_buf, size);
    CHECK(ret == 0, "phxfs_deregmem 1MiB");

    cudaFree(gpu_buf);
}

// ---------------------------------------------------------------------------
// Test 2: Multiple concurrent registrations
// ---------------------------------------------------------------------------
static void test_multiple_regmem(int dev_id) {
    printf("\n=== Test 2: Multiple concurrent registrations ===\n");

    const int N = 4;
    std::vector<void*> gpu_bufs(N);
    std::vector<void*> targets(N);
    std::vector<size_t> sizes = { 1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB };

    // Register all
    for (int i = 0; i < N; i++) {
        cudaMalloc(&gpu_bufs[i], sizes[i]);
        cudaMemset(gpu_bufs[i], 0, sizes[i]);
        cudaDeviceSynchronize();

        int ret = phxfs_regmem(dev_id, gpu_bufs[i], sizes[i], &targets[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "regmem region %d (%zuMiB)", i, sizes[i] / MiB);
        CHECK(ret == 0, "%s", msg);
    }

    // Deregister in reverse order
    for (int i = N - 1; i >= 0; i--) {
        int ret = phxfs_deregmem(dev_id, gpu_bufs[i], sizes[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "deregmem region %d (reverse)", i);
        CHECK(ret == 0, "%s", msg);
    }

    for (int i = 0; i < N; i++)
        cudaFree(gpu_bufs[i]);
}

// ---------------------------------------------------------------------------
// Test 3: Various sizes
// ---------------------------------------------------------------------------
static void test_various_sizes(int dev_id) {
    printf("\n=== Test 3: Various sizes ===\n");

    size_t test_sizes[] = {
        GPU_PAGE_SIZE,           // 64KiB (minimum)
        GPU_PAGE_SIZE * 2,       // 128KiB
        1 * MiB,
        16 * MiB,
        256 * MiB,
    };
    int n = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int i = 0; i < n; i++) {
        size_t size = test_sizes[i];
        void *gpu_buf = nullptr, *target = nullptr;

        cudaMalloc(&gpu_buf, size);
        cudaMemset(gpu_buf, 0, size);
        cudaDeviceSynchronize();

        int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
        char msg[128];
        snprintf(msg, sizeof(msg), "regmem %zuKiB", size / 1024);
        CHECK(ret == 0, "%s", msg);

        ret = phxfs_deregmem(dev_id, gpu_buf, size);
        snprintf(msg, sizeof(msg), "deregmem %zuKiB", size / 1024);
        CHECK(ret == 0, "%s", msg);

        cudaFree(gpu_buf);
    }
}

// ---------------------------------------------------------------------------
// Test 4: Misaligned size (should fail)
// ---------------------------------------------------------------------------
static void test_misaligned_size(int dev_id) {
    printf("\n=== Test 4: Misaligned size (expect failure) ===\n");

    void *gpu_buf = nullptr, *target = nullptr;
    size_t size = GPU_PAGE_SIZE + 1;  // Not 64KiB aligned

    cudaMalloc(&gpu_buf, GPU_PAGE_SIZE * 2);
    cudaMemset(gpu_buf, 0, GPU_PAGE_SIZE * 2);
    cudaDeviceSynchronize();

    int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
    CHECK(ret != 0, "regmem with misaligned size correctly fails");

    // Should not need deregmem if regmem failed
    if (ret == 0)
        phxfs_deregmem(dev_id, gpu_buf, size);

    cudaFree(gpu_buf);
}

// ---------------------------------------------------------------------------
// Test 5: Deregister unregistered memory (should fail)
// ---------------------------------------------------------------------------
static void test_dereg_unregistered(int dev_id) {
    printf("\n=== Test 5: Deregister unregistered memory (expect failure) ===\n");

    void *gpu_buf = nullptr;
    size_t size = 1 * MiB;

    cudaMalloc(&gpu_buf, size);
    cudaMemset(gpu_buf, 0, size);
    cudaDeviceSynchronize();

    // Deregister without registering first
    int ret = phxfs_deregmem(dev_id, gpu_buf, size);
#ifndef PHXFS_MAP_MODE_STAGING
      CHECK(ret != 0, "deregmem of unregistered memory correctly fails");
#else
      CHECK(ret == 0, "deregmem of unregistered memory correctly succeeds");
#endif
    cudaFree(gpu_buf);
}

// ---------------------------------------------------------------------------
// Test 6: Register, deregister, re-register same GPU buffer
// ---------------------------------------------------------------------------
static void test_reregister(int dev_id) {
    printf("\n=== Test 6: Re-register after deregister ===\n");

    void *gpu_buf = nullptr, *target1 = nullptr, *target2 = nullptr;
    size_t size = 2 * MiB;

    cudaMalloc(&gpu_buf, size);
    cudaMemset(gpu_buf, 0, size);
    cudaDeviceSynchronize();

    // First registration
    int ret = phxfs_regmem(dev_id, gpu_buf, size, &target1);
    CHECK(ret == 0, "first regmem");
    CHECK(target1 != nullptr, "first target non-null");

    ret = phxfs_deregmem(dev_id, gpu_buf, size);
    CHECK(ret == 0, "first deregmem");

    // Re-register same buffer
    ret = phxfs_regmem(dev_id, gpu_buf, size, &target2);
    CHECK(ret == 0, "second regmem (reuse)");
    CHECK(target2 != nullptr, "second target non-null");

    ret = phxfs_deregmem(dev_id, gpu_buf, size);
    CHECK(ret == 0, "second deregmem");

    cudaFree(gpu_buf);
}

// ---------------------------------------------------------------------------
// Test 7: Large registration (256MiB)
// ---------------------------------------------------------------------------
static void test_large_regmem(int dev_id) {
    printf("\n=== Test 7: Large registration (256MiB) ===\n");

    void *gpu_buf = nullptr, *target = nullptr;
    size_t size = 256 * MiB;

    cudaMalloc(&gpu_buf, size);
    cudaMemset(gpu_buf, 0, size);
    cudaDeviceSynchronize();

    int ret = phxfs_regmem(dev_id, gpu_buf, size, &target);
    CHECK(ret == 0, "regmem 256MiB");
    CHECK(target != nullptr, "256MiB target non-null");

    ret = phxfs_deregmem(dev_id, gpu_buf, size);
    CHECK(ret == 0, "deregmem 256MiB");

    cudaFree(gpu_buf);
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    int cuda_gpu_id = (argc > 1) ? atoi(argv[1]) : 0;

    printf("=== Phoenix Memory Registration Tests ===\n");
    printf("CUDA GPU ID: %d\n", cuda_gpu_id);

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

    test_basic_regmem(dev_id);
    test_multiple_regmem(dev_id);
    test_various_sizes(dev_id);
    test_misaligned_size(dev_id);
    test_dereg_unregistered(dev_id);
    test_reregister(dev_id);
    test_large_regmem(dev_id);

    phxfs_close(dev_id);

    printf("\n=== Summary ===\n");
    printf("Total: %d, Passed: %d, Failed: %d\n",
           tests_run, tests_passed, tests_failed);
    printf("Result: %s\n", tests_failed == 0 ? "ALL PASSED" : "SOME FAILED");
    return tests_failed ? 1 : 0;
}
