#include <cuda_runtime_api.h>
#include "test_ioctl.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

static const uint64_t page_bytes = 65536;

static bool cuda_ok(cudaError_t err, const char *operation)
{
    if (err == cudaSuccess)
        return true;
    std::fprintf(stderr, "FAIL %s: %s\n", operation, cudaGetErrorString(err));
    return false;
}

static int read_integer(const char *path)
{
    FILE *file = std::fopen(path, "r");
    if (!file)
        return -1;
    int value = -1;
    if (std::fscanf(file, "%d", &value) != 1)
        value = -1;
    std::fclose(file);
    return value;
}

static bool run_case(const char *device, uint64_t bytes)
{
    size_t available = 0, total = 0;
    if (!cuda_ok(cudaMemGetInfo(&available, &total), "cudaMemGetInfo"))
        return false;
    if (bytes + page_bytes > available) {
        std::fprintf(stderr, "FAIL insufficient GPU memory: need=%llu free=%zu\n",
                     static_cast<unsigned long long>(bytes + page_bytes), available);
        return false;
    }
    void *allocation = nullptr;
    if (!cuda_ok(cudaMalloc(&allocation, bytes + page_bytes), "cudaMalloc"))
        return false;
    uint64_t address = (reinterpret_cast<uintptr_t>(allocation) + page_bytes - 1)
                       & ~(page_bytes - 1);
    std::vector<uint64_t> addresses(bytes / page_bytes, 0);
    bool ok = true;
    for (int cycle = 0; cycle < 2 && ok; ++cycle) {
        int fd = open(device, O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            std::fprintf(stderr, "FAIL open %s: %s\n", device, std::strerror(errno));
            ok = false;
            break;
        }
        phxfs_p2p_test_request request = {};
        request.gpu_vaddr = address;
        request.length = bytes;
        request.n_addrs = addresses.size();
        request.ioaddrs = reinterpret_cast<uintptr_t>(addresses.data());
        std::fill(addresses.begin(), addresses.end(), 0);
        int baseline = read_integer("/sys/module/phoenixfs/refcnt");
        if (baseline < 0) {
            std::fprintf(stderr, "FAIL cannot read Phoenix module reference count\n");
            close(fd);
            ok = false;
            break;
        }
        double ms = 0;
        auto begin = std::chrono::steady_clock::now();
        if (ioctl(fd, PHXFS_P2P_TEST_REGISTER, &request) < 0) {
            std::fprintf(stderr, "FAIL registration bytes=%llu errno=%d (%s)\n",
                         static_cast<unsigned long long>(bytes), errno, std::strerror(errno));
            ok = false;
        } else {
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - begin).count();
            if (read_integer("/sys/module/phoenixfs/refcnt") != baseline + 1) {
                std::fprintf(stderr, "FAIL registration did not add exactly one Phoenix handle reference\n");
                ok = false;
            }
            for (size_t i = 0; i < addresses.size(); ++i) {
                if (!addresses[i] || addresses[i] % page_bytes) {
                    std::fprintf(stderr, "FAIL address[%zu]=0x%llx\n", i,
                                 static_cast<unsigned long long>(addresses[i]));
                    ok = false;
                    break;
                }
            }
            if (cycle == 0 && ioctl(fd, PHXFS_P2P_TEST_DEREGISTER, 0UL) < 0) {
                std::fprintf(stderr, "FAIL deregistration: %s\n", std::strerror(errno));
                ok = false;
            }
            if (cycle == 0 && read_integer("/sys/module/phoenixfs/refcnt") != baseline) {
                std::fprintf(stderr, "FAIL explicit unmap leaked a Phoenix handle reference\n");
                ok = false;
            }
        }
        if (close(fd) < 0) {
            std::fprintf(stderr, "FAIL close: %s\n", std::strerror(errno));
            ok = false;
        }
        if (read_integer("/sys/module/phoenixfs/refcnt") != baseline) {
            std::fprintf(stderr, "FAIL fd close leaked a Phoenix handle reference\n");
            ok = false;
        }
        std::printf("%s bytes=%llu pages=%zu table_bytes=%zu register_ms=%.3f cleanup=%s\n",
                    ok ? "PASS" : "FAIL", static_cast<unsigned long long>(bytes),
                    addresses.size(), addresses.size() * sizeof(uint64_t), ms,
                    cycle == 0 ? "explicit-unmap" : "fd-close");
    }
    return cuda_ok(cudaFree(allocation), "cudaFree") && ok;
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("Usage: %s GPU /dev/phxfs_p2p_test [BYTES]\n"
                    "Requires fixed Phoenix and the signed phxfs_p2p_test module. No Tutti/daemon needed.\n"
                    "No module loading, disk IO, or CUDA kernels are performed.\n"
                    "Without BYTES, tests 64KiB, 1GiB, 32GiB, 32GiB+64KiB and 55878615040 bytes.\n"
                    "Each size checks explicit unmap and fd-close cleanup before cudaFree.\n"
                    "Run on an idle host: concurrent registrations change the global reference count.\n"
                    "Exit: 0 pass, 1 test failure, 2 invalid arguments/prerequisites.\n", argv[0]);
        return 0;
    }
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "Usage: %s GPU /dev/phxfs_p2p_test [BYTES] (see --help)\n", argv[0]);
        return 2;
    }
    char *end = nullptr;
    errno = 0;
    long gpu = std::strtol(argv[1], &end, 10);
    if (errno || !*argv[1] || *end || gpu < 0 || gpu > INT32_MAX)
        return 2;
    std::vector<uint64_t> sizes = {page_bytes, 1ULL << 30, 32ULL << 30,
                                   (32ULL << 30) + page_bytes, 55878615040ULL};
    if (argc == 4) {
        if (!*argv[3] || std::strspn(argv[3], "0123456789") != std::strlen(argv[3]))
            return 2;
        errno = 0;
        uint64_t bytes = std::strtoull(argv[3], &end, 10);
        if (errno || *end || !bytes ||
            bytes % page_bytes || bytes > SIZE_MAX - page_bytes ||
            bytes / page_bytes > UINT32_MAX)
            return 2;
        sizes = {bytes};
    }
    int fd = open(argv[2], O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "Prerequisite failed: open %s: %s\n"
                             "Load Phoenix, then the signed phxfs_p2p_test module.\n",
                     argv[2], std::strerror(errno));
        return 2;
    }
    int probe = ioctl(fd, PHXFS_P2P_TEST_DEREGISTER, 0UL);
    close(fd);
    if (probe < 0) {
        std::fprintf(stderr, "Prerequisite failed: not a Phoenix P2P test device.\n");
        return 2;
    }
    if (!cuda_ok(cudaSetDevice(static_cast<int>(gpu)), "cudaSetDevice"))
        return 2;
    for (uint64_t bytes : sizes) {
        if (!run_case(argv[2], bytes))
            return 1;
    }
    std::puts("PASS all P2P registration cases (no DMA data verification)");
    return 0;
}
