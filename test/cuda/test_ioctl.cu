// test_ioctl.cu — raw /dev/phxfs_dev GPU encoding + mmap + MAP/UNMAP contract
// tests. Talks to the kernel module directly (no libphoenix), so it covers
// inputs libphoenix never produces: bad lengths, foreign/split/moved VMAs,
// repeated MAP, wrong GPU command, fork, process death, GPU memory freed while
// mapped (vendor force-reclaim), and concurrent MAP vs munmap/cudaFree.
//
// Mode aware: reads /sys/class/phxfs-generic/phxfs_devN/map_mode.
//   FULL    : any 64KiB-aligned GPU buffer can be mapped.
//   STAGING : only a buffer whose BAR span is 2MiB aligned/sized and
//             contiguous can be mapped; map_buffer() scans for such a window
//             the same way libphoenix's staging setup does.
//
// Usage: test_ioctl <cuda_gpu> [suite|case ...]
//   suites : all (default) | contract | reclaim | race
//   special: die | die_reset | hold <seconds>
//   env    : TEST_IOCTL_ITERS (race iterations, default 100)
// Exit status: 0 = no failure, 1 = failure, 2 = environment error.
#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../../libphoenix/include/common.h"

#define KiB (1024ULL)
#define MiB (1024ULL * 1024)

static int tests_run = 0, tests_passed = 0, tests_failed = 0, tests_skipped = 0;
#define CHECK(cond, msg, ...) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [PASS] " msg "\n", ##__VA_ARGS__); } \
    else { tests_failed++; printf("  [FAIL] " msg "\n", ##__VA_ARGS__); } \
} while (0)
#define SKIP(msg, ...) do { \
    tests_skipped++; \
    printf("  [SKIP] " msg "\n", ##__VA_ARGS__); \
} while (0)

static const size_t L     = 4 * MiB;   // default mapping size
static const size_t SLACK = 2 * MiB;   // room to find a 2MiB-aligned BAR window
static const size_t DPAGE = 64 * KiB;  // device page (checked against sysfs)

enum { MODE_FULL = 0, MODE_STAGING = 1 };

struct Ctx {
    int cuda = -1;
    int phx = -1;
    int fd = -1;
    int mode = -1;
    int iters = 100;
};
static Ctx g;

// ---------------------------------------------------------------- helpers

static std::string sysfs(int phx, const char *attr) {
    std::ifstream f("/sys/class/phxfs-generic/phxfs_dev" + std::to_string(phx) + "/" + attr);
    std::string s;
    f >> s;
    return s;
}

static int find_phx_dev(int cuda_dev) {
    char bdf[32] = {0};
    if (cudaDeviceGetPCIBusId(bdf, sizeof(bdf), cuda_dev) != cudaSuccess)
        return -1;
    for (int i = 0; i < 64; i++) {
        std::string s = sysfs(i, "pci_bdf");
        if (!s.empty() && strcasecmp(s.c_str(), bdf) == 0)
            return i;
    }
    return -1;
}

static int open_phx() {
    return open("/dev/phxfs_dev", O_RDWR | O_CLOEXEC);
}

static int phx_ioctl(int fd, unsigned long cmd, const void *c, size_t clen,
                     const void *d, size_t dlen) {
    phxfs_ioctl_para_t p;
    memset(&p, 0, sizeof(p));
    p.map_param.c_vaddr = (u64)(uintptr_t)c;
    p.map_param.c_size = clen;
    p.map_param.n_vaddr = (u64)(uintptr_t)d;
    p.map_param.n_size = dlen;
    int r = ioctl(fd, cmd, &p);
    return r < 0 ? -errno : r;
}
static int MAP(int fd, const void *c, size_t cl, const void *d, size_t dl, int gpu = g.phx) {
    return phx_ioctl(fd, PHXFS_IOCTL_MAP_GPU(gpu), c, cl, d, dl);
}
static int UNMAP(int fd, const void *c, size_t cl, const void *d, size_t dl, int gpu = g.phx) {
    return phx_ioctl(fd, PHXFS_IOCTL_UNMAP_GPU(gpu), c, cl, d, dl);
}

static char *vmap(int fd, size_t len) {
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? NULL : (char *)p;
}

static inline uint32_t pat(size_t word) { return 0x9E3779B9u * (uint32_t)(word + 1) ^ 0x5A5Au; }

static uintptr_t align_up(uintptr_t v, uintptr_t a) { return (v + a - 1) / a * a; }

// GPU allocation pre-filled with pat() by byte offset from `raw`.
struct DevBuf {
    char *raw = nullptr;
    size_t len = 0;
    bool alloc(size_t want) {
        len = want + SLACK;
        if (cudaMalloc(&raw, len) != cudaSuccess) { raw = nullptr; return false; }
        std::vector<uint32_t> h(len / 4);
        for (size_t i = 0; i < h.size(); i++) h[i] = pat(i);
        if (cudaMemcpy(raw, h.data(), len, cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
        return cudaDeviceSynchronize() == cudaSuccess;
    }
    void release() { if (raw) cudaFree(raw); raw = nullptr; }
};

// MAP [c, c+len) onto some window of `b`, returning the device address used
// in *dptr. FULL: the first 64KiB-aligned address. STAGING: scan 64KiB steps
// (2MiB-aligned virtual first) until the kernel accepts the BAR span; -EINVAL
// is its "layout not acceptable" answer, anything else is final.
static int map_buffer(int fd, char *c, size_t len, const DevBuf &b, char **dptr) {
    uintptr_t raw = (uintptr_t)b.raw, end = raw + b.len;
    if (g.mode != MODE_STAGING) {
        *dptr = (char *)align_up(raw, DPAGE);
        return MAP(fd, c, len, *dptr, len);
    }
    uintptr_t pref = align_up(raw, 2 * MiB);
    int rc = -EINVAL;
    if (pref + len <= end) {
        *dptr = (char *)pref;
        rc = MAP(fd, c, len, *dptr, len);
    }
    for (uintptr_t cand = align_up(raw, DPAGE); rc == -EINVAL && cand + len <= end; cand += DPAGE) {
        if (cand == pref) continue;
        *dptr = (char *)cand;
        rc = MAP(fd, c, len, *dptr, len);
    }
    return rc;
}

// Sampled check that the BAR view [c, c+len) shows the device data at dptr.
static bool verify(const char *c, size_t len, const DevBuf &b, const char *dptr) {
    size_t base = (size_t)(dptr - b.raw) / 4;
    const volatile uint32_t *u = (const volatile uint32_t *)c;
    for (size_t o = 0; o < len; o += KiB)
        if (u[o / 4] != pat(base + o / 4)) return false;
    return u[len / 4 - 1] == pat(base + len / 4 - 1);
}

// ---------------------------------------------------------------- contract

static void t_encoding() {
    printf("\n[encoding] command GPU/op bits / unbound zero-offset VMA\n");
    unsigned long cmd = PHXFS_IOCTL_MAP_GPU(g.phx);
    CHECK((_IOC_NR(cmd) >> PHXFS_IOCTL_GPU_SHIFT) == (unsigned)g.phx &&
          (_IOC_NR(cmd) & 0xfU) == 1 && _IOC_TYPE(cmd) == PHXFS_IOCTL &&
          _IOC_SIZE(cmd) == sizeof(phxfs_ioctl_map_t), "GPU/op encoding preserves ioctl fields");
    int fd = open_phx();
    CHECK(fd >= 0, "open unified node without SELECT");
    if (fd < 0) return;
    void *bad = mmap(NULL, L, PROT_READ | PROT_WRITE, MAP_SHARED, fd, getpagesize());
    CHECK(bad == MAP_FAILED && errno == EINVAL, "nonzero mmap offset -> EINVAL");
    if (bad != MAP_FAILED) munmap(bad, L);
    int r = ioctl(fd, cmd ^ (1U << _IOC_TYPESHIFT), NULL);
    CHECK(r < 0 && errno == ENOTTY, "malformed ioctl type -> ENOTTY");
    r = ioctl(fd, cmd ^ (1U << _IOC_SIZESHIFT), NULL);
    CHECK(r < 0 && errno == ENOTTY, "malformed ioctl size -> ENOTTY");
    char *c = vmap(fd, L);
    CHECK(c != NULL, "mmap reserves an address range without a GPU");
    if (!c) { close(fd); return; }
    CHECK(UNMAP(fd, c, L, NULL, 0) == 0, "UNMAP of an unbound VMA is a no-op");
    munmap(c, L);
    close(fd);
}

static void t_basic() {
    printf("\n[basic] MAP / data visible both ways / UNMAP idempotent\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *c = vmap(g.fd, L), *d = nullptr;
    CHECK(c != nullptr, "mmap %zu MiB", L / MiB);
    if (!c) { b.release(); return; }
    int rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "MAP (rc=%d)", rc);
    if (rc == 0) {
        CHECK(verify(c, L, b, d), "device data visible through the BAR mapping");
        volatile uint32_t *u = (volatile uint32_t *)c;
        u[0] = 0xDEADBEEFu;
        u[L / 4 - 1] = 0xFEEDFACEu;
        __sync_synchronize();
        (void)u[0];
        uint32_t first = 0, last = 0;
        cudaMemcpy(&first, d, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&last, d + L - 4, 4, cudaMemcpyDeviceToHost);
        CHECK(first == 0xDEADBEEFu && last == 0xFEEDFACEu,
              "CPU writes through the BAR mapping reach device memory");
    }
    CHECK(UNMAP(g.fd, c, L, d, L) == 0, "UNMAP");
    CHECK(UNMAP(g.fd, c, L, d, L) == 0, "second UNMAP is a no-op");
    CHECK(munmap(c, L) == 0, "munmap");
    b.release();
}

static void t_double_map() {
    printf("\n[double_map] MAP twice / MAP after UNMAP on the same VMA\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = nullptr;
    char *c = vmap(g.fd, L);
    int rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "first MAP (rc=%d)", rc);
    rc = MAP(g.fd, c, L, d, L);
    CHECK(rc == -EBUSY, "second MAP on a mapped VMA -> -EBUSY (rc=%d)", rc);
    CHECK(verify(c, L, b, d), "first mapping intact after the rejected MAP");
    UNMAP(g.fd, c, L, d, L);
    rc = MAP(g.fd, c, L, d, L);
    CHECK(rc == -EBUSY, "MAP after UNMAP on the same VMA -> -EBUSY (rc=%d)", rc);
    munmap(c, L);
    c = vmap(g.fd, L);
    rc = MAP(g.fd, c, L, d, L);
    CHECK(rc == 0, "fresh VMA maps the same device range again (rc=%d)", rc);
    CHECK(rc != 0 || verify(c, L, b, d), "data visible on the fresh VMA");
    UNMAP(g.fd, c, L, d, L);
    munmap(c, L);
    b.release();
}

static void t_small() {
    printf("\n[small] lengths below the device page (64KiB) / 2MiB span\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = (char *)align_up((uintptr_t)b.raw, 2 * MiB);
    struct { size_t vma, len; } cases[] = { {4 * KiB, 4 * KiB}, {DPAGE, 4 * KiB}, {DPAGE, DPAGE} };
    for (auto &k : cases) {
        char *c = vmap(g.fd, k.vma);
        int rc = MAP(g.fd, c, k.vma, d, k.len);
        if (g.mode == MODE_STAGING) {
            CHECK(rc == -EINVAL, "STAGING rejects a %zuKiB span (vma %zuKiB) -> -EINVAL (rc=%d)",
                  k.len / KiB, k.vma / KiB, rc);
        } else {
            CHECK(rc == 0, "FULL maps %zuKiB into a %zuKiB VMA (rc=%d)", k.len / KiB, k.vma / KiB, rc);
            if (rc == 0) {
                size_t n = (k.len < k.vma ? k.len : k.vma);
                CHECK(verify(c, n, b, d), "  first %zuKiB visible", n / KiB);
                UNMAP(g.fd, c, k.vma, d, k.len);
            }
        }
        munmap(c, k.vma);
    }
    b.release();
}

static void t_bounds() {
    printf("\n[bounds] malformed MAP arguments are rejected, VMA stays usable\n");
    DevBuf b;
    if (!b.alloc(2 * L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = (char *)align_up((uintptr_t)b.raw, 2 * MiB);
    char *c = vmap(g.fd, L);
    int rc;
    rc = MAP(g.fd, c, L, d, 2 * L);
    CHECK(rc == -EINVAL, "dev_len larger than the VMA -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, c, L, NULL, L);
    CHECK(rc == -EINVAL, "devaddr 0 -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, c, L, d, 0);
    CHECK(rc == -EINVAL, "dev_len 0 -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, c, L / 2, d, L / 2);
    CHECK(rc == -EINVAL, "c_size != VMA length -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, c + 4 * KiB, L - 4 * KiB, d, L - 4 * KiB);
    CHECK(rc == -EINVAL, "c_vaddr inside (not at) the VMA -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, c + 1, L, d, L);
    CHECK(rc == -EINVAL, "unaligned c_vaddr -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, NULL, L, d, L);
    CHECK(rc == -EINVAL, "c_vaddr 0 -> -EINVAL (rc=%d)", rc);
    rc = MAP(g.fd, c, L, d + 4 * KiB, L);
    CHECK(rc < 0, "devaddr not 64KiB aligned -> error (rc=%d)", rc);
    rc = MAP(g.fd, c, L, (void *)0x10000, L);
    CHECK(rc < 0, "devaddr that is not GPU memory -> error (rc=%d)", rc);
    char *dd = nullptr;
    rc = map_buffer(g.fd, c, L, b, &dd);
    CHECK(rc == 0, "VMA still mappable after the rejections (rc=%d)", rc);
    CHECK(rc != 0 || verify(c, L, b, dd), "data visible");
    UNMAP(g.fd, c, L, dd, L);
    munmap(c, L);
    b.release();
}

static void t_foreign() {
    printf("\n[foreign] VMAs not created by this device are rejected\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = (char *)align_up((uintptr_t)b.raw, 2 * MiB);
    int rc;

    char *a = (char *)mmap(NULL, L, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    rc = MAP(g.fd, a, L, d, L);
    CHECK(rc == -EINVAL, "anonymous VMA -> -EINVAL (rc=%d)", rc);
    munmap(a, L);

    int zfd = open("/dev/zero", O_RDWR);
    a = (char *)mmap(NULL, L, PROT_READ | PROT_WRITE, MAP_SHARED, zfd, 0);
    rc = MAP(g.fd, a, L, d, L);
    CHECK(rc == -EINVAL, "/dev/zero shared VMA -> -EINVAL (rc=%d)", rc);
    munmap(a, L);
    close(zfd);

    int other = -1, other_phx = -1;
    for (int i = 0; i < 64 && other < 0; i++) {
        if (i == g.phx || sysfs(i, "pci_bdf").empty()) continue;
        other = open_phx();
        if (other >= 0) other_phx = i;
    }
    if (other >= 0) {
        char *c = vmap(other, L), *dd = nullptr;
        CHECK(c != nullptr, "reserve an unbound VMA through another fd");
        if (c) {
            CHECK(UNMAP(other, c, L, NULL, 0, other_phx) == 0, "UNMAP does not bind a GPU");
            CHECK(MAP(other, c, L, NULL, L, other_phx) == -EINVAL,
                  "basic argument rejection does not bind a GPU");
            rc = MAP(g.fd, c, L, (void *)0x10000, L);
            CHECK(rc < 0, "invalid GPU address fails without inserting pages (rc=%d)", rc);
            CHECK(MAP(other, c, L, d, L, other_phx) == -EINVAL,
                  "failed MAP keeps its original GPU binding");
            rc = map_buffer(g.fd, c, L, b, &dd);
            CHECK(rc == 0, "same-GPU retry after clean failure (rc=%d)", rc);
            if (rc == 0) {
                CHECK(UNMAP(other, c, L, dd, L, other_phx) == 0, "wrong-GPU UNMAP is a no-op");
                CHECK(verify(c, L, b, dd), "mapping intact after wrong-GPU UNMAP");
                UNMAP(g.fd, c, L, dd, L);
                CHECK(MAP(g.fd, c, L, dd, L) == -EBUSY, "consumed VMA cannot be remapped");
                CHECK(MAP(other, c, L, dd, L, other_phx) == -EINVAL,
                      "UNMAP does not release the original GPU binding");
            }
            munmap(c, L);
        }
        c = vmap(g.fd, L);
        CHECK(c != nullptr, "reserve a separate unbound VMA");
        if (c) {
            rc = MAP(other, c, L, (void *)0x10000, L, other_phx);
            CHECK(rc < 0, "first MAP attempt on GPU %d fails cleanly (rc=%d)", other_phx, rc);
            CHECK(MAP(g.fd, c, L, d, L) == -EINVAL, "another GPU cannot rebind that VMA");
            munmap(c, L);
        }
        close(other);
    } else {
        SKIP("only one phxfs GPU: cross-GPU binding cases");
    }

    int fd2 = open_phx();
    char *c = vmap(fd2, L), *dd = nullptr;
    rc = map_buffer(g.fd, c, L, b, &dd);
    CHECK(rc == 0, "VMA of a second fd on the same device is accepted (rc=%d)", rc);
    UNMAP(fd2, c, L, dd, L);
    munmap(c, L);
    close(fd2);
    b.release();
}

static void t_vma_changes() {
    printf("\n[vma_changes] split / mremap grow / mremap move\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = nullptr;
    int rc;

    char *c = vmap(g.fd, L);
    munmap(c + L - DPAGE, DPAGE);
    rc = MAP(g.fd, c, L, (char *)align_up((uintptr_t)b.raw, 2 * MiB), L);
    CHECK(rc == -EINVAL, "MAP on a VMA shrunk by munmap -> -EINVAL (rc=%d)", rc);
    munmap(c, L - DPAGE);

    c = vmap(g.fd, L);
    munmap(c + L / 2, DPAGE);   // splits into two VMAs sharing one buffer
    rc = MAP(g.fd, c, L, (char *)align_up((uintptr_t)b.raw, 2 * MiB), L);
    CHECK(rc == -EINVAL, "MAP across a hole -> -EINVAL (rc=%d)", rc);
    munmap(c, L);

    c = vmap(g.fd, L);
    void *m = mremap(c, L, 2 * L, MREMAP_MAYMOVE);
    CHECK(m == MAP_FAILED, "mremap cannot grow a phony VMA");
    if (m != MAP_FAILED) c = (char *)m;
    munmap(c, m == MAP_FAILED ? L : 2 * L);

    // Moved before MAP: the buffer still records the old address.
    char *resv = (char *)mmap(NULL, 2 * L, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    c = vmap(g.fd, L);
    m = mremap(c, L, L, MREMAP_MAYMOVE | MREMAP_FIXED, resv);
    CHECK(m == resv, "mremap move of an unmapped phony VMA");
    if (m == resv) {
        rc = MAP(g.fd, resv, L, (char *)align_up((uintptr_t)b.raw, 2 * MiB), L);
        CHECK(rc == -EINVAL, "MAP at the moved address -> -EINVAL (rc=%d)", rc);
    } else {
        munmap(c, L);
    }
    munmap(resv, 2 * L);

    // Moved after MAP: PTEs travel with it, teardown still clean.
    resv = (char *)mmap(NULL, 2 * L, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    c = vmap(g.fd, L);
    rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "MAP before move (rc=%d)", rc);
    m = mremap(c, L, L, MREMAP_MAYMOVE | MREMAP_FIXED, resv);
    CHECK(m == resv, "mremap move of a mapped phony VMA");
    if (m == resv) {
        CHECK(rc != 0 || verify(resv, L, b, d), "data visible at the new address");
        UNMAP(g.fd, resv, L, d, L);   // old address recorded: expected no-op
    } else {
        UNMAP(g.fd, c, L, d, L);
        munmap(c, L);
    }
    munmap(resv, 2 * L);
    b.release();
}

static void t_bad_ioctl() {
    printf("\n[bad_ioctl] unknown command / bad argument pointer / UNMAP no-ops\n");
    int r = ioctl(g.fd, _IOW(PHXFS_IOCTL, 99, struct phxfs_ioctl_map_s), NULL);
    CHECK(r < 0 && errno == ENOTTY, "unknown ioctl -> ENOTTY (r=%d errno=%d)", r, errno);
    r = ioctl(g.fd, PHXFS_IOCTL_MAP_GPU(g.phx), (void *)8);
    CHECK(r < 0 && errno == EFAULT, "MAP with a bad arg pointer -> EFAULT (errno=%d)", errno);
    r = ioctl(g.fd, PHXFS_IOCTL_UNMAP_GPU(g.phx), (void *)8);
    CHECK(r < 0 && errno == EFAULT, "UNMAP with a bad arg pointer -> EFAULT (errno=%d)", errno);
    char *c = vmap(g.fd, L);
    CHECK(UNMAP(g.fd, c, L, (void *)0x1000000, L) == 0, "UNMAP of a never-mapped VMA -> 0");
    CHECK(UNMAP(g.fd, c, L / 2, NULL, 0) == 0, "UNMAP with a mismatched length -> 0");
    munmap(c, L);
    CHECK(UNMAP(g.fd, c, L, NULL, 0) == 0, "UNMAP of an unmapped address -> 0");
}

static void t_lifetime() {
    printf("\n[lifetime] munmap without UNMAP / fd closed while mapped / fork\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = nullptr;
    int rc;

    char *c = vmap(g.fd, L);
    rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "MAP (rc=%d)", rc);
    CHECK(munmap(c, L) == 0, "munmap without UNMAP (VMA close tears it down)");

    int fd2 = open_phx();
    c = vmap(fd2, L);
    rc = map_buffer(fd2, c, L, b, &d);
    CHECK(rc == 0, "MAP via a second fd (rc=%d)", rc);
    close(fd2);
    CHECK(rc != 0 || verify(c, L, b, d), "mapping survives close(fd)");
    CHECK(munmap(c, L) == 0, "munmap after close(fd)");

    c = vmap(g.fd, L);
    rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "MAP before fork (rc=%d)", rc);
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0)
        _exit(rc == 0 && verify(c, L, b, d) ? 0 : 3);
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "fork child sees the mapping (status=0x%x)", st);
    CHECK(rc != 0 || verify(c, L, b, d), "parent mapping intact after the child exited");
    UNMAP(g.fd, c, L, d, L);
    munmap(c, L);

    c = vmap(g.fd, L);
    rc = map_buffer(g.fd, c, L, b, &d);
    fflush(stdout);
    pid = fork();
    if (pid == 0) {
        munmap(c, L);   // child drops its copy first
        _exit(0);
    }
    waitpid(pid, &st, 0);
    CHECK(rc == 0 && verify(c, L, b, d), "parent mapping intact after the child munmapped");
    UNMAP(g.fd, c, L, d, L);
    munmap(c, L);

    int other_gpu = -1;
    for (unsigned i = 0; i < PHXFS_IOCTL_GPU_LIMIT; ++i)
        if ((int)i != g.phx && !sysfs(i, "pci_bdf").empty()) { other_gpu = i; break; }
    if (other_gpu >= 0) {
        c = vmap(g.fd, L);
        int gate[2];
        if (!c || pipe(gate) != 0) {
            CHECK(false, "prepare fork-before-MAP case");
            if (c) munmap(c, L);
            b.release();
            return;
        }
        fflush(stdout);
        pid = fork();
        if (pid == 0) {
            close(gate[1]);
            char ready = 0;
            bool ok = read(gate[0], &ready, 1) == 1 &&
                      MAP(g.fd, c, L, (void *)0x10000, L, other_gpu) == -EINVAL &&
                      UNMAP(g.fd, c, L, NULL, 0, other_gpu) == 0;
            close(gate[0]);
            munmap(c, L);
            _exit(ok ? 0 : 3);
        }
        close(gate[0]);
        if (pid > 0) {
            rc = map_buffer(g.fd, c, L, b, &d);
            CHECK(rc == 0, "parent MAP after fork (rc=%d)", rc);
            char ready = 1;
            CHECK(write(gate[1], &ready, 1) == 1, "publish the parent's GPU binding");
            close(gate[1]);
            CHECK(waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                  "child cannot rebind or unpin the parent's GPU after fork-before-MAP");
            CHECK(rc != 0 || verify(c, L, b, d), "parent data intact after child teardown");
            UNMAP(g.fd, c, L, d, L);
        } else {
            CHECK(false, "fork before MAP");
            close(gate[1]);
        }
        munmap(c, L);
    } else {
        SKIP("only one phxfs GPU: fork-before-MAP cross-GPU binding");
    }
    b.release();
}

// ---------------------------------------------------------------- reclaim

static void t_reclaim() {
    printf("\n[reclaim] GPU memory freed while mapped (vendor force-reclaim)\n");
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *d = nullptr;
    char *c = vmap(g.fd, L);
    int rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "MAP (rc=%d)", rc);
    cudaError_t e = cudaFree(b.raw);
    b.raw = nullptr;
    CHECK(e == cudaSuccess, "cudaFree while mapped returns (callback did not deadlock)");
    CHECK(UNMAP(g.fd, c, L, d, L) == 0, "UNMAP after reclaim");
    CHECK(munmap(c, L) == 0, "munmap after reclaim");

    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    c = vmap(g.fd, L);
    rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0, "MAP (rc=%d)", rc);
    b.release();
    CHECK(munmap(c, L) == 0, "reclaim then munmap without UNMAP");

    DevBuf b2;
    if (!b2.alloc(L)) { CHECK(false, "cudaMalloc after reclaim"); return; }
    c = vmap(g.fd, L);
    rc = map_buffer(g.fd, c, L, b2, &d);
    CHECK(rc == 0 && verify(c, L, b2, d), "new registration works after reclaim (rc=%d)", rc);
    UNMAP(g.fd, c, L, d, L);
    munmap(c, L);
    b2.release();
}

// ---------------------------------------------------------------- races

static void t_race_munmap() {
    printf("\n[race_munmap] MAP racing munmap of the same VMA (%d iters)\n", g.iters);
    DevBuf b;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    int ok = 0, rejected = 0, unexpected = 0, last_bad = 0;
    for (int i = 0; i < g.iters; i++) {
        char *c = vmap(g.fd, L), *d = nullptr;
        std::thread t([c, i] { usleep((i * 37) % 400); munmap(c, L); });
        int rc = map_buffer(g.fd, c, L, b, &d);
        t.join();
        if (rc == 0) ok++;
        else if (rc == -EINVAL || rc == -EFAULT) rejected++;
        else { unexpected++; last_bad = rc; }
    }
    printf("    mapped=%d rejected=%d unexpected=%d\n", ok, rejected, unexpected);
    CHECK(unexpected == 0, "MAP vs munmap only yields 0/-EINVAL/-EFAULT (last bad rc=%d)", last_bad);
    b.release();
}

static void t_race_parallel() {
    const int T = 8;
    printf("\n[race_parallel] %d threads x %d MAP/verify/UNMAP/munmap cycles\n", T, g.iters / 2);
    std::vector<DevBuf> bufs(T);
    for (auto &b : bufs)
        if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    std::atomic<int> fails{0}, maps{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < T; t++) {
        ts.emplace_back([&, t] {
            for (int i = 0; i < g.iters / 2; i++) {
                char *c = vmap(g.fd, L), *d = nullptr;
                if (!c) { fails++; continue; }
                int rc = map_buffer(g.fd, c, L, bufs[t], &d);
                if (rc == 0 && verify(c, L, bufs[t], d)) maps++;
                else fails++;
                UNMAP(g.fd, c, L, d, L);
                munmap(c, L);
            }
        });
    }
    for (auto &th : ts) th.join();
    printf("    good=%d bad=%d\n", maps.load(), fails.load());
    CHECK(fails.load() == 0, "concurrent registrations on one device all succeed and verify");
    for (auto &b : bufs) b.release();
}

static void t_race_reclaim() {
    int n = g.iters / 2 > 0 ? g.iters / 2 : 1;
    printf("\n[race_reclaim] MAP racing cudaFree of the target buffer (%d iters)\n", n);
    int ok = 0, err = 0, cuda_err = 0;
    for (int i = 0; i < n; i++) {
        DevBuf b;
        if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
        char *c = vmap(g.fd, L), *d = nullptr;
        cudaError_t fe = cudaSuccess;
        std::thread t([&, i] {
            cudaSetDevice(g.cuda);
            usleep((i * 53) % 600);
            fe = cudaFree(b.raw);
        });
        int rc = map_buffer(g.fd, c, L, b, &d);
        t.join();
        b.raw = nullptr;
        if (fe != cudaSuccess) cuda_err++;
        if (rc == 0) ok++; else err++;
        UNMAP(g.fd, c, L, d, L);
        munmap(c, L);
    }
    printf("    mapped=%d failed=%d cudaFree_errors=%d\n", ok, err, cuda_err);
    CHECK(cuda_err == 0, "cudaFree never fails or hangs while racing MAP");
    DevBuf b;
    char *d = nullptr;
    if (!b.alloc(L)) { CHECK(false, "cudaMalloc"); return; }
    char *c = vmap(g.fd, L);
    int rc = map_buffer(g.fd, c, L, b, &d);
    CHECK(rc == 0 && verify(c, L, b, d), "device still healthy afterwards (rc=%d)", rc);
    UNMAP(g.fd, c, L, d, L);
    munmap(c, L);
    b.release();
}

// ---------------------------------------------------------------- special

// Leave a live mapping behind and exit without any cleanup.
static int s_die(bool reset_first) {
    DevBuf b;
    if (!b.alloc(L)) return 2;
    char *c = vmap(g.fd, L), *d = nullptr;
    int rc = map_buffer(g.fd, c, L, b, &d);
    printf("die%s: MAP rc=%d, exiting without UNMAP/munmap\n", reset_first ? "_reset" : "", rc);
    fflush(stdout);
    if (reset_first)
        cudaDeviceReset();   // destroys the context first -> vendor reclaim
    _exit(rc == 0 ? 0 : 1);
}

// Keep a mapping (and the module) busy for `secs` seconds.
static int s_hold(int secs) {
    DevBuf b;
    if (!b.alloc(L)) return 2;
    char *c = vmap(g.fd, L), *d = nullptr;
    int rc = map_buffer(g.fd, c, L, b, &d);
    if (rc != 0) { printf("hold: MAP failed rc=%d\n", rc); return 1; }
    printf("HOLDING\n");
    fflush(stdout);
    sleep(secs);
    bool ok = verify(c, L, b, d);
    UNMAP(g.fd, c, L, d, L);
    munmap(c, L);
    b.release();
    printf("hold: released (data %s)\n", ok ? "ok" : "CORRUPT");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <cuda_gpu> [all|contract|reclaim|race|<case>... | die | die_reset | hold <s>]\n"
                        "cases: encoding basic double_map small bounds foreign vma_changes bad_ioctl lifetime\n"
                        "       reclaim race_munmap race_parallel race_reclaim\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    g.cuda = atoi(argv[1]);
    if (const char *it = getenv("TEST_IOCTL_ITERS")) g.iters = atoi(it) > 0 ? atoi(it) : g.iters;
    if (cudaSetDevice(g.cuda) != cudaSuccess) { printf("FATAL: cudaSetDevice(%d)\n", g.cuda); return 2; }
    cudaFree(0);
    g.phx = find_phx_dev(g.cuda);
    if (g.phx < 0) { printf("FATAL: no phxfs device for CUDA %d (module loaded?)\n", g.cuda); return 2; }
    g.fd = open_phx();
    if (g.fd < 0) { printf("FATAL: open /dev/phxfs_dev: %s\n", strerror(errno)); return 2; }
    std::string mode = sysfs(g.phx, "map_mode"), page = sysfs(g.phx, "page_size");
    g.mode = mode.empty() ? -1 : atoi(mode.c_str());
    printf("=== phxfs raw ioctl tests: CUDA %d -> /dev/phxfs_dev GPU index %d, map_mode=%s, page_size=%s ===\n",
           g.cuda, g.phx, g.mode == MODE_FULL ? "FULL" : g.mode == MODE_STAGING ? "STAGING" : mode.c_str(),
           page.c_str());
    if (g.mode != MODE_FULL && g.mode != MODE_STAGING) { printf("FATAL: unknown map_mode\n"); return 2; }
    if (page != std::to_string(DPAGE)) { printf("FATAL: test assumes a 64KiB device page\n"); return 2; }

    std::vector<std::string> args(argv + 2, argv + argc);
    if (args.empty()) args.push_back("all");
    if (args[0] == "die") return s_die(false);
    if (args[0] == "die_reset") return s_die(true);
    if (args[0] == "hold") return s_hold(args.size() > 1 ? atoi(args[1].c_str()) : 10);

    const std::vector<std::pair<std::string, std::function<void()>>> cases = {
        {"encoding", t_encoding}, {"basic", t_basic}, {"double_map", t_double_map}, {"small", t_small},
        {"bounds", t_bounds}, {"foreign", t_foreign}, {"vma_changes", t_vma_changes},
        {"bad_ioctl", t_bad_ioctl}, {"lifetime", t_lifetime},
        {"reclaim", t_reclaim},
        {"race_munmap", t_race_munmap}, {"race_parallel", t_race_parallel},
        {"race_reclaim", t_race_reclaim},
    };
    auto in_suite = [](const std::string &suite, const std::string &name) {
        if (suite == "all") return true;
        if (suite == "reclaim") return name == "reclaim";
        if (suite == "race") return name.rfind("race_", 0) == 0;
        if (suite == "contract") return name != "reclaim" && name.rfind("race_", 0) != 0;
        return suite == name;
    };
    for (auto &a : args) {
        bool any = false;
        for (auto &c : cases)
            if (in_suite(a, c.first)) { c.second(); any = true; }
        if (!any) { printf("unknown suite/case '%s'\n", a.c_str()); return 2; }
    }

    close(g.fd);
    printf("\n=== Summary ===\nTotal: %d, Passed: %d, Failed: %d, Skipped: %d\nResult: %s\n",
           tests_run, tests_passed, tests_failed, tests_skipped,
           tests_failed == 0 ? "ALL PASSED" : "SOME FAILED");
    return tests_failed ? 1 : 0;
}
