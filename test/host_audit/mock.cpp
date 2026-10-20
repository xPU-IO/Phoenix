// CPU memory and memfd only. Simulates API contracts, never GPU DMA or P2P.
#include "mock.h"
#include <cerrno>
#include <cstdarg>
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/ioctl.h>

namespace mock {
int find_result[PHXFS_MAX_DEVICES] = {-1,-1,-1,-1,-1,-1,-1,-1};
int device_count = 2, kernel_mode = PHX_MAP_MODE_FULL;
std::atomic<int> open_error{0}, map_error{0}, unmap_error{0}, munmap_error{0}, host_error{0};
std::atomic<int> maps{0}, unmaps{0}, host_live{0}, device_live{0}, launches{0};
std::atomic<bool> fail_threads{false};
std::atomic<int> fail_fd{-1};
std::atomic<off_t> fail_from{0}, fail_until{0};
void (*alloc_gate)() = nullptr;
void (*map_gate)() = nullptr;
int (*map_check)(uint64_t, size_t) = nullptr;
int file(size_t size) {
    int fd = memfd_create("phoenix-audit", MFD_CLOEXEC);
    CHECK(fd >= 0 && ftruncate(fd, size) == 0);
    return fd;
}
phxfs_io_req_t req(int fd, int d, void *buf, size_t n, off_t fo, off_t bo) {
    return {fd, d, buf, bo, n, fo, -999};
}
void host_open(int d) {
    setenv("PHXFS_MODE", "host", 1);
    CHECK(phxfs_open(d) == 0);
}
void check_idle(int d) {
    auto *pb = &mbuffer[d];
    CHECK(pb->active_ops == 0);
    for (auto *node = pb->head; node; node = node->next) CHECK(node->refcount == 0);
}
}
static bool fake_fd[4096]{};
struct Mapping { uintptr_t device; void *host; size_t size; };
static Mapping mapping[128]{};
static pthread_mutex_t mapping_lock = PTHREAD_MUTEX_INITIALIZER;
static void *translate(const void *p, size_t n) {
    void *r = const_cast<void *>(p);
    pthread_mutex_lock(&mapping_lock);
    for (auto &m : mapping) {
        uintptr_t off = (uintptr_t)p - m.device;
        if (m.host && off < m.size && n <= m.size - off) { r = (char *)m.host + off; break; }
    }
    pthread_mutex_unlock(&mapping_lock);
    return r;
}
static int allocate(size_t n, void **p) { return posix_memalign(p, 65536, n) == 0 ? 0 : -ENOMEM; }
static int device_alloc(int d, size_t n, void **p) {
    if (d < 0 || d >= mock::device_count) return -ENODEV;
    int rc = allocate(n, p); if (!rc) ++mock::device_live; return rc;
}
static void device_free(void *p) { if (p) { --mock::device_live; free(p); } }
static int host_alloc(size_t n, void **p) {
    if (mock::alloc_gate) mock::alloc_gate();
    if (mock::host_error) return -mock::host_error.load();
    int rc = allocate(n, p); if (!rc) ++mock::host_live; return rc;
}
static void host_free(void *p) { if (p) { --mock::host_live; free(p); } }
static int copy(void *d, const void *s, size_t n) { memcpy(translate(d,n), translate(s,n), n); return 0; }
static int set_device(int d) { return d >= 0 && d < mock::device_count ? 0 : -ENODEV; }
static int find(int d) { return d >= 0 && d < PHXFS_MAX_DEVICES ? mock::find_result[d] : -1; }
static int launch(void *, void (*fn)(void *), void *p) { ++mock::launches; fn(p); return 0; }
// Compile identical tests against headers before and after this optional hook.
template <class T> static auto validator(T &o, int)
    -> decltype(o.validate_device = set_device, void()) { o.validate_device = set_device; }
template <class T> static void validator(T &, long) {}
static devconn_ops ops{};
extern "C" {
devconn_ops *devconn = &ops;
int devconn_init() {
    validator(ops,0);
    ops.name = "audit-cpu"; ops.page_size = 65536; ops.find_device = find;
    ops.mem_alloc = device_alloc; ops.mem_free = device_free; ops.memcpy_dtod = copy;
    ops.host_alloc = host_alloc; ops.host_free = host_free; ops.set_device = set_device;
    ops.memcpy_h2d = copy; ops.memcpy_d2h = copy; ops.launch_host_func = launch;
    return 0;
}
#ifdef AUDIT_LEGACY_PROBE
int phx_connector_probe_alloc(int d, size_t n, void **p) { return device_alloc(d,n,p); }
void phx_connector_probe_free(void *p) { device_free(p); }
#endif
int __real_open(const char *, int, ...);
int __wrap_open(const char *path, int flags, ...) {
    if (!strncmp(path, "/dev/phxfs_dev", 14)) {
        if (mock::open_error) { errno = mock::open_error; return -1; }
        int fd = mock::file(0); CHECK(fd < 4096); fake_fd[fd] = true; return fd;
    }
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }
    return __real_open(path, flags, mode);
}
// ifstream calls libc through libstdc++; symbol interposition covers that call.
FILE *fopen64(const char *path, const char *mode) {
    if (!strncmp(path, "/sys/class/phxfs-generic/phxfs_dev", 32)) {
        FILE *f = tmpfile(); if (f) { fprintf(f, "%d\n", mock::kernel_mode); rewind(f); } return f;
    }
    static auto real = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "fopen64");
    return real(path, mode);
}
void *__real_mmap(void *, size_t, int, int, int, off_t);
void *__wrap_mmap(void *p, size_t n, int prot, int flags, int fd, off_t off) {
    if (fd >= 0 && fd < 4096 && fake_fd[fd])
        return __real_mmap(p,n,prot,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    return __real_mmap(p,n,prot,flags,fd,off);
}
int __real_munmap(void *, size_t);
int __wrap_munmap(void *p, size_t n) {
    if (int err = mock::munmap_error.exchange(0)) { errno = err; return -1; }
    pthread_mutex_lock(&mapping_lock);
    for (auto &m : mapping) if (m.host == p) m = {};
    pthread_mutex_unlock(&mapping_lock);
    return __real_munmap(p,n);
}
int __real_ioctl(int, unsigned long, ...);
int __wrap_ioctl(int fd, unsigned long cmd, ...) {
    va_list ap; va_start(ap,cmd); void *arg = va_arg(ap,void *); va_end(ap);
    if (fd < 0 || fd >= 4096 || !fake_fd[fd]) return __real_ioctl(fd,cmd,arg);
    if (cmd == PHXFS_IOCTL_MAP) {
        ++mock::maps; if (mock::map_gate) mock::map_gate();
        auto *p = static_cast<phxfs_ioctl_para_t *>(arg);
        if (mock::map_check) {
            int err = mock::map_check(p->map_param.n_vaddr, p->map_param.n_size);
            if (err) { errno = err; return -1; }
        }
        if (mock::map_error) { errno = mock::map_error; return -1; }
        pthread_mutex_lock(&mapping_lock);
        bool found = false;
        for (auto &m : mapping) if (!m.host) {
            m = {p->map_param.n_vaddr, (void *)p->map_param.c_vaddr, p->map_param.c_size}; found=true; break;
        }
        pthread_mutex_unlock(&mapping_lock); CHECK(found);
    } else if (cmd == PHXFS_IOCTL_UNMAP) {
        ++mock::unmaps;
        if (mock::unmap_error) { errno = mock::unmap_error; return -1; }
    }
    return 0;
}
int __real_close(int);
int __wrap_close(int fd) { if (fd >= 0 && fd < 4096) fake_fd[fd]=false; return __real_close(fd); }
ssize_t __real_pread(int, void *, size_t, off_t);
ssize_t __real_pwrite(int, const void *, size_t, off_t);
ssize_t __wrap_pread(int fd, void *p, size_t n, off_t off) {
    if (fd == mock::fail_fd && off >= mock::fail_from && off < mock::fail_until) { errno=EIO; return -1; }
    return __real_pread(fd,p,n,off);
}
ssize_t __wrap_pwrite(int fd, const void *p, size_t n, off_t off) {
    if (fd == mock::fail_fd && off >= mock::fail_from && off < mock::fail_until) { errno=EIO; return -1; }
    return __real_pwrite(fd,p,n,off);
}
int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    return mock::fail_threads ? EAGAIN : __real_pthread_create(t,a,fn,arg);
}
}
