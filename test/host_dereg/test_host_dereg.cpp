// Same production lifecycle and test for CPU mocks and the CUDA connector.
// memfd stores only 4 KiB in memory: no filesystem/disk benchmark or module.
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
#include "phoenix.h"
#include "phx_internal.h"

static std::mutex lock;
static std::condition_variable changed;
static bool copy_entered, copy_done, hold_copy, wait_entered, dereg_done;
static thread_local bool dereg_thread = false;
static int failures = 0;
static int (*real_h2d)(void *, const void *, size_t);
static int (*real_d2h)(void *, const void *, size_t);
static constexpr size_t bytes = 4096;

extern "C" int __real_pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
extern "C" int __wrap_pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    if (dereg_thread) {
        std::lock_guard<std::mutex> guard(lock);
        wait_entered = true;
        changed.notify_all();
    }
    return __real_pthread_cond_wait(c, m);
}

static void check(bool ok, const char *label) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++failures;
}
static void require(bool ok, const char *label) {
    check(ok, label);
    if (!ok) std::exit(2);
}
template<class Predicate> static void await(Predicate predicate) {
    std::unique_lock<std::mutex> guard(lock);
    if (!changed.wait_for(guard, std::chrono::seconds(10), predicate)) {
        fprintf(stderr, "FAIL: synchronization timeout\n");
        std::_Exit(3);
    }
}
static int gated_copy(void *d, const void *s, size_t n,
                      int (*copy)(void *, const void *, size_t)) {
    {
        std::unique_lock<std::mutex> guard(lock);
        copy_entered = true;
        changed.notify_all();
        changed.wait(guard, [] { return !hold_copy; });
    }
    int rc = copy(d, s, n);
    {
        std::lock_guard<std::mutex> guard(lock);
        copy_done = true;
        changed.notify_all();
    }
    return rc;
}
static int h2d(void *d, const void *s, size_t n) { return gated_copy(d, s, n, real_h2d); }
static int d2h(void *d, const void *s, size_t n) { return gated_copy(d, s, n, real_d2h); }

static void scenario(int fd, void *gpu, bool write, bool destroy,
                     bool completed_copy, bool bad_fd, bool synchronous) {
    printf("CASE write=%d destroy=%d completed_copy=%d bad_fd=%d sync=%d\n",
           write, destroy, completed_copy, bad_fd, synchronous);
    void *target = nullptr;
    require(phxfs_regmem(0, gpu, bytes, &target) == 0, "register");
    require(phxfs_regmem(0, gpu, bytes, &target) == 0, "duplicate registration");
    {
        std::lock_guard<std::mutex> guard(lock);
        copy_entered = copy_done = wait_entered = dereg_done = false;
        hold_copy = !completed_copy && !bad_fd;
    }
    phxfs_io_req_t req{};
    req.fd = bad_fd ? -1 : fd;
    req.device_id = 0;
    req.buf = gpu;
    req.nbytes = bytes;
    phxfs_batch_t *batch = nullptr;
    std::thread sync_worker;
    ssize_t sync_result = -1;
    if (synchronous) {
        sync_worker = std::thread([&] {
            devconn->set_device(0);
            sync_result = phxfs_read(fd, 0, gpu, 0, bytes, 0);
        });
    } else {
        batch = write ? phxfs_batch_submit_write(&req, 1)
                      : phxfs_batch_submit_read(&req, 1);
        require(batch != nullptr, "submit");
    }
    if (!bad_fd)
        await([&] { return completed_copy ? copy_done : copy_entered; });

    require(phxfs_deregmem(0, gpu, bytes) == 0,
            "non-final registration reference drops without blocking");
    int dereg_result = -1;
    std::thread dereg([&] {
        dereg_thread = true;
        dereg_result = phxfs_deregmem(0, gpu, bytes);
        dereg_thread = false;
        std::lock_guard<std::mutex> guard(lock);
        dereg_done = true;
        changed.notify_all();
    });
    // The wrapper observes an actual condition wait; the baseline instead
    // returns early. Neither outcome depends on sleeping or worker speed.
    await([] { return wait_entered || dereg_done; });
    {
        std::lock_guard<std::mutex> guard(lock);
        check(wait_entered && !dereg_done, "last deregistration waits for held I/O refs");
        hold_copy = false;
        changed.notify_all();
    }
    // Keep going on the baseline: ASan must see the stale-node dereference
    // when wait/destroy (or synchronous completion) releases its reference.
    if (synchronous) {
        sync_worker.join();
        check(sync_result == (ssize_t)bytes, "synchronous read completed");
    } else if (destroy) {
        check(phxfs_batch_destroy(batch) == 0, "destroy releases refs");
    } else {
        check(phxfs_batch_wait(batch) == (bad_fd ? 1 : 0), "wait reports result and releases refs");
        check(bad_fd ? req.result < 0 : req.result == (ssize_t)bytes, "per-request result");
    }
    dereg.join();
    check(dereg_result == 0, "last deregistration completes");
    check(phxfs_deregmem(0, gpu, bytes) != 0, "registration removed");
    require(phxfs_regmem(0, gpu, bytes, &target) == 0, "re-register after completed deregistration");
    require(phxfs_deregmem(0, gpu, bytes) == 0, "idle deregistration");
    if (!bad_fd) {
        unsigned char actual[bytes], expected[bytes];
        memset(expected, 0x5a, bytes);
        require(real_d2h(actual, gpu, bytes) == 0, "copy back for verification");
        check(memcmp(actual, expected, bytes) == 0, "GPU/mock buffer payload preserved");
        require(pread(fd, actual, bytes, 0) == (ssize_t)bytes, "read memfd for verification");
        check(memcmp(actual, expected, bytes) == 0, "file payload preserved");
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    alarm(40);
    const char *engine = argc > 1 ? argv[1] : "sync";
    setenv("PHXFS_MODE", "host_staging", 1);
    setenv("PHXFS_IO_ENGINE", engine, 1);
    require(strcmp(phxfs_io_engine_name(), engine) == 0, "requested engine active (no silent fallback)");
    require(devconn->set_device(0) == 0, "select allocated device ordinal 0");
    require(phxfs_open(0) == 0 && phxfs_get_map_mode(0) == PHX_MAP_MODE_HOST, "open HOST");
    void *gpu = nullptr;
    require(devconn->mem_alloc(0, bytes, &gpu) == 0, "allocate device/mock buffer");
    unsigned char seed[bytes];
    memset(seed, 0x5a, bytes);
    real_h2d = devconn->memcpy_h2d;
    real_d2h = devconn->memcpy_d2h;
    require(real_h2d(gpu, seed, bytes) == 0, "initialize buffer");
    int fd = memfd_create("phoenix-host-dereg", MFD_CLOEXEC);
    require(fd >= 0 && pwrite(fd, seed, bytes, 0) == (ssize_t)bytes, "initialize memory-only file");
    devconn->memcpy_h2d = h2d;
    devconn->memcpy_d2h = d2h;
    scenario(fd, gpu, false, false, false, false, false);
    scenario(fd, gpu, false, true, false, false, false);
    scenario(fd, gpu, true, false, false, false, false);
    scenario(fd, gpu, false, false, true, false, false);
    scenario(fd, gpu, false, false, false, true, false);
    scenario(fd, gpu, false, false, false, false, true);
    devconn->memcpy_h2d = real_h2d;
    devconn->memcpy_d2h = real_d2h;
    close(fd);
    devconn->mem_free(gpu);
    check(phxfs_close(0) == 0, "close after all handles/registrations consumed");
    printf("RESULT failures=%d\n", failures);
    return failures ? 1 : 0;
}
