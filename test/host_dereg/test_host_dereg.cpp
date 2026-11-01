// Same production lifecycle and test for CPU mocks and the CUDA connector.
// memfd uses RAM only: 4 KiB for lifetime tests, 24 MiB plus guards for merged tails.
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include "phoenix.h"
#include "phx_internal.h"

static std::mutex lock;
static std::condition_variable changed;
static bool copy_entered, copy_done, hold_copy, wait_entered, dereg_done;
static int copies_entered;
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
        copies_entered++;
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
    if (!synchronous && !bad_fd)
        check(phxfs_batch_cancel(batch) == 0, "cancel keeps refs until wait/destroy");

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

static void cancel_scenario(int fd, void *gpu, bool write, bool destroy) {
    printf("CASE cancel write=%d destroy=%d\n", write, destroy);
    void *target = nullptr;
    require(phxfs_regmem(0, gpu, bytes, &target) == 0, "register for cancellation");
    {
        std::lock_guard<std::mutex> guard(lock);
        copies_entered = 0;
        hold_copy = true;
    }
    phxfs_io_req_t blockers[4]{};
    phxfs_batch_t *handles[4]{};
    for (int i = 0; i < 4; i++) {
        blockers[i].fd = fd;
        blockers[i].device_id = 0;
        blockers[i].buf = gpu;
        blockers[i].nbytes = bytes;
        handles[i] = phxfs_batch_submit_read(&blockers[i], 1);
        require(handles[i] != nullptr, "submit HOST worker blocker");
    }
    await([] { return copies_entered == 4; });
    phxfs_io_req_t reqs[2]{};
    for (int i = 0; i < 2; i++) {
        reqs[i].fd = fd;
        reqs[i].device_id = 0;
        reqs[i].buf = gpu;
        reqs[i].buf_offset = i * (bytes / 2);
        reqs[i].f_offset = i * (bytes / 2);
        reqs[i].nbytes = bytes / 2;
    }
    auto *batch = write ? phxfs_batch_submit_write(reqs, 2)
                        : phxfs_batch_submit_read(reqs, 2);
    require(batch != nullptr, "submit merged HOST requests");
    check(phxfs_batch_cancel(batch) == 0, "cancel pending HOST work");
    check(phxfs_batch_cancel(batch) == 0, "repeat HOST cancellation");
    if (destroy) {
        check(phxfs_batch_destroy(batch) == 0, "destroy canceled HOST batch");
        check(reqs[0].result == 0 && reqs[1].result == 0, "HOST destroy skips result copy-back");
    } else {
        check(phxfs_batch_wait(batch) == 2, "HOST cancellation counts original requests");
        check(reqs[0].result == -ECANCELED && reqs[1].result == -ECANCELED,
              "merged HOST requests report cancellation");
    }
    check(phxfs_batch_cancel(handles[0]) == 0, "cancel claimed HOST work without waiting");
    {
        std::lock_guard<std::mutex> guard(lock);
        check(copies_entered == 4, "canceled HOST work starts no copies");
        hold_copy = false;
        changed.notify_all();
    }
    for (int i = 0; i < 4; i++) {
        check(phxfs_batch_wait(handles[i]) == 0, "claimed HOST work drains normally");
        check(blockers[i].result == (ssize_t)bytes, "claimed HOST result preserved");
    }
    check(phxfs_deregmem(0, gpu, bytes) == 0, "cancellation releases mapping refs");
}

static void cancel_staging_scenario(int fd, void *gpu) {
    mbuffer[0].map_mode = PHX_MAP_MODE_STAGING;
    phxfs_io_req_t req{};
    req.fd = fd;
    req.device_id = 0;
    req.buf = gpu;
    req.nbytes = bytes;
    req.result = -1;
    auto *batch = phxfs_batch_submit_read(&req, 1);
    require(batch != nullptr, "submit deferred STAGING batch");
    check(phxfs_batch_cancel(batch) == 0, "cancel deferred STAGING batch");
    check(phxfs_batch_cancel(batch) == 0, "repeat STAGING cancellation");
    check(req.result == -1, "STAGING cancel defers results to wait");
    check(phxfs_batch_wait(batch) == 1 && req.result == -ECANCELED,
          "STAGING wait skips transfer and reports cancellation");
    req.result = -1;
    batch = phxfs_batch_submit_write(&req, 1);
    require(batch != nullptr, "submit deferred STAGING write");
    check(phxfs_batch_cancel(batch) == 0 && phxfs_batch_destroy(batch) == 0,
          "cancel/destroy deferred STAGING write");
    check(req.result == -1, "STAGING destroy does not copy results");
    mbuffer[0].map_mode = PHX_MAP_MODE_HOST;
}

// A=20 MiB and B=4 MiB produce a 16 MiB slice and an 8 MiB merged tail.
// Use the existing HOST connector and copy gates, not device_id=-1 CPU I/O.
static void merged_tail(bool write, bool async) {
    constexpr size_t mib = 1024 * 1024, first = 20 * mib, second = 4 * mib;
    constexpr size_t guard = 4096, total = first + second + 2 * guard;
    std::vector<unsigned char> expected(total, 0xa5), initial(total, 0xa5), actual(total);
    for (size_t i = 0; i < first + second; ++i) {
        expected[guard + i] = (unsigned char)(1 + (i % 113) + 23 * (i / (4 * mib)));
        initial[guard + i] = 0;
    }
    void *gpu = nullptr, *target = nullptr;
    require(devconn->mem_alloc(0, total, &gpu) == 0, "allocate merged-tail buffer");
    require(phxfs_regmem(0, gpu, total, &target) == 0, "register merged-tail buffer");
    int fd = memfd_create("phoenix-host-merged-tail", MFD_CLOEXEC);
    require(fd >= 0, "create merged-tail memfd");
    const auto &file_seed = write ? initial : expected;
    const auto &gpu_seed = write ? expected : initial;
    require(pwrite(fd, file_seed.data(), total, 0) == (ssize_t)total, "reset file including guards");
    real_h2d = devconn->memcpy_h2d;
    real_d2h = devconn->memcpy_d2h;
    require(real_h2d(gpu, gpu_seed.data(), total) == 0, "reset device including guards");
    copies_entered = 0;
    hold_copy = false;
    devconn->memcpy_h2d = h2d;
    devconn->memcpy_d2h = d2h;
    phxfs_io_req_t reqs[2]{};
    for (int i = 0; i < 2; ++i) {
        reqs[i].fd = fd;
        reqs[i].device_id = 0;
        reqs[i].buf = gpu;
        reqs[i].buf_offset = reqs[i].f_offset = guard + (i ? first : 0);
        reqs[i].nbytes = i ? second : first;
    }
    int rc;
    if (async) {
        auto *h = write ? phxfs_batch_submit_write(reqs, 2) : phxfs_batch_submit_read(reqs, 2);
        require(h != nullptr, "submit merged-tail HOST batch");
        rc = phxfs_batch_wait(h);
    } else {
        rc = write ? phxfs_write_batch(reqs, 2) : phxfs_read_batch(reqs, 2);
    }
    printf("merged-tail rc=%d A=%zd B=%zd copies=%d\n", rc, reqs[0].result, reqs[1].result, copies_entered);
    check(rc == 0, "merged-tail batch succeeds");
    check(reqs[0].result == (ssize_t)first, "A reports exactly 20 MiB");
    check(reqs[1].result == (ssize_t)second, "B reports exactly 4 MiB");
    check(copies_entered == 2, "HOST callbacks execute two coalesced slices");
    require(real_d2h(actual.data(), gpu, total) == 0, "read back device");
    check(actual == expected, "device data and both guards");
    require(pread(fd, actual.data(), total, 0) == (ssize_t)total, "read back file");
    check(actual == expected, "file data and both guards");
    devconn->memcpy_h2d = real_h2d;
    devconn->memcpy_d2h = real_d2h;
    require(phxfs_deregmem(0, gpu, total) == 0, "deregister merged-tail buffer");
    devconn->mem_free(gpu);
    close(fd);
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
    if (argc == 5 && strcmp(argv[2], "merged-tail") == 0) {
        merged_tail(strcmp(argv[3], "write") == 0, strcmp(argv[4], "async") == 0);
        check(phxfs_close(0) == 0, "close merged-tail device");
        return failures ? 1 : 0;
    }
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
    cancel_scenario(fd, gpu, false, false);
    cancel_scenario(fd, gpu, true, false);
    cancel_scenario(fd, gpu, true, true);
    cancel_staging_scenario(fd, gpu);
    devconn->memcpy_h2d = real_h2d;
    devconn->memcpy_d2h = real_d2h;
    close(fd);
    devconn->mem_free(gpu);
    check(phxfs_close(0) == 0, "close after all handles/registrations consumed");
    printf("RESULT failures=%d\n", failures);
    return failures ? 1 : 0;
}
