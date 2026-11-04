// Isolated CPU-only fault injection. Reuses the existing HOST connector/core;
// the new override is never linked into test_host_dereg or the production library.
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sys/mman.h>
#include <unistd.h>
#include "phoenix.h"
#include "phx_internal.h"

static thread_local bool armed, fired;
static thread_local int held_ops, held_maps;
static std::atomic<int> copies{0};
static int (*real_copy)(void *, const void *, size_t);
static int counted_copy(void *d, const void *s, size_t n) {
    ++copies;
    return real_copy(d, s, n);
}

void *operator new(size_t n) {
    if (armed) {
        // Fail the first allocation after this registration and its device
        // are retained; no allocation-size or ordinal-count assumptions.
        auto *pb = &mbuffer[0];
        pthread_mutex_lock(&pb->lock);
        held_ops = pb->active_ops;
        held_maps = pb->head ? pb->head->refcount : 0;
        pthread_mutex_unlock(&pb->lock);
        if (held_ops > 0 && held_maps > 0) {
            armed = false;
            fired = true;
            throw std::bad_alloc();
        }
    }
    void *p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, size_t) noexcept { std::free(p); }
void *operator new(size_t n, const std::nothrow_t &) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void operator delete(void *p, const std::nothrow_t &) noexcept { std::free(p); }

static bool idle() {
    auto *pb = &mbuffer[0];
    pthread_mutex_lock(&pb->lock);
    bool ok = pb->active_ops == 0 && pb->head && pb->head->refcount == 0;
    pthread_mutex_unlock(&pb->lock);
    return ok;
}
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main(int argc, char **argv) {
    alarm(40);
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool write = argc > 1 && std::strcmp(argv[1], "write") == 0;
    const bool async = argc > 2 && std::strcmp(argv[2], "async") == 0;
    setenv("PHXFS_MODE", "host_staging", 1);
    setenv("PHXFS_IO_ENGINE", "sync", 1);
    CHECK(phxfs_open(0) == 0 && phxfs_get_map_mode(0) == PHX_MAP_MODE_HOST);
    CHECK(std::strcmp(phxfs_io_engine_name(), "sync") == 0);
    constexpr size_t guard = 4096, payload = 4096, total = payload + 2 * guard;
    unsigned char expected[total], initial[total], actual[total];
    std::memset(expected, 0xa5, total);
    std::memset(initial, 0xa5, total);
    for (size_t i = 0; i < payload; ++i) {
        expected[guard + i] = (unsigned char)(1 + i % 113);
        initial[guard + i] = 0;
    }
    void *gpu = nullptr, *target = nullptr;
    CHECK(devconn->mem_alloc(0, total, &gpu) == 0);
    CHECK(phxfs_regmem(0, gpu, total, &target) == 0 && idle());
    int fd = memfd_create("phoenix-host-allocation", MFD_CLOEXEC);
    CHECK(fd >= 0);
    const auto *file_seed = write ? initial : expected;
    const auto *gpu_seed = write ? expected : initial;
    CHECK(pwrite(fd, file_seed, total, 0) == (ssize_t)total);
    CHECK(devconn->memcpy_h2d(gpu, gpu_seed, total) == 0);
    real_copy = devconn->memcpy_h2d; // Existing CPU mock uses the same copy both ways.
    devconn->memcpy_h2d = devconn->memcpy_d2h = counted_copy;
    phxfs_io_req_t req{};
    req.fd = fd; req.device_id = 0; req.buf = gpu;
    req.buf_offset = req.f_offset = guard; req.nbytes = payload;
    auto submit = [&] { return write ? phxfs_batch_submit_write(&req, 1) : phxfs_batch_submit_read(&req, 1); };
    auto run = [&] { return write ? phxfs_write_batch(&req, 1) : phxfs_read_batch(&req, 1); };
    int rc = 0, error = 0;
    bool escaped = false;
    phxfs_batch_t *h = nullptr;
    armed = true;
    try { if (async) h = submit(); else rc = run(); error = errno; }
    catch (const std::bad_alloc &) { escaped = true; }
    armed = false;
    bool released = idle();
    printf("failure write=%d async=%d fired=%d held_ops=%d held_maps=%d escaped=%d rc=%d errno=%d idle=%d copies=%d\n",
           write, async, fired, held_ops, held_maps, escaped, rc, error, released, copies.load());
    CHECK(fired && held_ops > 0 && held_maps > 0);
    CHECK(!escaped && released && (async ? !h && error == ENOMEM : rc == -ENOMEM));
    CHECK(copies == 0); // This is preparation failure, not a worker partial failure.
    CHECK(std::memcmp(gpu, gpu_seed, total) == 0);
    CHECK(pread(fd, actual, total, 0) == (ssize_t)total && std::memcmp(actual, file_seed, total) == 0);
    if (async) { h = submit(); CHECK(h); rc = phxfs_batch_wait(h); }
    else rc = run();
    CHECK(rc == 0 && req.result == (ssize_t)payload && copies > 0 && idle());
    CHECK(std::memcmp(gpu, expected, total) == 0);
    CHECK(pread(fd, actual, total, 0) == (ssize_t)total && std::memcmp(actual, expected, total) == 0);
    CHECK(phxfs_deregmem(0, gpu, total) == 0);
    devconn->mem_free(gpu);
    CHECK(phxfs_close(0) == 0);
    close(fd);
    puts("PASS retained-reference failure, no I/O, cleanup and successful HOST retry with guards");
}
