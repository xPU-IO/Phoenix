// SPDX-License-Identifier: Apache-2.0
// CPU-only unit tests for the asynchronous batch API: submit / wait /
// destroy lifecycle, per-request result copy-back, failure isolation and
// worker-pool queueing (pipelined submits). It forces the sync engine
// internally, so it needs neither a GPU nor the phxfs kernel module.

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "phoenix.h"
#include "../libphoenix/io_engine/io_engine.h"

namespace {

int tests_run;
int tests_failed;

#define CHECK(cond, fmt, ...) do {                                      \
    tests_run++;                                                        \
    if (cond) {                                                        \
        std::printf("  [PASS] " fmt "\n", ##__VA_ARGS__);             \
    } else {                                                           \
        tests_failed++;                                                \
        std::printf("  [FAIL] " fmt "\n", ##__VA_ARGS__);             \
    }                                                                  \
} while (0)

constexpr int kRequests = 4;
constexpr size_t kStride = 8192;
constexpr size_t kPrefix = 37;
constexpr off_t kPipeOffset = 64 * 1024;

int make_temp_file() {
    const char *tmpdir = std::getenv("TMPDIR");
    std::string path = std::string(tmpdir ? tmpdir : "/tmp") +
                       "/phx_batch_async_XXXXXX";
    std::vector<char> writable(path.begin(), path.end());
    writable.push_back('\0');
    int fd = mkstemp(writable.data());
    if (fd >= 0)
        unlink(writable.data());
    return fd;
}

void fill_pattern(std::vector<uint8_t> &buf, int request, size_t len) {
    for (size_t i = 0; i < len; i++)
        buf[kPrefix + i] = static_cast<uint8_t>((request * 67 + i) & 0xff);
}

struct WorkerGate {
    std::mutex mutex;
    std::condition_variable cv;
    int entered = 0;
    bool released = false;
};

int gate_prepare(phxfs_io_op_req *op, phxfs_io_op,
                 phxfs_io_worker_ctx *, int) {
    auto *gate = static_cast<WorkerGate *>(op->private_data);
    std::unique_lock<std::mutex> guard(gate->mutex);
    gate->entered++;
    gate->cv.notify_all();
    gate->cv.wait(guard, [&] { return gate->released; });
    return 0;
}

void wait_for_workers(WorkerGate &gate) {
    std::unique_lock<std::mutex> guard(gate.mutex);
    if (!gate.cv.wait_for(guard, std::chrono::seconds(10), [&] {
            return gate.entered == PHXFS_POOL_THREADS;
        })) {
        std::fprintf(stderr, "worker gate timed out\n");
        std::_Exit(2);
    }
}

void release_workers(WorkerGate &gate) {
    std::lock_guard<std::mutex> guard(gate.mutex);
    gate.released = true;
    gate.cv.notify_all();
}

void test_cancel(int fd) {
    WorkerGate gate;
    uint8_t scratch[PHXFS_POOL_THREADS]{};
    phxfs_io_op_req blockers[PHXFS_POOL_THREADS]{};
    for (int i = 0; i < PHXFS_POOL_THREADS; i++) {
        blockers[i].fd = fd;
        blockers[i].host_addr = &scratch[i];
        blockers[i].nbytes = 1;
        blockers[i].prepare = gate_prepare;
        blockers[i].private_data = &gate;
    }
    auto *block = phxfs_pool_submit(blockers, PHXFS_POOL_THREADS,
                                    PHXFS_IO_READ, true);
    CHECK(block != nullptr, "submit worker blockers");
    if (!block)
        return;
    wait_for_workers(gate);

    uint8_t buffer[32];
    std::memset(buffer, 0xa5, sizeof(buffer));
    phxfs_io_req_t reqs[2]{};
    reqs[0].fd = fd;
    reqs[0].device_id = -1;
    reqs[0].buf = buffer;
    reqs[0].nbytes = sizeof(buffer);
    reqs[0].result = -1;
    reqs[1] = reqs[0];
    reqs[1].buf_offset = -1;
    auto *h = phxfs_batch_submit_read(reqs, 2);
    CHECK(h && phxfs_batch_cancel(h) == 0, "cancel queued read");
    CHECK(h && phxfs_batch_cancel(h) == 0, "repeated cancel is idempotent");
    CHECK(reqs[0].result == -1, "cancel defers result copy-back to wait");
    CHECK(h && phxfs_batch_wait(h) == 2, "wait counts canceled and invalid requests");
    CHECK(reqs[0].result == -ECANCELED && reqs[1].result == -EFAULT,
          "cancel preserves submission-time errors");
    bool unchanged = true;
    for (uint8_t byte : buffer)
        unchanged = unchanged && byte == 0xa5;
    CHECK(unchanged, "canceled read leaves buffer untouched");

    off_t end = lseek(fd, 0, SEEK_END);
    reqs[0].f_offset = end + 4096;
    h = phxfs_batch_submit_write(reqs, 1);
    CHECK(h && phxfs_batch_cancel(h) == 0, "cancel queued write");
    CHECK(h && phxfs_batch_wait(h) == 1 && reqs[0].result == -ECANCELED,
          "canceled write reports ECANCELED");
    CHECK(lseek(fd, 0, SEEK_END) == end, "canceled write does not extend file");

    bool recycled = true;
    for (int i = 0; i < 32; i++) {
        reqs[0].result = -1;
        h = phxfs_batch_submit_write(reqs, 1);
        if (!h) {
            recycled = false;
            break;
        }
        recycled = phxfs_batch_cancel(h) == 0 && recycled;
        recycled = phxfs_batch_destroy(h) == 0 && recycled;
        recycled = reqs[0].result == -1 && recycled;
    }
    CHECK(recycled, "cancel/destroy recycles queue slots without copying results");
    release_workers(gate);
    CHECK(phxfs_pool_cancel(block) == 0, "cancel with all work already claimed");
    CHECK(phxfs_pool_wait(block) == 0, "claimed work completes normally");

    WorkerGate partial_gate;
    constexpr int n = 32;
    uint8_t partial_buf[n]{};
    phxfs_io_op_req ops[n]{};
    for (int i = 0; i < n; i++) {
        ops[i].fd = fd;
        ops[i].host_addr = &partial_buf[i];
        ops[i].nbytes = 1;
        ops[i].prepare = gate_prepare;
        ops[i].private_data = &partial_gate;
    }
    auto *partial = phxfs_pool_submit(ops, n, PHXFS_IO_READ, true);
    CHECK(partial != nullptr, "submit partially claimed job");
    if (!partial)
        return;
    wait_for_workers(partial_gate);
    CHECK(phxfs_pool_cancel(partial) == 0, "cancel partially claimed job without waiting");
    release_workers(partial_gate);
    int expected = n - PHXFS_POOL_THREADS * PHXFS_HOST_STAGE_SLOTS;
    CHECK(phxfs_pool_wait(partial) == expected, "partial cancellation failure count");
    bool correct = true;
    for (int i = 0; i < n; i++)
        correct = correct && ops[i].result == (i < n - expected ? 1 : -ECANCELED);
    CHECK(correct, "claimed ops complete; unclaimed ops are canceled");
}

void test_handle_lifecycle(int fd) {
    uint8_t buffer[32]{};
    struct LifecycleCase {
        const char *name;
        int fd;
        off_t buf_offset;
        size_t nbytes;
        off_t f_offset;
        ssize_t result;
    };
    const LifecycleCase cases[] = {
        {"success", fd, 0, sizeof(buffer), 0, sizeof(buffer)},
        {"bad fd", -1, 0, sizeof(buffer), 0, -EBADF},
        {"unresolved buffer", fd, -1, sizeof(buffer), 0, -EFAULT},
        {"short read", fd, 0, sizeof(buffer), lseek(fd, 0, SEEK_END) - 8, 8},
        {"zero length", fd, 0, 0, 0, 0},
    };
    for (const auto &entry : cases) {
        phxfs_io_req_t req{};
        req.fd = entry.fd;
        req.device_id = -1;
        req.buf = buffer;
        req.buf_offset = entry.buf_offset;
        req.nbytes = entry.nbytes;
        req.f_offset = entry.f_offset;
        bool waited = true, destroyed = true;
        for (int r = 0; r < 256 && waited && destroyed; r++) {
            req.result = -1;
            phxfs_batch_t *h = phxfs_batch_submit_read(&req, 1);
            if (!h) {
                waited = false;
                break;
            }
            int rc = phxfs_batch_wait(h);
            h = nullptr;
            int expected = entry.result == (ssize_t)entry.nbytes ? 0 : 1;
            waited = rc == expected && req.result == entry.result;

            req.result = -1;
            h = phxfs_batch_submit_read(&req, 1);
            if (!h) {
                destroyed = false;
                break;
            }
            rc = phxfs_batch_destroy(h);
            h = nullptr;
            ssize_t expected_result = entry.result == -EFAULT ? -EFAULT : -1;
            destroyed = rc == 0 && req.result == expected_result;
        }
        CHECK(waited, "repeated submit/wait lifecycle: %s", entry.name);
        CHECK(destroyed, "repeated submit/destroy without copy-back: %s", entry.name);
    }
    phxfs_batch_t *h = phxfs_batch_submit_write(nullptr, 0);
    CHECK(h != nullptr, "empty batch for destroy returns a handle");
    if (h) {
        int rc = phxfs_batch_destroy(h);
        h = nullptr;
        CHECK(rc == 0, "empty batch destroy releases the handle");
    }
}

void test_independent_handles(int fd) {
    constexpr int nthreads = 4;
    bool completed[nthreads]{};
    std::thread workers[nthreads];
    for (int i = 0; i < nthreads; i++) {
        workers[i] = std::thread([&, i] {
            uint8_t buffer[32]{};
            phxfs_io_req_t req{};
            req.fd = fd;
            req.device_id = -1;
            req.buf = buffer;
            req.nbytes = sizeof(buffer);
            for (int r = 0; r < 256; r++) {
                req.result = -1;
                phxfs_batch_t *h = phxfs_batch_submit_read(&req, 1);
                if (!h)
                    return;
                int rc = phxfs_batch_wait(h);
                h = nullptr;
                if (rc != 0 || req.result != (ssize_t)sizeof(buffer))
                    return;
            }
            completed[i] = true;
        });
    }
    for (auto &worker : workers)
        worker.join();
    bool ok = true;
    for (bool done : completed)
        ok = ok && done;
    CHECK(ok, "independent handles complete concurrently with separate request storage");
}

}  // namespace

int main() {
    setenv("PHXFS_IO_ENGINE", "sync", 1);

    std::printf("=== Phoenix asynchronous batch API tests ===\n");
    CHECK(std::strcmp(phxfs_io_engine_name(), "sync") == 0,
          "forced sync engine selected");

    /* Empty batches still yield a consumable handle. */
    phxfs_batch_t *h = phxfs_batch_submit_read(nullptr, 0);
    CHECK(h != nullptr, "empty read batch returns a handle");
    CHECK(h && phxfs_batch_cancel(h) == 0, "cancel empty batch");
    CHECK(h && phxfs_batch_cancel(h) == 0, "repeat cancel empty batch");
    CHECK(h && phxfs_batch_wait(h) == 0, "empty read batch wait");
    h = phxfs_batch_submit_write(nullptr, 0);
    CHECK(h != nullptr, "empty write batch returns a handle");
    CHECK(h && phxfs_batch_wait(h) == 0, "empty write batch wait");

    /* Bad-handle plumbing. */
    CHECK(phxfs_batch_cancel(nullptr) == -EINVAL, "cancel on NULL handle");
    CHECK(phxfs_batch_wait(nullptr) == -EINVAL, "wait on NULL handle");
    CHECK(phxfs_batch_wait_all(nullptr, 0) == 0,
          "wait_all accepts an empty handle list");
    CHECK(phxfs_batch_wait_all(nullptr, -1) == -EINVAL,
          "wait_all rejects a negative handle count");
    CHECK(phxfs_batch_wait_all(nullptr, 1) == -EINVAL,
          "wait_all rejects a missing handle list");
    phxfs_batch_t *empty_slots[] = {nullptr, nullptr};
    CHECK(phxfs_batch_wait_all(empty_slots, 2) == 0,
          "wait_all skips NULL handle slots");
    CHECK(phxfs_batch_destroy(nullptr) == -EINVAL, "destroy on NULL handle");

    int fd = make_temp_file();
    CHECK(fd >= 0, "create temporary file");
    if (fd < 0)
        return 1;

    /* Write -> wait -> per-request results copied back. */
    const size_t lengths[kRequests] = {1024, 2048, 3072, 4096};
    std::vector<std::vector<uint8_t>> sources(kRequests);
    std::vector<phxfs_io_req_t> writes(kRequests);
    for (int i = 0; i < kRequests; i++) {
        sources[i].assign(kPrefix + lengths[i] + 16, 0);
        fill_pattern(sources[i], i, lengths[i]);
        writes[i] = {};
        writes[i].fd = fd;
        writes[i].device_id = -1;
        writes[i].buf = sources[i].data();
        writes[i].buf_offset = kPrefix;
        writes[i].nbytes = lengths[i];
        writes[i].f_offset = static_cast<off_t>(i * kStride);
        writes[i].result = -1;
    }

    h = phxfs_batch_submit_write(writes.data(), kRequests);
    CHECK(h != nullptr, "submit write batch");
    CHECK(h && writes[0].result == -1,
          "results stay untouched between submit and wait");
    int rc = h ? phxfs_batch_wait(h) : -1;
    CHECK(rc == 0, "multi-request write batch wait rc=%d", rc);
    for (int i = 0; i < kRequests; i++)
        CHECK(writes[i].result == static_cast<ssize_t>(lengths[i]),
              "write result[%d]=%zd", i, writes[i].result);

    /* Read back and compare. */
    std::vector<std::vector<uint8_t>> destinations(kRequests);
    std::vector<phxfs_io_req_t> reads(kRequests);
    for (int i = 0; i < kRequests; i++) {
        destinations[i].assign(kPrefix + lengths[i] + 16, 0xa5);
        reads[i] = {};
        reads[i].fd = fd;
        reads[i].device_id = -1;
        reads[i].buf = destinations[i].data();
        reads[i].buf_offset = kPrefix;
        reads[i].nbytes = lengths[i];
        reads[i].f_offset = static_cast<off_t>(i * kStride);
        reads[i].result = -1;
    }

    h = phxfs_batch_submit_read(reads.data(), kRequests);
    CHECK(h != nullptr, "submit read batch");
    rc = h ? phxfs_batch_wait(h) : -1;
    CHECK(rc == 0, "multi-request read batch wait rc=%d", rc);
    for (int i = 0; i < kRequests; i++) {
        CHECK(reads[i].result == static_cast<ssize_t>(lengths[i]),
              "read result[%d]=%zd", i, reads[i].result);
        CHECK(std::memcmp(sources[i].data() + kPrefix,
                          destinations[i].data() + kPrefix,
                          lengths[i]) == 0,
              "read data[%d] matches", i);
    }

    /* Pipelining: two batches submitted without waiting in between are both
     * queued and each completes with its own results. The pool deliberately
     * gives NO ordering between independent batches (workers service live
     * jobs round-robin, see io_pool.cpp), so the read targets a region
     * seeded synchronously beforehand and the pipelined write goes to a
     * disjoint region verified afterwards. */
    std::vector<uint8_t> pipe_src(kPrefix + 512 + 16, 0);
    fill_pattern(pipe_src, 11, 512);
    std::vector<uint8_t> pipe_seed(kPrefix + 512 + 16, 0);
    fill_pattern(pipe_seed, 12, 512);
    CHECK(pwrite(fd, pipe_seed.data() + kPrefix, 512, kPipeOffset) == 512,
          "seed pipelined-read region");
    std::vector<uint8_t> pipe_dst(512, 0);
    phxfs_io_req_t pipe_write = {};
    pipe_write.fd = fd;
    pipe_write.device_id = -1;
    pipe_write.buf = pipe_src.data();
    pipe_write.buf_offset = kPrefix;
    pipe_write.nbytes = 512;
    pipe_write.f_offset = kPipeOffset + 4096;
    pipe_write.result = -1;
    phxfs_io_req_t pipe_read = {};
    pipe_read.fd = fd;
    pipe_read.device_id = -1;
    pipe_read.buf = pipe_dst.data();
    pipe_read.buf_offset = 0;
    pipe_read.nbytes = 512;
    pipe_read.f_offset = kPipeOffset;
    pipe_read.result = -1;

    phxfs_batch_t *hw = phxfs_batch_submit_write(&pipe_write, 1);
    phxfs_batch_t *hr = phxfs_batch_submit_read(&pipe_read, 1);
    CHECK(hw != nullptr && hr != nullptr, "pipelined submits queue");
    phxfs_batch_t *pipelined[] = {hw, hr};
    CHECK(phxfs_batch_wait_all(pipelined, 2) == 0,
          "pipelined wait_all");
    CHECK(pipelined[0] == nullptr && pipelined[1] == nullptr,
          "wait_all consumes every handle");
    CHECK(pipe_write.result == 512 && pipe_read.result == 512,
          "pipelined results copied back");
    CHECK(std::memcmp(pipe_seed.data() + kPrefix, pipe_dst.data(), 512) == 0,
          "pipelined read returns the seeded data");
    std::vector<uint8_t> pipe_verify(512, 0);
    CHECK(pread(fd, pipe_verify.data(), 512, kPipeOffset + 4096) == 512 &&
          std::memcmp(pipe_src.data() + kPrefix, pipe_verify.data(), 512) == 0,
          "pipelined write landed on disk");

    /* Failure isolation inside one batch. */
    std::vector<uint8_t> good(lengths[0]);
    std::vector<uint8_t> bad(lengths[0]);
    phxfs_io_req_t mixed[2] = {};
    mixed[0].fd = fd;
    mixed[0].device_id = -1;
    mixed[0].buf = good.data();
    mixed[0].nbytes = good.size();
    mixed[0].f_offset = 0;
    mixed[1].fd = -1;
    mixed[1].device_id = -1;
    mixed[1].buf = bad.data();
    mixed[1].nbytes = bad.size();
    mixed[1].f_offset = 0;

    h = phxfs_batch_submit_read(mixed, 2);
    rc = h ? phxfs_batch_wait(h) : -1;
    CHECK(rc == 1, "bad fd isolated as one failed request rc=%d", rc);
    CHECK(mixed[0].result == static_cast<ssize_t>(good.size()),
          "valid request completes beside bad fd");
    CHECK(mixed[1].result == -EBADF, "bad fd result=%zd", mixed[1].result);

    /* wait_all must consume every supplied handle even when an earlier
     * handle returns an engine/request error. */
    phxfs_io_req_t bad_batch_req = {};
    bad_batch_req.fd = -1;
    bad_batch_req.device_id = -1;
    bad_batch_req.buf = bad.data();
    bad_batch_req.nbytes = bad.size();
    bad_batch_req.f_offset = 0;
    phxfs_io_req_t good_batch_req = {};
    good_batch_req.fd = fd;
    good_batch_req.device_id = -1;
    good_batch_req.buf = good.data();
    good_batch_req.nbytes = good.size();
    good_batch_req.f_offset = 0;
    phxfs_batch_t *bad_handle = phxfs_batch_submit_read(&bad_batch_req, 1);
    phxfs_batch_t *good_handle = phxfs_batch_submit_read(&good_batch_req, 1);
    phxfs_batch_t *mixed_handles[] = {bad_handle, good_handle};
    CHECK(bad_handle != nullptr && good_handle != nullptr,
          "submit handles for wait_all error drain");
    if (bad_handle && good_handle) {
        int all_rc = phxfs_batch_wait_all(mixed_handles, 2);
        CHECK(all_rc == 1, "wait_all returns accumulated failures=%d", all_rc);
        CHECK(mixed_handles[0] == nullptr && mixed_handles[1] == nullptr,
              "wait_all consumes handles after an error");
        CHECK(bad_batch_req.result == -EBADF &&
              good_batch_req.result == static_cast<ssize_t>(good.size()),
              "wait_all preserves results for every handle");
    } else {
        if (bad_handle)
            phxfs_batch_destroy(bad_handle);
        if (good_handle)
            phxfs_batch_destroy(good_handle);
        CHECK(false, "wait_all error-drain submissions succeeded");
    }

    /* A GPU device_id without an opened device must fail, not fall back. */
    phxfs_io_req_t gpu = {};
    gpu.fd = fd;
    gpu.device_id = 0;
    gpu.buf = good.data();
    gpu.nbytes = good.size();
    gpu.f_offset = 0;
    gpu.result = -1;
    h = phxfs_batch_submit_read(&gpu, 1);
    rc = h ? phxfs_batch_wait(h) : -1;
    CHECK(rc == 1, "unopened device request counted rc=%d", rc);
    CHECK(gpu.result == -EFAULT,
          "unopened device result=%zd (no CPU fallback)", gpu.result);

    /* Invalid buffer offset counted once. */
    phxfs_io_req_t invalid[2] = {};
    invalid[0] = reads[0];
    invalid[0].buf_offset = -1;
    invalid[0].result = -1;
    invalid[1] = reads[1];
    invalid[1].result = -1;
    h = phxfs_batch_submit_read(invalid, 2);
    rc = h ? phxfs_batch_wait(h) : -1;
    CHECK(rc == 1, "invalid buffer offset counted once rc=%d", rc);
    CHECK(invalid[0].result == -EFAULT,
          "invalid buffer offset result=%zd", invalid[0].result);
    CHECK(invalid[1].result == static_cast<ssize_t>(invalid[1].nbytes),
          "valid request completes beside invalid offset");

    /* Short read at EOF. The offset is relative to the current end of
     * file because earlier sections extended it. */
    std::vector<uint8_t> eof_buf(256, 0);
    phxfs_io_req_t eof = {};
    eof.fd = fd;
    eof.device_id = -1;
    eof.buf = eof_buf.data();
    eof.nbytes = eof_buf.size();
    eof.f_offset = lseek(fd, 0, SEEK_END) - 64;
    h = phxfs_batch_submit_read(&eof, 1);
    rc = h ? phxfs_batch_wait(h) : -1;
    CHECK(rc == 1, "short EOF read counted as failure rc=%d", rc);
    CHECK(eof.result == 64, "short EOF result=%zd", eof.result);

    /* Handle lifetime: each handle is consumed exactly once. */
    eof.result = -1;
    h = phxfs_batch_submit_read(&eof, 1);
    CHECK(h != nullptr, "resubmit for lifetime test");
    CHECK(h && phxfs_batch_wait(h) == 1, "first wait consumes the handle");
    h = nullptr;

    /* destroy() frees a batch without copying results back, but the
     * already-submitted I/O still lands. */
    std::vector<uint8_t> drop_src(kPrefix + 512 + 16, 0);
    fill_pattern(drop_src, 12, 512);
    phxfs_io_req_t drop = {};
    drop.fd = fd;
    drop.device_id = -1;
    drop.buf = drop_src.data();
    drop.buf_offset = kPrefix;
    drop.nbytes = 512;
    drop.f_offset = kPipeOffset + 4096;
    drop.result = -1;
    h = phxfs_batch_submit_write(&drop, 1);
    CHECK(h != nullptr, "submit write batch for destroy");
    CHECK(phxfs_batch_destroy(h) == 0, "destroy abandons the handle");
    CHECK(drop.result == -1, "destroy does not copy results back");
    std::vector<uint8_t> drop_dst(512, 0);
    ssize_t n = pread(fd, drop_dst.data(), 512, drop.f_offset);
    CHECK(n == 512 && std::memcmp(drop_src.data() + kPrefix,
                                   drop_dst.data(), 512) == 0,
          "destroyed write batch still landed on disk");

    test_cancel(fd);
    test_handle_lifecycle(fd);
    test_independent_handles(fd);
    close(fd);
    std::printf("=== %d checks, %d failed ===\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}
