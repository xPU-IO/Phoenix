// CPU-only unit tests for the asynchronous batch API: submit / wait /
// destroy lifecycle, per-request result copy-back, failure isolation and
// worker-pool queueing (pipelined submits). It forces the sync engine
// internally, so it needs neither a GPU nor the phxfs kernel module.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "phoenix.h"

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

}  // namespace

int main() {
    setenv("PHXFS_IO_ENGINE", "sync", 1);

    std::printf("=== Phoenix asynchronous batch API tests ===\n");
    CHECK(std::strcmp(phxfs_io_engine_name(), "sync") == 0,
          "forced sync engine selected");

    /* Empty batches still yield a consumable handle. */
    phxfs_batch_t *h = phxfs_batch_submit_read(nullptr, 0);
    CHECK(h != nullptr, "empty read batch returns a handle");
    CHECK(h && phxfs_batch_wait(h) == 0, "empty read batch wait");
    h = phxfs_batch_submit_write(nullptr, 0);
    CHECK(h != nullptr, "empty write batch returns a handle");
    CHECK(h && phxfs_batch_wait(h) == 0, "empty write batch wait");

    /* Bad-handle plumbing. */
    CHECK(phxfs_batch_wait(nullptr) == -EINVAL, "wait on NULL handle");
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

    /* Pipelining: two batches submitted without waiting in between; the
     * pool services them in FIFO order, so the read observes the write. */
    std::vector<uint8_t> pipe_src(kPrefix + 512 + 16, 0);
    fill_pattern(pipe_src, 11, 512);
    std::vector<uint8_t> pipe_dst(512, 0);
    phxfs_io_req_t pipe_write = {};
    pipe_write.fd = fd;
    pipe_write.device_id = -1;
    pipe_write.buf = pipe_src.data();
    pipe_write.buf_offset = kPrefix;
    pipe_write.nbytes = 512;
    pipe_write.f_offset = kPipeOffset;
    pipe_write.result = -1;
    phxfs_io_req_t pipe_read = pipe_write;
    pipe_read.buf = pipe_dst.data();
    pipe_read.buf_offset = 0;

    phxfs_batch_t *hw = phxfs_batch_submit_write(&pipe_write, 1);
    phxfs_batch_t *hr = phxfs_batch_submit_read(&pipe_read, 1);
    CHECK(hw != nullptr && hr != nullptr, "pipelined submits queue");
    CHECK(hw && phxfs_batch_wait(hw) == 0, "pipelined write wait");
    CHECK(hr && phxfs_batch_wait(hr) == 0, "pipelined read wait");
    CHECK(std::memcmp(pipe_src.data() + kPrefix, pipe_dst.data(), 512) == 0,
          "pipelined read observes the earlier write");

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
    CHECK(phxfs_batch_wait(h) == -EINVAL, "second wait rejected");
    CHECK(phxfs_batch_destroy(h) == -EINVAL, "destroy after wait rejected");

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

    close(fd);
    std::printf("=== %d checks, %d failed ===\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}
