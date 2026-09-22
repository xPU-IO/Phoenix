// CPU-only unit tests for the synchronous batch API and sync I/O engine.

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

int make_temp_file() {
    const char *tmpdir = std::getenv("TMPDIR");
    std::string path = std::string(tmpdir ? tmpdir : "/tmp") +
                       "/phx_batch_sync_XXXXXX";
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

    std::printf("=== Phoenix synchronous batch API tests ===\n");
    CHECK(std::strcmp(phxfs_io_engine_name(), "sync") == 0,
          "forced sync engine selected");
    CHECK(phxfs_read_batch(nullptr, 0) == 0, "empty read batch");
    CHECK(phxfs_write_batch(nullptr, 0) == 0, "empty write batch");

    int fd = make_temp_file();
    CHECK(fd >= 0, "create temporary file");
    if (fd < 0)
        return 1;

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

    int rc = phxfs_write_batch(writes.data(), kRequests);
    CHECK(rc == 0, "multi-request write batch rc=%d", rc);
    for (int i = 0; i < kRequests; i++)
        CHECK(writes[i].result == static_cast<ssize_t>(lengths[i]),
              "write result[%d]=%zd", i, writes[i].result);

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

    rc = phxfs_read_batch(reads.data(), kRequests);
    CHECK(rc == 0, "multi-request read batch rc=%d", rc);
    for (int i = 0; i < kRequests; i++) {
        CHECK(reads[i].result == static_cast<ssize_t>(lengths[i]),
              "read result[%d]=%zd", i, reads[i].result);
        CHECK(std::memcmp(sources[i].data() + kPrefix,
                          destinations[i].data() + kPrefix,
                          lengths[i]) == 0,
              "read data[%d] matches", i);
    }

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

    rc = phxfs_read_batch(mixed, 2);
    CHECK(rc == 1, "bad fd isolated as one failed request rc=%d", rc);
    CHECK(mixed[0].result == static_cast<ssize_t>(good.size()),
          "valid request completes beside bad fd");
    CHECK(mixed[1].result == -EBADF, "bad fd result=%zd", mixed[1].result);

    phxfs_io_req_t invalid[2] = {};
    invalid[0] = reads[0];
    invalid[0].buf_offset = -1;
    invalid[0].result = -1;
    invalid[1] = reads[1];
    invalid[1].result = -1;
    rc = phxfs_read_batch(invalid, 2);
    CHECK(rc == 1, "invalid buffer offset counted once rc=%d", rc);
    CHECK(invalid[0].result == -EFAULT,
          "invalid buffer offset result=%zd", invalid[0].result);
    CHECK(invalid[1].result == static_cast<ssize_t>(invalid[1].nbytes),
          "valid request completes beside invalid offset");

    std::vector<uint8_t> eof_buf(256, 0);
    phxfs_io_req_t eof = {};
    eof.fd = fd;
    eof.device_id = -1;
    eof.buf = eof_buf.data();
    eof.nbytes = eof_buf.size();
    eof.f_offset = static_cast<off_t>((kRequests - 1) * kStride +
                                     lengths[kRequests - 1] - 64);
    rc = phxfs_read_batch(&eof, 1);
    CHECK(rc == 1, "short EOF read counted as failure rc=%d", rc);
    CHECK(eof.result == 64, "short EOF result=%zd", eof.result);

    close(fd);
    std::printf("=== %d checks, %d failed ===\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}
