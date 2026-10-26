#include "mock.h"
#include "fail_new.h"
#include <cerrno>
int main() {
    mock::host_open(); mock::fail_threads = true;
    char data[4096], verify[4096]; memset(data, 37, sizeof(data)); void *target;
    CHECK(phxfs_regmem(0, data, sizeof(data), &target) == 0);
    int fd = mock::file(sizeof(data)); auto r = mock::req(fd, 0, data, sizeof(data));
    // Five vector allocations precede host_worker_state for this one-slice job.
    // The request result distinguishes a worker preparation error from an
    // earlier batch construction failure; no memory exhaustion is involved.
    fail_after = 5; bool escaped = false; int rc = 0;
    try { rc = phxfs_write_batch(&r, 1); }
    catch (const std::bad_alloc &) { escaped = true; }
    fail_after = -1;
    CHECK(!escaped && rc == -ENOMEM && r.result == -ENOMEM);
    mock::check_idle(); CHECK(mock::host_live == PHX_HOST_STAGING_SLOTS);
    CHECK(phxfs_write_batch(&r, 1) == 0 && r.result == sizeof(data));
    CHECK(pread(fd, verify, sizeof(verify), 0) == sizeof(verify));
    CHECK(!memcmp(data, verify, sizeof(data)));
    mock::check_idle(); CHECK(mock::host_live == PHX_HOST_STAGING_SLOTS);
    CHECK(phxfs_deregmem(0, data, sizeof(data)) == 0 && phxfs_close(0) == 0);
    CHECK(mock::host_live == 0); close(fd);
    puts("PASS inline worker allocation failure, reference cleanup and retry");
}
