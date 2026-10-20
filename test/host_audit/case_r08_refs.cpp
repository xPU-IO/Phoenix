#include "mock.h"
#include "fail_new.h"
#include <cerrno>
int main() {
    mock::host_open(); char data[4096]{}; void *target;
    CHECK(phxfs_regmem(0,data,sizeof(data),&target)==0);
    int fd=mock::file(sizeof(data)); auto r=mock::req(fd,0,data,sizeof(data));
    // nodes/devices/addresses allocate first; fail ops.reserve after resolution.
    fail_after=3; int rc=0; bool escaped=false;
    try { rc=phxfs_write_batch(&r,1); } catch(const std::bad_alloc &) { escaped=true; }
    fail_after=-1; mock::check_idle(); CHECK(!escaped && rc==-ENOMEM);
    CHECK(phxfs_deregmem(0,data,sizeof(data))==0); CHECK(phxfs_close(0)==0); close(fd);
    puts("PASS allocation failure after mapping/device references were acquired");
}
