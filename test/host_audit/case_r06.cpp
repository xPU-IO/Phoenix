#include "mock.h"
#include <cerrno>
int main() {
    unsetenv("PHXFS_MODE"); mock::find_result[0]=0; CHECK(phxfs_open(0)==0);
    // A connector implementing only the operation table can service the probe.
    mock::map_error=EOPNOTSUPP;
    CHECK(phx_probe_direct_device(0)==-EOPNOTSUPP && mock::maps==1);
    CHECK(mock::device_live==0); mock::check_idle();
    devconn->mem_alloc=nullptr;
    CHECK(phx_probe_direct_device(0)==-EOPNOTSUPP);
    CHECK(mock::maps==1); mock::check_idle(); CHECK(phxfs_close(0)==0);
    puts("PASS generic probe operations and unsupported connector");
}
