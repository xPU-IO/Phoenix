#include "mock.h"
#include <cerrno>
int main() {
    unsetenv("PHXFS_MODE"); mock::find_result[0]=0; CHECK(phxfs_open(0)==0);
    CHECK(phx_probe_direct_device(0)==0);
    CHECK(mbuffer[0].head==nullptr && mbuffer[0].last_hit==nullptr);
    CHECK(mock::device_live==0 && mock::maps==1 && mock::unmaps==1);
    int unmapped=mock::unmaps;
    CHECK(phxfs_close(0)==0); CHECK(mock::unmaps==unmapped);
    CHECK(phxfs_open(0)==0); mock::map_error=EOPNOTSUPP;
    CHECK(phx_probe_direct_device(0)==-EOPNOTSUPP);
    CHECK(!mbuffer[0].head && mock::device_live==0); mock::map_error=0;
    // Failed cleanup must retain the allocation until its mapping is released.
    mock::unmap_error=EIO;
    CHECK(phx_probe_direct_device(0)!=0);
    CHECK(mbuffer[0].head && mock::device_live==1);
    mock::unmap_error=0; CHECK(phxfs_close(0)==0);
    CHECK(!mbuffer[0].head && !mbuffer[0].last_hit && mock::device_live==0);
    CHECK(phxfs_open(0)==0);
    for(int i=0;i<16;i++) CHECK(phx_probe_direct_device(0)==0);
    CHECK(!mbuffer[0].head && mock::device_live==0); mock::check_idle();
    CHECK(phxfs_close(0)==0);
    puts("PASS probe cleanup, MAP/UNMAP failures, repeated probe and close");
}
