#include "mock.h"
#include <cerrno>
int main() {
    unsetenv("PHXFS_MODE"); mock::find_result[0]=0;
    CHECK(phxfs_open(0)==0);
    mock::munmap_error=EIO;
    CHECK(phx_probe_direct_device(0)!=0);
    CHECK(mbuffer[0].head && mock::device_live==1);
    CHECK(!mbuffer[0].head->has_reg && mbuffer[0].head->mapped);
    CHECK(phxfs_close(0)==0);
    CHECK(!mbuffer[0].head && mock::device_live==0);
    puts("PASS probe munmap failure retains allocation until close");
}
