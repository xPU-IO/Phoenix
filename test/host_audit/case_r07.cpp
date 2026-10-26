#include "mock.h"
#include <cerrno>
static int switches=0;
static int change_device(int) { ++switches; return 0; }
int main(int argc, char **argv) {
    CHECK(argc==3); const char *mode=argv[1], *scenario=argv[2];
    setenv("PHXFS_MODE",mode,1); devconn->set_device=change_device;
    bool zero=!strncmp(scenario,"zero-",5), valid=!strcmp(scenario,"valid");
    mock::device_count=zero?0:1; int d=zero||valid?0:1;
    if (!strcmp(scenario,"invalid-find") || !strcmp(scenario,"zero-find")) {
        int rc=phxfs_find_dev(d);
        fprintf(stderr,"mode=%s scenario=%s find=%d host_live=%d switches=%d\n",mode,scenario,rc,mock::host_live.load(),switches);
        CHECK(rc==-1 && mock::host_live==0 && switches==0);
    } else {
        if(valid) CHECK(phxfs_find_dev(d)==0);
        int rc=phxfs_open(d), allocated=mock::host_live, changed=switches;
        bool initialized=mbuffer[d].init_stat;
        fprintf(stderr,"mode=%s scenario=%s open=%d host_live=%d init=%d switches=%d\n",mode,scenario,rc,allocated,initialized,changed);
        if(rc==0) CHECK(phxfs_close(d)==0); // Record upstream allocation before cleanup.
        if(valid) CHECK(rc==0 && initialized && allocated==PHX_HOST_STAGING_SLOTS);
        else CHECK(rc==-ENODEV && !initialized && allocated==0);
        CHECK(mock::host_live==0 && changed==0);
    }
    printf("PASS visible-device scenario %s %s\n",mode,scenario);
}
