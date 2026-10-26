#include "mock.h"
#include <cerrno>
#include <initializer_list>
#include <set>
#include <vector>
static void attempt(int open_err, int map_err, int mode, bool fallback) {
    mock::open_error=open_err; mock::map_error=map_err; mock::kernel_mode=mode;
    int rc=phxfs_open(0);
    fprintf(stderr,"open_error=%d map_error=%d kernel_mode=%d: rc=%d map_mode=%d host_live=%d device_live=%d expected_fallback=%d\n",
            open_err,map_err,mode,rc,phxfs_get_map_mode(0),mock::host_live.load(),mock::device_live.load(),fallback);
    if(fallback) { CHECK(rc==0 && phxfs_get_map_mode(0)==PHX_MAP_MODE_HOST); CHECK(phxfs_close(0)==0); }
    else { CHECK(rc==-(open_err?open_err:map_err)); CHECK(!mbuffer[0].init_stat); }
    CHECK(mock::host_live==0 && mock::device_live==0 && !mbuffer[0].head);
}
static uintptr_t allocation_base;
static size_t allocation_size;
static int (*real_alloc)(int, size_t, void **);
static std::vector<uintptr_t> windows;
static bool reject_all;
static int observe_allocation(int d, size_t n, void **p) {
    int rc = real_alloc(d, n, p);
    if (!rc) { allocation_base = (uintptr_t)*p; allocation_size = n; }
    return rc;
}
static int check_window(uint64_t addr, size_t size) {
    CHECK(size == 2 * 1024 * 1024);
    CHECK(addr >= allocation_base && addr + size <= allocation_base + allocation_size);
    CHECK(addr % devconn->page_size == 0);
    windows.push_back(addr);
    return reject_all || windows.size() == 1 ? EINVAL : 0;
}
static void mapping_windows(bool all_rejected) {
    mock::open_error = 0; mock::map_error = 0;
    mock::kernel_mode = PHX_MAP_MODE_STAGING;
    windows.clear(); reject_all = all_rejected;
    real_alloc = devconn->mem_alloc; devconn->mem_alloc = observe_allocation;
    mock::map_check = check_window;
    int before = mock::maps;
    int rc = phxfs_open(0);
    fprintf(stderr,"window open: rc=%d all_rejected=%d attempted=%zu\n",rc,all_rejected,windows.size());
    CHECK(rc == 0);
    CHECK(phxfs_get_map_mode(0) == (all_rejected ? PHX_MAP_MODE_HOST : PHX_MAP_MODE_STAGING));
    CHECK((size_t)(mock::maps - before) == windows.size());
    std::set<uintptr_t> actual(windows.begin(), windows.end()), expected;
    // Independently enumerate every aligned start whose entire pool fits.
    const size_t pool = 2 * 1024 * 1024;
    for (uintptr_t a = allocation_base; a + pool <= allocation_base + allocation_size; ++a)
        if (a % devconn->page_size == 0) expected.insert(a);
    CHECK(!windows.empty() && windows.front() % pool == 0);
    CHECK(actual.size() == windows.size()); // no duplicate attempts
    if (all_rejected) CHECK(actual == expected && actual.size() > 1);
    else CHECK(windows.size() == 2 && expected.count(windows.back()) == 1);
    fprintf(stderr, "window scenario all_rejected=%d: attempted=%zu legal=%zu mode=%d\n",
            all_rejected, windows.size(), expected.size(), phxfs_get_map_mode(0));
    mock::map_check = nullptr; devconn->mem_alloc = real_alloc;
    CHECK(phxfs_close(0) == 0);
    CHECK(mock::host_live == 0 && mock::device_live == 0 && !mbuffer[0].head);
}
static int error_number(const char *s) {
    struct Entry { const char *name; int value; };
    for (auto e : {Entry{"ENOENT",ENOENT}, Entry{"ENODEV",ENODEV}, Entry{"ENXIO",ENXIO},
                   Entry{"EOPNOTSUPP",EOPNOTSUPP}, Entry{"ENOSYS",ENOSYS}, Entry{"EACCES",EACCES},
                   Entry{"EPERM",EPERM}, Entry{"ENOMEM",ENOMEM}, Entry{"EIO",EIO}, Entry{"EINVAL",EINVAL}})
        if (!strcmp(s,e.name)) return e.value;
    CHECK(false); return 0;
}
int main(int argc, char **argv) {
    CHECK(argc==2); const char *scenario=argv[1];
    setenv("PHXFS_MODE","auto",1); setenv("PHX_STAGING_SIZE_MB","2",1); mock::find_result[0]=0;
    if (!strncmp(scenario,"open-",5) || !strncmp(scenario,"full-",5) || !strncmp(scenario,"stage-",6)) {
        bool stage=!strncmp(scenario,"stage-",6), open=!strncmp(scenario,"open-",5);
        int err=error_number(scenario+(stage?6:5));
        bool allowed=err==ENOENT || err==ENODEV || err==ENXIO || err==EOPNOTSUPP || err==ENOSYS;
        attempt(open?err:0,open?0:err,stage?PHX_MAP_MODE_STAGING:PHX_MAP_MODE_FULL,allowed);
    } else if (!strcmp(scenario,"all-windows")) mapping_windows(true);
    else if (!strcmp(scenario,"second-window")) mapping_windows(false);
    else if (!strcmp(scenario,"invalid-page")) {
        mock::kernel_mode=PHX_MAP_MODE_STAGING; auto page=devconn->page_size; devconn->page_size=3;
        int rc=phxfs_open(0); fprintf(stderr,"invalid page: rc=%d host_live=%d\n",rc,mock::host_live.load());
        CHECK(rc==-EINVAL && mock::host_live==0 && mock::device_live==0); devconn->page_size=page;
    } else if (!strcmp(scenario,"alloc-ENOMEM")) {
        mock::kernel_mode=PHX_MAP_MODE_STAGING; auto alloc=devconn->mem_alloc;
        devconn->mem_alloc=[](int,size_t,void**){return -ENOMEM;};
        int rc=phxfs_open(0); fprintf(stderr,"setup allocation: rc=%d host_live=%d\n",rc,mock::host_live.load());
        CHECK(rc==-ENOMEM && mock::host_live==0 && mock::device_live==0); devconn->mem_alloc=alloc;
    } else if (!strncmp(scenario,"discovery-",10)) {
        // Characterize the unresolved discovery boundary, not a fixed contract.
        mock::find_result[0]=!strcmp(scenario+10,"missing")?-1:-error_number(scenario+10);
        CHECK(phxfs_open(0)==0 && phxfs_get_map_mode(0)==PHX_MAP_MODE_HOST);
        CHECK(phxfs_close(0)==0 && mock::host_live==0);
    } else if (!strcmp(scenario,"direct-open")) {
        setenv("PHXFS_MODE","direct",1); attempt(ENOENT,0,PHX_MAP_MODE_FULL,false);
    } else if (!strcmp(scenario,"direct-stage")) {
        setenv("PHXFS_MODE","direct",1); attempt(0,ENODEV,PHX_MAP_MODE_STAGING,false);
    } else CHECK(false);
    printf("PASS fallback scenario %s\n",scenario);
}
