#include "mock.h"
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>

static std::mutex mu;
static std::condition_variable cv;
static bool entered=false, release_gate=false, second_waiting=false;
static thread_local bool second_opener=false;
extern "C" int __real_pthread_cond_wait(pthread_cond_t *,pthread_mutex_t *);
extern "C" int __wrap_pthread_cond_wait(pthread_cond_t *c,pthread_mutex_t *m) {
    if (second_opener) {
        std::lock_guard<std::mutex> l(mu); second_waiting=true; cv.notify_all();
    }
    return __real_pthread_cond_wait(c,m);
}
static int allocation=0;
static void fail_third_allocation() { if (++allocation==3) mock::host_error=ENOMEM; }
static void gate() {
    std::unique_lock<std::mutex> l(mu); entered=true; cv.notify_all();
    cv.wait(l,[]{return release_gate;});
}
static void scenario(bool fail, bool automatic) {
    entered=false; release_gate=false; second_waiting=false;
    mock::host_error=fail?ENOMEM:0;
    mock::map_error=automatic?EOPNOTSUPP:0;
    mock::find_result[0]=automatic?0:-1;
    setenv("PHXFS_MODE",automatic?"auto":"host",1);
    if(automatic) mock::map_gate=gate; else mock::alloc_gate=gate;
    int first=-999, second=-999;
    std::thread a([&]{first=phxfs_open(0);});
    { std::unique_lock<std::mutex> l(mu); cv.wait(l,[]{return entered;}); }
    std::thread b([&]{second_opener=true; second=phxfs_open(0);});
    { std::unique_lock<std::mutex> l(mu); cv.wait(l,[]{return second_waiting;}); }
    { std::lock_guard<std::mutex> l(mu); release_gate=true; cv.notify_all(); }
    a.join(); b.join(); mock::alloc_gate=nullptr; mock::map_gate=nullptr;
    if(fail) {
        CHECK(first==-ENOMEM && second==-ENOMEM);
        CHECK(!mbuffer[0].init_stat && !mbuffer[0].head && mock::host_live==0);
    } else {
        CHECK(first==0 && second==0); CHECK(mbuffer[0].open_count==2);
        CHECK(phxfs_get_map_mode(0)==PHX_MAP_MODE_HOST);
        CHECK(mbuffer[0].host_staging_size!=0); mock::check_idle();
        CHECK(phxfs_close(0)==0 && mbuffer[0].init_stat);
        CHECK(phxfs_close(0)==0 && !mbuffer[0].init_stat);
        CHECK(mock::host_live==0 && mock::device_live==0);
    }
}
int main() {
    scenario(false,false); scenario(true,false); scenario(false,true);
    mock::host_error=0; mock::host_open(); CHECK(phxfs_close(0)==0);
    auto allocate=devconn->host_alloc; devconn->host_alloc=nullptr;
    CHECK(phxfs_open(0)==-EOPNOTSUPP && !mbuffer[0].init_stat);
    devconn->host_alloc=allocate;
    mock::alloc_gate=fail_third_allocation;
    CHECK(phxfs_open(0)==-ENOMEM && mock::host_live==0 && !mbuffer[0].init_stat);
    mock::alloc_gate=nullptr; mock::host_error=0;
    CHECK(phxfs_open(0)==0 && phxfs_close(0)==0);
    puts("PASS concurrent public open, rollback and retry");
}
