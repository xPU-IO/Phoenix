#include "mock.h"
#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>
static std::mutex mu;
static std::condition_variable cv;
static bool resolved=false, closing=false, proceed=false;
static thread_local bool armed=false;
static std::atomic<int> copies{0};
void *operator new(size_t n) {
    if(armed && mbuffer[0].head && mbuffer[0].head->refcount>0) {
        armed=false;
        std::unique_lock<std::mutex> l(mu); resolved=true; cv.notify_all();
        cv.wait(l,[]{return proceed;});
    }
    void*p=malloc(n?n:1); if(!p) throw std::bad_alloc(); return p;
}
void operator delete(void*p) noexcept { free(p); }
void operator delete(void*p,size_t) noexcept { free(p); }
void *operator new(size_t n,const std::nothrow_t&) noexcept { try { return ::operator new(n); } catch(...) { return nullptr; } }
void operator delete(void*p,const std::nothrow_t&) noexcept { free(p); }
extern "C" int __real_pthread_cond_wait(pthread_cond_t*,pthread_mutex_t*);
extern "C" int __wrap_pthread_cond_wait(pthread_cond_t*c,pthread_mutex_t*m) {
    if(c==&mbuffer[0].drain_cv) { std::lock_guard<std::mutex> l(mu); closing=true; cv.notify_all(); }
    return __real_pthread_cond_wait(c,m);
}
static int copy(void*d,const void*s,size_t n) { ++copies; memcpy(d,s,n); return 0; }
int main() {
    mock::host_open(); char gpu[4096]{}, pattern[4096]; memset(pattern,39,4096);
    void*target; CHECK(phxfs_regmem(0,gpu,4096,&target)==0); devconn->memcpy_h2d=copy;
    int fd=mock::file(4096); CHECK(pwrite(fd,pattern,4096,0)==4096);
    auto r=mock::req(fd,0,gpu,4096);
    std::thread io([&]{armed=true; CHECK(phxfs_read_batch(&r,1)==0);});
    { std::unique_lock<std::mutex> l(mu); cv.wait(l,[]{return resolved;}); }
    std::thread closer([]{CHECK(phxfs_close(0)==0);});
    { std::unique_lock<std::mutex> l(mu); cv.wait(l,[]{return closing;}); proceed=true; cv.notify_all(); }
    io.join(); closer.join();
    CHECK(copies==1 && r.result==4096 && !memcmp(gpu,pattern,4096)); close(fd);
    puts("PASS resolved HOST batch keeps copy callbacks when close starts");
}
