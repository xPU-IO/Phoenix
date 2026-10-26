#pragma once
#include <new>
#include <atomic>
#include <pthread.h>
#include <cstdlib>
static thread_local int fail_after=-1;
static thread_local bool failure_fired=false;
static std::atomic<bool> fail_workers{false};
static pthread_t main_thread;
void *operator new(size_t n) {
    if (fail_workers && !pthread_equal(pthread_self(),main_thread)) throw std::bad_alloc();
    if(fail_after==0) { fail_after=-1; failure_fired=true; throw std::bad_alloc(); }
    if(fail_after>0) --fail_after;
    void *p=malloc(n?n:1); if(!p) throw std::bad_alloc(); return p;
}
void operator delete(void *p) noexcept { free(p); }
void operator delete(void *p, size_t) noexcept { free(p); }
void *operator new(size_t n, const std::nothrow_t &) noexcept { try { return ::operator new(n); } catch(...) { return nullptr; } }
void operator delete(void *p, const std::nothrow_t &) noexcept { free(p); }
