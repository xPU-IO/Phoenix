#include "mock.h"
#include "fail_new.h"
#include <cerrno>
#include <vector>
int main() {
    main_thread=pthread_self(); mock::host_open();
    const size_t n=20*1024*1024, g=4096, size=n+5*g;
    std::vector<char> data(size), expected(size), verify(size), initial(size,(char)0xe3); void *target;
    CHECK(phxfs_regmem(0,data.data(),size,&target)==0); int fd=mock::file(size);
    int injected=0, retried=0, successes=0;
    for(bool async : {false,true}) for(int at=0;at<16;at++) {
        expected=initial;
        for(size_t i=0;i<n;i++) expected[g+i]=(char)((i/g*37+i%251+at*19+async*71)%201);
        for(size_t i=0;i<g;i++) expected[n+3*g+i]=(char)((i%251+at*23+109)%201);
        data=expected;
        CHECK(pwrite(fd,initial.data(),size,0)==(ssize_t)size);
        phxfs_io_req_t r[]={mock::req(fd,0,data.data(),n,g,g),mock::req(fd,0,data.data(),g,n+3*g,n+3*g)};
        bool escaped=false, submitted=false; int rc=0;
        int io_before=mock::io_calls;
        failure_fired=false; fail_after=at;
        try {
            if(async) {
                auto*h=phxfs_batch_submit_write(r,2); submitted=h!=nullptr;
                if(h) rc=phxfs_batch_wait(h);
                else { CHECK(errno==ENOMEM); rc=-ENOMEM; }
            } else rc=phxfs_write_batch(r,2);
        } catch(const std::bad_alloc &) { escaped=true; }
        fail_after=-1; CHECK(!escaped); mock::check_idle();
        // Only the submitting thread's C++ allocations are injected here.
        // Worker failures are tested separately and may have partial progress.
        if(failure_fired) {
            ++injected; CHECK(rc==-ENOMEM && !submitted);
            CHECK(mock::io_calls==io_before);
            CHECK(pread(fd,verify.data(),size,0)==(ssize_t)size && verify==initial);
            fprintf(stderr,"preparation failure async=%d allocation=%d: no I/O, references zero; retry\n",async,at);
            for(auto &req:r) req.result=-999;
            if(async) { auto*h=phxfs_batch_submit_write(r,2); CHECK(h); rc=phxfs_batch_wait(h); }
            else rc=phxfs_write_batch(r,2);
            ++retried;
        } else ++successes;
        CHECK(rc==0 && r[0].result==(ssize_t)n && r[1].result==(ssize_t)g);
        CHECK(pread(fd,verify.data(),size,0)==(ssize_t)size);
        CHECK(verify==expected && data==expected); // Both requests, chunks and guards.
        mock::check_idle();
    }
    CHECK(injected>0 && retried==injected && successes>0);
    fprintf(stderr,"allocation cases: preparation_failures=%d explicit_retries=%d uninjected_successes=%d\n",injected,retried,successes);
    CHECK(phxfs_deregmem(0,data.data(),size)==0); CHECK(phxfs_close(0)==0); close(fd);
    puts("PASS deterministic allocation failures across sync/async preparation and explicit retry");
}
