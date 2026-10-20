#include "mock.h"
#include <vector>
int main() {
    constexpr size_t M=1024*1024, n=20*M, g=4096, bytes=n+5*g;
    unsetenv("PHXFS_MODE"); mock::find_result[1]=1; CHECK(phxfs_open(1)==0);
    mock::host_open(0);
    void *full, *mapped, *target;
    CHECK(devconn->mem_alloc(1,n,&full)==0);
    CHECK(phxfs_regmem(1,full,n,&mapped)==0);
    char host[g]{}; CHECK(phxfs_regmem(0,host,sizeof(host),&target)==0);
    std::vector<char> cpu(n), pattern(n), expected(bytes), verify(bytes);
    int fd=mock::file(bytes); int scenario=0;
    for(bool async : {false,true}) for(bool use_full : {false,true}) for(bool write : {false,true}) {
        ++scenario;
        void *buf=use_full?full:cpu.data(); void *data=use_full?mapped:cpu.data();
        // Distinguish 4 KiB blocks, the HOST request, and successive scenarios.
        for(size_t i=0;i<n;i++) pattern[i]=(char)((i/4096*37+i%251+scenario*19)%201);
        std::fill(expected.begin(),expected.end(),(char)0xe3);
        memcpy(expected.data()+g,pattern.data(),n);
        for(size_t i=0;i<g;i++) expected[n+3*g+i]=(char)((i%251+scenario*23+89)%201);
        if(write) {
            std::fill(verify.begin(),verify.end(),(char)0xe3);
            CHECK(pwrite(fd,verify.data(),bytes,0)==(ssize_t)bytes);
            memcpy(data,pattern.data(),n); memcpy(host,expected.data()+n+3*g,g);
        } else {
            CHECK(pwrite(fd,expected.data(),bytes,0)==(ssize_t)bytes);
            memset(data,0xff,n); memset(host,0xff,g);
        }
        phxfs_io_req_t r[]={mock::req(fd,0,host,g,n+3*g),mock::req(fd,use_full?1:-1,buf,n,g)};
        int rc;
        if(async) { auto*h=write?phxfs_batch_submit_write(r,2):phxfs_batch_submit_read(r,2); CHECK(h); rc=phxfs_batch_wait(h); }
        else rc=write?phxfs_write_batch(r,2):phxfs_read_batch(r,2);
        CHECK(rc==0 && r[0].result==(ssize_t)g && r[1].result==(ssize_t)n);
        CHECK(!memcmp(data,pattern.data(),n));
        CHECK(!memcmp(host,expected.data()+n+3*g,g));
        CHECK(pread(fd,verify.data(),bytes,0)==(ssize_t)bytes);
        CHECK(verify==expected); // Both requests plus leading, gap and trailing guards.
        mock::check_idle(0); mock::check_idle(1);
    }
    CHECK(phxfs_deregmem(0,host,sizeof(host))==0); CHECK(phxfs_close(0)==0);
    CHECK(phxfs_deregmem(1,full,n)==0); CHECK(phxfs_close(1)==0);
    devconn->mem_free(full); close(fd);
    puts("PASS chunk addresses: read/write, sync/async, HOST+CPU/HOST+simulated FULL");
}
