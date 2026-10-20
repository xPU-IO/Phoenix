#include "mock.h"
#include <vector>
int main() {
    const size_t M=1024*1024, n=24*M, guard=4096, total=n+2*guard; mock::host_open();
    std::vector<char> data(n), pattern(n), verify(total), initial(total), expected(total); void *target;
    auto make_pattern=[&](size_t first) {
        for(size_t i=0;i<n;i++) pattern[i]=(char)(i<first ? 1+i%113 : 129+(i-first)%113);
    };
    CHECK(phxfs_regmem(0,data.data(),n,&target)==0);
    int fd=mock::file(total);
    for(bool async : {false,true}) for(bool write : {false,true}) for(size_t first : {16*M,20*M}) {
        make_pattern(first); data=pattern;
        std::fill(initial.begin(),initial.end(),0x6d); expected=initial;
        memcpy(expected.data()+guard,pattern.data(),n);
        if(write) memset(initial.data()+guard,0x0c,n);
        else { initial=expected; memset(data.data(),0,n); }
        CHECK(pwrite(fd,initial.data(),total,0)==(ssize_t)total);
        phxfs_io_req_t r[]={mock::req(fd,0,data.data(),first,guard), mock::req(fd,0,data.data()+first,n-first,guard+first)};
        int rc;
        if(async) { auto*h=write?phxfs_batch_submit_write(r,2):phxfs_batch_submit_read(r,2); CHECK(h); rc=phxfs_batch_wait(h); }
        else rc=write?phxfs_write_batch(r,2):phxfs_read_batch(r,2);
        CHECK(rc==0 && r[0].result==(ssize_t)first && r[1].result==(ssize_t)(n-first));
        CHECK(data==pattern);
        CHECK(pread(fd,verify.data(),total,0)==(ssize_t)total);
        CHECK(verify==expected); // request-specific bytes and outside-range guards
    }
    make_pattern(20*M);
    CHECK(ftruncate(fd,guard+22*M)==0);
    std::fill(initial.begin(),initial.end(),0x6d);
    memcpy(initial.data()+guard,pattern.data(),22*M);
    CHECK(pwrite(fd,initial.data(),guard+22*M,0)==(ssize_t)(guard+22*M));
    memset(data.data(),0,n);
    phxfs_io_req_t short_read[]={mock::req(fd,0,data.data(),20*M,guard),mock::req(fd,0,data.data()+20*M,4*M,guard+20*M)};
    CHECK(phxfs_read_batch(short_read,2)==1);
    CHECK(short_read[0].result==20*(ssize_t)M && short_read[1].result==2*(ssize_t)M);
    CHECK(!memcmp(data.data(),pattern.data(),22*M));
    for(size_t i=22*M;i<n;i++) CHECK(data[i]==0);
    mock::check_idle(); CHECK(phxfs_deregmem(0,data.data(),n)==0); CHECK(phxfs_close(0)==0); close(fd);
    puts("PASS merged tail ownership and short read, read/write sync/async");
}
