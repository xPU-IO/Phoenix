#include "mock.h"
#include <cerrno>
#include <vector>
// Fortified glibc calls must pass through the same controlled I/O failure.
extern "C" ssize_t __wrap_pread(int, void *, size_t, off_t);
extern "C" ssize_t __wrap___pread_chk(int fd, void *p, size_t n, off_t off, size_t cap) {
    CHECK(n<=cap); return __wrap_pread(fd,p,n,off);
}
int main() {
    const size_t M=1024*1024, half=16*M, n=2*half; mock::host_open();
    std::vector<char> data(n), pattern(n,71), verify(n); void *target;
    CHECK(phxfs_regmem(0,data.data(),n,&target)==0); int fd=mock::file(n);
    for(bool async : {false,true}) for(bool write : {false,true}) for(int failure : {0,1,2}) {
        mock::fail_fd=-1; std::fill(verify.begin(),verify.end(),0);
        CHECK(pwrite(fd,write?verify.data():pattern.data(),n,0)==(ssize_t)n);
        data=write?pattern:verify;
        mock::fail_from=failure==1?half:0; mock::fail_until=failure==0?half:n; mock::fail_fd=fd;
        auto r=mock::req(fd,0,data.data(),n); int rc;
        if(async) { auto*h=write?phxfs_batch_submit_write(&r,1):phxfs_batch_submit_read(&r,1); CHECK(h); rc=phxfs_batch_wait(h); }
        else rc=write?phxfs_write_batch(&r,1):phxfs_read_batch(&r,1);
        fprintf(stderr,"case async=%d write=%d failure=%d rc=%d result=%zd\n",async,write,failure,rc,r.result);
        CHECK(rc==1 && r.result==(failure==2?-EIO:(ssize_t)half));
        mock::fail_fd=-1;
        if(write) CHECK(pread(fd,verify.data(),n,0)==(ssize_t)n); else verify=data;
        for(size_t i=0;i<n;i++) CHECK(verify[i]==((failure==0&&i>=half)||(failure==1&&i<half)?71:0));
        mock::check_idle();
    }
    CHECK(phxfs_deregmem(0,data.data(),n)==0); CHECK(phxfs_close(0)==0); close(fd);
    puts("PASS split errors: first/last/all, read/write, sync/async, bytes and content");
}
