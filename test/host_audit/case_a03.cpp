#include "mock.h"
#include <cerrno>
#include <vector>
int main() {
    mock::host_open(); mock::fail_threads=true;
    const size_t chunk=4096, size=19*chunk;
    std::vector<char> data(size), expected(size), verify(size), file_expected(size); void *target;
    CHECK(phxfs_regmem(0,data.data(),size,&target)==0);
    int fd=mock::file(size); int scenario=0;
    for(int n : {4,5,9}) for(bool write : {false,true}) {
        ++scenario;
        std::fill(expected.begin(),expected.end(),(char)0xe3);
        std::vector<phxfs_io_req_t> r;
        for(int i=0;i<n;i++) {
            size_t off=(2*i+1)*chunk;
            for(size_t j=0;j<chunk;j++) expected[off+j]=(char)((i*31+j%251+scenario*17)%201);
            r.push_back(mock::req(fd,0,data.data(),chunk,off,off));
        }
        // Independently reset every scenario, including all unrequested gaps.
        std::fill(data.begin(),data.end(),(char)0xe3);
        file_expected=expected;
        if(write) {
            data=expected; std::fill(verify.begin(),verify.end(),(char)0xe3);
            CHECK(pwrite(fd,verify.data(),size,0)==(ssize_t)size);
        } else CHECK(pwrite(fd,expected.data(),size,0)==(ssize_t)size);
        CHECK((write?phxfs_write_batch(r.data(),n):phxfs_read_batch(r.data(),n))==0);
        for(int i=0;i<n;i++) CHECK(r[i].result==(ssize_t)chunk);
        CHECK(data==expected);
        CHECK(pread(fd,verify.data(),size,0)==(ssize_t)size);
        CHECK(verify==file_expected); // Requested ranges and all outside guards.
        CHECK(mock::host_live==PHX_HOST_STAGING_SLOTS); mock::check_idle();
    }
    auto r=mock::req(fd,0,data.data(),chunk);
    errno=0; CHECK(!phxfs_batch_submit_read(&r,1) && errno==ENOTSUP); mock::check_idle();
    CHECK(phxfs_deregmem(0,data.data(),size)==0); CHECK(phxfs_close(0)==0); CHECK(mock::host_live==0); close(fd);
    puts("PASS no-worker inline path: 4/5/9 callbacks, read/write and cleanup");
}
