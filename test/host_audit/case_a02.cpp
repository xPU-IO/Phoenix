#include "mock.h"
#include <cerrno>
#include <initializer_list>
int main() {
    mock::host_open(); char gpu[4096]{}, cpu[4096]; memset(cpu,51,sizeof(cpu));
    void *target; CHECK(phxfs_regmem(0,gpu,sizeof(gpu),&target)==0);
    size_t n=sizeof(gpu); off_t bo=0,fo=0; ssize_t done=0;
    int fd=mock::file(n); CHECK(pwrite(fd,cpu,n,0)==(ssize_t)n);
    for(bool write : {false,true}) {
        int rc=write?phxfs_write_stream(fd,gpu,&n,&bo,&fo,&done,(void*)1)
                    :phxfs_read_stream(fd,gpu,&n,&bo,&fo,&done,(void*)1);
        CHECK(rc==-EOPNOTSUPP && done==-EOPNOTSUPP && mock::launches==0);
        mock::check_idle();
    }
    CHECK(phxfs_read(fd,0,gpu,0,n,0)==(ssize_t)n && !memcmp(gpu,cpu,n));
    CHECK(phxfs_read_stream(fd,cpu,&n,&bo,&fo,&done,(void*)1)==0 && done==(ssize_t)n);
    CHECK(mock::launches==1);
    CHECK(phxfs_deregmem(0,gpu,n)==0); CHECK(phxfs_close(0)==0); close(fd);
    puts("PASS HOST stream rejection, CPU stream and synchronous HOST controls");
}
