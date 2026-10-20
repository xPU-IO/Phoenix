#include "mock.h"
#include "fail_new.h"
#include <cerrno>
int main() {
    main_thread=pthread_self(); mock::host_open();
    char buf[4096]; memset(buf,17,sizeof(buf)); void *target;
    CHECK(phxfs_regmem(0,buf,sizeof(buf),&target)==0); int fd=mock::file(sizeof(buf));
    auto r=mock::req(fd,0,buf,sizeof(buf)); fail_workers=true;
    auto*h=phxfs_batch_submit_write(&r,1); CHECK(h);
    CHECK(phxfs_batch_wait(h)==-ENOMEM && r.result==-ENOMEM);
    fail_workers=false; mock::check_idle();
    CHECK(phxfs_write_batch(&r,1)==0 && r.result==sizeof(buf));
    char verify[4096]; CHECK(pread(fd,verify,sizeof(verify),0)==sizeof(verify)); CHECK(!memcmp(buf,verify,sizeof(buf)));
    CHECK(phxfs_deregmem(0,buf,sizeof(buf))==0); CHECK(phxfs_close(0)==0); close(fd);
    puts("PASS worker allocation failure returns ENOMEM without terminating the process");
}
