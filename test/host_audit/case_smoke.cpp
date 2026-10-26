#include "mock.h"
int main() {
    mock::host_open();
    char a[4096], b[4096]{}; memset(a, 37, sizeof(a));
    void *target; CHECK(phxfs_regmem(0,a,sizeof(a),&target)==0);
    int fd=mock::file(sizeof(a));
    CHECK(phxfs_write(fd,0,a,0,sizeof(a),0)==sizeof(a));
    CHECK(pread(fd,b,sizeof(b),0)==sizeof(b)); CHECK(!memcmp(a,b,sizeof(a)));
    mock::check_idle(); CHECK(phxfs_deregmem(0,a,sizeof(a))==0);
    CHECK(phxfs_close(0)==0); CHECK(mock::host_live==0); close(fd);
    puts("PASS host smoke (CPU connector, memfd, sync engine)");
}
