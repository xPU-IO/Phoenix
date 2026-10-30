#ifndef PHXFS_P2P_TEST_IOCTL_H
#define PHXFS_P2P_TEST_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

struct phxfs_p2p_test_request {
    __u64 gpu_vaddr;
    __u64 length;
    __u64 ioaddrs;
    __u32 n_addrs;
    __u32 reserved;
};

#define PHXFS_P2P_TEST_REGISTER _IOW('P', 1, struct phxfs_p2p_test_request)
#define PHXFS_P2P_TEST_DEREGISTER _IO('P', 2)

#endif
