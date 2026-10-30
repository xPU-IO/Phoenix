#include <linux/compat.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "phxfs-p2p-service.h"
#include "test_ioctl.h"

#define GPU_PAGE_BYTES 65536ULL

struct test_session {
    struct mutex lock;
    struct phxfs_p2p_handle *handle;
};

static int test_open(struct inode *inode, struct file *file)
{
    struct test_session *session;

    session = kzalloc(sizeof(*session), GFP_KERNEL);
    if (!session)
        return -ENOMEM;
    mutex_init(&session->lock);
    file->private_data = session;
    return 0;
}

static int test_release(struct inode *inode, struct file *file)
{
    struct test_session *session = file->private_data;

    phxfs_p2p_deregister(session->handle);
    kfree(session);
    return 0;
}

static long test_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct test_session *session = file->private_data;
    struct phxfs_p2p_test_request request;
    struct phxfs_p2p_handle *handle = NULL;
    const uint64_t *ioaddrs = NULL;
    uint32_t n_addrs = 0;
    int err;

    if (cmd != PHXFS_P2P_TEST_REGISTER && cmd != PHXFS_P2P_TEST_DEREGISTER)
        return -ENOTTY;
    mutex_lock(&session->lock);
    if (cmd == PHXFS_P2P_TEST_DEREGISTER) {
        phxfs_p2p_deregister(session->handle);
        session->handle = NULL;
        err = 0;
        goto out;
    }
    if (session->handle) {
        err = -EBUSY;
        goto out;
    }
    if (copy_from_user(&request, (void __user *)arg, sizeof(request))) {
        err = -EFAULT;
        goto out;
    }
    if (!request.gpu_vaddr || !request.length || !request.ioaddrs ||
        request.gpu_vaddr % GPU_PAGE_BYTES || request.length % GPU_PAGE_BYTES ||
        request.gpu_vaddr > U64_MAX - request.length || request.reserved ||
        request.length / GPU_PAGE_BYTES != request.n_addrs ||
        request.n_addrs > SIZE_MAX / sizeof(*ioaddrs)) {
        err = -EINVAL;
        goto out;
    }
    err = phxfs_p2p_register(request.gpu_vaddr, request.length,
                            &handle, &ioaddrs, &n_addrs);
    if (!err && (!handle || !ioaddrs || n_addrs != request.n_addrs))
        err = -EIO;
    if (!err && copy_to_user(u64_to_user_ptr(request.ioaddrs), ioaddrs,
                             (size_t)n_addrs * sizeof(*ioaddrs)))
        err = -EFAULT;
    if (err)
        phxfs_p2p_deregister(handle);
    else
        session->handle = handle;
out:
    mutex_unlock(&session->lock);
    return err;
}

#ifdef CONFIG_COMPAT
static long test_compat_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    return test_ioctl(file, cmd, (unsigned long)compat_ptr(arg));
}
#endif

static const struct file_operations test_fops = {
    .owner = THIS_MODULE,
    .open = test_open,
    .release = test_release,
    .unlocked_ioctl = test_ioctl,
#ifdef CONFIG_COMPAT
    .compat_ioctl = test_compat_ioctl,
#endif
    .llseek = no_llseek,
};

static struct miscdevice test_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "phxfs_p2p_test",
    .fops = &test_fops,
    .mode = 0600,
};

static int __init test_init(void)
{
    return misc_register(&test_device);
}

static void __exit test_exit(void)
{
    misc_deregister(&test_device);
}

module_init(test_init);
module_exit(test_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Independent Phoenix P2P registration test consumer; no DMA");
