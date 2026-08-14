#include <linux/module.h>

#include "phxfs-backend.h"
#include "phxfs.h"

/* The active P2P backend (global, set at module init). */
struct phxfs_p2p_ops *phxfs_p2p;

int phxfs_p2p_register_backend(struct phxfs_p2p_ops *ops)
{
	if (!ops || !ops->init || !ops->get_pages || !ops->put_pages ||
	    !ops->get_n_pages || !ops->get_phys_addrs ||
	    !ops->free_page_table || ops->page_size == 0) {
		phxfs_err("phxfs: invalid P2P backend ops\n");
		return -EINVAL;
	}

	phxfs_p2p = ops;
	phxfs_info("phxfs: registered P2P backend: %s (page_size=%llu)\n",
		   ops->name, ops->page_size);
	return 0;
}

/*
 * Vendor-specific backend init is selected at compile time.
 * Each vendor provides a <vendor>_backend_register() function
 * in its own source file (e.g. nvidia-backend.c).
 *
 * If no CONFIG_PHXFS_VENDOR_* is defined, default to NVIDIA.
 */
#if defined(CONFIG_PHXFS_VENDOR_METAX)
extern int metax_backend_register(void);
#elif defined(CONFIG_PHXFS_VENDOR_AMD)
extern int amd_backend_register(void);
#elif defined(CONFIG_PHXFS_VENDOR_HUAWEI)
extern int huawei_backend_register(void);
#else
extern int nvidia_backend_register(void);
#endif

int phxfs_p2p_backend_init(void)
{
	int ret;

#if defined(CONFIG_PHXFS_VENDOR_METAX)
	ret = metax_backend_register();
#elif defined(CONFIG_PHXFS_VENDOR_AMD)
	ret = amd_backend_register();
#elif defined(CONFIG_PHXFS_VENDOR_HUAWEI)
	ret = huawei_backend_register();
#else
	ret = nvidia_backend_register();
#endif

	if (ret) {
		phxfs_err("phxfs: P2P backend registration failed: %d\n", ret);
		return ret;
	}

	if (!phxfs_p2p) {
		phxfs_err("phxfs: no P2P backend registered\n");
		return -ENODEV;
	}

	return 0;
}

void phxfs_p2p_backend_exit(void)
{
	if (phxfs_p2p && phxfs_p2p->exit)
		phxfs_p2p->exit();
	phxfs_p2p = NULL;
}
