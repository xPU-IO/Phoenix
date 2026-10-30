/*
 * Phoenix P2P Service Export
 *
 * When Phoenix has remapped a device's BAR via devm_memremap_pages
 * (MEMORY_DEVICE_PCI_P2PDMA), vendor-specific DMA map calls may fail
 * for that device. This service bypasses DMA mapping and returns the
 * physical addresses from the backend's get_pages directly as DMA bus
 * addresses (valid under IOMMU=pt).
 */

#include <linux/module.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/compiler.h>
#include <linux/completion.h>
#include <linux/refcount.h>

#include "phxfs-p2p-service.h"
#include "phxfs-backend.h"
#include "phxfs.h"        /* phxfs_err / phxfs_warn / phxfs_info */
#include "config-host.h"

enum {
	PHXFS_P2P_LIVE = 0,	/* pages pinned; callback and owner may race */
	PHXFS_P2P_RUNNING,	/* callback owns the page table */
	PHXFS_P2P_DROPPED,	/* page table gone (reclaimed or put) */
};

struct phxfs_p2p_handle {
	struct phxfs_page_table       *pt;
	uint64_t                     *ioaddrs;    /* kvmalloc'd */
	uint32_t                      n_addrs;
	uint64_t                      gpu_vaddr;
	int                           invalidated; /* set by free_callback */
	atomic_t                      state;      /* PHXFS_P2P_* */
	refcount_t                    refs;       /* owner + registered callback */
	struct completion             reg_done;   /* register() done with pt */
};

/* Last reference frees the handle and releases the module pin taken in
 * phxfs_p2p_register(). */
static void phxfs_p2p_handle_put(struct phxfs_p2p_handle *h)
{
	if (!refcount_dec_and_test(&h->refs))
		return;
	kvfree(h->ioaddrs);
	kfree(h);
	module_put(THIS_MODULE);
}

/*
 * Owner-side release of the pin. Wins against the callback through the state
 * word: if the callback already claimed the page table it owns it, otherwise
 * put_pages() here and drop the callback's reference on its behalf (it will
 * see !LIVE and do nothing).
 */
static void phxfs_p2p_owner_release(struct phxfs_p2p_handle *h)
{
	struct phxfs_page_table *pt;

	if (atomic_cmpxchg(&h->state, PHXFS_P2P_LIVE, PHXFS_P2P_DROPPED) !=
	    PHXFS_P2P_LIVE)
		return;
	pt = xchg(&h->pt, NULL);
	if (pt)
		phxfs_p2p->put_pages(h->gpu_vaddr, pt);
	phxfs_p2p_handle_put(h);	/* callback reference */
}

/*
 * phxfs_p2p_free_cb - called by the vendor driver when it forcefully
 * reclaims the GPU pages (e.g. GPU context destroyed).
 *
 * We must NOT call put_pages here (the driver is already
 * reclaiming). We free the page_table struct via free_page_table and
 * mark the handle as invalidated so the caller's subsequent deregister
 * skips put_pages.
 *
 * The vendor can fire this before get_pages() has returned to
 * phxfs_p2p_register() (NVIDIA registers the callback as the last step of
 * nv_p2p_get_pages()) and frees its page table right after we return, so
 * wait until registration is done reading it.
 *
 * We do NOT free ioaddrs or the handle while the owner still holds it -- the
 * handle is refcounted and freed by whoever drops the last reference.
 */
static void phxfs_p2p_free_cb(void *data)
{
	struct phxfs_p2p_handle *h = data;
	struct phxfs_page_table *pt;

	if (!h) {
		phxfs_warn("phxfs_p2p_free_cb: NULL data\n");
		return;
	}

	wait_for_completion(&h->reg_done);

	if (atomic_cmpxchg(&h->state, PHXFS_P2P_LIVE, PHXFS_P2P_RUNNING) !=
	    PHXFS_P2P_LIVE)
		return;		/* owner already released; it dropped our ref */

	WRITE_ONCE(h->invalidated, 1);
	pt = xchg(&h->pt, NULL);
	if (pt)
		phxfs_p2p->free_page_table(pt);

	phxfs_warn("phxfs_p2p_free_cb: GPU pages force-reclaimed for vaddr=0x%llx\n",
		   (unsigned long long)h->gpu_vaddr);

	atomic_set(&h->state, PHXFS_P2P_DROPPED);
	phxfs_p2p_handle_put(h);	/* callback reference */
}

int phxfs_p2p_register(uint64_t gpu_vaddr,
                       uint64_t length,
                       struct phxfs_p2p_handle **out_handle,
                       const uint64_t **out_ioaddrs,
                       uint32_t *out_n_addrs)
{
	struct phxfs_p2p_handle *h;
	const char *stage = "page_count";
	int err;

	if (!out_handle || !out_ioaddrs || !out_n_addrs)
		return -EINVAL;
	*out_handle = NULL;
	*out_ioaddrs = NULL;
	*out_n_addrs = 0;
	if (gpu_vaddr == 0 || length == 0)
		return -EINVAL;
	if (!phxfs_p2p)
		return -ENODEV;

	/* Prevent Phoenix from being unloaded while handles are active */
	if (!try_module_get(THIS_MODULE))
		return -ENODEV;

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	if (!h) {
		module_put(THIS_MODULE);
		return -ENOMEM;
	}
	h->gpu_vaddr   = gpu_vaddr;
	h->invalidated = 0;
	atomic_set(&h->state, PHXFS_P2P_LIVE);
	refcount_set(&h->refs, 2);	/* owner + callback */
	init_completion(&h->reg_done);

	/* Step 1: pin device pages */
	err = phxfs_p2p->get_pages(gpu_vaddr, length,
	                           &h->pt, phxfs_p2p_free_cb, h);
	if (err || !h->pt) {
		int ret = err ? err : -ENOMEM;
		phxfs_err("phxfs_p2p_register: get_pages failed, "
			  "vaddr=0x%llx len=%llu err=%d\n",
			  (unsigned long long)gpu_vaddr,
			  (unsigned long long)length, ret);
		kfree(h);
		module_put(THIS_MODULE);
		return ret;
	}

	/* Step 2: extract physical addresses as bus addresses.
	 * Under IOMMU=pt (deployment contract), phys == bus address.
	 * We deliberately skip vendor DMA map — it fails when Phoenix
	 * has remapped the BAR.
	 * A reclaim callback that fires meanwhile waits on reg_done, so h->pt
	 * stays valid until complete_all() below. */
	h->n_addrs = phxfs_p2p->get_n_pages(h->pt);
	if (!h->n_addrs) {
		err = -EIO;
	} else {
		stage = "ioaddrs_alloc";
		h->ioaddrs = kvmalloc_array(h->n_addrs, sizeof(*h->ioaddrs), GFP_KERNEL);
		if (!h->ioaddrs) {
			err = -ENOMEM;
		} else {
			stage = "get_phys_addrs";
			err = phxfs_p2p->get_phys_addrs(h->pt, h->ioaddrs, h->n_addrs);
		}
	}
	complete_all(&h->reg_done);
	if (err) {
		phxfs_err("phxfs_p2p_register: %s failed, vaddr=0x%llx "
			  "len=%llu n_addrs=%u table_bytes=%llu err=%d\n",
			  stage, (unsigned long long)gpu_vaddr,
			  (unsigned long long)length, h->n_addrs,
			  (unsigned long long)h->n_addrs * sizeof(*h->ioaddrs), err);
		phxfs_p2p_owner_release(h);
		phxfs_p2p_handle_put(h);	/* owner; frees + module_put */
		return err;
	}

	*out_handle  = h;
	*out_ioaddrs = h->ioaddrs;   /* borrowed */
	*out_n_addrs = h->n_addrs;

	phxfs_info("phxfs_p2p_register: success vaddr=0x%llx len=%llu n_addrs=%u\n",
		   (unsigned long long)gpu_vaddr,
		   (unsigned long long)length, h->n_addrs);
	return 0;
}
EXPORT_SYMBOL_GPL(phxfs_p2p_register);

void phxfs_p2p_deregister(struct phxfs_p2p_handle *handle)
{
	uint64_t gpu_vaddr;
	int was_invalidated;

	if (!handle)
		return;

	/* Save fields needed for logging before freeing the handle */
	gpu_vaddr = handle->gpu_vaddr;

	/* If the callback has not claimed the page table, do the normal
	 * put_pages; otherwise the callback owns (or already freed) it and
	 * this only drops the owner's reference. Whichever side drops the
	 * last reference frees the heap and releases the module pin. */
	phxfs_p2p_owner_release(handle);
	was_invalidated = READ_ONCE(handle->invalidated);
	phxfs_p2p_handle_put(handle);

	phxfs_info("phxfs_p2p_deregister: vaddr=0x%llx invalidated=%d\n",
		   (unsigned long long)gpu_vaddr, was_invalidated);
}
EXPORT_SYMBOL_GPL(phxfs_p2p_deregister);
