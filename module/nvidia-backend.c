// SPDX-License-Identifier: Apache-2.0
/*
 * NVIDIA P2P Backend
 *
 * Wraps the NVIDIA driver's P2P API (nv-p2p.h) behind the
 * vendor-agnostic phxfs_p2p_ops interface.
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/pci.h>

#include "phxfs-backend.h"
#include "phxfs.h"       /* phxfs_err / phxfs_info */

/* NVIDIA driver P2P API — nv-p2p.h is provided by the NVIDIA driver source */
#include "nv-p2p.h"

/* Function pointer types for NVIDIA P2P symbols (loaded via __symbol_get) */
typedef int (*nvidia_p2p_get_pages_fptr)(uint64_t, uint32_t,
		uint64_t, uint64_t,
		struct nvidia_p2p_page_table **,
		void (*free_callback)(void *data), void *);
typedef int (*nvidia_p2p_put_pages_fptr)(uint64_t, uint32_t,
		uint64_t, struct nvidia_p2p_page_table *);
typedef int (*nvidia_p2p_free_page_table_fptr)(struct nvidia_p2p_page_table *);
typedef int (*nvidia_p2p_dma_map_pages_fptr)(struct pci_dev *,
		struct nvidia_p2p_page_table *,
		struct nvidia_p2p_dma_mapping **);
typedef int (*nvidia_p2p_dma_unmap_pages_fptr)(struct pci_dev *,
		struct nvidia_p2p_page_table *,
		struct nvidia_p2p_dma_mapping *);
typedef int (*nvidia_p2p_free_dma_mapping_fptr)(struct nvidia_p2p_dma_mapping *);

/* ------------------------------------------------------------------ */
/* NVIDIA P2P symbol pointers (loaded via __symbol_get)               */
/* ------------------------------------------------------------------ */

static nvidia_p2p_get_pages_fptr         nvidia_p2p_get_pages_p;
static nvidia_p2p_put_pages_fptr         nvidia_p2p_put_pages_p;
static nvidia_p2p_free_page_table_fptr   nvidia_p2p_free_page_table_p;
/* DMA map/unmap symbols — kept for completeness, currently unused by
 * the active Phoenix data path but may be needed by future code. */
static nvidia_p2p_dma_map_pages_fptr     nvidia_p2p_dma_map_pages_p;
static nvidia_p2p_dma_unmap_pages_fptr   nvidia_p2p_dma_unmap_pages_p;
static nvidia_p2p_free_dma_mapping_fptr  nvidia_p2p_free_dma_mapping_p;

#ifdef HAVE_MODULE_MUTEX
extern struct mutex module_mutex;
#endif

static void nvidia_put_symbols(void)
{
	if (nvidia_p2p_dma_unmap_pages_p)  __symbol_put("nvidia_p2p_dma_unmap_pages");
	if (nvidia_p2p_get_pages_p)        __symbol_put("nvidia_p2p_get_pages");
	if (nvidia_p2p_put_pages_p)        __symbol_put("nvidia_p2p_put_pages");
	if (nvidia_p2p_dma_map_pages_p)    __symbol_put("nvidia_p2p_dma_map_pages");
	if (nvidia_p2p_free_dma_mapping_p) __symbol_put("nvidia_p2p_free_dma_mapping");
	if (nvidia_p2p_free_page_table_p)  __symbol_put("nvidia_p2p_free_page_table");

	nvidia_p2p_dma_unmap_pages_p  = NULL;
	nvidia_p2p_get_pages_p        = NULL;
	nvidia_p2p_put_pages_p        = NULL;
	nvidia_p2p_dma_map_pages_p    = NULL;
	nvidia_p2p_free_dma_mapping_p = NULL;
	nvidia_p2p_free_page_table_p  = NULL;
}

static int nvidia_load_symbols(void)
{
#ifdef HAVE_MODULE_MUTEX
	mutex_lock(&module_mutex);
#endif
	nvidia_p2p_dma_unmap_pages_p = __symbol_get("nvidia_p2p_dma_unmap_pages");
	if (!nvidia_p2p_dma_unmap_pages_p) goto err;

	nvidia_p2p_get_pages_p = __symbol_get("nvidia_p2p_get_pages");
	if (!nvidia_p2p_get_pages_p) goto err;

	nvidia_p2p_put_pages_p = __symbol_get("nvidia_p2p_put_pages");
	if (!nvidia_p2p_put_pages_p) goto err;

	nvidia_p2p_dma_map_pages_p = __symbol_get("nvidia_p2p_dma_map_pages");
	if (!nvidia_p2p_dma_map_pages_p) goto err;

	nvidia_p2p_free_dma_mapping_p = __symbol_get("nvidia_p2p_free_dma_mapping");
	if (!nvidia_p2p_free_dma_mapping_p) goto err;

	nvidia_p2p_free_page_table_p = __symbol_get("nvidia_p2p_free_page_table");
	if (!nvidia_p2p_free_page_table_p) goto err;

#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	return 0;

err:
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	nvidia_put_symbols();
	return -ENOSYS;
}

/* ------------------------------------------------------------------ */
/* phxfs_p2p_ops implementation                                       */
/* ------------------------------------------------------------------ */

static int nvidia_init(void)
{
	int ret = nvidia_load_symbols();
	if (ret)
		phxfs_err("phxfs: failed to load nvidia_p2p symbols\n");
	return ret;
}

static void nvidia_exit(void)
{
	nvidia_put_symbols();
}

static int nvidia_get_pages(uint64_t vaddr, uint64_t length,
			    struct phxfs_page_table **pt,
			    void (*free_cb)(void *), void *data)
{
	struct phxfs_page_table *h;
	int err;

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	if (!h)
		return -ENOMEM;

	err = nvidia_p2p_get_pages_p(0, 0, vaddr, length,
				     (struct nvidia_p2p_page_table **)&h->priv,
				     free_cb, data);
	if (err) {
		kfree(h);
		return err;
	}
	*pt = h;
	return 0;
}

static void nvidia_put_pages(uint64_t vaddr, struct phxfs_page_table *pt)
{
	nvidia_p2p_put_pages_p(0, 0, vaddr, pt->priv);
	kfree(pt);
}

static uint32_t nvidia_get_n_pages(struct phxfs_page_table *pt)
{
	struct nvidia_p2p_page_table *npt = pt->priv;
	return npt->entries;
}

static int nvidia_get_phys_addrs(struct phxfs_page_table *pt,
				 uint64_t *addrs, uint32_t n_addrs)
{
	struct nvidia_p2p_page_table *npt = pt->priv;
	uint32_t i;

	if (npt->entries != n_addrs)
		return -EINVAL;

	for (i = 0; i < n_addrs; i++) {
		if (!npt->pages[i])
			return -ENOMEM;
		addrs[i] = npt->pages[i]->physical_address;
	}
	return 0;
}

static void nvidia_free_page_table(struct phxfs_page_table *pt)
{
	if (pt->priv)
		nvidia_p2p_free_page_table_p(pt->priv);
	kfree(pt);
}

static struct phxfs_p2p_ops nvidia_p2p_ops = {
	.name            = "nvidia",
	.init            = nvidia_init,
	.exit            = nvidia_exit,
	.get_pages       = nvidia_get_pages,
	.put_pages       = nvidia_put_pages,
	.get_n_pages     = nvidia_get_n_pages,
	.get_phys_addrs = nvidia_get_phys_addrs,
	.free_page_table = nvidia_free_page_table,
	.page_size       = 64 * 1024,  /* NVIDIA GPU page = 64 KiB */
};

int nvidia_backend_register(void)
{
	int ret = nvidia_init();
	if (ret)
		return ret;
	return phxfs_p2p_register_backend(&nvidia_p2p_ops);
}
