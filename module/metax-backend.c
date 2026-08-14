/*
 * METAX P2P Backend
 *
 * Wraps the METAX driver's P2P API behind the
 * vendor-agnostic phxfs_p2p_ops interface.
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>

#include "phxfs-backend.h"
#include "phxfs.h"       /* phxfs_err / phxfs_info */

struct metax_p2p_page_table {
	void *handle;
	struct sg_table *sgt;
	uint32_t virtual_entries;
	uint32_t page_size;
	void (*free_cb)(void *);
    void *data;
	uint32_t entries;
};

/* Function pointer types for METAX P2P symbols (loaded via __symbol_get) */
typedef int (*metax_p2p_acquire_mem_fptr)(uint64_t, size_t, void **, int (*)(void *), void *);
typedef int (*metax_p2p_get_mem_fptr)(void *, struct sg_table **);
typedef int (*metax_p2p_put_mem_fptr)(void *, struct sg_table *);
typedef uint64_t (*metax_p2p_get_bus_offset_fptr)(void *);
typedef uint32_t (*metax_p2p_get_page_size_fptr)(void *);
typedef void (*metax_p2p_release_mem_fptr)(void *);

/* ------------------------------------------------------------------ */
/* METAX P2P symbol pointers (loaded via __symbol_get)               */
/* ------------------------------------------------------------------ */

static metax_p2p_acquire_mem_fptr    metax_p2p_acquire_mem_p;
static metax_p2p_get_mem_fptr        metax_p2p_get_mem_p;
static metax_p2p_put_mem_fptr        metax_p2p_put_mem_p;
static metax_p2p_get_bus_offset_fptr metax_p2p_get_bus_offset_p;
static metax_p2p_get_page_size_fptr  metax_p2p_get_page_size_p;
static metax_p2p_release_mem_fptr    metax_p2p_release_mem_p;

#ifdef HAVE_MODULE_MUTEX
extern struct mutex module_mutex;
#endif

static void metax_put_symbols(void)
{
	if (metax_p2p_acquire_mem_p)    __symbol_put("metax_p2p_acquire_mem");
	if (metax_p2p_get_mem_p)        __symbol_put("metax_p2p_get_mem");
	if (metax_p2p_put_mem_p)        __symbol_put("metax_p2p_put_mem");
	if (metax_p2p_get_bus_offset_p) __symbol_put("metax_p2p_get_bus_offset");
	if (metax_p2p_get_page_size_p)  __symbol_put("metax_p2p_get_page_size");
	if (metax_p2p_release_mem_p)    __symbol_put("metax_p2p_release_mem");

	metax_p2p_acquire_mem_p    = NULL;
	metax_p2p_get_mem_p        = NULL;
	metax_p2p_put_mem_p        = NULL;
	metax_p2p_get_bus_offset_p = NULL;
	metax_p2p_get_page_size_p  = NULL;
	metax_p2p_release_mem_p    = NULL;
}

static int metax_load_symbols(void)
{
#ifdef HAVE_MODULE_MUTEX
	mutex_lock(&module_mutex);
#endif
	metax_p2p_acquire_mem_p = __symbol_get("metax_p2p_acquire_mem");
	if (!metax_p2p_acquire_mem_p) goto err;

	metax_p2p_get_mem_p = __symbol_get("metax_p2p_get_mem");
	if (!metax_p2p_get_mem_p) goto err;

	metax_p2p_put_mem_p = __symbol_get("metax_p2p_put_mem");
	if (!metax_p2p_put_mem_p) goto err;

	metax_p2p_get_bus_offset_p = __symbol_get("metax_p2p_get_bus_offset");
	if (!metax_p2p_get_bus_offset_p) goto err;

	metax_p2p_get_page_size_p = __symbol_get("metax_p2p_get_page_size");
	if (!metax_p2p_get_page_size_p) goto err;

	metax_p2p_release_mem_p = __symbol_get("metax_p2p_release_mem");
	if (!metax_p2p_release_mem_p) goto err;

#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	return 0;

err:
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	metax_put_symbols();
	return -ENOSYS;
}

/* ------------------------------------------------------------------ */
/* phxfs_p2p_ops implementation                                       */
/* ------------------------------------------------------------------ */

static int metax_init(void)
{
	int ret = metax_load_symbols();
	if (ret)
		phxfs_err("phxfs: failed to load metax_p2p symbols\n");
	return ret;
}

static void metax_exit(void)
{
	metax_put_symbols();
}

static int metax_acquire_callback_wrapper(void *arg)
{
	struct metax_p2p_page_table *pt = (struct metax_p2p_page_table *)arg;
	if (pt && pt->free_cb) {
		pt->free_cb(pt->data);
	}
	return 0;
}

static int metax_p2p_get_pages_p(uint64_t vaddr,
		uint64_t length,
		struct metax_p2p_page_table **page_table,
		void (*free_cb)(void *),
		void *data)
{
	int err;
	struct metax_p2p_page_table *pt;
	uint32_t page_size;
	uint32_t virtual_entries;

	if(metax_p2p_acquire_mem_p == NULL || metax_p2p_get_mem_p == NULL ||
			metax_p2p_get_page_size_p == NULL) {
		return -ENOMEM;
	}

	pt = kmalloc(sizeof(struct metax_p2p_page_table), GFP_KERNEL);
	if (pt == NULL) {
		phxfs_err("Failed to allocate page table\n");
		return -ENOMEM;
	}

	pt->free_cb = free_cb;
	pt->data = data;

	err = metax_p2p_acquire_mem_p(vaddr, length, &pt->handle, metax_acquire_callback_wrapper, pt);
	phxfs_info("metax_p2p_acquire_mem_p: vaddr=0x%llx, length=0x%llx, handle=%p, pt=%p, err=%d\n",
		 vaddr, length, pt->handle, pt, err);
	if (err != 0) {
		phxfs_err("metax_p2p_acquire_mem failed: %d\n", err);
		kfree(pt);
		return err;
	}

	err = metax_p2p_get_mem_p(pt->handle, &pt->sgt);
	phxfs_info("metax_p2p_get_mem_p: handle=%p, sgt=%p, err=%d\n", pt->handle, pt->sgt, err);
	if (err != 0) {
		phxfs_err("metax_p2p_get_mem failed: %d\n", err);
		if(metax_p2p_release_mem_p)
			metax_p2p_release_mem_p(pt->handle);
		kfree(pt);
		return err;
	}

	page_size = metax_p2p_get_page_size_p(pt->handle);
	phxfs_info("metax_p2p_get_page_size_p: handle=%p, page_size=%u\n", pt->handle, page_size);
	if (page_size == 0) {
		page_size = 1 << 16;
	}
	virtual_entries = length / page_size;
	pt->virtual_entries = virtual_entries;
    phxfs_info("metax_p2p_get_page_size_p: handle=%p, virtual_entries=%u\n", pt->handle, virtual_entries);

	pt->page_size = ((struct p2p_vmap*)data)->page_size;
	*page_table = pt;
	return 0;
}

static int metax_get_pages(uint64_t vaddr, uint64_t length,
		struct phxfs_page_table **pt,
		void (*free_cb)(void *), void *data)
{
	struct phxfs_page_table *h;
	int err;

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	if (!h)
		return -ENOMEM;

	err = metax_p2p_get_pages_p(vaddr, length,
				     (struct metax_p2p_page_table **)&h->priv,
				     free_cb, data);
	if (err) {
		kfree(h);
		return err;
	}
	*pt = h;
	return 0;
}

static int metax_p2p_put_pages_p(uint64_t vaddr,
		struct metax_p2p_page_table *page_table)
{
	if (page_table == NULL) {
		return 0;
	}

	if (page_table->sgt != NULL) {
		if(metax_p2p_put_mem_p) {
			phxfs_info("metax_p2p_put_mem_p: handle=%p, sgt=%p\n", page_table->handle, page_table->sgt);
			metax_p2p_put_mem_p(page_table->handle, page_table->sgt);
		}
	}

	if (page_table->handle != NULL) {
		if(metax_p2p_release_mem_p) {
			phxfs_info("metax_p2p_release_mem_p: handle=%p\n", page_table->handle);
			metax_p2p_release_mem_p(page_table->handle);
		}
	}

	kfree(page_table);
	page_table = NULL;
	return 0;
}

static void metax_put_pages(uint64_t vaddr, struct phxfs_page_table *pt)
{
	metax_p2p_put_pages_p(vaddr, (struct metax_p2p_page_table *)pt->priv);
	kfree(pt);
	pt = NULL;
}

static int metax_p2p_dma_map_pages_p(struct metax_p2p_page_table *page_table,
		uint32_t request_page_size,
		uint32_t request_naddr,
	    uint64_t** dma_addresses)
{
	uint64_t offset;
	uint32_t i;
	struct scatterlist *sg;
	uint64_t *dma_addrs;
	uint32_t entries = 0;

	if (page_table == NULL || page_table->sgt == NULL) {
		return -EINVAL;
	}

	if(metax_p2p_get_bus_offset_p == NULL) {
		return -ENOMEM;
	}

	offset = metax_p2p_get_bus_offset_p(page_table->handle);
	phxfs_info("metax_p2p_get_bus_offset_p: handle=%p, offset=0x%llx\n",
		 page_table->handle, offset);

	dma_addrs = kmalloc(request_naddr * sizeof(uint64_t), GFP_KERNEL);
	if (dma_addrs == NULL) {
		phxfs_err("Failed to allocate dma addresses array\n");
		return -ENOMEM;
	}

	for_each_sg(page_table->sgt->sgl, sg, page_table->sgt->nents, i) {
		uint64_t addr = sg->dma_address;
		uint32_t len = sg->length;
		uint32_t pages = (len - offset) / request_page_size;
		uint32_t j;

		for (j = 0; j < pages; j++) {
			if (entries < request_naddr) {
				dma_addrs[entries] = addr + offset + j * request_page_size;
				phxfs_info("metax_p2p calc dma: handle=%p, sg=%p, index=%u, phy_addr=%llx, dma_addr=%llx, off 0x%x\n",
					page_table->handle, sg, entries, addr, dma_addrs[entries], sg->offset);
				entries++;
			}
		}
	}

    if (entries < request_naddr) {
		phxfs_err("Mem allocation not success: required page number %u, allocated page number %u\n", request_naddr, entries);
		return -ENOMEM;
	}

	*dma_addresses = dma_addrs;
	return 0;
}

static uint32_t metax_get_n_pages(struct phxfs_page_table *pt)
{
	struct metax_p2p_page_table *mpt = pt->priv;
	return mpt->entries;
}

static int metax_get_phys_addrs(struct phxfs_page_table *pt,
				 uint64_t *addrs, uint32_t n_addrs)
{
	struct metax_p2p_page_table *mpt = pt->priv;
	uint64_t *dma_addrs;
	int i, err;

	err = metax_p2p_dma_map_pages_p(mpt, mpt->page_size, n_addrs, &dma_addrs);
	for (i = 0; i < n_addrs; i++) {
		addrs[i] = dma_addrs[i];
	}
	mpt->entries = n_addrs;
    return err;
}

static int metax_p2p_free_page_table_p(struct metax_p2p_page_table *page_table)
{
	if (page_table == NULL)
		return 0;

	kfree(page_table);
	page_table = NULL;
	return 0;
}

static void metax_free_page_table(struct phxfs_page_table *pt)
{
	if (pt->priv)
		metax_p2p_free_page_table_p((struct metax_p2p_page_table *)pt->priv);
	kfree(pt);
	pt = NULL;
}

static struct phxfs_p2p_ops metax_p2p_ops =
{
	.name            = "metax",
	.init            = metax_init,
	.exit            = metax_exit,
	.get_pages       = metax_get_pages,
	.put_pages       = metax_put_pages,
	.get_n_pages     = metax_get_n_pages,
	.get_phys_addrs  = metax_get_phys_addrs,
	.free_page_table = metax_free_page_table,
	.page_size       = 64 * 1024,  /* METAX GPU page = 64 KiB */
};

int metax_backend_register(void)
{
	int ret = metax_init();
	if (ret)
		return ret;
	return phxfs_p2p_register_backend(&metax_p2p_ops);
}
