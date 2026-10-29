/*
 * Iluvatar P2P backend
 *
 * Pins a cudaMalloc device VA via itr_p2p_mem_ctx_init() plus that ctx's
 * dma_map_pages(). Keep the range only when its bus address sits in the
 * same pci_dev's MEM BAR.
 * itr_peer_mem.h has no free_callback slot. __peer_mem_put_pages does not
 * call out, and the kref_put on the allocation is commented out in the
 * driver. free_cb is ignored. cudaFree while an I/O is in flight is a
 * use-after-free until the driver grows a reclaim callback.
 *
 * struct layouts below must match the installed ixdriver:
 *   kmd/itr/itr_peer_mem.h          itr_peer_mem_ctx
 *   kmd/itr/kmdlib/itr_mem_handle.h mem_handle prefix through ctx
 *   kmd/itr/kmdlib/itr_common_def.h itr_addr_range
 * A wrong mem_handle prefix fails closed: dva must cover the VA we looked up.
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/kernel.h>
#include <linux/pci.h>

#include "phxfs-backend.h"
#include "phxfs.h"

#define ILUVATAR_PAGE_SIZE 4096ULL
#define ILUVATAR_MAX_MINOR 16

struct itr_addr_range {
	u64 addr_start;
	u64 size;
};

struct itr_address_range {
	u64 addr_start;
	u64 size;
};

/* Must match kmd/itr/itr_peer_mem.h. */
struct itr_peer_mem_ctx {
	unsigned long va;
	unsigned long size;
	unsigned long page_size;
	void *private_data;
	union {
		unsigned long dma_addr;
		struct {
			struct itr_address_range *dma_addr_ranges;
			int chunk_count;
		} sg_lm;
	};
	unsigned long (*get_page_size)(struct itr_peer_mem_ctx *ctx);
	int (*get_pages)(struct itr_peer_mem_ctx *ctx, unsigned long addr,
			 unsigned long size, void *sg_head);
	int (*put_pages)(struct itr_peer_mem_ctx *ctx, void *sg_head);
	int (*dma_map_pages)(struct itr_peer_mem_ctx *ctx, void *sg_head,
			     void *dma_device);
	int (*dma_unmap_pages)(struct itr_peer_mem_ctx *ctx, void *sg_head,
			       void *dma_device);
};

/*
 * Prefix of kmd/itr/kmdlib/itr_mem_handle.h struct mem_handle on x86_64 SysV.
 * dva at 16, size at 32, ctx at 40. Checked at runtime against the VA.
 */
struct iluvatar_mem_prefix {
	u32 id;
	u8 hdl_type;
	void *sys_va_addr;
	unsigned long dva;
	void *dva_allocator;
	u64 size;
	void *ctx;
};

struct iluvatar_pt {
	u64 *pages;
	u32 n_pages;
};

typedef int (*itr_p2p_mem_ctx_init_fptr)(struct itr_peer_mem_ctx *ctx);

static itr_p2p_mem_ctx_init_fptr itr_p2p_mem_ctx_init_p;

static void iluvatar_put_symbols(void)
{
	if (itr_p2p_mem_ctx_init_p)
		__symbol_put("itr_p2p_mem_ctx_init");
	itr_p2p_mem_ctx_init_p = NULL;
}

static int iluvatar_load_symbols(void)
{
	itr_p2p_mem_ctx_init_p = __symbol_get("itr_p2p_mem_ctx_init");
	if (itr_p2p_mem_ctx_init_p)
		return 0;
	return -ENOSYS;
}

static int iluvatar_init(void)
{
	int ret = iluvatar_load_symbols();

	BUILD_BUG_ON(offsetof(struct iluvatar_mem_prefix, dva) != 16);
	BUILD_BUG_ON(offsetof(struct iluvatar_mem_prefix, size) != 32);
	BUILD_BUG_ON(offsetof(struct iluvatar_mem_prefix, ctx) != 40);
	if (ret)
		phxfs_err("phxfs: iluvatar symbols missing (load iluvatar.ko first)\n");
	return ret;
}

static void iluvatar_exit(void)
{
	iluvatar_put_symbols();
}

static int iluvatar_expand(struct itr_addr_range *ranges, int chunks,
			   u64 **out, u32 *n_out)
{
	u64 bytes = 0;
	u32 n, filled;
	u64 *pages;
	int c;

	for (c = 0; c < chunks; c++) {
		if (!ranges[c].size ||
		    (ranges[c].addr_start & (ILUVATAR_PAGE_SIZE - 1)) ||
		    (ranges[c].size & (ILUVATAR_PAGE_SIZE - 1)))
			return -EINVAL;
		if (bytes > U64_MAX - ranges[c].size)
			return -EINVAL;
		bytes += ranges[c].size;
	}
	if (!bytes || (bytes & (ILUVATAR_PAGE_SIZE - 1)))
		return -EINVAL;
	n = bytes / ILUVATAR_PAGE_SIZE;
	pages = kvcalloc(n, sizeof(*pages), GFP_KERNEL);
	if (!pages)
		return -ENOMEM;
	filled = 0;
	for (c = 0; c < chunks; c++) {
		u32 np = ranges[c].size / ILUVATAR_PAGE_SIZE;
		u32 page;

		for (page = 0; page < np; page++) {
			if (filled >= n) {
				kvfree(pages);
				return -EINVAL;
			}
			pages[filled++] = ranges[c].addr_start +
					  page * ILUVATAR_PAGE_SIZE;
		}
	}
	if (filled != n) {
		kvfree(pages);
		return -EINVAL;
	}
	*out = pages;
	*n_out = n;
	return 0;
}

static bool iluvatar_addr_in_bars(struct pci_dev *pdev, u64 addr)
{
	int i;

	for (i = 0; i < PCI_STD_NUM_BARS; i++) {
		resource_size_t start, len;

		if (!(pci_resource_flags(pdev, i) & IORESOURCE_MEM))
			continue;
		start = pci_resource_start(pdev, i);
		len = pci_resource_len(pdev, i);
		if (!len)
			continue;
		if (addr >= start && addr < start + len)
			return true;
	}
	return false;
}

/* Map against one pci_dev, copy the clipped ranges, then drop the DMA map. */
static int iluvatar_map_one(struct itr_peer_mem_ctx *pctx, struct device *dma_dev,
			    u64 length, struct itr_addr_range **out, int *n_out)
{
	struct itr_addr_range *ranges;
	int dummy = 0;
	int rc, i, used;
	u64 left = length;

	rc = pctx->dma_map_pages(pctx, &dummy, dma_dev);
	if (rc || pctx->sg_lm.chunk_count <= 0 || !pctx->sg_lm.dma_addr_ranges) {
		if (pctx->dma_unmap_pages)
			pctx->dma_unmap_pages(pctx, &dummy, dma_dev);
		return rc ? rc : -EINVAL;
	}
	ranges = kvmalloc_array(pctx->sg_lm.chunk_count, sizeof(*ranges),
				 GFP_KERNEL);
	if (!ranges) {
		pctx->dma_unmap_pages(pctx, &dummy, dma_dev);
		return -ENOMEM;
	}
	used = 0;
	for (i = 0; i < pctx->sg_lm.chunk_count && left; i++) {
		u64 sz = pctx->sg_lm.dma_addr_ranges[i].size;

		if (sz > left)
			sz = left;
		ranges[used].addr_start = pctx->sg_lm.dma_addr_ranges[i].addr_start;
		ranges[used].size = sz;
		left -= sz;
		used++;
	}
	if (pctx->dma_unmap_pages)
		pctx->dma_unmap_pages(pctx, &dummy, dma_dev);
	if (left || used == 0) {
		kvfree(ranges);
		return -EINVAL;
	}
	*out = ranges;
	*n_out = used;
	return 0;
}

static int iluvatar_map_bar(struct itr_peer_mem_ctx *pctx, u64 length,
			   struct itr_addr_range **out, int *n_out)
{
	struct pci_dev *pdev = NULL;
	int rc = -ENODEV;

	if (!pctx->dma_map_pages)
		return -ENOSYS;
	/*
	 * The VA's HBM lives in one card's BAR. dma_map_pages() is asked
	 * once per 0x1e3e device; keep the result only when the bus address
	 * falls inside that same device's MEM BAR.
	 */
	while ((pdev = pci_get_device(0x1e3e, PCI_ANY_ID, pdev)) != NULL) {
		struct itr_addr_range *ranges = NULL;
		int n = 0;

		rc = iluvatar_map_one(pctx, &pdev->dev, length, &ranges, &n);
		if (!rc && n > 0 && iluvatar_addr_in_bars(pdev, ranges[0].addr_start)) {
			*out = ranges;
			*n_out = n;
			pci_dev_put(pdev);
			return 0;
		}
		kvfree(ranges);
		if (!rc)
			rc = -EINVAL;
	}
	return rc;
}

static int iluvatar_get_pages(uint64_t vaddr, uint64_t length,
			      struct phxfs_page_table **pt,
			      void (*free_cb)(void *), void *data)
{
	struct itr_peer_mem_ctx *pctx;
	struct iluvatar_mem_prefix *mem;
	struct phxfs_page_table *h;
	struct iluvatar_pt *ipt;
	struct itr_addr_range *ranges = NULL;
	int chunk_count = 0;
	int rc;

	if (!vaddr || !length || (length & (ILUVATAR_PAGE_SIZE - 1)) ||
	    (vaddr & (ILUVATAR_PAGE_SIZE - 1)))
		return -EINVAL;

	pctx = kzalloc(sizeof(*pctx), GFP_KERNEL);
	if (!pctx)
		return -ENOMEM;
	pctx->va = (unsigned long)vaddr;
	pctx->size = (unsigned long)length;
	rc = itr_p2p_mem_ctx_init_p(pctx);
	if (rc) {
		phxfs_err("phxfs: itr_p2p_mem_ctx_init(%llx,%llx) = %d\n",
			  vaddr, length, rc);
		kfree(pctx);
		return rc;
	}
	mem = pctx->private_data;
	if (!mem || vaddr < mem->dva ||
	    length > mem->size || vaddr + length > mem->dva + mem->size) {
		phxfs_err("phxfs: iluvatar mem prefix does not cover VA %llx (dva=%lx size=%llx)\n",
			  vaddr, mem ? mem->dva : 0, mem ? mem->size : 0);
		kfree(pctx);
		return -EPROTO;
	}

	/*
	 * itr_p2p_get_dev_pages() wants a userspace context handle, not the
	 * kernel itr_ctx pointer. The peer-mem ctx already found this VA.
	 * dma_map_pages() turns that allocation into BAR bus addresses.
	 * With IOMMU off those addresses are the BAR physical addresses.
	 */
	rc = iluvatar_map_bar(pctx, length, &ranges, &chunk_count);
	kfree(pctx);
	if (rc || !ranges || chunk_count <= 0) {
		phxfs_err("phxfs: iluvatar dma_map_pages(%llx,%llx) failed: %d\n",
			  vaddr, length, rc);
		return rc ? rc : -EINVAL;
	}

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	ipt = kzalloc(sizeof(*ipt), GFP_KERNEL);
	if (!h || !ipt) {
		kvfree(ranges);
		kfree(h);
		kfree(ipt);
		return -ENOMEM;
	}
	(void)free_cb;
	(void)data;
	rc = iluvatar_expand(ranges, chunk_count, &ipt->pages, &ipt->n_pages);
	kvfree(ranges);
	if (rc || ipt->n_pages != length / ILUVATAR_PAGE_SIZE) {
		kvfree(ipt->pages);
		kfree(ipt);
		kfree(h);
		phxfs_err("phxfs: iluvatar ranges are not %llu-byte pages (%d)\n",
			  ILUVATAR_PAGE_SIZE, rc);
		return rc ? rc : -EINVAL;
	}
	h->priv = ipt;
	*pt = h;
	return 0;
}

static void iluvatar_free_page_table(struct phxfs_page_table *pt);

static void iluvatar_put_pages(uint64_t vaddr, struct phxfs_page_table *pt)
{
	(void)vaddr;
	/* dma_map_pages was unmapped before we copied the bus addresses. */
	iluvatar_free_page_table(pt);
}

static uint32_t iluvatar_get_n_pages(struct phxfs_page_table *pt)
{
	struct iluvatar_pt *ipt = pt->priv;

	return ipt ? ipt->n_pages : 0;
}

static int iluvatar_get_phys_addrs(struct phxfs_page_table *pt,
				   uint64_t *addrs, uint32_t n_addrs)
{
	struct iluvatar_pt *ipt = pt->priv;
	u32 i;

	if (!ipt || ipt->n_pages != n_addrs)
		return -EINVAL;
	for (i = 0; i < n_addrs; i++) {
		if (!ipt->pages[i])
			return -ENOMEM;
		addrs[i] = ipt->pages[i];
	}
	return 0;
}

static void iluvatar_free_page_table(struct phxfs_page_table *pt)
{
	struct iluvatar_pt *ipt = pt->priv;

	/* The driver never calls free_cb, so this only drops our copy. */
	if (ipt)
		kvfree(ipt->pages);
	kfree(ipt);
	kfree(pt);
}

static struct phxfs_p2p_ops iluvatar_p2p_ops = {
	.name            = "iluvatar",
	.init            = iluvatar_init,
	.exit            = iluvatar_exit,
	.get_pages       = iluvatar_get_pages,
	.put_pages       = iluvatar_put_pages,
	.get_n_pages     = iluvatar_get_n_pages,
	.get_phys_addrs  = iluvatar_get_phys_addrs,
	.free_page_table = iluvatar_free_page_table,
	.page_size       = ILUVATAR_PAGE_SIZE,
};

int iluvatar_backend_register(void)
{
	int ret = iluvatar_init();

	if (ret)
		return ret;
	return phxfs_p2p_register_backend(&iluvatar_p2p_ops);
}
