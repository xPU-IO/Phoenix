/*
 * AMD P2P Backend
 *
 * Wraps the amdkfd peer-direct interface — exported by the amdgpu kernel
 * driver as amdkfd_query_rdma_interface() — behind the vendor-agnostic
 * phxfs_p2p_ops interface, mirroring the structure of nvidia-backend.c.
 *
 * The interface is defined upstream in the ROCm kernel driver:
 *   include/drm/amd_rdma.h        (MIT OR GPL-2.0 dual licensed)
 *   drivers/gpu/drm/amd/amdkfd/kfd_peerdirect.c
 * The pieces we need are copied locally below so no vendor header is
 * required at build time (same approach the METAX backend takes).
 *
 * Semantics (verified against kfd_peerdirect.c):
 *
 *  - get_pages() looks the KFD process buffer object covering
 *    [address, address+length) up by USER virtual address (a hipMalloc
 *    pointer), pins it in its ORIGINAL domain — VRAM stays VRAM, there is
 *    no migration to GTT — and builds an sg_table whose dma addresses are
 *    PCIe BAR bus addresses. IOMMU must be disabled for those to be plain
 *    bus addresses; AMD's own comment in kfd_peerdirect.c says:
 *      "returning assumes that iommu functionality should be disabled so
 *       we can assume that sg_table already contains DMA addresses."
 *
 *  - The free_callback fires when the driver force-reclaims the memory
 *    (process teardown, GECC, ...). After it returns, every resource tied
 *    to the amd_p2p_info is gone, so free_page_table() must only free our
 *    own heap — exactly the NVIDIA free_page_table contract.
 *
 *  - get_page_size() reports the GPU page granularity. The current KFD
 *    implementation returns PAGE_SIZE; we query it at init time instead
 *    of hard-coding anything, so a future driver reporting 64KiB or 2MiB
 *    keeps working unchanged.
 *
 * Works for both map modes: FULL and STAGING pin through the same
 * get_pages/put_pages path.
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/pci.h>
#include <linux/pid.h>
#include <linux/scatterlist.h>

#include "phxfs-backend.h"
#include "phxfs.h"       /* phxfs_err / phxfs_info */

/* ------------------------------------------------------------------ */
/* amdkfd rdma interface — local copy of ROCm include/drm/amd_rdma.h  */
/* ------------------------------------------------------------------ */

/** P2P handle: the sg_table is the "physical page table". */
struct amd_p2p_info {
	uint64_t         va;      /* user virtual address */
	uint64_t         size;    /* total size of allocation */
	struct pid      *pid;     /* process pid the VA belongs to */
	struct sg_table *pages;   /* DMA/Bus addresses */
	void            *priv;    /* AMD kernel driver private */
};

struct amd_rdma_interface {
	int (*get_pages)(uint64_t address, uint64_t length, struct pid *pid,
			 struct device *dma_dev,
			 struct amd_p2p_info **amd_p2p_data,
			 void (*free_callback)(void *client_priv),
			 void *client_priv);
	int (*put_pages)(struct amd_p2p_info **amd_p2p_data);
	int (*is_gpu_address)(uint64_t address, struct pid *pid);
	int (*get_page_size)(uint64_t address, uint64_t length,
			     struct pid *pid, unsigned long *page_size);
};

typedef int (*amdkfd_query_rdma_interface_fptr)(
	const struct amd_rdma_interface **ops);

/* ------------------------------------------------------------------ */
/* Interface pointer, loaded via __symbol_get (loose coupling)        */
/* ------------------------------------------------------------------ */

static amdkfd_query_rdma_interface_fptr p_amdkfd_query_rdma_interface;
static const struct amd_rdma_interface *p_iface;

/* GPU page granularity, queried from the driver at init — never
 * hard-coded (AMD is not NVIDIA: the value is whatever KFD reports). */
static uint64_t amd_page_size = PAGE_SIZE;

/* One AMD accelerator pci_dev, used as the dma_dev argument of
 * get_pages(). Any of them works: with the IOMMU off, dma_map_resource()
 * is an identity mapping, so the sg entries are BAR bus addresses no
 * matter which 64-bit DMA device we name. Reference held until exit. */
static struct pci_dev *amd_gpu_pdev;

static struct device *amd_get_dma_dev(void)
{
	struct pci_dev *pdev = NULL;

	if (amd_gpu_pdev)
		return &amd_gpu_pdev->dev;

	while ((pdev = pci_get_device(PHXFS_PCI_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PHXFS_PCI_ACCEL_CLASS)
			break;
	}
	if (!pdev) {
		phxfs_err("phxfs/amd: no AMD accelerator (class 0x%04x) found\n",
			  PHXFS_PCI_ACCEL_CLASS);
		return NULL;
	}
	amd_gpu_pdev = pdev;
	return &pdev->dev;
}

/* ------------------------------------------------------------------ */
/* Force-reclaim callback bridge                                      */
/* ------------------------------------------------------------------ */

/*
 * Wrapper we hand to the KFD interface together with each pin. When the
 * driver force-reclaims the memory it calls us back; we only forward to
 * the Phoenix core callback (which frees the core-side descriptors and
 * the page-table wrapper via free_page_table()).
 *
 * On return, everything behind w->info is already released by the driver
 * — never dereference it here and never call put_pages afterwards.
 */
struct amd_pt_wrap {
	struct amd_p2p_info *info;
	void (*free_cb)(void *);   /* Phoenix core callback */
	void *cb_data;
};

static void amd_force_reclaim_cb(void *client_priv)
{
	struct amd_pt_wrap *w = client_priv;

	if (!w)
		return;
	if (w->free_cb)
		w->free_cb(w->cb_data);
	/* w itself is freed by amd_free_page_table() through the core's
	 * callback path; w->info is dead, do not touch it. */
}

/* ------------------------------------------------------------------ */
/* phxfs_p2p_ops implementation                                       */
/* ------------------------------------------------------------------ */

static int amd_init(void)
{
	const struct amd_rdma_interface *iface = NULL;
	unsigned long ps = 0;
	int ret;

	p_amdkfd_query_rdma_interface =
		(amdkfd_query_rdma_interface_fptr)
		__symbol_get("amdkfd_query_rdma_interface");
	if (!p_amdkfd_query_rdma_interface) {
		phxfs_err("phxfs/amd: amdkfd_query_rdma_interface unavailable "
			  "(is the amdgpu driver loaded?)\n");
		return -ENOSYS;
	}

	if (p_amdkfd_query_rdma_interface(&iface) || !iface) {
		phxfs_err("phxfs/amd: amdkfd_query_rdma_interface failed\n");
		ret = -ENOSYS;
		goto err_put;
	}
	if (!iface->get_pages || !iface->put_pages || !iface->get_page_size) {
		phxfs_err("phxfs/amd: incomplete rdma interface\n");
		ret = -ENOSYS;
		goto err_put;
	}
	p_iface = iface;

	/* Ask the driver for the real GPU page granularity. The current KFD
	 * implementation answers PAGE_SIZE (local memory is physically
	 * contiguous, so the granularity is by its own comment "arbitrary"),
	 * but we take whatever it reports — no NVIDIA-style 64KiB assumed. */
	if (iface->get_page_size(0, PAGE_SIZE, NULL, &ps) == 0 && ps > 0)
		amd_page_size = ps;
	if (amd_page_size < PAGE_SIZE || (amd_page_size % PAGE_SIZE) != 0) {
		phxfs_err("phxfs/amd: unsupported page size %llu "
			  "(must be a multiple of %lu)\n",
			  (unsigned long long)amd_page_size, PAGE_SIZE);
		ret = -EINVAL;
		goto err_put;
	}
	phxfs_info("phxfs/amd: rdma interface ok, page_size=%llu\n",
		   (unsigned long long)amd_page_size);
	return 0;

err_put:
	__symbol_put("amdkfd_query_rdma_interface");
	p_amdkfd_query_rdma_interface = NULL;
	p_iface = NULL;
	return ret;
}

static void amd_exit(void)
{
	if (amd_gpu_pdev) {
		pci_dev_put(amd_gpu_pdev);
		amd_gpu_pdev = NULL;
	}
	if (p_amdkfd_query_rdma_interface)
		__symbol_put("amdkfd_query_rdma_interface");
	p_amdkfd_query_rdma_interface = NULL;
	p_iface = NULL;
}

static int amd_get_pages(uint64_t vaddr, uint64_t length,
			 struct phxfs_page_table **pt,
			 void (*free_cb)(void *), void *data)
{
	struct phxfs_page_table *h;
	struct amd_pt_wrap *w;
	struct device *dma_dev;
	int err;

	*pt = NULL;

	if (!p_iface || !p_iface->get_pages)
		return -ENOSYS;

	/* The KFD side aligns [addr, addr+len) up to PAGE_SIZE; require the
	 * caller to already be page-aligned so nothing is silently pinned
	 * beyond the requested range. */
	if (vaddr & (PAGE_SIZE - 1)) {
		phxfs_err("phxfs/amd: vaddr 0x%llx not page-aligned\n",
			  (unsigned long long)vaddr);
		return -EINVAL;
	}

	dma_dev = amd_get_dma_dev();
	if (!dma_dev)
		return -ENODEV;

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	w = kzalloc(sizeof(*w), GFP_KERNEL);
	if (!h || !w) {
		kfree(h);
		kfree(w);
		return -ENOMEM;
	}
	w->free_cb  = free_cb;
	w->cb_data  = data;
	w->info     = NULL;

	/* pid = NULL -> the calling process' address space. */
	err = p_iface->get_pages(vaddr, length, NULL, dma_dev,
				 &w->info, amd_force_reclaim_cb, w);
	if (err || !w->info) {
		if (!err)
			err = -ENOMEM;
		phxfs_err("phxfs/amd: get_pages(0x%llx, %llu) failed: %d\n",
			  (unsigned long long)vaddr,
			  (unsigned long long)length, err);
		kfree(w);
		kfree(h);
		return err;
	}
	h->priv = w;
	*pt = h;
	return 0;
}

static void amd_put_pages(uint64_t vaddr, struct phxfs_page_table *pt)
{
	struct amd_pt_wrap *w = pt->priv;

	/* Releases the sg_table, unpins the BO and drops the BO reference —
	 * the exact inverse of get_pages(). */
	p_iface->put_pages(&w->info);
	kfree(w);
	kfree(pt);
}

static uint32_t amd_get_n_pages(struct phxfs_page_table *pt)
{
	struct amd_pt_wrap *w = pt->priv;

	if (!w || !w->info)
		return 0;
	return (uint32_t)(w->info->size / amd_page_size);
}

/*
 * Flatten the sg_table into one bus address per device page.
 *
 * Defensive checks borrowed from the uGDS experience: every segment must
 * be page-size aligned, no non-page residual may remain, and the total
 * page count must match the caller's expectation exactly — anything else
 * is -EINVAL (fail-fast) rather than a silently shifted page list.
 *
 * BAR containment is NOT checked here: the backend does not know which
 * GPU's BAR backs this buffer. The core's phxfs_bar_offset_to_va() /
 * phxfs_staging_ensure_span() already reject any address that does not
 * fall inside the discovered device's BAR window.
 */
static int amd_get_phys_addrs(struct phxfs_page_table *pt,
			      uint64_t *addrs, uint32_t n_addrs)
{
	struct amd_pt_wrap *w = pt->priv;
	struct scatterlist *sg;
	uint32_t idx = 0;
	int i;

	if (!w || !w->info || !w->info->pages || !addrs)
		return -EINVAL;

	for_each_sg(w->info->pages->sgl, sg, w->info->pages->nents, i) {
		uint64_t addr = sg_dma_address(sg);
		uint64_t len  = sg_dma_len(sg);

		if (addr & (amd_page_size - 1)) {
			phxfs_err("phxfs/amd: sg entry %d addr 0x%llx not "
				  "%llu-byte aligned\n", i,
				  (unsigned long long)addr,
				  (unsigned long long)amd_page_size);
			return -EINVAL;
		}
		while (len >= amd_page_size && idx < n_addrs) {
			addrs[idx++] = addr;
			addr += amd_page_size;
			len  -= amd_page_size;
		}
		if (len > 0 && idx < n_addrs) {
			phxfs_err("phxfs/amd: sg entry %d has %llu-byte "
				  "non-page residual\n", i,
				  (unsigned long long)len);
			return -EINVAL;
		}
		if (idx >= n_addrs)
			break;
	}
	if (idx != n_addrs) {
		phxfs_err("phxfs/amd: page count mismatch: got %u, "
			  "expected %u\n", idx, n_addrs);
		return -EINVAL;
	}
	return 0;
}

static void amd_free_page_table(struct phxfs_page_table *pt)
{
	struct amd_pt_wrap *w = pt->priv;

	/* Force-reclaim path: the driver already released everything behind
	 * w->info — do NOT call put_pages (double free) and do NOT deref
	 * w->info. Only our own heap goes away here. */
	kfree(w);
	kfree(pt);
}

static struct phxfs_p2p_ops amd_p2p_ops = {
	.name            = "amd",
	.init            = amd_init,
	.exit            = amd_exit,
	.get_pages       = amd_get_pages,
	.put_pages       = amd_put_pages,
	.get_n_pages     = amd_get_n_pages,
	.get_phys_addrs = amd_get_phys_addrs,
	.free_page_table = amd_free_page_table,
	/* Overwritten at register time with the value the driver reports —
	 * see amd_backend_register(). */
	.page_size       = PAGE_SIZE,
};

int amd_backend_register(void)
{
	int ret = amd_init();
	if (ret)
		return ret;
	amd_p2p_ops.page_size = amd_page_size;
	return phxfs_p2p_register_backend(&amd_p2p_ops);
}
