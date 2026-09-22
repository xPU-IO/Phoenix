#ifndef __PHOENIX_H__
#define __PHOENIX_H__

#include <linux/types.h>
#include <linux/version.h>
#include <linux/blk-mq.h>
#include <linux/nvme.h>
#include <linux/memremap.h>
#include <linux/genalloc.h>
#include <linux/cdev.h>
#include <linux/mmzone.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>
#include <linux/printk.h>
#include <linux/refcount.h>
#include <linux/workqueue.h>

#define MAX_DEV_NUM 16
#define MAX_GPU_DEVS 64

/* Forward declaration — full definition in phxfs-backend.h */
struct phxfs_page_table;

/* Phoenix logging macros */
extern int phxfs_debug;

#define phxfs_info(fmt, ...)					\
	do {							\
		if (phxfs_debug)				\
			printk(KERN_INFO fmt, ##__VA_ARGS__);	\
	} while (0)

#define phxfs_warn(fmt, ...)					\
	printk(KERN_WARNING fmt, ##__VA_ARGS__)

#define phxfs_err(fmt, ...)					\
	printk(KERN_ERR fmt, ##__VA_ARGS__)

/*
 * FULL mode only: block size at which the BAR is probed for PAT conflicts and
 * split into segments at probe time. Pure probe bookkeeping.
 */
#define PHXFS_REMAP_UNIT_SIZE  ((u64)16 * 1024 * 1024)  /* 16 MiB probe block */

/*
 * STAGING mode: alignment (and size multiple) every remapped span must have.
 *
 * struct-page presence is tracked per sparsemem sub-section: pfn_valid()
 * resolves to pfn_section_valid(), which tests one bit per sub-section of
 * ms->usage->subsection_map (mmzone.h). A span that is not sub-section
 * aligned/sized therefore also marks whatever GPU memory shares its first or
 * last sub-section as pfn_valid, and that memory then fails
 * nvidia_p2p_dma_map_pages() (dma_map_resource() rejects pfn_valid pages),
 * i.e. RDMA/peermem registration of unrelated buffers breaks.
 *
 * So instead of growing the remap to a fixed grid, we require the staging
 * pool to be sub-section aligned and sized (libphoenix guarantees this) and
 * remap exactly the pool's span -- nothing more.
 */
#define PHXFS_REMAP_ALIGN      ((u64)PAGES_PER_SUBSECTION << PAGE_SHIFT)

/*
 * A BAR range that some other mapping already holds a non-write-back memtype
 * for, as parsed out of the kernel's PAT memtype list. devm_memremap_pages()
 * refuses to remap such a range, so the segment builder has to route around it.
 */
struct phxfs_pat_conflict {
	u64 start;
	u64 end;
};

/*
 * Read the PAT memtype list and extract the conflict ranges overlapping
 * [bar_start, bar_start + bar_len). Returns the number of conflicts found, or
 * a negative errno. conflicts[] is caller-allocated with max_entries capacity.
 */
int phxfs_read_pat_conflicts(u64 bar_start, u64 bar_len,
			     struct phxfs_pat_conflict *conflicts,
			     int max_entries);

/*
 * BAR mapping mode (module param phxfs_map_mode).
 *
 *   FULL    : remap the whole GPU BAR at probe. Any registered user GPU
 *             buffer gets a struct page and DMAs directly (SSD -> user GPU).
 *             This gives every BAR page a struct page (pfn_valid == true),
 *             which is what prevents nvidia_p2p_dma_map_pages() (RDMA/peermem)
 *             from mapping the same GPU afterwards.
 *   STAGING : do NOT remap at probe. Remap, on demand at each registration,
 *             exactly the BAR span the registered buffer occupies -- in
 *             practice Phoenix's own staging pool, which libphoenix allocates
 *             PHXFS_REMAP_ALIGN-aligned and -sized for this reason. A span
 *             already remapped by an earlier registration is reused.
 *             The rest of the BAR keeps pfn_valid == false, so user GPU memory
 *             stays registerable by RDMA/peermem. Data is DMA'd into the
 *             staging pool and copied D2D to the user buffer by libphoenix.
 *
 *             On-demand (rather than once-per-device) remapping is required
 *             because the BAR aperture offsets a buffer is pinned at are
 *             chosen by the GPU driver per pin: two processes, or two runs
 *             with a different allocation history (e.g. under a profiler),
 *             legitimately land on different BAR units.
 */
#define PHXFS_MAP_MODE_FULL     0
#define PHXFS_MAP_MODE_STAGING  1
/*
 * Compile-time default map mode, set by the build (CMake PHXFS_MAP_MODE).
 * STAGING is the default; FULL must be opted into (cmake -DPHXFS_MAP_MODE=full
 * or, at load time, insmod phoenixfs.ko phxfs_map_mode=0).
 */
#ifndef PHXFS_MAP_MODE_DEFAULT
#define PHXFS_MAP_MODE_DEFAULT  PHXFS_MAP_MODE_STAGING
#endif
extern int phxfs_map_mode;

/*
 * Mirror of the kernel-private struct pci_p2pdma_pagemap, which lives in
 * drivers/pci/p2pdma.c and in no header at all.
 *
 * The kernel's to_p2p_pgmap() container_of()s a pgmap pointer back to the
 * enclosing struct, so this layout has to match the running kernel exactly --
 * and the kernel only ever reads those fields once pdev->p2pdma is set, i.e.
 * after phxfs_p2pdma_bootstrap() has registered a bootstrap slice.
 *
 * Upstream reshuffled it twice:
 *   <= 6.6    { pgmap, struct pci_dev *provider, u64 bus_offset }
 *   6.7 - 6.18{ struct pci_dev *provider, u64 bus_offset, pgmap }  (4a7ce8334965)
 *   >= 7.0    { pgmap, struct p2pdma_provider *mem }
 * phxfs_p2pdma_verify_layout() checks this assumption at load time against a
 * pgmap the kernel built itself, so the next change fails the load instead of
 * dereferencing garbage from the NVMe submit path.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
struct phxfs_pgmap {
	struct dev_pagemap pgmap;
	struct p2pdma_provider *mem;
};
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(6, 7, 0)
struct phxfs_pgmap {
	struct pci_dev *provider;
	u64 bus_offset;
	struct dev_pagemap pgmap;
};
#else
struct phxfs_pgmap {
	struct dev_pagemap pgmap;
	struct pci_dev *provider;
	u64 bus_offset;
};
#endif

struct phxfs_bar_segment {
	u64 phys_start;    /* physical start address of this segment */
	u64 size;          /* segment size (multiple of PHXFS_REMAP_UNIT_SIZE) */
	void *va;          /* virtual address from devm_memremap_pages */
	int refcount;      /* live registrations covering this unit (staging mode).
			    * 0 means "reclaimable": the release worker may unmap
			    * it, and until it does the unit stays valid and is
			    * re-adopted by the next registration that needs it. */
	struct phxfs_pgmap *p2p_pgmap;
};

struct phxfs_dev {
    struct pci_dev *dev; /*pci device */
    int domain;
    unsigned int bus;
    unsigned int devfn;
    u64 size; /* HBM pci bar 4 size */
    u64 paddr; /* HBM bus address space addr */
    int bar; /* PCI BAR index backing paddr/size, -1 if not resolved */
    u64 bus_offset; /* pci_bus_address(bar) - pci_resource_start(bar) */
    u64 p2p_slice_start; /* p2pdma bootstrap slice, 0 if none */
    u64 p2p_slice_size;
    struct resource pgmap_res;
    struct device device; /* char device. */
    struct cdev cdev;
    int idx;
    struct phxfs_pgmap *p2p_pgmap; /* legacy single-segment pgmap (kept for compat) */
    void __iomem *pci_mem_va; /* legacy single-segment VA (kept for compat) */
    bool remap;
    struct phxfs_bar_segment *segments; /* dynamically allocated segment array,
                                         * kept sorted by phys_start */
    int num_segments;    /* number of successfully mapped segments */
    int seg_capacity;    /* allocated entries in segments[] (>= num_segments) */
    struct mutex seg_lock; /* guards segments/num_segments/seg_capacity */
    struct delayed_work seg_release_work; /* unmaps refcount==0 units */
    int seg_release_tries; /* remaining retries for the release worker */
};

/*
 * Point one of our pgmaps at the provider the kernel will look for. Only
 * meaningful once phxfs_p2pdma_bootstrap() has established pdev->p2pdma.
 *
 * <= 6.18 reads provider->p2pdma, so a plain pointer is enough and an
 * un-bootstrapped device degrades into NOT_SUPPORTED (-EREMOTEIO per IO)
 * rather than a fault. >= 7.0 dereferences the provider pointer itself, so
 * the lookup has to succeed or the caller must not proceed.
 */
static inline int phxfs_pgmap_set_provider(struct phxfs_pgmap *pg,
					   struct phxfs_dev *dev)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
	/*
	 * Without CONFIG_PCI_P2PDMA no provider can exist and the kernel never
	 * routes our pages through p2pdma, so NULL is both safe and correct.
	 */
	pg->mem = NULL;
#ifdef CONFIG_PCI_P2PDMA
	pg->mem = pcim_p2pdma_provider(dev->dev, dev->bar);
	if (!pg->mem)
		return -ENODEV;
#endif
	return 0;
#else
	pg->provider = dev->dev;
	pg->bus_offset = dev->bus_offset;
	return 0;
#endif
}

/*
 * Set up whatever the kernel's p2pdma mapping path needs before any BAR page
 * gets a struct page, and verify our assumptions about it. Called once per
 * device before the first remap. Implemented in phxfs-p2pdma.c; a no-op on
 * kernels built without CONFIG_PCI_P2PDMA.
 */
int phxfs_p2pdma_setup(struct phxfs_dev *phx_dev);

struct phxfs_ctrl {
    struct phxfs_dev phx_dev[MAX_DEV_NUM];
    int dev_num;
};

/* P2P mapping descriptor (vendor-agnostic) */
struct p2p_vmap;

struct gpu_region {
    struct phxfs_page_table *pt;
};

struct p2p_vmap {
    refcount_t   refs;            /* VMA owner + registered reclaim callback */
    atomic_t     callback_state;  /* live/running/dropped; see phxfs-mem.c */
    u64          gpuvaddr;
    u64          size;
    unsigned long page_size;
    void        *data;           /* points to struct gpu_region */
    unsigned long n_addrs;
};

struct phxfs_dev_info_s {
    u64 dev_id;
} __attribute__((packed, aligned(8)));

struct phxfs_ioctl_map_s {
    struct phxfs_dev_info_s dev;
    u64 c_vaddr;
    u64 c_size;
    u64 n_vaddr;
    u64 n_size;
    u64 end_addr;
    u32 sbuf_block;
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_map_s phxfs_ioctl_map_t;


#define PHXFS_IOCTL 0x88 /* 0x4c */
#define PHXFS_IOCTL_MAP _IOW(PHXFS_IOCTL, 1, struct phxfs_ioctl_map_s)
#define PHXFS_IOCTL_UNMAP _IOW(PHXFS_IOCTL, 2, struct phxfs_ioctl_map_s)

void phxfs_map_dev_release(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);

/*
 * Staging mode: give the registered buffer's BAR pages a ZONE_DEVICE mapping.
 * `phys` holds one BAR address per device page of `page_size` bytes.
 *
 * The buffer must occupy exactly one BAR span: `phys` has to be a dense,
 * ascending run, PHXFS_REMAP_ALIGN-aligned and -sized. That span is remapped
 * as-is -- what the caller registered is what gets struct pages, never more --
 * so GPU memory outside the pool keeps pfn_valid == false and stays
 * registerable by RDMA/peermem. A layout that would force a wider remap is
 * rejected with -EINVAL rather than silently breaking foreign registrations;
 * libphoenix allocates the staging pool so that it holds.
 *
 * Takes a reference on the span (a span an earlier registration already
 * mapped is reused) and returns its start address, which the caller must hand
 * back to phxfs_staging_put_span() when the registration goes away.
 *
 * Takes dev->seg_lock. Returns 0, or a negative errno with no reference taken.
 */
int phxfs_staging_ensure_span(struct phxfs_dev *dev, const u64 *phys,
			      unsigned long n, size_t page_size,
			      u64 *out_span_start);

/*
 * Drop the reference taken by phxfs_staging_ensure_span(). A span that reaches
 * zero references is handed to the release worker, which unmaps it once its
 * pages are idle.
 */
void phxfs_staging_put_span(struct phxfs_dev *dev, u64 span_start);

/* Stop the release worker (module unload / device teardown). */
void phxfs_staging_release_cancel(struct phxfs_dev *dev);

#endif
