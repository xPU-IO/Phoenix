#ifndef __PHOENIX_H__
#define __PHOENIX_H__

#include <linux/types.h>
#include <linux/blk-mq.h>
#include <linux/nvme.h>
#include <linux/memremap.h>
#include <linux/genalloc.h>
#include <linux/cdev.h>
#include <linux/mmzone.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/printk.h>
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
#ifndef CONFIG_PHXFS_VENDOR_METAX
#define PHXFS_RESERVED_SIZE    ((u64)128 * 1024 * 1024)  /* 128 MiB reserved at head/tail */
#else
#define PHXFS_RESERVED_SIZE    0
#endif

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

struct phxfs_bar_segment {
	u64 phys_start;    /* physical start address of this segment */
	u64 size;          /* segment size (multiple of PHXFS_REMAP_UNIT_SIZE) */
	void *va;          /* virtual address from devm_memremap_pages */
	int refcount;      /* live registrations covering this unit (staging mode).
			    * 0 means "reclaimable": the release worker may unmap
			    * it, and until it does the unit stays valid and is
			    * re-adopted by the next registration that needs it. */
	struct pci_p2pdma_pagemap *p2p_pgmap;
};

struct pci_p2pdma_pagemap {
    struct dev_pagemap pgmap;
    struct pci_dev provider;
    u64 bus_offset;
};

struct phxfs_dev {
    struct pci_dev *dev; /*pci device */
    int domain;
    unsigned int bus;
    unsigned int devfn;
    u64 size; /* HBM pci bar 4 size */
    u64 paddr; /* HBM bus address space addr */
    struct resource pgmap_res;
    struct device device; /* char device. */
    struct cdev cdev;
    int idx;
    struct pci_p2pdma_pagemap *p2p_pgmap; /* legacy single-segment pgmap (kept for compat) */
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

struct phxfs_ctrl {
    struct phxfs_dev phx_dev[MAX_DEV_NUM];
    int dev_num;
};

/* P2P mapping descriptor (vendor-agnostic) */
struct p2p_vmap;
typedef void (*release_fn)(struct p2p_vmap*);

struct gpu_region {
    struct phxfs_page_table *pt;
};

struct p2p_vmap {
    u64          gpuvaddr;
    u64          gpupaddr;
    u64          size;
    u64          cpuvaddr;
    release_fn   release;
    struct page **pages;
    unsigned long page_size;
    void        *data;           /* points to struct gpu_region */
    unsigned long n_addrs;
    uint64_t     addrs[1];
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

struct phxfs_ioctl_io_s {
    u64 cpuvaddr; /* cpu vaddr */
    loff_t offset; /* file offset */
    u64 size; /* Read/Write length */
    u64 end_fence_value; /* End fence value for DMA completion */
    s64 ioctl_return;
    int fd; /* File descriptor */
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_io_s phxfs_ioctl_io_t;

struct phxfs_ioctl_ret_s {
    s64 ret;
    u8 padding[40];
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_ret_s phxfs_ioctl_ret_t;

union phxfs_ioctl_para_s {
    struct phxfs_ioctl_map_s map_param;
    struct phxfs_ioctl_io_s io_para;
    struct phxfs_ioctl_ret_s ret;
} __attribute__((packed, aligned(8)));
typedef union phxfs_ioctl_para_s phxfs_ioctl_para_t;


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