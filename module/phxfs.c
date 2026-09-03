#include <asm/page.h>
#include <linux/version.h>
#include <linux/cdev.h>
#include <linux/ctype.h> //for isdigit()
#include <linux/device.h>
#include <linux/fcntl.h>
#include <linux/fdtable.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/ioport.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/memory.h>
#include <linux/module.h>
#include <linux/nvme_ioctl.h>
#include <linux/pci-p2pdma.h>
#include <linux/pci.h>
#include <linux/pci_ids.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/seq_buf.h>
#include <linux/thread_info.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/xarray.h>

#include "config-host.h"
#include "phxfs-mem.h"
#include "phxfs.h"

#include "phxfs-backend.h"

static DEFINE_IDA(phxfs_chr_minor_ida);
static dev_t phxfs_chr_devt;
static struct class *phxfs_chr_class;
struct device phxfs_chr_dev_device;
struct cdev phxfs_chr_dev;

#define PHXFS_MINORS 1

struct phxfs_ctrl ctrl;

#define NUM_THREADS 128
u32 npu_num;
uint64_t gpu_info_table[MAX_GPU_DEVS];

int phxfs_numa_node = -1;
module_param(phxfs_numa_node, int, 0644);
MODULE_PARM_DESC(phxfs_numa_node, "Target NUMA node for GPU filtering (-1 = no filter)");

int phxfs_debug = 0;
module_param(phxfs_debug, int, 0644);
MODULE_PARM_DESC(phxfs_debug, "Enable info-level logging (0=off, 1=on)");

int phxfs_map_mode = PHXFS_MAP_MODE_DEFAULT;
module_param(phxfs_map_mode, int, 0644);
MODULE_PARM_DESC(phxfs_map_mode,
	"BAR mapping mode: 1=staging (remap only a Phoenix staging pool on "
	"demand; leaves user GPU memory unmapped so RDMA/peermem can still "
	"register it) [default], 0=full BAR remap at load (direct SSD->GPU DMA; "
	"opt-in via cmake -DPHXFS_MAP_MODE=full or phxfs_map_mode=0)");

int phxfs_staging_release = 1;
module_param(phxfs_staging_release, int, 0644);
MODULE_PARM_DESC(phxfs_staging_release,
	"Staging mode: unmap a BAR unit once no registration references it "
	"(1=on [default], 0=keep every unit mapped until module unload)");

#define PHXFS_PAT_PATH "/sys/kernel/debug/x86/pat_memtype_list"
#define PHXFS_PAT_BUF_SIZE (64 * 1024) /* PAT file typically < 16 KiB */

/* PAT conflict range recorded during parsing */
struct phxfs_pat_conflict {
	u64 start;
	u64 end;
};

/*
 * Read PAT memtype list and extract conflict ranges that overlap
 * with [bar_start, bar_start + bar_len).
 * Returns number of conflicts found, or negative errno.
 * conflicts array is allocated by caller with max_entries capacity.
 */
static int phxfs_read_pat_conflicts(u64 bar_start, u64 bar_len,
				    struct phxfs_pat_conflict *conflicts,
				    int max_entries)
{
	struct file *filp;
	loff_t pos = 0;
	char *buf;
	int ret, n_conflicts = 0;
	u64 bar_end = bar_start + bar_len;

	buf = kzalloc(PHXFS_PAT_BUF_SIZE, GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	filp = filp_open(PHXFS_PAT_PATH, O_RDONLY, 0);
	if (IS_ERR(filp)) {
		phxfs_warn("phxfs: cannot open %s (err=%ld), "
		       "skipping PAT conflict detection\n",
		       PHXFS_PAT_PATH, PTR_ERR(filp));
		kfree(buf);
		return 0; /* graceful: no conflicts detected */
	}

	ret = kernel_read(filp, buf, PHXFS_PAT_BUF_SIZE - 1, &pos);
	filp_close(filp, NULL);

	if (ret <= 0) {
		phxfs_warn("phxfs: failed to read %s (ret=%d)\n",
		       PHXFS_PAT_PATH, ret);
		kfree(buf);
		return 0;
	}
	buf[ret] = '\0';

	/* Parse lines like: "write-combining @ 0x21a000800000-0x21a000900000" */
	{
		char *line = buf;
		while (line && *line) {
			char *nl = strchr(line, '\n');
			u64 cs, ce;
			char memtype[64] = {0};

			if (nl)
				*nl = '\0';

			/* Skip write-back entries - they don't conflict with cached mapping */
			if (strncmp(line, "write-back", 10) == 0 ||
			    strncmp(line, "PAT", 3) == 0) {
				line = nl ? nl + 1 : NULL;
				continue;
			}

			/* Try to parse: <memtype> @ 0x<start>-0x<end> */
			if (sscanf(line, "%63s @ 0x%llx-0x%llx", memtype, &cs, &ce) == 3) {
				/* Check if this range overlaps with our BAR window */
				if (cs < bar_end && ce > bar_start) {
					/* Clip to BAR range */
					u64 clipped_start = max(cs, bar_start);
					u64 clipped_end = min(ce, bar_end);

					if (n_conflicts < max_entries) {
						conflicts[n_conflicts].start = clipped_start;
						conflicts[n_conflicts].end = clipped_end;
						n_conflicts++;
					} else {
						phxfs_warn("phxfs: too many PAT conflicts "
						       "(>%d), some skipped\n", max_entries);
						break;
					}
				}
			}
			line = nl ? nl + 1 : NULL;
		}
	}

	kfree(buf);
	return n_conflicts;
}

/*
 * Compute which REMAP_UNIT_SIZE-aligned blocks within the HBM region are free of
 * PAT conflicts, then merge adjacent free blocks into segments.
 *
 * Strategy:
 *   - Reserve [paddr, paddr + PHXFS_RESERVED_SIZE) at the head
 *   - Reserve [paddr + hbm_size - PHXFS_RESERVED_SIZE, paddr + hbm_size) at the tail
 *   - Divide the middle region into REMAP_UNIT_SIZE blocks
 *   - Mark blocks that overlap any PAT conflict as "skip"
 *   - Merge consecutive non-skip blocks into segments
 *
 * Returns number of segments, or negative errno.
 * Caller must kfree(*out_segments) when done.
 */
static int phxfs_compute_bar_segments(
	u64 paddr, u64 hbm_size,
	struct phxfs_pat_conflict *conflicts, int n_conflicts,
	struct phxfs_bar_segment **out_segments)
{
	u64 usable_start, usable_end, region_size;
	int n_blocks, i, s;
	int n_segments = 0;
	bool *block_skip; /* true = has PAT conflict, skip this block */
	struct phxfs_bar_segment *segs;

	if (n_conflicts < 0) {
		return -EINVAL;
	}

	usable_start = paddr + PHXFS_RESERVED_SIZE;
	usable_end = paddr + hbm_size - PHXFS_RESERVED_SIZE;

	if (usable_end <= usable_start) {
		phxfs_warn("phxfs: HBM too small for head/tail reservation "
		       "(hbm_size=%llu MiB)\n", hbm_size / (1024 * 1024));
		return -ENOSPC;
	}

	region_size = usable_end - usable_start;
	n_blocks = (int)(region_size / PHXFS_REMAP_UNIT_SIZE);
	if (n_blocks <= 0)
		return -ENOSPC;

	block_skip = kcalloc(n_blocks, sizeof(bool), GFP_KERNEL);
	if (!block_skip)
		return -ENOMEM;

	/* Mark blocks that overlap with any PAT conflict */
	for (i = 0; i < n_blocks; i++) {
		u64 blk_start = usable_start + (u64)i * PHXFS_REMAP_UNIT_SIZE;
		u64 blk_end = blk_start + PHXFS_REMAP_UNIT_SIZE;
		int c;

		for (c = 0; c < n_conflicts; c++) {
			if (conflicts[c].start < blk_end && conflicts[c].end > blk_start) {
				block_skip[i] = true;
				break;
			}
		}
	}

	/* Count how many merged segments we need */
	for (i = 0; i < n_blocks; ) {
		if (!block_skip[i]) {
			/* Start of a run of non-skip blocks */
			while (i < n_blocks && !block_skip[i])
				i++;
			n_segments++;
		} else {
			i++;
		}
	}

	if (n_segments == 0) {
		phxfs_warn("phxfs: no remappable segments found\n");
		kfree(block_skip);
		return -ENOSPC;
	}

	segs = kcalloc(n_segments, sizeof(struct phxfs_bar_segment), GFP_KERNEL);
	if (!segs) {
		kfree(block_skip);
		return -ENOMEM;
	}

	/* Build merged segments */
	s = 0;
	for (i = 0; i < n_blocks && s < n_segments; ) {
		if (!block_skip[i]) {
			u64 seg_start = usable_start + (u64)i * PHXFS_REMAP_UNIT_SIZE;
			int run_len = 0;

			while (i < n_blocks && !block_skip[i]) {
				run_len++;
				i++;
			}

			segs[s].phys_start = seg_start;
			segs[s].size = (u64)run_len * PHXFS_REMAP_UNIT_SIZE;
			segs[s].va = NULL;
			segs[s].p2p_pgmap = NULL;
			s++;
		} else {
			i++;
		}
	}

	kfree(block_skip);
	*out_segments = segs;
	return s;
}

int extract_trailing_number(const char str[]) {
	int number = 0;
	int multiplier = 1;
	size_t len;
	int found_digit = 0;
	int i;
	len = strlen(str);

	for (i = len - 1; i >= 0; --i) {
		if (isdigit(str[i])) {
			number += (str[i] - '0') * multiplier;
			found_digit = 1;
			if (multiplier == 1) {
				multiplier = 10;
			} else if (found_digit) {
				break;
			}
		} else if (found_digit) {
			break;
		}
	}

	if (found_digit) {
		return number;
	} else {
		return -1;
	}
}

static int phxfs_devm_memremap(struct phxfs_dev *phx_dev) {
	struct phxfs_pat_conflict *conflicts = NULL;
	struct phxfs_bar_segment *segs = NULL;
	int n_conflicts, n_segments;
	int i, ret = 1;

	if (!phx_dev)
		return -EINVAL;

	phxfs_info("phxfs%d: BAR size=%llu MiB, paddr=0x%llx\n",
	       phx_dev->idx, phx_dev->size / (1024 * 1024), phx_dev->paddr);

	/* Max blocks = usable region / unit size, used as upper bound for conflicts */
	{
		int max_blocks = (int)((phx_dev->size - 2 * PHXFS_RESERVED_SIZE) / PHXFS_REMAP_UNIT_SIZE);
		if (max_blocks <= 0) {
			phxfs_warn("phxfs%d: BAR too small for head/tail reservation\n",
			       phx_dev->idx);
			return -ENOSPC;
		}

		/* Detect PAT conflicts within [paddr, paddr + size) */
		conflicts = kcalloc(max_blocks, sizeof(struct phxfs_pat_conflict), GFP_KERNEL);
		if (!conflicts)
			return -ENOMEM;

		n_conflicts = phxfs_read_pat_conflicts(phx_dev->paddr, phx_dev->size,
						       conflicts, max_blocks);
	}
	if (n_conflicts < 0) {
		phxfs_warn("phxfs%d: PAT conflict detection failed (%d), "
		       "falling back to full BAR remap\n", phx_dev->idx, n_conflicts);
		kfree(conflicts);
		/* Fallback: remap entire BAR as single segment */
		goto fallback_single;
	}

	phxfs_info("phxfs%d: found %d PAT conflict(s) in BAR region\n",
	       phx_dev->idx, n_conflicts);

	for (i = 0; i < n_conflicts; i++) {
		u64 off_start = conflicts[i].start - phx_dev->paddr;
		u64 off_end = conflicts[i].end - phx_dev->paddr;
		phxfs_info("phxfs%d:   conflict %d: offset 0x%llx-0x%llx "
		       "(%llu MiB - %llu MiB)\n",
		       phx_dev->idx, i, off_start, off_end,
		       off_start / (1024 * 1024), off_end / (1024 * 1024));
	}

	/* Compute segments (skip conflicts, merge adjacent free blocks) */
	n_segments = phxfs_compute_bar_segments(phx_dev->paddr, phx_dev->size,
						conflicts, n_conflicts, &segs);
	kfree(conflicts);

	if (n_segments <= 0) {
		phxfs_warn("phxfs%d: no valid segments computed, "
		       "falling back to full BAR remap\n", phx_dev->idx);
		goto fallback_single;
	}

	phxfs_info("phxfs%d: computed %d segment(s) after conflict skipping + merging\n",
	       phx_dev->idx, n_segments);
	for (i = 0; i < n_segments; i++) {
		u64 off = segs[i].phys_start - phx_dev->paddr;
		phxfs_info("phxfs%d:   segment %d: offset 0x%llx, size %llu MiB\n",
		       phx_dev->idx, i, off, segs[i].size / (1024 * 1024));
	}

	/* Perform devm_memremap_pages for each segment */
	phx_dev->segments = segs;
	phx_dev->num_segments = 0;
	phx_dev->seg_capacity = n_segments;

	for (i = 0; i < n_segments; i++) {
		struct dev_pagemap *pgmap;
		struct pci_p2pdma_pagemap *p2p_pgmap;

		p2p_pgmap = devm_kzalloc(&phx_dev->dev->dev,
					  sizeof(struct pci_p2pdma_pagemap), GFP_KERNEL);
		if (!p2p_pgmap) {
			ret = -ENOMEM;
			goto err_cleanup;
		}

		pgmap = &p2p_pgmap->pgmap;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
		pgmap->range.start = segs[i].phys_start;
		pgmap->range.end = segs[i].phys_start + segs[i].size - 1;
		pgmap->nr_range = 1;
#else
		pgmap->res.start = segs[i].phys_start;
		pgmap->res.end = segs[i].phys_start + segs[i].size - 1;
		pgmap->res.flags = IORESOURCE_MEM;
#endif
		pgmap->type = MEMORY_DEVICE_PCI_P2PDMA;

		segs[i].va = devm_memremap_pages(&phx_dev->dev->dev, pgmap);
		if (IS_ERR_OR_NULL(segs[i].va)) {
			phxfs_warn("phxfs%d: devm_memremap_pages failed for "
			       "segment %d (offset 0x%llx, size %llu MiB), err=%ld\n",
			       phx_dev->idx, i,
			       segs[i].phys_start - phx_dev->paddr,
			       segs[i].size / (1024 * 1024),
			       PTR_ERR(segs[i].va));
			segs[i].va = NULL;
			devm_kfree(&phx_dev->dev->dev, p2p_pgmap);
			/* Continue with remaining segments */
			continue;
		}

		segs[i].p2p_pgmap = p2p_pgmap;
		phx_dev->num_segments++;
		phxfs_info("phxfs%d: segment %d remapped: offset 0x%llx, "
		       "size %llu MiB, va=0x%lx\n",
		       phx_dev->idx, i,
		       segs[i].phys_start - phx_dev->paddr,
		       segs[i].size / (1024 * 1024),
		       (unsigned long)segs[i].va);
	}

	if (phx_dev->num_segments == 0) {
		phxfs_err("phxfs%d: all segment remaps failed\n", phx_dev->idx);
		ret = -ENOMEM;
		goto err_cleanup;
	}

	/* For compatibility: set pci_mem_va to first segment's VA */
	phx_dev->pci_mem_va = segs[0].va;
	phx_dev->remap = 1;

	/* Also set legacy p2p_pgmap to first segment's for any old cleanup paths */
	phx_dev->p2p_pgmap = segs[0].p2p_pgmap;

	/* Calculate total remapped size */
	{
		u64 total_remapped = 0;
		for (i = 0; i < phx_dev->num_segments; i++)
			total_remapped += segs[i].size;
		phxfs_info("phxfs%d: successfully remapped %d segments, "
		       "total %llu MiB out of %llu MiB HBM\n",
		       phx_dev->idx, phx_dev->num_segments,
		       total_remapped / (1024 * 1024),
		       phx_dev->size / (1024 * 1024));
	}

	return 0;

fallback_single:
	/* Legacy single-segment full BAR remap */
	{
		struct dev_pagemap *pgmap;

		phx_dev->p2p_pgmap = devm_kzalloc(&phx_dev->dev->dev,
						    sizeof(struct pci_p2pdma_pagemap), GFP_KERNEL);
		if (!phx_dev->p2p_pgmap)
			return -ENOMEM;

	phxfs_info("npu_devm_memremap 1\n");
		pgmap = &phx_dev->p2p_pgmap->pgmap;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
		pgmap->range.start = phx_dev->paddr;
		pgmap->range.end = phx_dev->paddr + phx_dev->size - 1;
		pgmap->nr_range = 1;
#else
		phx_dev->pgmap_res.start = phx_dev->paddr;
		phx_dev->pgmap_res.end = phx_dev->paddr + phx_dev->size - 1;
		phx_dev->pgmap_res.flags = IORESOURCE_MEM;
	phxfs_info("npu->pgmap->res.start is %llx, end is %llx\n",
		phx_dev->pgmap_res.start, phx_dev->pgmap_res.end);
		pgmap->res = phx_dev->pgmap_res;
#endif
		pgmap->type = MEMORY_DEVICE_PCI_P2PDMA;

		phx_dev->pci_mem_va = devm_memremap_pages(&phx_dev->dev->dev, pgmap);

	phxfs_info("npu numa is %d\n", phx_dev->dev->dev.numa_node);

		if (IS_ERR_OR_NULL(phx_dev->pci_mem_va)) {
			phxfs_err("phxfs%d: fallback devm_memremap_pages failed\n",
			       phx_dev->idx);
			devm_kfree(&phx_dev->dev->dev, phx_dev->p2p_pgmap);
			return -ENOMEM;
		}

	phxfs_info("npu devm_memremap_pages success, addr is %lx\n",
			(uintptr_t)phx_dev->pci_mem_va);
		phx_dev->remap = 1;
		phx_dev->segments = NULL;
		phx_dev->num_segments = 0;
		phx_dev->seg_capacity = 0;

		phxfs_info("phxfs%d: fallback single-segment remap, va=0x%lx\n",
		       phx_dev->idx, (unsigned long)phx_dev->pci_mem_va);
		return 0;
	}

err_cleanup:
	/* Clean up any segments that were successfully remapped */
	for (i = 0; i < n_segments; i++) {
		if (segs[i].p2p_pgmap && segs[i].va) {
			devm_memunmap_pages(&phx_dev->dev->dev, &segs[i].p2p_pgmap->pgmap);
		}
		if (segs[i].p2p_pgmap) {
			devm_kfree(&phx_dev->dev->dev, segs[i].p2p_pgmap);
		}
	}
	kfree(segs);
	phx_dev->segments = NULL;
	phx_dev->num_segments = 0;
	phx_dev->seg_capacity = 0;
	return ret;
}

/* ------------------------------------------------------------------ */
/* Segment table (staging mode)                                */
/*                                                                    */
/* dev->segments is kept sorted by phys_start so that                 */
/* phxfs_bar_offset_to_va() can binary-search it. In staging mode each */
/* entry covers exactly one PHXFS_REMAP_UNIT_SIZE unit of the BAR      */
/* (possibly clipped at the BAR end), so entries never partially       */
/* overlap. All accesses are under dev->seg_lock.                      */
/* ------------------------------------------------------------------ */

/*
 * Index of the segment containing `phys`, or -1 if none. *ins (when non-NULL)
 * receives the position at which a segment starting at `phys` must be inserted
 * to keep the array sorted. Caller holds dev->seg_lock.
 */
static int seg_find_locked(struct phxfs_dev *dev, u64 phys, int *ins)
{
	int lo = 0, hi = dev->num_segments - 1;

	while (lo <= hi) {
		int mid = lo + (hi - lo) / 2;
		u64 seg_start = dev->segments[mid].phys_start;
		u64 seg_end = seg_start + dev->segments[mid].size;

		if (phys < seg_start) {
			hi = mid - 1;
		} else if (phys >= seg_end) {
			lo = mid + 1;
		} else {
			if (ins)
				*ins = mid;
			return mid;
		}
	}
	if (ins)
		*ins = lo;
	return -1;
}

/* Make room for one more segment. Caller holds dev->seg_lock. */
static int seg_reserve_one_locked(struct phxfs_dev *dev)
{
	struct phxfs_bar_segment *grown;
	int newcap;

	if (dev->num_segments + 1 <= dev->seg_capacity)
		return 0;

	newcap = dev->seg_capacity ? dev->seg_capacity * 2
				: dev->num_segments + 8;
	grown = krealloc(dev->segments, (size_t)newcap * sizeof(*grown),
			 GFP_KERNEL);
	if (!grown) {
		phxfs_err("phxfs%d: segment array grow to %d failed\n",
		       dev->idx, newcap);
		return -ENOMEM;
	}
	dev->segments = grown;
	dev->seg_capacity = newcap;
	return 0;
}

/*
 * Remap one BAR physical span [phys_start, phys_end) as an additional
 * ZONE_DEVICE segment and insert it into dev->segments. The span must not
 * overlap an existing segment. Caller holds dev->seg_lock.
 */
static int phxfs_remap_span_locked(struct phxfs_dev *phx_dev, u64 phys_start,
				   u64 phys_end)
{
	struct dev_pagemap *pgmap;
	struct pci_p2pdma_pagemap *p2p_pgmap;
	void *va;
	u64 size;
	int ins = 0, ret;

	if (!phx_dev || phys_end <= phys_start)
		return -EINVAL;
	size = phys_end - phys_start;

	/*
	 * An overlap would be rejected by devm_memremap_pages() anyway and
	 * would break the "entries never partially overlap" invariant the
	 * binary search relies on, so reject it up front.
	 */
	if (seg_find_locked(phx_dev, phys_start, &ins) >= 0 ||
	    seg_find_locked(phx_dev, phys_end - 1, NULL) >= 0) {
		phxfs_err("phxfs%d: span [0x%llx-0x%llx) overlaps an existing "
		       "segment\n", phx_dev->idx, phys_start, phys_end);
		return -EEXIST;
	}

	ret = seg_reserve_one_locked(phx_dev);
	if (ret)
		return ret;

	p2p_pgmap = devm_kzalloc(&phx_dev->dev->dev,
				 sizeof(struct pci_p2pdma_pagemap), GFP_KERNEL);
	if (!p2p_pgmap)
		return -ENOMEM;

	pgmap = &p2p_pgmap->pgmap;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
	pgmap->range.start = phys_start;
	pgmap->range.end = phys_end - 1;
	pgmap->nr_range = 1;
#else
	pgmap->res.start = phys_start;
	pgmap->res.end = phys_end - 1;
	pgmap->res.flags = IORESOURCE_MEM;
#endif
	pgmap->type = MEMORY_DEVICE_PCI_P2PDMA;

	va = devm_memremap_pages(&phx_dev->dev->dev, pgmap);
	if (IS_ERR_OR_NULL(va)) {
		long err = PTR_ERR(va);
		phxfs_err("phxfs%d: staging devm_memremap_pages failed for "
		       "[0x%llx-0x%llx), err=%ld\n",
		       phx_dev->idx, phys_start, phys_end, err);
		devm_kfree(&phx_dev->dev->dev, p2p_pgmap);
		return err ? (int)err : -ENOMEM;
	}

	if (ins < phx_dev->num_segments)
		memmove(&phx_dev->segments[ins + 1], &phx_dev->segments[ins],
			(size_t)(phx_dev->num_segments - ins) *
				sizeof(phx_dev->segments[0]));
	phx_dev->segments[ins].phys_start = phys_start;
	phx_dev->segments[ins].size = size;
	phx_dev->segments[ins].va = va;
	phx_dev->segments[ins].refcount = 0;   /* caller takes the reference */
	phx_dev->segments[ins].p2p_pgmap = p2p_pgmap;
	phx_dev->num_segments++;
	/* Legacy compat field: keep it pointing at a pgmap owned by segments[],
	 * so phxfs_cdev_del() does not free it twice. */
	phx_dev->p2p_pgmap = phx_dev->segments[0].p2p_pgmap;
	phx_dev->remap = 1;

	phxfs_info("phxfs%d: staging segment remapped: [0x%llx-0x%llx), "
	       "%llu MiB, va=0x%lx (segments=%d)\n",
	       phx_dev->idx, phys_start, phys_end, size / (1024 * 1024),
	       (unsigned long)va, phx_dev->num_segments);
	return 0;
}

int phxfs_staging_ensure_span(struct phxfs_dev *dev, const u64 *phys,
			      unsigned long n, size_t page_size,
			      u64 *out_span_start)
{
	u64 span_start, span_size, bar_end;
	unsigned long i;
	int idx, ret = 0;

	if (!dev || !phys || !out_span_start || n == 0 || page_size == 0)
		return -EINVAL;

	*out_span_start = 0;
	span_start = phys[0];
	span_size = (u64)n * page_size;
	bar_end = dev->paddr + dev->size;

	if (span_start < dev->paddr || span_start >= bar_end ||
	    span_size > bar_end - span_start) {
		phxfs_err("phxfs%d: pinned span [0x%llx+0x%llx) outside BAR "
		       "[0x%llx-0x%llx)\n", dev->idx, span_start, span_size,
		       dev->paddr, bar_end);
		return -ERANGE;
	}

	/*
	 * Remap what was registered and nothing more. That needs the pinned
	 * pages to form one dense ascending run, aligned and sized to the
	 * granularity at which struct pages can be handed out at all (see
	 * PHXFS_REMAP_ALIGN). Any other layout would force the remap to also
	 * cover foreign GPU pages, which would then fail RDMA/peermem
	 * registration -- refuse loudly instead of breaking them silently.
	 */
	for (i = 1; i < n; i++) {
		if (phys[i] == span_start + (u64)i * page_size)
			continue;
		phxfs_err("phxfs%d: staging pool is not one contiguous BAR span "
		       "(page %lu at 0x%llx, expected 0x%llx)\n",
		       dev->idx, i, phys[i], span_start + (u64)i * page_size);
		return -EINVAL;
	}
	if (!IS_ALIGNED(span_start, PHXFS_REMAP_ALIGN) ||
	    !IS_ALIGNED(span_size, PHXFS_REMAP_ALIGN)) {
		phxfs_err("phxfs%d: staging span [0x%llx+0x%llx) is not %llu MiB "
		       "aligned and sized; remapping it would also expose "
		       "neighbouring GPU memory\n", dev->idx, span_start,
		       span_size, PHXFS_REMAP_ALIGN >> 20);
		return -EINVAL;
	}

	mutex_lock(&dev->seg_lock);
	idx = seg_find_locked(dev, span_start, NULL);
	if (idx >= 0) {
		/*
		 * Re-adopt the span an earlier registration mapped, but only if
		 * it is the very same span: anything else means the two
		 * registrations disagree about the BAR range, and growing or
		 * splitting a live segment is exactly what we refuse to do.
		 */
		if (dev->segments[idx].phys_start != span_start ||
		    dev->segments[idx].size != span_size) {
			phxfs_err("phxfs%d: staging span [0x%llx+0x%llx) "
			       "conflicts with mapped segment [0x%llx+0x%llx)\n",
			       dev->idx, span_start, span_size,
			       dev->segments[idx].phys_start,
			       dev->segments[idx].size);
			ret = -EINVAL;
		}
	} else {
		ret = phxfs_remap_span_locked(dev, span_start,
					      span_start + span_size);
		if (!ret) {
			idx = seg_find_locked(dev, span_start, NULL);
			if (idx < 0) {   /* cannot happen; be loud, not silent */
				phxfs_err("phxfs%d: span 0x%llx missing right "
				       "after remap\n", dev->idx, span_start);
				ret = -EFAULT;
			}
		}
	}
	if (!ret)
		dev->segments[idx].refcount++;
	mutex_unlock(&dev->seg_lock);

	if (!ret)
		*out_span_start = span_start;
	return ret;
}

/*
 * True if every page of [phys_start, phys_start + size) is idle: refcount back
 * to the single reference devm_memremap_pages() established and no user mapping
 * left. memunmap_pages() waits for exactly this condition with an
 * uninterruptible wait_for_completion(), so checking first turns a would-be
 * hang into a skipped (retried) release.
 */
static bool unit_pages_idle(u64 phys_start, u64 size)
{
	unsigned long pfn = PHYS_PFN(phys_start);
	unsigned long end = PHYS_PFN(phys_start + size);

	for (; pfn < end; pfn++) {
		struct page *page = pfn_to_page(pfn);

		if (page_count(page) != 1 || page_mapcount(page) > 0)
			return false;
	}
	return true;
}

/* Drop segment `idx` (caller holds dev->seg_lock and verified it is idle). */
static void seg_unmap_locked(struct phxfs_dev *dev, int idx)
{
	struct phxfs_bar_segment *seg = &dev->segments[idx];
	u64 phys_start = seg->phys_start;
	u64 size = seg->size;

	devm_memunmap_pages(&dev->dev->dev, &seg->p2p_pgmap->pgmap);
	devm_kfree(&dev->dev->dev, seg->p2p_pgmap);

	if (idx + 1 < dev->num_segments)
		memmove(&dev->segments[idx], &dev->segments[idx + 1],
			(size_t)(dev->num_segments - idx - 1) *
				sizeof(dev->segments[0]));
	dev->num_segments--;

	/* Legacy compat field must never dangle (phxfs_cdev_del() reads it). */
	dev->p2p_pgmap = dev->num_segments ? dev->segments[0].p2p_pgmap : NULL;
	if (dev->num_segments == 0)
		dev->remap = 0;

	phxfs_info("phxfs%d: staging segment released: [0x%llx-0x%llx), "
	       "%llu MiB (segments=%d)\n",
	       dev->idx, phys_start, phys_start + size, size / (1024 * 1024),
	       dev->num_segments);
}

static void phxfs_seg_release_work_fn(struct work_struct *work)
{
	struct phxfs_dev *dev = container_of(to_delayed_work(work),
					     struct phxfs_dev, seg_release_work);
	int busy = 0;
	int i;

	/* Full-mode segments are owned by probe and never refcounted. */
	if (phxfs_map_mode != PHXFS_MAP_MODE_STAGING)
		return;

	mutex_lock(&dev->seg_lock);
	for (i = 0; i < dev->num_segments; ) {
		struct phxfs_bar_segment *seg = &dev->segments[i];

		if (seg->refcount > 0 || !seg->va || !seg->p2p_pgmap) {
			i++;
			continue;
		}
		if (!unit_pages_idle(seg->phys_start, seg->size)) {
			busy++;
			i++;
			continue;
		}
		seg_unmap_locked(dev, i);   /* array shifted: revisit index i */
	}
	if (busy && dev->seg_release_tries > 0) {
		dev->seg_release_tries--;
		schedule_delayed_work(&dev->seg_release_work,
				      msecs_to_jiffies(200));
	} else if (busy) {
		phxfs_warn("phxfs%d: %d staging segment(s) still busy, leaving "
		       "them mapped and reusable\n", dev->idx, busy);
	}
	mutex_unlock(&dev->seg_lock);
}

void phxfs_staging_put_span(struct phxfs_dev *dev, u64 span_start)
{
	bool freed = false;
	int idx;

	if (!dev || !span_start)
		return;

	mutex_lock(&dev->seg_lock);
	idx = seg_find_locked(dev, span_start, NULL);
	if (idx < 0)
		phxfs_warn("phxfs%d: put of unknown span 0x%llx\n",
		       dev->idx, span_start);
	else if (dev->segments[idx].refcount <= 0)
		phxfs_warn("phxfs%d: unbalanced put on span 0x%llx\n",
		       dev->idx, span_start);
	else if (--dev->segments[idx].refcount == 0)
		freed = true;

	if (freed && phxfs_staging_release) {
		dev->seg_release_tries = 25;   /* ~5 s of retries, then give up */
		schedule_delayed_work(&dev->seg_release_work, 0);
	}
	mutex_unlock(&dev->seg_lock);
}

void phxfs_staging_release_cancel(struct phxfs_dev *dev)
{
	if (dev)
		cancel_delayed_work_sync(&dev->seg_release_work);
}

static int phxfs_ctrl_init(struct phxfs_ctrl *dev_ctrl, u32 dev_num) {
	int i, j, ret;
	u64 size;
	u16 bus, fn;
	int domain;

	if (!dev_ctrl)
		return -EINVAL;

	dev_ctrl->dev_num = dev_num;
	for (i = 0; i < dev_ctrl->dev_num; i++) {
		domain = (int)(gpu_info_table[i] >> 32);
		bus = (gpu_info_table[i] >> 8) & 0xFF;
		fn = gpu_info_table[i] & 0xFF;
		dev_ctrl->phx_dev[i].dev = pci_get_domain_bus_and_slot(domain, bus, fn);
		if (dev_ctrl->phx_dev[i].dev == NULL) {
			phxfs_warn("npu%u: pci_get_domain_bus_and_slot failed\n", i);
			return -1;
		}
		// for (j = 0; j < PCI_STD_NUM_BARS; j++) {
		for (j = 0; j <= PCI_STD_RESOURCE_END; j++) {
			size = pci_resource_len(dev_ctrl->phx_dev[i].dev, j);
			if (size > dev_ctrl->phx_dev[i].size){
				dev_ctrl->phx_dev[i].paddr = pci_resource_start(dev_ctrl->phx_dev[i].dev, j);
				dev_ctrl->phx_dev[i].size = size;
			}
		}
		dev_ctrl->phx_dev[i].idx = i;
		dev_ctrl->phx_dev[i].remap = 0;
		dev_ctrl->phx_dev[i].segments = NULL;
		dev_ctrl->phx_dev[i].num_segments = 0;
		dev_ctrl->phx_dev[i].seg_capacity = 0;
		mutex_init(&dev_ctrl->phx_dev[i].seg_lock);
		dev_ctrl->phx_dev[i].seg_release_tries = 0;
		INIT_DELAYED_WORK(&dev_ctrl->phx_dev[i].seg_release_work,
				  phxfs_seg_release_work_fn);
		phxfs_info("npu%u: bus is %x, size is %llu, paddr is %llx\n", i,
			dev_ctrl->phx_dev[i].dev->bus->number, dev_ctrl->phx_dev[i].size,
			dev_ctrl->phx_dev[i].paddr);
		/*
		 * Full mode remaps the whole BAR up front. Staging mode defers
		 * the remap to the first registration (phxfs_map_dev_addr_inner),
		 * where only the registered buffer's BAR span is remapped, so the
		 * rest of the BAR keeps pfn_valid == false and stays registerable
		 * by RDMA/peermem.
		 */
		if (phxfs_map_mode == PHXFS_MAP_MODE_FULL) {
			ret = phxfs_devm_memremap(&dev_ctrl->phx_dev[i]);
			if (ret)
				return ret;
		} else {
			phxfs_info("phxfs%d: staging mode -- deferring BAR remap "
			       "until first registration\n", i);
		}
	}
	return 0;
}

static int phxfs_open(struct inode *inode, struct file *filp) {
	int ret = 0;
	int dev_idx;
	char *file_name;

	if (WARN_ON(!filp))
		return -EINVAL;

	file_name = filp->f_path.dentry->d_iname; 

	if (file_name != NULL) {
		dev_idx = extract_trailing_number(file_name);
		phxfs_info("phxfs_open %s, npu_idx is %d\n", file_name, dev_idx);
		if (dev_idx < 0 || dev_idx >= ctrl.dev_num) {
			ret = -1;
			goto out;
		}
		filp->private_data = &ctrl.phx_dev[dev_idx];
	}
out:
	phxfs_info("phxfs_open %d\n", ret);
	return ret;
}

static int phxfs_release(struct inode *inode, struct file *filp) { return 0; }

static long phxfs_ioctl(struct file *filp, unsigned int cmd,
                        unsigned long arg) {
	void __user *argp = (void *)arg;
	switch (cmd) {
		case PHXFS_IOCTL_MAP: {
			struct phxfs_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct phxfs_ioctl_map_s)))
				return -EFAULT;
			return phxfs_map_dev_addr(&map_param, map_param.n_vaddr, map_param.n_size,
									map_param.c_vaddr, map_param.c_size);
		}
		case PHXFS_IOCTL_UNMAP: {
			struct phxfs_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct phxfs_ioctl_map_s)))
				return -EFAULT;
			phxfs_map_dev_release(&map_param, map_param.n_vaddr, map_param.n_size,
								map_param.c_vaddr, map_param.c_size);
			return 0;
		}
		default:
			return -ENOTTY;
	}
}

static const struct file_operations phxfs_chr_fops = {
    .owner = THIS_MODULE,
    .open = phxfs_open,
    .release = phxfs_release,
    .unlocked_ioctl = phxfs_ioctl,
    .mmap = phxfs_mmap,
};

static ssize_t pci_bdf_show(struct device *cdev_device,
                            struct device_attribute *attr, char *buf) {
	struct phxfs_dev *phxdev;

	if (WARN_ON(!cdev_device || !buf))
		return -ENODEV;

	phxdev = dev_get_drvdata(cdev_device);
	if (phxdev == NULL || phxdev->dev == NULL)
		return -ENODEV;
	return sprintf(buf, "%04x:%02x:%02x.%x\n",
		       pci_domain_nr(phxdev->dev->bus),
		       phxdev->dev->bus->number,
		       PCI_SLOT(phxdev->dev->devfn),
		       PCI_FUNC(phxdev->dev->devfn));
}
static DEVICE_ATTR_RO(pci_bdf);

/*
 * Read-only sysfs handle exposing the active BAR mapping mode (global module
 * param) per device, so libphoenix can adapt at open time via the same
 * /sys/class/phxfs-generic/phxfs_devN/ path it already uses for pci_bdf.
 */
static ssize_t map_mode_show(struct device *cdev_device,
                             struct device_attribute *attr, char *buf) {
	if (WARN_ON(!buf))
		return -EINVAL;
	return sprintf(buf, "%d\n", phxfs_map_mode);
}
static DEVICE_ATTR_RO(map_mode);

void phxfs_cdev_del(struct cdev *cdev, struct device *cdev_device,
                    struct phxfs_dev *dev) {
	if (WARN_ON(!cdev || !cdev_device || !dev))
		return;

	device_remove_file(cdev_device, &dev_attr_pci_bdf);
	device_remove_file(cdev_device, &dev_attr_map_mode);
	cdev_device_del(cdev, cdev_device);
	/* No release worker may run past this point: it touches dev->segments. */
	phxfs_staging_release_cancel(dev);
	if (dev->remap) {
		if (dev->segments && dev->num_segments > 0) {
			/* Multi-segment cleanup */
			int i;
			for (i = 0; i < dev->num_segments; i++) {
				if (dev->segments[i].p2p_pgmap && dev->segments[i].va) {
					devm_memunmap_pages(&dev->dev->dev,
							    &dev->segments[i].p2p_pgmap->pgmap);
				}
				if (dev->segments[i].p2p_pgmap) {
					devm_kfree(&dev->dev->dev, dev->segments[i].p2p_pgmap);
				}
			}
			kfree(dev->segments);
			dev->segments = NULL;
			dev->num_segments = 0;
			dev->seg_capacity = 0;
			dev->p2p_pgmap = NULL; /* already freed per-segment above */
		} else if (dev->p2p_pgmap) {
			/* Legacy single-segment cleanup */
			devm_memunmap_pages(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
		}
		dev->pci_mem_va = NULL;
	}
	if (dev->p2p_pgmap != NULL && !(dev->segments && dev->num_segments > 0)) {
		devm_kfree(&dev->dev->dev, dev->p2p_pgmap);
	}
	dev->dev = NULL;
	ida_simple_remove(&phxfs_chr_minor_ida, dev->idx);
}

int phxfs_cdev_add(struct cdev *cdev, struct device *cdev_device,
                   const struct file_operations *fops, struct module *owner,
                   struct phxfs_dev *dev) {
	int ret;
	ret = ida_simple_get(&phxfs_chr_minor_ida, 0, MAX_DEV_NUM, GFP_KERNEL);
	if (ret < 0)
		return ret;
	dev->idx = ret;
	ret = dev_set_name(cdev_device, "phxfs_dev%d", dev->idx);
	if (ret) {
		ida_simple_remove(&phxfs_chr_minor_ida, dev->idx);
		return ret;
	}
	cdev_device->devt = MKDEV(MAJOR(phxfs_chr_devt), dev->idx);
	cdev_device->class = phxfs_chr_class;
	device_initialize(cdev_device);
	cdev_init(cdev, fops);
	cdev->owner = owner;
	ret = cdev_device_add(cdev, cdev_device);
	if (ret) {
		ida_simple_remove(&phxfs_chr_minor_ida, dev->idx);
		return ret;
	}
	dev_set_drvdata(cdev_device, dev);
	ret = device_create_file(cdev_device, &dev_attr_pci_bdf);
	if (ret) {
		cdev_device_del(cdev, cdev_device);
		ida_simple_remove(&phxfs_chr_minor_ida, dev->idx);
		return ret;
	}
	ret = device_create_file(cdev_device, &dev_attr_map_mode);
	if (ret) {
		device_remove_file(cdev_device, &dev_attr_pci_bdf);
		cdev_device_del(cdev, cdev_device);
		ida_simple_remove(&phxfs_chr_minor_ida, dev->idx);
	}
	return ret;
}

int phxfs_cdev_init(struct phxfs_ctrl *ctrl) {
	int ret = -ENOMEM;
	int i;

	if (!ctrl)
		return -EINVAL;

	ret = alloc_chrdev_region(&phxfs_chr_devt, 0, ctrl->dev_num,
								"phxfs-generic");
	if (ret < 0)
		goto destroy_subsys_class;
#ifdef CLASS_CREATE_HAS_TWO_PARAMS
  	phxfs_chr_class = class_create(THIS_MODULE, "phxfs-generic");
#else
  	phxfs_chr_class = class_create("phxfs-generic");
#endif
	if (IS_ERR(phxfs_chr_class)) {
		ret = PTR_ERR(phxfs_chr_class);
		goto unregister_generic_phxfs;
	}
	for (i = 0; i < ctrl->dev_num; i++) {
		ret = phxfs_cdev_add(&ctrl->phx_dev[i].cdev, &ctrl->phx_dev[i].device,
							&phxfs_chr_fops, THIS_MODULE, &ctrl->phx_dev[i]);
		if (ret) {
		kfree_const(ctrl->phx_dev[i].device.kobj.name);
		goto unregister_generic_phxfs;
		}
	}
	phxfs_info("phxfs_cdev_init success!\n");
	return 0;

unregister_generic_phxfs:
  	unregister_chrdev_region(phxfs_chr_devt, ctrl->dev_num);

destroy_subsys_class:
	class_destroy(phxfs_chr_class);
	return ret;
}

static void phxfs_discover_devices(void)
{
	struct pci_dev *pdev = NULL;

	memset(gpu_info_table, 0, sizeof(gpu_info_table));
	npu_num = 0;
#ifndef CONFIG_PHXFS_VENDOR_METAX
	/* Scan 3D display controllers */
	while ((pdev = pci_get_class(PCI_CLASS_DISPLAY_3D << 8, pdev)) != NULL) {
		if (pdev->vendor != PHXFS_PCI_VENDOR_ID || !pdev->bus)
			continue;
		if (phxfs_numa_node >= 0 &&
		    pcibus_to_node(pdev->bus) != phxfs_numa_node) {
			phxfs_info("phxfs: skip GPU %04x:%02x:%02x.%d (numa mismatch)\n",
				   pci_domain_nr(pdev->bus), pdev->bus->number,
				   PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
			continue;
		}
		if (npu_num >= MAX_GPU_DEVS)
			break;
		gpu_info_table[npu_num] =
			((uint64_t)pci_domain_nr(pdev->bus) << 32) |
			PCI_DEVID(pdev->bus->number, pdev->devfn);
		phxfs_info("phxfs: found GPU %04x:%02x:%02x.%d (index=%u)\n",
			   pci_domain_nr(pdev->bus), pdev->bus->number,
			   PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn), npu_num);
		npu_num++;
	}

	/* Scan VGA display controllers */
	while ((pdev = pci_get_class(PCI_CLASS_DISPLAY_VGA << 8, pdev)) != NULL) {
		if (pdev->vendor != PHXFS_PCI_VENDOR_ID || !pdev->bus)
			continue;
		if (phxfs_numa_node >= 0 &&
		    pcibus_to_node(pdev->bus) != phxfs_numa_node)
			continue;
		if (npu_num >= MAX_GPU_DEVS)
			break;
		gpu_info_table[npu_num] =
			((uint64_t)pci_domain_nr(pdev->bus) << 32) |
			PCI_DEVID(pdev->bus->number, pdev->devfn);
		phxfs_info("phxfs: found GPU %04x:%02x:%02x.%d (index=%u)\n",
			   pci_domain_nr(pdev->bus), pdev->bus->number,
			   PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn), npu_num);
		npu_num++;
	}
#else
	/* Scan METAX display controllers */
	while ((pdev = pci_get_class(PCI_CLASS_DISPLAY << 8, pdev)) != NULL) {
		if (pdev->vendor != PHXFS_PCI_VENDOR_ID || !pdev->bus)
			continue;
		if (phxfs_numa_node >= 0 &&
		    pcibus_to_node(pdev->bus) != phxfs_numa_node) {
			phxfs_info("phxfs: skip GPU %04x:%02x:%02x.%d (numa mismatch)\n",
				   pci_domain_nr(pdev->bus), pdev->bus->number,
				   PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
			continue;
		}
		if (npu_num >= MAX_GPU_DEVS)
			break;
		gpu_info_table[npu_num] =
			((uint64_t)pci_domain_nr(pdev->bus) << 32) |
			PCI_DEVID(pdev->bus->number, pdev->devfn);
		phxfs_info("phxfs: found GPU %04x:%02x:%02x.%d (index=%u)\n",
			   pci_domain_nr(pdev->bus), pdev->bus->number,
			   PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn), npu_num);
		npu_num++;
	}
#endif
}

static int __init phxfs_init(void) {
	int ret;

	if (phxfs_p2p_backend_init()) {
		phxfs_warn("Could not initialize P2P backend\n");
		return -1;
	}

	phxfs_discover_devices();

	phxfs_info("found %u GPU device(s)\n", npu_num);

	if (npu_num <= 0 || npu_num > MAX_DEV_NUM) {
		phxfs_err("devdrv_get_devnum error:%u\n", npu_num);
		return -1;
	}
	ret = phxfs_ctrl_init(&ctrl, npu_num);
	if (ret != 0) {
		phxfs_err("npu_ctrl_init error:%d\n", ret);
		return -1;
	}
	ret = phxfs_cdev_init(&ctrl);
	if (ret) {
		phxfs_err("phxfs_init error!\n");
		return -1;
	}
	phxfs_mbuffer_init();
	return 0;
}

static void __exit phxfs_exit(void) {
	int i;
	for (i = 0; i < ctrl.dev_num; i++) {
		phxfs_cdev_del(&ctrl.phx_dev[i].cdev, &ctrl.phx_dev[i].device, &ctrl.phx_dev[i]);
	}

	phxfs_p2p_backend_exit();

	class_destroy(phxfs_chr_class);
	unregister_chrdev_region(phxfs_chr_devt, PHXFS_MINORS);
	ida_destroy(&phxfs_chr_minor_ida);

	phxfs_info("Good bye!\n");
}

module_init(phxfs_init);
module_exit(phxfs_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("qiushi <qiushijsxs@outlook.com>");
MODULE_DESCRIPTION("Phoenix direct storage for multi-vendor accelerators");
MODULE_VERSION("0.0.1");
