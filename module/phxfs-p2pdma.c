// SPDX-License-Identifier: Apache-2.0
/*
 * p2pdma provider bootstrap for phoenixfs.
 *
 * On kernels built with CONFIG_PCI_P2PDMA, every BAR page phoenixfs hands a
 * struct page to has pgmap->type == MEMORY_DEVICE_PCI_P2PDMA, and nvme_map_data()
 * tests exactly that (is_pci_p2pdma_page()) before routing the request through
 * pci_p2pdma_map_sg_attrs() instead of the generic dma_map_sg() path.
 *
 * That path dereferences the provider mirrored in struct phxfs_pgmap and, since
 * 6.0, refuses to map at all (NOT_SUPPORTED -> -EREMOTEIO on every IO) unless
 * pdev->p2pdma exists. The only exported way to establish it is
 * pci_p2pdma_add_resource(), which conveniently also leaves behind a pgmap the
 * kernel built itself -- the ground truth phxfs_p2pdma_verify_layout() checks
 * our layout assumption against.
 *
 * So register one PHXFS_REMAP_ALIGN slice of the BAR that phoenixfs never remaps
 * itself, purely to bootstrap the provider. The slice's pgmap is devm-owned by
 * the GPU device, so it outlives a phoenixfs rmmod; on the next load
 * pdev->p2pdma is still set and we skip re-adding. Those 2 MiB are reclaimed
 * only on GPU driver reload or reboot.
 *
 * On kernels without CONFIG_PCI_P2PDMA none of this exists, is_pci_p2pdma_page()
 * is a compile-time false, and our pages take the generic dma_map_sg() path --
 * which is why phxfs_p2pdma_setup() is a no-op there.
 */

#include <linux/sched.h>
#include <linux/vmalloc.h>

#include "phxfs.h"

static void pgmap_range_of(struct dev_pagemap *pg, u64 *start, u64 *end)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
	*start = pg->range.start;
	*end = pg->range.end;
#else
	*start = pg->res.start;
	*end = pg->res.end;
#endif
}

/*
 * ZONE_DEVICE memory is hot-added in sub-section units (check_pfn_span()
 * rejects anything finer on SPARSEMEM_VMEMMAP), so probing one PFN per
 * PHXFS_REMAP_ALIGN finds every pgmap in the range.
 */
bool phxfs_find_pgmap(u64 start, u64 len, u64 skip_start, u64 skip_len,
		      struct phxfs_pgmap_hit *hit)
{
	u64 addr, end = start + len;
	unsigned long n = 0;

	for (addr = ALIGN_DOWN(start, PHXFS_REMAP_ALIGN); addr < end;
	     addr += PHXFS_REMAP_ALIGN) {
		struct dev_pagemap *pg;

		if (skip_len && addr >= skip_start && addr < skip_start + skip_len)
			continue;
		if ((++n & 4095) == 0)
			cond_resched();
		pg = get_dev_pagemap(PHYS_PFN(addr), NULL);
		if (!pg)
			continue;
		if (hit) {
			hit->addr = addr;
			hit->type = pg->type;
			pgmap_range_of(pg, &hit->start, &hit->end);
		}
		put_dev_pagemap(pg);
		return true;
	}
	return false;
}

void phxfs_report_pgmap(struct phxfs_dev *dev, const char *what,
			const struct phxfs_pgmap_hit *hit)
{
	phxfs_err("phxfs%d: refusing to %s: BAR offset 0x%llx already has struct "
		  "pages (pgmap [0x%llx-0x%llx], type %d%s) owned by another "
		  "mapping -- most likely the GPU driver's own BAR remap. "
		  "Remapping over it would corrupt that driver's memory map; "
		  "disable that driver's BAR remap to use phoenixfs on this GPU\n",
		  dev->idx, what, hit->addr - dev->paddr, hit->start, hit->end,
		  hit->type,
		  hit->type == MEMORY_DEVICE_PCI_P2PDMA ? " PCI_P2PDMA" : "");
}

#ifdef CONFIG_PCI_P2PDMA

static int phxfs_p2pdma_create_slice(struct phxfs_dev *phx_dev)
{
	struct phxfs_pat_conflict *conflicts = NULL;
	struct pci_dev *pdev = phx_dev->dev;
	struct phxfs_pgmap_hit hit;
	u64 span_start, span_len, slice = 0;
	int n_blocks, n_conflicts, i, ret;

	if (phx_dev->bar < 0)
		return -EINVAL;

	/*
	 * No provider yet, so any pgmap in the BAR belongs to someone else
	 * (e.g. the GPU driver remapped its own BAR). Bail out before
	 * pci_p2pdma_add_resource(): it attaches a p2pdma provider to the GPU
	 * device first and does not take it back when the remap then fails,
	 * and its error path itself WARNs on some kernels. This is the hard
	 * safety boundary: foreign mappings that carry struct pages must
	 * never be overlapped. Foreign PAT-only entries (plain ioremaps,
	 * no struct pages) are handled below by letting the kernel arbitrate.
	 */
	if (phxfs_find_pgmap(phx_dev->paddr, phx_dev->size, 0, 0, &hit)) {
		phxfs_report_pgmap(phx_dev, "use this GPU", &hit);
		return -EBUSY;
	}

	/*
	 * The slice is drawn from the whole BAR as the first PHXFS_REMAP_ALIGN
	 * block no PAT conflict overlaps; phxfs_devm_memremap() injects it into
	 * the conflict list so the segment builder routes around it.
	 */
	span_start = phx_dev->paddr;
	span_len = phx_dev->size;
	if (span_len < PHXFS_REMAP_ALIGN)
		return -ENOSPC;

	n_blocks = (int)(span_len / PHXFS_REMAP_ALIGN);
	conflicts = kvcalloc(n_blocks, sizeof(*conflicts), GFP_KERNEL);
	if (!conflicts)
		return -ENOMEM;

	n_conflicts = phxfs_read_pat_conflicts(span_start, span_len, conflicts,
					       n_blocks);
	if (n_conflicts < 0) {
		ret = n_conflicts;
		goto out;
	}

	/* First PHXFS_REMAP_ALIGN-sized block that no PAT conflict overlaps. */
	for (i = 0; i < n_blocks; i++) {
		u64 blk = span_start + (u64)i * PHXFS_REMAP_ALIGN;
		int c;

		for (c = 0; c < n_conflicts; c++) {
			if (conflicts[c].start < blk + PHXFS_REMAP_ALIGN &&
			    conflicts[c].end > blk)
				break;
		}
		if (c == n_conflicts) {
			slice = blk;
			break;
		}
	}

	if (!slice) {
		/*
		 * Every block is covered by foreign PAT entries -- e.g. the
		 * GPU driver's own whole-BAR write-combining ioremap. Those
		 * carry no struct pages (struct pages were already refused by
		 * the pre-check above), so overlapping them is not the
		 * corruption case; the kernel arbitrates the memtype conflict
		 * and fails pci_p2pdma_add_resource() cleanly where the
		 * overlap is not allowed on this kernel.
		 */
		slice = span_start;
		phxfs_warn("phxfs%d: every %llu MiB block is covered by foreign "
		       "PAT entries (whole-BAR foreign memtype?); trying the "
		       "first block and letting the kernel decide\n",
		       phx_dev->idx, PHXFS_REMAP_ALIGN >> 20);
	}

	/* offset is relative to the start of the BAR */
	ret = pci_p2pdma_add_resource(pdev, phx_dev->bar, PHXFS_REMAP_ALIGN,
				      slice - phx_dev->paddr);
	if (ret) {
		phxfs_err("phxfs%d: p2pdma bootstrap failed (%d); a foreign "
		       "memtype (e.g. the GPU driver's own whole-BAR "
		       "write-combining remap) that this kernel refuses to "
		       "overlap is a likely cause; on this kernel every DMA "
		       "map of our BAR pages would fail with -EREMOTEIO\n",
		       phx_dev->idx, ret);
		goto out;
	}

	phx_dev->p2p_slice_start = slice;
	phx_dev->p2p_slice_size = PHXFS_REMAP_ALIGN;
	phxfs_info("phxfs%d: p2pdma bootstrapped via slice [0x%llx+0x%llx)\n",
	       phx_dev->idx, slice, (u64)PHXFS_REMAP_ALIGN);
	ret = 0;
out:
	kvfree(conflicts);
	return ret;
}

/* pci_p2pdma_add_resource() installs a devm-owned pgmap that survives a
 * phoenixfs unload. Recover its BAR location on the next load so FULL mode
 * can exclude that range instead of attempting a conflicting remap. */
static int phxfs_p2pdma_recover_slice(struct phxfs_dev *phx_dev)
{
	struct phxfs_pgmap_hit foreign = { 0 };
	bool have_foreign = false;
	u64 off, found = 0;
	int matches = 0;

	for (off = 0; off + PHXFS_REMAP_ALIGN <= phx_dev->size;
	     off += PHXFS_REMAP_ALIGN) {
		struct dev_pagemap *pg;
		u64 range_start, range_end;

		if (((off / PHXFS_REMAP_ALIGN) & 4095) == 4095)
			cond_resched();
		pg = get_dev_pagemap(PHYS_PFN(phx_dev->paddr + off), NULL);
		if (!pg)
			continue;
		pgmap_range_of(pg, &range_start, &range_end);
		/* Our slice: a P2PDMA pgmap covering exactly this sub-section. */
		if (pg->type == MEMORY_DEVICE_PCI_P2PDMA &&
		    range_start == phx_dev->paddr + off &&
		    range_end == phx_dev->paddr + off + PHXFS_REMAP_ALIGN - 1) {
			found = phx_dev->paddr + off;
			matches++;
		} else if (!have_foreign) {
			have_foreign = true;
			foreign.addr = phx_dev->paddr + off;
			foreign.start = range_start;
			foreign.end = range_end;
			foreign.type = pg->type;
		}
		put_dev_pagemap(pg);
	}

	if (have_foreign) {
		phxfs_report_pgmap(phx_dev, "use this GPU", &foreign);
		return -EBUSY;
	}

	if (matches == 1) {
		phx_dev->p2p_slice_start = found;
		phx_dev->p2p_slice_size = PHXFS_REMAP_ALIGN;
		phxfs_info("phxfs%d: recovered existing p2pdma slice "
		       "[0x%llx+0x%llx)\n", phx_dev->idx,
		       phx_dev->p2p_slice_start, phx_dev->p2p_slice_size);
		return 0;
	}
	if (matches > 1) {
		phxfs_err("phxfs%d: found %d candidate p2pdma slices; "
			   "refusing ambiguous BAR recovery\n",
			   phx_dev->idx, matches);
		return -EEXIST;
	}

	/*
	 * Provider exists but no slice: a previous load's add_resource attempt
	 * failed after attaching the provider (e.g. into a kernel that
	 * enforces PAT on memory-add). Self-heal by creating the slice now --
	 * the foreign-pgmap scan above already cleared the BAR.
	 */
	phxfs_warn("phxfs%d: p2pdma provider exists but no slice found, "
	       "re-creating the bootstrap slice\n", phx_dev->idx);
	return phxfs_p2pdma_create_slice(phx_dev);
}

static int phxfs_p2pdma_bootstrap(struct phxfs_dev *phx_dev)
{
	struct pci_dev *pdev = phx_dev->dev;

	if (phx_dev->bar < 0)
		return -EINVAL;

	if (pdev->p2pdma)
		return phxfs_p2pdma_recover_slice(phx_dev);

	return phxfs_p2pdma_create_slice(phx_dev);
}

/*
 * Verify that our mirror of struct pci_p2pdma_pagemap matches the running
 * kernel, using the pgmap the kernel built for the bootstrap slice as ground
 * truth. A mismatch means a future kernel reshuffled that private struct: fail
 * the load instead of dereferencing garbage from the NVMe submit path.
 */
static int phxfs_p2pdma_verify_layout(struct phxfs_dev *phx_dev)
{
	struct dev_pagemap *pg;
	struct phxfs_pgmap *mirror;
	u64 bus_offset;
	bool mismatch;
	int ret = 0;

	if (!phx_dev->p2p_slice_size)
		return 0;	/* nothing to compare against */

	pg = get_dev_pagemap(PHYS_PFN(phx_dev->p2p_slice_start), NULL);
	if (!pg) {
		phxfs_err("phxfs%d: no pgmap at the p2pdma bootstrap slice "
		       "(0x%llx); refusing to load with an unverifiable layout\n",
		       phx_dev->idx, phx_dev->p2p_slice_start);
		return -ENODEV;
	}
	if (pg->type != MEMORY_DEVICE_PCI_P2PDMA) {
		ret = -EINVAL;
		goto out;
	}

	mirror = container_of(pg, struct phxfs_pgmap, pgmap);
	bus_offset = pci_bus_address(phx_dev->dev, phx_dev->bar) -
		     pci_resource_start(phx_dev->dev, phx_dev->bar);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
	mismatch = !mirror->mem ||
		   mirror->mem->owner != &phx_dev->dev->dev ||
		   mirror->mem->bus_offset != bus_offset;
#else
	mismatch = mirror->provider != phx_dev->dev ||
		   mirror->bus_offset != bus_offset;
#endif

	if (mismatch) {
		phxfs_err("phxfs%d: struct pci_p2pdma_pagemap does not match this "
		       "kernel (%u.%u) -- refusing to load rather than corrupt "
		       "the p2pdma mapping path\n",
		       phx_dev->idx, LINUX_VERSION_CODE >> 16,
		       (LINUX_VERSION_CODE >> 8) & 0xff);
		ret = -EINVAL;
	}
out:
	put_dev_pagemap(pg);
	return ret;
}

#endif /* CONFIG_PCI_P2PDMA */

int phxfs_p2pdma_setup(struct phxfs_dev *phx_dev)
{
#ifdef CONFIG_PCI_P2PDMA
	int ret;

	ret = phxfs_p2pdma_bootstrap(phx_dev);
	if (ret)
		return ret;

	return phxfs_p2pdma_verify_layout(phx_dev);
#else
	struct phxfs_pgmap_hit hit;

	/* No bootstrap slice here: every pgmap in the BAR is foreign. */
	if (phxfs_find_pgmap(phx_dev->paddr, phx_dev->size, 0, 0, &hit)) {
		phxfs_report_pgmap(phx_dev, "use this GPU", &hit);
		return -EBUSY;
	}
	return 0;
#endif
}
