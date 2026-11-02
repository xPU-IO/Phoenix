#include <asm/page_types.h>
#include <linux/kernel.h> 
#include <linux/blkdev.h> 
#include <linux/blk_types.h> 
#include <linux/random.h> 
#include <linux/file.h> 

#include <linux/memory.h> 
#include <linux/slab.h>
#include <linux/vmalloc.h>

 
#include "phxfs-mem.h"  
#include "phxfs-backend.h"
#include "config-host.h"

#define PHXFS_PAT_BUF_SIZE (64 * 1024) /* PAT file typically < 16 KiB */

/*
 * Read the PAT memtype snapshot and extract conflict ranges that overlap
 * with [bar_start, bar_start + bar_len).
 * Returns number of conflicts found, or negative errno.
 * conflicts array is allocated by caller with max_entries capacity.
 */
int phxfs_read_pat_conflicts(u64 bar_start, u64 bar_len,
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
		phxfs_warn("phxfs: cannot open PAT snapshot %s (err=%ld); "
		       "load the module with 'make insmod' to generate it; "
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

	/*
	 * Two line formats, both with an exclusive end:
	 *   >= 5.5: "PAT: [mem 0x000021a000080000-0x000021a000090000] write-combining"
	 *   older : "write-combining @ 0x21a000800000-0x21a000900000"
	 * plus a "PAT memtype list:" header that matches neither.
	 */
	{
		char *line = buf;
		while (line && *line) {
			char *nl = strchr(line, '\n');
			u64 cs, ce;
			char memtype[64] = {0};
			bool parsed;

			if (nl)
				*nl = '\0';

			parsed = sscanf(line, "PAT: [mem 0x%llx-0x%llx] %63s",
					&cs, &ce, memtype) == 3 ||
				 sscanf(line, "%63s @ 0x%llx-0x%llx",
					memtype, &cs, &ce) == 3;

			/* Write-back entries don't conflict with a cached mapping */
			if (parsed && strcmp(memtype, "write-back") != 0) {
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

enum phxfs_callback_state {
    PHXFS_CALLBACK_LIVE = 0,
    PHXFS_CALLBACK_RUNNING,
    PHXFS_CALLBACK_DROPPED,
};

static void p2p_vmap_put(struct p2p_vmap *map)
{
    if (refcount_dec_and_test(&map->refs))
        kfree(map);
}

/* Drop the callback's map reference if no force-reclaim callback claimed it. */
static void p2p_vmap_drop_callback_ref(struct p2p_vmap *map)
{
    if (atomic_cmpxchg(&map->callback_state, PHXFS_CALLBACK_LIVE,
                       PHXFS_CALLBACK_DROPPED) == PHXFS_CALLBACK_LIVE)
        p2p_vmap_put(map);
}

/*
 * Look up the virtual address for a BAR offset using multi-segment mapping.
 * Uses binary search on segments array for efficiency.
 * For legacy single-segment devices (segments == NULL), falls back to
 * pci_mem_va + bar_offset.
 */
void *phxfs_bar_offset_to_va(struct phxfs_dev *dev, u64 bar_offset)
{
	int lo, hi, mid;
	u64 phys_addr;

	if (!dev)
		return NULL;

	/* Legacy single-segment path */
	if (!dev->segments || dev->num_segments <= 0) {
		if (dev->pci_mem_va && bar_offset < dev->size)
			return dev->pci_mem_va + bar_offset;
		return NULL;
	}

	phys_addr = dev->paddr + bar_offset;

	/* Binary search: find the segment containing phys_addr */
	lo = 0;
	hi = dev->num_segments - 1;
	while (lo <= hi) {
		u64 seg_start, seg_end;
		mid = lo + (hi - lo) / 2;
		seg_start = dev->segments[mid].phys_start;
		seg_end = seg_start + dev->segments[mid].size;

		if (phys_addr < seg_start) {
			hi = mid - 1;
		} else if (phys_addr >= seg_end) {
			lo = mid + 1;
		} else {
			/* Found: phys_addr is within [seg_start, seg_end) */
			return dev->segments[mid].va + (phys_addr - seg_start);
		}
	}

	return NULL;
}

void release_gpu_memory(struct p2p_vmap* map)
{
    struct gpu_region* gd;
    struct phxfs_page_table *pt;
    bool owns_callback;

    if (!map)
        return;

    /* Win callback ownership before calling put_pages(). If force reclaim is
     * already running, that callback exclusively owns the vendor page table. */
    owns_callback = atomic_cmpxchg(&map->callback_state, PHXFS_CALLBACK_LIVE,
                                   PHXFS_CALLBACK_DROPPED) ==
                    PHXFS_CALLBACK_LIVE;
    if (owns_callback) {
        gd = (struct gpu_region *)xchg(&map->data, NULL);
        if (gd != NULL) {
            pt = xchg(&gd->pt, NULL);
            if (pt != NULL)
                phxfs_p2p->put_pages(map->gpuvaddr, pt);
            kfree(gd);
        }
        p2p_vmap_put(map);  /* registered reclaim callback */
    }
    p2p_vmap_put(map);  /* VMA/registration owner */
}

static void force_release_gpu_memory(struct p2p_vmap* map)
{
    struct gpu_region* gd;
    struct phxfs_page_table *pt;

    if (WARN_ON(!map))
        return;

    /*
     * The vendor may fire this as soon as it has registered the callback,
     * i.e. before get_pages() has even returned to phxfs_map_dev_addr_inner()
     * (NVIDIA registers it as the last step of nv_p2p_get_pages()), and it
     * frees its own page table right after we return. Hold the vendor off
     * until registration has published gd->pt and finished reading the page
     * table; from then on the ownership protocol below is sufficient.
     */
    wait_for_completion(&map->reg_done);

    if (atomic_cmpxchg(&map->callback_state, PHXFS_CALLBACK_LIVE,
                       PHXFS_CALLBACK_RUNNING) != PHXFS_CALLBACK_LIVE)
        return;

    gd = (struct gpu_region *)xchg(&map->data, NULL);


    if (gd != NULL)
    {

        pt = xchg(&gd->pt, NULL);
        if (pt != NULL)
            phxfs_p2p->free_page_table(pt);

        kfree(gd);

        phxfs_warn("Device driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    	
	}
	/* The phony-buffer VMA still owns `map`. Keep the descriptor alive until
	 * its ->close path; only the vendor page table was reclaimed here. */
	atomic_set(&map->callback_state, PHXFS_CALLBACK_DROPPED);
	p2p_vmap_put(map);  /* registered reclaim callback */
}

#ifndef PHXFS_HAVE_MMAP_READ_LOCK
static inline void mmap_read_lock(struct mm_struct *mm) { down_read(&mm->mmap_sem); }
static inline void mmap_read_unlock(struct mm_struct *mm) { up_read(&mm->mmap_sem); }
#endif

static const struct vm_operations_struct phxfs_vm_ops;
void phxfs_mbuffer_get_ref(phxfs_mmap_buffer_t mbuffer);

/*
 * The phony-buffer VMA that exactly covers [addr, addr + len), or NULL. Only
 * VMAs created by phxfs_mmap() qualify, so another driver's vm_private_data is
 * never reinterpreted as a phony buffer. Caller holds mmap_lock.
 */
static struct vm_area_struct *phxfs_find_phony_vma(struct mm_struct *mm,
                                                   u64 addr, u64 len)
{
    struct vm_area_struct *vma = find_vma(mm, addr);

    if (!vma || vma->vm_start != addr || vma->vm_end - vma->vm_start != len ||
        vma->vm_ops != &phxfs_vm_ops)
        return NULL;
    return vma;
}

/*
 * Look up the caller's phony mapping [cpuvaddr, +length) and take a reference
 * before dropping mmap_lock. GPU binding is checked under mbuffer->lock.
 * Release with phxfs_mbuffer_put().
 */
static phxfs_mmap_buffer_t phxfs_get_phony_buffer(u64 cpuvaddr, u64 length)
{
    phxfs_mmap_buffer_t mbuffer = NULL;
    struct mm_struct *mm = current->mm;
    struct vm_area_struct *vma;

    if (!mm || !cpuvaddr || !length) {
        phxfs_err("phxfs: phony buffer lookup: bad address/length\n");
        return NULL;
    }
    if (cpuvaddr % PAGE_SIZE) {
        phxfs_err("phxfs: phony buffer lookup: cpuvaddr not aligned\n");
        return NULL;
    }

    mmap_read_lock(mm);
    vma = phxfs_find_phony_vma(mm, cpuvaddr, length);
    if (vma) {
        mbuffer = (phxfs_mmap_buffer_t)vma->vm_private_data;
        if (mbuffer && mbuffer->c_vaddr == cpuvaddr && mbuffer->map_len == length)
            phxfs_mbuffer_get_ref(mbuffer);
        else
            mbuffer = NULL;
    }
    mmap_read_unlock(mm);

    if (!mbuffer)
        phxfs_warn("reg region is not same as mmap region\n");
    return mbuffer;
}

int phxfs_map_dev_addr_inner(phxfs_mmap_buffer_t mbuffer, u64 devaddr, u64 dev_len) {
    struct phxfs_dev *dev = NULL;
    struct gpu_region* gd = NULL;
    struct vm_area_struct *vma;
    struct mm_struct *mm = current->mm;
    u64 *dev_page_addrs = NULL;
    u64 page_size;
    u64 nr_dev_pages;
    u64 pci_bar_off;
    u64 cpu_vaddr;
    unsigned long total_pages;
    unsigned long k;
    bool pages_pinned = false;
    bool keep_span = false;
    int ret = 0, i, j;

    if (!mbuffer)
        return -EINVAL;

    page_size = phxfs_p2p->page_size;
    if (page_size < PAGE_SIZE || page_size % PAGE_SIZE != 0) {
        phxfs_err("phxfs: backend page size %llu is not a multiple of PAGE_SIZE\n",
                  page_size);
        return -EINVAL;
    }
    mbuffer->subpage_num = page_size / PAGE_SIZE;
    dev = mbuffer->dev;
    
    if (dev == NULL) {
        phxfs_err("phxfs_map_dev_addr_inner get npu info error\n");
        ret = -ENOMEM;
        goto out;
    }
    /*
     * Full mode remapped the whole BAR at probe, so pci_mem_va must be set.
     * Staging mode remaps this buffer's span lazily below, so a NULL
     * pci_mem_va (and empty segments) is expected on the first registration.
     */
    if (phxfs_map_mode != PHXFS_MAP_MODE_STAGING && dev->pci_mem_va == NULL) {
        phxfs_err("phxfs_map_dev_addr_inner get npu info error\n");
        ret = -ENOMEM;
        goto out;
    }
    
    nr_dev_pages = DIV_ROUND_UP(dev_len, page_size);

    mbuffer->dev_page_num = nr_dev_pages;
    if (dev_len < page_size) {
        if (dev_len % PAGE_SIZE != 0){
            ret = -EINVAL;
            goto out;
        }
        mbuffer->host_page_num = DIV_ROUND_UP(dev_len, PAGE_SIZE);
    }else{
        mbuffer->host_page_num = nr_dev_pages * (mbuffer->subpage_num);
    }
    /*
     * host_page_num is how many host pages get inserted into the VMA. Make
     * sure all of them fit before pinning anything: vm_insert_page() past
     * vm_end would fail half way with PTEs already in.
     * (The caller bounds dev_len by map_len, so this cannot overflow.)
     */
    if (mbuffer->host_page_num > (mbuffer->map_len >> PAGE_SHIFT)) {
        phxfs_err("phxfs: registration of %llu bytes needs %lu host pages, "
                  "mapping only has %llu\n", dev_len, mbuffer->host_page_num,
                  mbuffer->map_len >> PAGE_SHIFT);
        ret = -EINVAL;
        goto out;
    }
    
    /*
     * The arrays below scale linearly with the registered region size.
     * A single kmalloc() is capped at KMALLOC_MAX_SIZE (~4MiB of physically
     * contiguous memory). ppages needs region/512 bytes, hitting 4MiB exactly
     * at a 2GiB registration -- this is the real root cause of the historical
     * "2GB per mmap" limit. Use kvmalloc() so large arrays transparently fall
     * back to vmalloc() and the limit becomes the BAR/HBM size instead.
     */
    dev_page_addrs = kvcalloc(nr_dev_pages, sizeof(u64), GFP_KERNEL);
    if (dev_page_addrs == NULL) {
        ret = -ENOMEM;
        goto out;
    }

    /* The fill loop below writes subpage_num entries for every device page,
     * which is more than host_page_num when dev_len < page_size: size the
     * array for what is written, not for what is inserted. */
    mbuffer->ppages = (struct page **) kvmalloc_array(nr_dev_pages * mbuffer->subpage_num,
                                                      sizeof(struct page *), GFP_KERNEL);
    if (mbuffer->ppages == NULL) {
        phxfs_err("Failed to allocate ppages array (%llu entries)\n",
                  nr_dev_pages * mbuffer->subpage_num);
        ret = -ENOMEM;
        goto out;
    }

    /* The descriptor contains fixed-size metadata; physical addresses live in
     * mbuffer->dev_page_addrs, which is already allocated separately. */
    mbuffer->map = kzalloc(sizeof(struct p2p_vmap), GFP_KERNEL);
    if (mbuffer->map == NULL)
    {
        phxfs_err("Failed to allocate mapping descriptor\n");
        ret = -ENOMEM;
        goto out;
    }

    refcount_set(&mbuffer->map->refs, 2);
    atomic_set(&mbuffer->map->callback_state, PHXFS_CALLBACK_LIVE);
    init_completion(&mbuffer->map->reg_done);
    mbuffer->map->page_size = page_size;
    mbuffer->map->size = dev_len;
    mbuffer->map->gpuvaddr = devaddr;
    mbuffer->map->n_addrs = mbuffer->dev_page_num;

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        phxfs_err("Failed to allocate gpu_region\n");
        ret = -ENOMEM;
        goto out;
    }
    gd->pt = NULL;
    mbuffer->map->data = (void*)gd;
    ret = phxfs_p2p->get_pages(mbuffer->map->gpuvaddr, page_size * mbuffer->map->n_addrs, &gd->pt, 
        (void (*)(void*)) force_release_gpu_memory, mbuffer->map);
    /*
     * A reclaim callback that fired in the meantime is parked on reg_done
     * (see force_release_gpu_memory), so gd and the vendor page table stay
     * valid until complete_all() below. Do not touch gd after it.
     */
    if (gd->pt != NULL)
        pages_pinned = true;
    if (ret == 0 && pages_pinned)
        ret = phxfs_p2p->get_phys_addrs(gd->pt, dev_page_addrs, mbuffer->map->n_addrs);
    else if (ret == 0)
        ret = -ENOMEM;
    complete_all(&mbuffer->map->reg_done);
    if (!pages_pinned) {
        phxfs_err("phxfs_p2p->get_pages failed, ret=%d\n", ret);
        goto out;
    }
    if (ret) {
        phxfs_err("get_phys_addrs failed, ret=%d\n", ret);
        goto out;
    }

    /*
     * Staging mode: the BAR was NOT remapped at probe. Remap exactly the BAR
     * span this buffer occupies, reusing a span an earlier registration
     * already mapped. This must run on *every* registration, not just the
     * first on the device: the BAR offsets a buffer is pinned at are picked by
     * the GPU driver per pin, so a second process (or the same program run
     * with a different allocation history) lands on a different span. Only
     * that span gets struct pages; the rest of the BAR keeps pfn_valid ==
     * false so user GPU memory stays registerable by RDMA/peermem.
     */
    if (phxfs_map_mode == PHXFS_MAP_MODE_STAGING) {
        ret = phxfs_staging_ensure_span(dev, dev_page_addrs, nr_dev_pages,
                                        page_size, &mbuffer->staging_span);
        if (ret) {
            phxfs_err("phxfs%d: staging remap failed: %d\n", dev->idx, ret);
            goto out;
        }
    }

    mbuffer->dev_page_addrs = dev_page_addrs;
    total_pages = mbuffer->host_page_num;
    if (IS_ERR_OR_NULL(mbuffer->ppages)) {
        ret = -ENOMEM;
        goto out;
    }

    /* seg_lock keeps the segment array stable while we walk it: a concurrent
     * registration on the same device may insert (and krealloc/memmove) it. */
    mutex_lock(&dev->seg_lock);
    for (i = 0; i < nr_dev_pages; i++) {
        pci_bar_off = dev_page_addrs[i] - mbuffer->dev->paddr;
        cpu_vaddr = (uint64_t)phxfs_bar_offset_to_va(mbuffer->dev, pci_bar_off);
        if (cpu_vaddr == 0) {
            mutex_unlock(&dev->seg_lock);
            phxfs_err("phxfs_map_dev_addr_inner: bar_offset 0x%llx not in any segment\n",
                   pci_bar_off);
            ret = -EFAULT;
            goto out;
        }
        for (j = 0; j < mbuffer->subpage_num; j++) {
            mbuffer->ppages[i * mbuffer->subpage_num + j] = virt_to_page(cpu_vaddr + j * PAGE_SIZE);
        }
    }
    mutex_unlock(&dev->seg_lock);

    /*
     * Insert under mmap_lock, into the VMA as it is *now*: the caller dropped
     * the lock while pinning, so the original VMA may have been unmapped,
     * split or moved. Read mode suffices: the VMA is VM_MIXEDMAP and
     * insert_page() serialises on the PTE lock, exactly like a fault.
     */
    mmap_read_lock(mm);
    vma = phxfs_find_phony_vma(mm, mbuffer->c_vaddr, mbuffer->map_len);
    if (!vma || vma->vm_private_data != mbuffer) {
        mmap_read_unlock(mm);
        phxfs_err("phxfs: phony mapping went away during registration\n");
        ret = -EFAULT;
        goto out;
    }
    for (k = 0; k < total_pages; k++) {
        ret = vm_insert_page(vma,
                             mbuffer->c_vaddr + k * PAGE_SIZE,
                             mbuffer->ppages[k]);
        if (ret)
            break;
    }
    mmap_read_unlock(mm);
    if (ret) {
        phxfs_err("vm_insert_page failed, k=%lu, ret=%d\n", k, ret);
        if (k > 0) {
            /*
             * Modules cannot zap a VM_MIXEDMAP range, so the k PTEs already
             * inserted stay until the VMA is torn down. They keep their BAR
             * pages busy: hold the staging span until phxfs_mbuffer_free()
             * (VMA close) instead of handing it to the release worker now,
             * and refuse any further MAP on this VMA.
             */
            mbuffer->consumed = true;
            keep_span = true;
        }
        goto out;
    }

    mbuffer->consumed = true;
    mbuffer->remap = 1;
    return 0;

out:
    /*
     * Error teardown. If GPU pages were pinned, release them ourselves
     * (put_pages() does not invoke the registered
     * force-release callback), then free the descriptors. This fixes the
     * previous NULL-deref / use-after-free that panicked the kernel when a
     * registration above ~2GiB failed (ppages kmalloc returned NULL and the
     * unchecked get_pages / out: path freed the still-referenced map & gd).
     */
    if (mbuffer->map != NULL) {
        if (pages_pinned) {
            release_gpu_memory(mbuffer->map);
        } else {
            struct gpu_region *owned;

            owned = (struct gpu_region *)xchg(&mbuffer->map->data, NULL);
            kfree(owned);
            p2p_vmap_drop_callback_ref(mbuffer->map);
            p2p_vmap_put(mbuffer->map);
        }
        mbuffer->map = NULL;
        gd = NULL;
    }
    if (mbuffer->ppages != NULL) {
        kvfree(mbuffer->ppages);
        mbuffer->ppages = NULL;
    }
    if (dev_page_addrs != NULL) {
        kvfree(dev_page_addrs);
        mbuffer->dev_page_addrs = NULL;
    }
    /* Give back the staging span this attempt referenced, unless PTEs into
     * it were left behind (then the VMA teardown gives it back). */
    if (mbuffer->staging_span != 0 && !keep_span) {
        phxfs_staging_put_span(dev, mbuffer->staging_span);
        mbuffer->staging_span = 0;
    }

    return ret;
}

int phxfs_map_dev_addr(struct phxfs_dev *dev, phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length) {
    int ret;
    phxfs_mmap_buffer_t mbuffer;

    mbuffer = phxfs_get_phony_buffer(cpuvaddr, length);
    if (mbuffer == NULL)
        return -EINVAL;

    if (devaddr == 0 || dev_len == 0 || dev_len > mbuffer->map_len) {
        phxfs_err("phxfs: bad MAP request: devaddr=0x%llx dev_len=%llu map_len=%llu\n",
                  devaddr, dev_len, mbuffer->map_len);
        ret = -EINVAL;
        goto put;
    }

    mutex_lock(&mbuffer->lock);
    if (mbuffer->dev && mbuffer->dev != dev) {
        ret = -EINVAL;
    } else if (mbuffer->remap || mbuffer->consumed) {
        /* The BAR PTEs of the earlier MAP stay in the VMA until it is
         * unmapped, so it cannot be mapped again (and silently re-mapping
         * would leak the earlier pin, arrays and span reference). */
        phxfs_err("phxfs: phony buffer 0x%llx is already mapped; mmap a new one\n",
                  cpuvaddr);
        ret = -EBUSY;
    } else {
        /* Keep the first MAP attempt's GPU until the VMA is destroyed. */
        mbuffer->dev = dev;
        mbuffer->dev_id = dev->idx;
        mbuffer->n_vaddr = devaddr;
        mbuffer->dev_len = dev_len;
        ret = phxfs_map_dev_addr_inner(mbuffer, devaddr, dev_len);
    }
    mutex_unlock(&mbuffer->lock);
put:
    phxfs_mbuffer_put(mbuffer);
    return ret;
}

/*
 * Release the GPU-side pin of a registration, leaving the buffer descriptor
 * itself alive. Idempotent: a second call (UNMAP twice, or UNMAP followed by
 * the VMA teardown path) is a no-op.
 */
static void phxfs_mbuffer_unpin(phxfs_mmap_buffer_t mbuffer) {
    if (mbuffer == NULL || !mbuffer->remap)
        return;
    mbuffer->remap = 0;
    release_gpu_memory(mbuffer->map);   /* put_pages + frees map/gd */
    mbuffer->map = NULL;
}

/*
 * PHXFS_IOCTL_UNMAP: drop the GPU pin now, but keep the buffer (and the BAR
 * unit references) until the phony-buffer VMA goes away. The BAR pages are
 * still inserted in that VMA, and unmapping a BAR unit while a page table
 * still references its pages would block forever in memunmap_pages().
 */
void phxfs_map_dev_release(struct phxfs_dev *dev, phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length) {
    phxfs_mmap_buffer_t mbuffer;

    mbuffer = phxfs_get_phony_buffer(cpuvaddr, length);
    if (mbuffer == NULL)
        return;
    mutex_lock(&mbuffer->lock);
    if (mbuffer->dev == dev)
        phxfs_mbuffer_unpin(mbuffer);
    mutex_unlock(&mbuffer->lock);
    phxfs_mbuffer_put(mbuffer);
}

static void phxfs_mbuffer_free(phxfs_mmap_buffer_t mbuffer) {
    if (WARN_ON(!mbuffer))
        return;

    /* Normally already done by PHXFS_IOCTL_UNMAP; this covers a process that
     * died without unmapping. */
    phxfs_mbuffer_unpin(mbuffer);

    /*
     * Only now, with the VMA (and therefore every page table entry pointing at
     * the BAR pages) gone, is it safe to give the staging BAR span back: the
     * release worker can find its pages idle and unmap them.
     */
    if (mbuffer->staging_span != 0) {
        phxfs_staging_put_span(mbuffer->dev, mbuffer->staging_span);
        mbuffer->staging_span = 0;
    }

    if (mbuffer->dev_page_addrs != NULL) {
        kvfree(mbuffer->dev_page_addrs);
        mbuffer->dev_page_addrs = NULL;
    }
    if (mbuffer->ppages != NULL) {
        kvfree(mbuffer->ppages);
        mbuffer->ppages = NULL;
    }

    mbuffer->dev = NULL;
    mbuffer->vma = NULL;
}

void phxfs_mbuffer_get_ref(phxfs_mmap_buffer_t mbuffer) {
    if (WARN_ON(!mbuffer))
        return;
    atomic_inc(&mbuffer->ref);
}

bool phxfs_mbuffer_put_ref(phxfs_mmap_buffer_t mbuffer) {
    if (WARN_ON(!mbuffer))
        return true;
    return atomic_dec_and_test(&mbuffer->ref);
}

static void phxfs_mbuffer_put_internal(phxfs_mmap_buffer_t mbuffer) {
    if (mbuffer == NULL) return;
    
    if (phxfs_mbuffer_put_ref(mbuffer)) {
        phxfs_mbuffer_free(mbuffer);
        mutex_destroy(&mbuffer->lock);
        kfree(mbuffer);
    }
}

void phxfs_mbuffer_put(phxfs_mmap_buffer_t mbuffer) {
    return phxfs_mbuffer_put_internal(mbuffer);
}

static int phxfs_add_phony_buffer(struct vm_area_struct *vma) {
    u64 buffer_len;
    int ret = -EINVAL;
    phxfs_mmap_buffer_t phxfs_mbuffer;

    if (WARN_ON(!vma))
        return -EINVAL;

    buffer_len = vma->vm_end - vma->vm_start;

    phxfs_mbuffer = (phxfs_mmap_buffer_t)kzalloc(sizeof(struct phxfs_mmap_buffer), GFP_KERNEL);
    if (!phxfs_mbuffer) {
        ret = -ENOMEM;
        goto error;
    }

    if (vma->vm_private_data != NULL) {
        phxfs_warn("vma->vm_private_data!=NULL\n");
        kfree(phxfs_mbuffer);
        goto error;
    }

    atomic_set(&phxfs_mbuffer->ref, 1);
    mutex_init(&phxfs_mbuffer->lock);
    phxfs_mbuffer->consumed = false;
    phxfs_mbuffer->vma = vma;
    phxfs_mbuffer->c_vaddr = vma->vm_start;
    phxfs_mbuffer->map_len = buffer_len;
    phxfs_mbuffer->remap = 0;
    phxfs_mbuffer->staging_span = 0;

    vma->vm_private_data = (void *)phxfs_mbuffer;

    return 0;

error:
    return ret;
}

/*
 * The phony-buffer VMA owns the buffer descriptor: the ref taken in
 * phxfs_add_phony_buffer() is the VMA's. ->open covers a VMA split (each half
 * gets its own reference), ->close drops one. The last drop tears the buffer
 * down, which is also what reclaims the staging BAR units -- and it runs from
 * remove_vma(), i.e. after unmap_vmas()/free_pgtables() have already dropped
 * every page-table reference to those BAR pages. It therefore covers a process
 * that exits or crashes without calling PHXFS_IOCTL_UNMAP.
 */
static void phxfs_vma_open(struct vm_area_struct *vma) {
    phxfs_mmap_buffer_t mbuffer = (phxfs_mmap_buffer_t)vma->vm_private_data;

    if (mbuffer)
        phxfs_mbuffer_get_ref(mbuffer);
}

static void phxfs_vma_close(struct vm_area_struct *vma) {
    phxfs_mmap_buffer_t mbuffer = (phxfs_mmap_buffer_t)vma->vm_private_data;

    if (mbuffer)
        phxfs_mbuffer_put(mbuffer);
}

static const struct vm_operations_struct phxfs_vm_ops = {
    .open  = phxfs_vma_open,
    .close = phxfs_vma_close,
};

int phxfs_mmap(struct file *filp, struct vm_area_struct *vma) {
    int ret;
    struct mm_struct *mm = current->mm;

    if (vma->vm_pgoff != 0)
        return -EINVAL;
#ifdef NVFS_VM_FLAGS_NOT_CONSTANT
    vma->vm_flags &= ~VM_PFNMAP;
    vma->vm_flags &= ~VM_IO;
    vma->vm_flags |= VM_MIXEDMAP;
    vma->vm_flags |= mm->def_flags;
    /* The phony buffer is sized at mmap() time (map_len) and maps device
     * BAR pages: it must not grow under mremap(), nor be read by a core dump. */
    vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
#else
    unsigned long vm_flags;
    vm_flags = ACCESS_PRIVATE(vma, __vm_flags);
    vm_flags &= ~VM_PFNMAP;
    vm_flags &= ~VM_IO;
    vm_flags |= VM_MIXEDMAP;
    vm_flags |= mm->def_flags;
    vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
    vm_flags_set(vma, vm_flags);
#endif
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    ret = phxfs_add_phony_buffer(vma);
    if (ret == 0)
        vma->vm_ops = &phxfs_vm_ops;   /* VMA now owns the buffer */
    return ret;
}
