#ifndef __PHXFS_MEM_H__
#define __PHXFS_MEM_H__

#include <linux/types.h>
#include <linux/mm.h>
#include <linux/file.h>
#include "phxfs.h" 

#define PAGE_SHIFT 12

struct phxfs_mmap_buffer {
    atomic_t ref;   /* one reference per VMA that carries this buffer */
    u64 c_vaddr; // mmap cpu vaddr
    u64 map_len; // mmap len
    u64 n_vaddr; // allocated by cann api
    u64 dev_len; // reg dev len
    unsigned long dev_id; // npu id
    u64 *dev_page_addrs; // dev page io addr list
    unsigned long dev_page_num; // indicate num of dev_page_addrs
    unsigned long subpage_num; // (dev_page_size / PAGE_SIZE)
    struct page **ppages; // host page which are mapped to dev page
    unsigned long host_page_num; // the corresponding host page num to dev page num, equal to dev_page_num
    struct vm_area_struct *vma;
    struct phxfs_dev *dev;
    bool remap; // if vma remap_pfn_range set true, otherwise false
    struct p2p_vmap* map;
    u64 staging_span;/* start of the staging BAR span this buffer holds a
                      * reference on (0 = none) */
};
typedef struct phxfs_mmap_buffer* phxfs_mmap_buffer_t;

int phxfs_map_dev_addr_inner(phxfs_mmap_buffer_t pbuffer, u64 devaddr, u64 dev_len);
int phxfs_map_dev_addr(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);
int phxfs_mmap(struct file *filp, struct vm_area_struct *vma);
void phxfs_mbuffer_put(phxfs_mmap_buffer_t pbuffer);

/*
 * Look up the virtual address for a BAR offset using multi-segment mapping.
 * Returns the virtual address, or NULL if the offset doesn't fall in any segment.
 * For legacy single-segment devices, falls back to pci_mem_va + bar_offset.
 */
void *phxfs_bar_offset_to_va(struct phxfs_dev *dev, u64 bar_offset);

#endif