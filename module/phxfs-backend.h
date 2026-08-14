#ifndef __PHXFS_BACKEND_H__
#define __PHXFS_BACKEND_H__

#include <linux/types.h>
#include <linux/pci.h>

#include "phxfs-mem.h"

/*
 * Compile-time vendor selection.
 * The build system defines exactly one of these via -D flag:
 *   CONFIG_PHXFS_VENDOR_NVIDIA
 *   CONFIG_PHXFS_VENDOR_METAX
 *   CONFIG_PHXFS_VENDOR_AMD
 *   CONFIG_PHXFS_VENDOR_HUAWEI
 *
 * Single-vendor-per-machine assumption: only one backend is compiled in.
 */

#ifdef CONFIG_PHXFS_VENDOR_NVIDIA
#define PHXFS_PCI_VENDOR_ID 0x10DE
#elif defined(CONFIG_PHXFS_VENDOR_METAX)
#define PHXFS_PCI_VENDOR_ID 0x9999
#define PCI_CLASS_DISPLAY 0x0380
#elif defined(CONFIG_PHXFS_VENDOR_AMD)
#define PHXFS_PCI_VENDOR_ID 0x1002
#elif defined(CONFIG_PHXFS_VENDOR_HUAWEI)
#define PHXFS_PCI_VENDOR_ID 0x19E5
#else
/* Default to NVIDIA for backward compatibility */
#define PHXFS_PCI_VENDOR_ID 0x10DE
#endif

/**
 * struct phxfs_page_table - vendor-agnostic page table handle.
 * @priv: vendor-specific page table (e.g. nvidia_p2p_page_table*)
 *
 * Allocated by backend get_pages(), freed by put_pages() or
 * free_page_table().  Core code treats this as opaque.
 */
struct phxfs_page_table {
	void *priv;
};

/**
 * struct phxfs_p2p_ops - P2P backend operations.
 *
 * Each vendor implements one instance and registers it at module
 * init via phxfs_p2p_register_backend().  The active backend is
 * stored in the global phxfs_p2p pointer.
 */
struct phxfs_p2p_ops {
	const char *name;

	/** Load vendor kernel symbols / init backend. 0 on success. */
	int  (*init)(void);
	void (*exit)(void);

	/**
	 * Pin device memory pages.
	 * @vaddr:    device virtual address
	 * @length:   bytes to pin
	 * @pt:       receives allocated phxfs_page_table (caller treats as opaque)
	 * @free_cb:  callback invoked when vendor driver force-reclaims pages
	 * @data:     opaque cookie passed to free_cb
	 * Return: 0 on success, negative errno on failure.
	 */
	int  (*get_pages)(uint64_t vaddr, uint64_t length,
			  struct phxfs_page_table **pt,
			  void (*free_cb)(void *), void *data);

	/**
	 * Unpin device memory pages and free the page_table.
	 * Called for normal (non-force-reclaim) release.
	 */
	void (*put_pages)(uint64_t vaddr, struct phxfs_page_table *pt);

	/**
	 * Get number of page entries in the page table.
	 */
	uint32_t (*get_n_pages)(struct phxfs_page_table *pt);

	/**
	 * Fill caller-allocated array with physical addresses.
	 * @addrs:   caller-allocated array (capacity >= get_n_pages())
	 * @n_addrs: number of entries to fill (must match get_n_pages())
	 * Return: 0 on success, -ENOMEM if any page entry is NULL.
	 */
	int (*get_phys_addrs)(struct phxfs_page_table *pt,
			      uint64_t *addrs, uint32_t n_addrs);

	/**
	 * Free page_table structure after force-reclaim.
	 * Do NOT call put_pages — the vendor driver already reclaimed
	 * the pages.  This only frees the heap-allocated struct.
	 */
	void (*free_page_table)(struct phxfs_page_table *pt);

	/** Device page size in bytes (NVIDIA = 64 KiB, etc.) */
	uint64_t page_size;
};

/* Global active backend — set at module init, never NULL after successful init. */
extern struct phxfs_p2p_ops *phxfs_p2p;

/* Register a backend (called by vendor-specific init code). */
int phxfs_p2p_register_backend(struct phxfs_p2p_ops *ops);

/* Initialize the compiled-in backend.  Returns 0 on success. */
int phxfs_p2p_backend_init(void);

/* Shutdown the compiled-in backend. */
void phxfs_p2p_backend_exit(void);

#endif /* __PHXFS_BACKEND_H__ */
