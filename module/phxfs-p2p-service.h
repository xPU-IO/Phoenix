#ifndef __PHXFS_P2P_SERVICE_H__
#define __PHXFS_P2P_SERVICE_H__

#include <linux/types.h>

/**
 * phxfs_p2p_handle - opaque handle returned by phxfs_p2p_register.
 * Caller treats as void*; Phoenix owns the internal definition.
 */
struct phxfs_p2p_handle;

/**
 * phxfs_p2p_register - pin GPU pages and return DMA bus addresses.
 *
 * Why this exists:
 *   When Phoenix has remapped a device's BAR via devm_memremap_pages
 *   (MEMORY_DEVICE_PCI_P2PDMA), vendor DMA map calls may fail for
 *   that device. This service bypasses DMA mapping and returns the
 *   physical addresses from the backend's get_pages directly as DMA
 *   bus addresses (valid under IOMMU=pt -- the deployment contract
 *   for bare-metal direct I/O).
 *
 * Lifecycle:
 *   - Caller must keep the GPU allocation/context alive until all DMA
 *     has drained and phxfs_p2p_deregister(handle) has completed.
 *   - Caller must call phxfs_p2p_deregister(handle) when done.
 *   - If the device driver forcefully reclaims the pages (context
 *     destroyed), Phoenix's internal free_callback fires and marks the
 *     handle as invalidated. The caller's subsequent deregister call
 *     is a safe no-op on the device side (just frees heap).
 *   - Phoenix's module refcount is held (try_module_get) for the
 *     lifetime of the handle, so Phoenix cannot be unloaded while
 *     active registrations exist.
 *
 * @gpu_vaddr:   device virtual address to pin (must be page-aligned)
 * @length:      size in bytes (must be multiple of device page size)
 * @out_handle:  receives opaque handle (caller frees via deregister)
 * @out_ioaddrs: receives BORROWED pointer to bus address array.
 *               Valid until phxfs_p2p_deregister is called.
 *               Caller must NOT kfree this pointer.
 * @out_n_addrs: receives number of entries in out_ioaddrs
 *
 * Return: 0 on success, negative errno on failure.
 *         -ENODEV  if Phoenix's P2P backend not loaded
 *         -ENOMEM  if allocation failed
 *         other    propagated from backend get_pages
 */
int phxfs_p2p_register(uint64_t gpu_vaddr,
                       uint64_t length,
                       struct phxfs_p2p_handle **out_handle,
                       const uint64_t **out_ioaddrs,
                       uint32_t *out_n_addrs);

/**
 * phxfs_p2p_deregister - release pinned GPU pages and free the handle.
 *
 * Safe to call after the free_callback has fired (internal no-op on
 * the NVIDIA side, just frees heap + module_put).
 *
 * @handle:  handle from phxfs_p2p_register. NULL is a safe no-op.
 *           After this call, the borrowed ioaddrs pointer is invalid.
 */
void phxfs_p2p_deregister(struct phxfs_p2p_handle *handle);

#endif /* __PHXFS_P2P_SERVICE_H__ */
