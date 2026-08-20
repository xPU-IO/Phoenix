#ifndef __DEVCONNECTOR_H__
#define __DEVCONNECTOR_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DevConnector — vendor-specific device operations.
 *
 * Each vendor provides one implementation (e.g. nvidia_connector.cc),
 * selected at build time by PHXFS_VENDOR.  Core libphoenix code calls
 * through the global *devconn pointer and never touches vendor APIs
 * (CUDA, HIP, CANN, ...) directly.
 *
 * Adding a new vendor:
 *   1. Create <vendor>_connector.cc implementing devconn_ops
 *   2. Add the source file to CMakeLists.txt for that vendor
 *   3. No changes to phoenix.h / phoenix.cpp
 */
struct devconn_ops {
    const char  *name;                  /* "nvidia", "amd", "huawei", ... */
    uint64_t     page_size;             /* device page size in bytes */

    /*
     * Initialize vendor runtime (optional, may be NULL).
     * Called once at library load.
     * Returns 0 on success.
     */
    int   (*init)(void);

    /*
     * Map a vendor-specific device ID to a phxfs device index.
     *   NVIDIA: device_id is a CUDA device ID
     *   AMD:    device_id is a HIP device ID
     *   Huawei: device_id is an NPU ID
     * Returns phxfs device index (>=0) or -1 on failure.
     */
    int   (*find_device)(int device_id);

    /*
     * Staging-mode device-memory operations (may be NULL for vendors that
     * do not support the staging path yet; the core checks before use).
     *
     * mem_alloc:   allocate `size` bytes of device memory on the accelerator
     *              backing phxfs device index `phxfs_dev`. Returns 0 and sets
     *              *dptr, or a negative errno.
     * mem_free:    release a buffer from mem_alloc.
     * memcpy_dtod: synchronous device-to-device copy (dst and src are both
     *              device pointers). Returns 0 or a negative errno.
     */
    int   (*mem_alloc)(int phxfs_dev, size_t size, void **dptr);
    void  (*mem_free)(void *dptr);
    int   (*memcpy_dtod)(void *dst, const void *src, size_t n);

    /*
     * Asynchronous device-to-device copy on a connector-owned queue, plus a
     * barrier for it. `slot` selects one of a small fixed set of queues per
     * device (0 .. PHX_STAGING_SLOTS-1), so the staging path can double-buffer
     * without knowing anything about vendor stream/queue types.
     *
     * memcpy_dtod_async returns as soon as the copy is enqueued;
     * queue_sync(phxfs_dev, slot) returns once every copy enqueued on that
     * queue has completed (and reports any error they raised).
     *
     * Both may be NULL: the core then falls back to the synchronous
     * memcpy_dtod above, so a vendor only needs the sync op to work.
     */
    int   (*memcpy_dtod_async)(int phxfs_dev, int slot, void *dst,
                               const void *src, size_t n);
    int   (*queue_sync)(int phxfs_dev, int slot);

    /*
     * Stream-ordered I/O primitive (phx_stream.cpp; REQUIRED for the
     * stream API — phxfs_read_stream / phxfs_write_stream fail with
     * -EOPNOTSUPP when it is NULL).
     *
     * Model: the DMA itself is host-driven (io_uring/P2P or pread) and
     * unknown to CUDA. Instead of bridging its completion into the stream
     * with events, the DMA runs INSIDE a host callback enqueued on the
     * user stream:
     *
     *   launch_host_func(stream, fn, arg)  ==  cudaLaunchHostFunc
     *
     * CUDA guarantees the callback is "called after currently enqueued work
     * and will block work added after it" — so a READ's consumers and a
     * WRITE's preceding gather are ordered correctly BY CONSTRUCTION, with
     * no events, no drain loop, and no submit-vs-consumer race. A bare
     * cudaStreamSynchronize is always safe for callers.
     *
     * launch_host_func: enqueue fn(arg) on `stream` (the user stream).
     *        The callback MUST NOT make CUDA API calls (CUDA rule); the
     *        core therefore runs only the pure pread/pwrite leg inside
     *        it.
     */
    int   (*launch_host_func)(void *stream,
                              void (*fn)(void *), void *arg);

    /*
     * Profiler range annotations (NVTX on NVIDIA, vendor equivalents
     * elsewhere). Both may be NULL — the core checks before calling, so a
     * vendor without a profiler API needs no stubs and pays nothing.
     * Calls nest per thread: every push has exactly one pop.
     * `name` only has to stay valid for the duration of the push call.
     */
    void  (*range_push)(const char *name);
    void  (*range_pop)(void);
};

/*
 * Global active connector — initialized at link time by the
 * compiled-in connector file.  Never NULL in a valid build.
 */
extern struct devconn_ops *devconn;

/*
 * Initialize the compiled-in connector.
 * Returns 0 on success.  Safe to call multiple times.
 */
int devconn_init(void);

#ifdef __cplusplus
}
#endif

#endif /* __DEVCONNECTOR_H__ */
