#ifndef __PHOENIX_H__
#define __PHOENIX_H__
/* Public C API: use C headers only so this compiles under a plain C compiler
 * as well as C++. */
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

int phxfs_open(int device_id);
int phxfs_close(int device_id);

/* Returns the device page size in bytes (NVIDIA=64KB, etc.). */
uint64_t phxfs_get_page_size(void);

/*
 * Returns the BAR mapping mode of a phxfs device:
 *   0 (FULL):    entire BAR remapped at probe; phxfs_regmem pins user pages
 *   1 (STAGING): kernel maps only an internal staging pool; phxfs_regmem is
 *                a no-op and reads are routed SSD->staging->D2D->user buffer
 * Returns negative errno if the device is not present.
 */
int phxfs_get_map_mode(int device_id);

/*
 * Map a vendor-specific device ID to a phxfs device index.
 * The vendor is selected at build time (PHXFS_VENDOR).
 *   NVIDIA: device_id is a CUDA device ID
 *   AMD:    device_id is a HIP device ID
 *   Huawei: device_id is an NPU ID
 */
int phxfs_find_dev(int device_id);

/* Backward-compatible wrapper for existing NVIDIA callers */
static inline int phxfs_find_dev_for_cuda_gpu(int cuda_gpu_id) {
    return phxfs_find_dev(cuda_gpu_id);
}

/*
 * Register/deregister a device (GPU) buffer for direct I/O.
 *
 * IMPORTANT: read/write and the batch API identify a buffer by its
 * original DEVICE address (`addr` here, `buf` there) — NOT by `*target_addr`.
 * `*target_addr` is an internal host-mapped handle returned for reference; do
 * not pass it back as the I/O buffer or the lookup will miss. `addr` must be
 * nonzero, `len` nonzero and 64KiB-aligned; overlapping a live registration
 * is rejected, while an exact-duplicate registration is reference-counted and
 * reused (deregister once per register).
 */
int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
int phxfs_deregmem(int device, const void *addr, size_t len);

/*
 * device_id selects the target buffer the same way as the batch API below:
 *   - device_id >= 0: `buf` must lie inside a GPU registration on that
 *     phxfs device (phxfs_regmem), or the call fails.
 *   - device_id  < 0: `buf` is a plain CPU (host) address.
 */
ssize_t phxfs_read(int fd, int device_id, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(int fd, int device_id, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

/* ------------------------------------------------------------------ *
 * Batch I/O
 *
 * A single call submits N independent I/O requests to the underlying I/O
 * engine (io_uring where available, sync pread/pwrite fallback otherwise),
 * fanned out over a small pool of worker threads/rings. This removes the
 * per-request syscall overhead of looping over phxfs_read/phxfs_write, and
 * lets the storage layer service the requests concurrently.
 *
 * NOTE on fork(): the internal worker pool is fork-safe via pthread_atfork
 * (the child lazily re-creates the pool on first use), but the kernel P2P
 * mappings and device fds created by phxfs_open/phxfs_regmem in the parent
 * are NOT inherited correctly across fork. Callers that need multi-process
 * should use spawn (re-open in the child), not fork-after-open.
 *
 * The target buffer is selected by the SIGN of device_id:
 *   - device_id >= 0: a GPU buffer previously registered with phxfs_regmem
 *     on that phxfs device. `buf` must lie inside a registration or the
 *     request fails (-EFAULT) — there is no silent CPU fallback.
 *   - device_id  < 0: an ordinary CPU buffer (e.g. pinned host memory used
 *     by a store path); `buf` is used as a plain host address.
 *
 * A batch may freely mix requests targeting different devices and CPU
 * buffers; all requests share the same worker pool.
 * ------------------------------------------------------------------ */
typedef struct phxfs_io_req {
    int      fd;          /* open file descriptor (O_DIRECT recommended) */
    int      device_id;   /* >=0: phxfs GPU device the buf is registered on; */
                          /*  <0: plain CPU buffer */
    void    *buf;         /* GPU addr (registered) or CPU addr */
    off_t    buf_offset;  /* byte offset within buf */
    size_t   nbytes;      /* transfer length */
    off_t    f_offset;    /* file offset */
    ssize_t  result;      /* OUT: bytes transferred, or negative errno */
} phxfs_io_req_t;

/*
 * Synchronous batch: submit all N requests, wait for completion, fill
 * each req->result. Returns 0 if every request completed with
 * result == nbytes; otherwise returns the number of failed requests
 * (>0), or a negative value on a submission-level error.
 *
 * Lifetime contract (applies to both sync and async batches): every
 * resource each request references must stay valid until the call (sync)
 * or phxfs_batch_wait() (async) returns:
 *   - reqs[i].fd must remain open (fds are NOT dup'd internally; if you
 *     close and the number is reused, I/O may hit the wrong file);
 *   - the CPU buffer (for host-buffer requests) must not be freed;
 *   - for registered GPU buffers, the phxfs_regmem registration is held by
 *     an internal reference for the batch's duration — a concurrent
 *     phxfs_deregmem() on it blocks until the batch completes rather than
 *     unmapping under an in-flight transfer.
 */
int phxfs_read_batch(phxfs_io_req_t *reqs, int n);
int phxfs_write_batch(phxfs_io_req_t *reqs, int n);

/*
 * Asynchronous batch — submit queues the batch on the worker pool and
 * returns immediately; the caller can run GPU compute meanwhile, then call
 * phxfs_batch_wait() to block for completion and get per-request results.
 * Enables compute/I/O overlap.
 *
 * Submitted batches queue up (bounded capacity) and each runs with the
 * pool's full worker set in turn — so pipelining several submits (e.g. to
 * prefetch ahead) is supported without waiting for the previous one first.
 * If the queue is full, submit fails immediately (non-blocking) with NULL
 * and errno == EBUSY rather than waiting for space.
 *
 *   reqs (and every resource it references — see the lifetime contract on
 *   phxfs_read_batch above: fd, CPU buffer, GPU registration) MUST remain
 *   valid until phxfs_batch_wait()/phxfs_batch_destroy() returns, and its
 *   fields MUST NOT be modified after submit.
 *   Each handle must be consumed by exactly one call to phxfs_batch_wait()
 *   or phxfs_batch_destroy(); after that call the handle is freed and must
 *   not be used again.
 *
 * phxfs_batch_submit_read/write return an opaque handle, or NULL on error
 * (errno == EBUSY if the pool's queue is full, ENOMEM on allocation
 * failure, ENOTSUP if the worker pool is unavailable).
 * phxfs_batch_wait returns the number of failed requests (>=0), or negative
 * on error, copies per-request results back, and frees the handle.
 * phxfs_batch_destroy abandons a batch: it waits for in-flight I/O to quiesce
 * (submitted I/O cannot be cancelled), then frees the handle WITHOUT copying
 * results back — for callers that drop a batch without needing its results.
 */
typedef struct phxfs_batch phxfs_batch_t;

phxfs_batch_t *phxfs_batch_submit_read(phxfs_io_req_t *reqs, int n);
phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n);
int phxfs_batch_wait(phxfs_batch_t *handle);
int phxfs_batch_destroy(phxfs_batch_t *handle);

/*
 * Name of the active I/O engine selected at runtime
 * ("io_uring", "sync", ...). For diagnostics / tests.
 */
const char *phxfs_io_engine_name(void);

/* ------------------------------------------------------------------ *
 * Stream-ordered asynchronous I/O (host-function model)
 *
 * phxfs_read_stream / phxfs_write_stream present a cuFile/uGDS-like async
 * API. Ordering model: the DMA runs INSIDE a host callback that CUDA
 * enqueues on the user stream (cudaLaunchHostFunc semantics). CUDA
 * guarantees the callback is "called after currently enqueued work and
 * will block work added after it", so
 *
 *   - a READ's later consumers on the stream see the DMA'd data,
 *   - a WRITE's preceding gather kernels complete before the DMA reads.
 *
 * ...by construction — no events, no drain loop, no submit-vs-consumer
 * race. A bare cudaStreamSynchronize() on the caller side is therefore
 * ALWAYS safe (this differs from the earlier event-bridge design).
 * Consecutive submissions on one stream execute in submission order
 * (consecutive host functions are officially supported).
 *
 * There is NO stream registration: like cuFileReadAsync / cuFileWriteAsync,
 * every submission carries the stream handle; there is no per-stream
 * state to set up or tear down.
 *
 * Staging-mode devices (map_mode=staging) are NOT supported by the stream
 * API: their two-hop path (SSD -> staging pool -> D2D -> user) would need
 * the D2D leg inside or around the callback, which CUDA forbids.
 * Submissions whose buffer device is staging fail with -EOPNOTSUPP;
 * staging devices keep using the synchronous phxfs_read / phxfs_write
 * (and the batch API). Stream-ordered staging support is planned.
 *
 * Submission contract (mirrors cuFileReadAsync/WriteAsync):
 *   - nbytes / buf_offset / f_offset point at caller-owned storage whose
 *     values are read at submission time; *bytes_done is written by the
 *     library. ALL of this storage must stay valid and unmodified until
 *     the submission has completed (stream synchronized).
 *   - after the stream passes the submission, *bytes_done holds the
 *     transferred byte count, or a negative errno of the failure. A
 *     failed DMA never stalls the stream: the callback always returns.
 *   - fd, the I/O buffer and (for GPU buffers) its phxfs_regmem
 *     registration must stay valid until the stream is synchronized
 *     past the submission. The library takes NO internal references —
 *     a deregistration racing an in-flight DMA is a caller error.
 *     Synchronize the stream before deregistering (the same rule
 *     cuFileReadAsync / cuFileWriteAsync users follow).
 *   - the buffer device is resolved from the buffer itself, like
 *     cuFileReadAsync: a buf inside any opened device's registration
 *     table is a GPU buffer (DMA against its P2P host address; a
 *     staging-mode device fails with -EOPNOTSUPP); a buf inside no
 *     registration is a plain CPU address. An extent sticking out of
 *     its registration fails with -EFAULT.
 *   - returns 0 if the submission was accepted (I/O outcome is reported
 *     through *bytes_done), or a negative errno for submission-level
 *     failures (bad args, callback enqueue failure...).
 *
 * Vendor requirement: the connector must provide the launch_host_func
 * primitive; without it submissions fail with -EOPNOTSUPP. There is no
 * synchronous fallback.
 * ------------------------------------------------------------------ */
int phxfs_read_stream(int fd, void *buf, size_t *nbytes,
                      off_t *buf_offset, off_t *f_offset,
                      ssize_t *bytes_done, void *stream);
int phxfs_write_stream(int fd, void *buf, size_t *nbytes,
                       off_t *buf_offset, off_t *f_offset,
                       ssize_t *bytes_done, void *stream);

#ifdef __cplusplus
}
#endif

#endif
