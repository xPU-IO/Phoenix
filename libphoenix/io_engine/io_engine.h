// SPDX-License-Identifier: Apache-2.0
#ifndef __PHXFS_IO_ENGINE_H__
#define __PHXFS_IO_ENGINE_H__

/*
 * Pluggable batch I/O engine abstraction (internal to libphoenix).
 *
 * The public phxfs_read_batch/phxfs_write_batch entry points dispatch to
 * whichever engine was selected at runtime. Engines are ranked by
 * preference; the first one that probes successfully wins:
 *
 *   io_uring  -> best on kernels that support it (default)
 *   sync      -> pread/pwrite loop, always-available fallback
 *
 * Each request's target buffer has already been resolved to a host
 * virtual address the kernel can DMA into (GPU P2P VMA or CPU buffer),
 * so engines only deal with (fd, host_addr, nbytes, f_offset).
 */

#include <sys/types.h>
#include <cstddef>

/*
 * Per-syscall I/O chunk size, shared by every engine and the single-request
 * API. The kernel module supports one arbitrarily large mmap+ioctl
 * registration, so this only chunks the *I/O*: a single read()/write()/
 * io_uring op transfers at most MAX_RW_COUNT (INT_MAX & PAGE_MASK, ~2GiB) per
 * call. 1GiB is a clean, 64KiB-aligned value well under that cap.
 */
#define PHXFS_IO_CHUNK (1024ULL * 1024 * 1024)  /* 1 GiB */

enum phxfs_io_op {
    PHXFS_IO_READ = 0,
    PHXFS_IO_WRITE = 1,
};

struct phxfs_io_worker_ctx {
    int index;
    void *opaque;
    void (*cleanup)(void *opaque);
};

struct phxfs_io_op_req;
typedef int (*phxfs_io_prepare_fn)(struct phxfs_io_op_req *req,
                                   enum phxfs_io_op op,
                                   struct phxfs_io_worker_ctx *worker,
                                   int slot);
typedef void (*phxfs_io_complete_fn)(struct phxfs_io_op_req *req,
                                     enum phxfs_io_op op,
                                     struct phxfs_io_worker_ctx *worker,
                                     int slot);

/* One resolved request: host_addr is ready to hand to the kernel. */
struct phxfs_io_op_req {
    int      fd;
    void    *host_addr;   /* resolved DMA-able host VA */
    size_t   nbytes;
    off_t    f_offset;
    ssize_t  result;      /* OUT */
    phxfs_io_prepare_fn  prepare;
    phxfs_io_complete_fn complete;
    void    *private_data;
    int      worker_slot;
};

#define PHXFS_HOST_STAGE_SLOTS 4

struct phxfs_io_engine {
    const char *name;

    /* Probe availability. Return 0 if usable, <0 otherwise. Called once. */
    int (*probe)(void);

    /*
     * Submit and complete a whole batch. Fills each req->result.
     * Returns 0 if all succeeded (result == nbytes), or the count of
     * failed requests (>0), or negative on a submission-level error.
     */
    int (*submit_batch)(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op);
};

/*
 * Return the active engine, probing/selecting on first use.
 * Never returns NULL (sync engine is the guaranteed fallback).
 */
const struct phxfs_io_engine *phxfs_io_engine_get(void);

/* Engine instances (defined in their respective .cpp files). */
extern const struct phxfs_io_engine phxfs_io_engine_sync;
#ifdef PHXFS_HAVE_LIBURING
extern const struct phxfs_io_engine phxfs_io_engine_uring;
#endif

/* ------------------------------------------------------------------ *
 * Batch I/O thread pool
 *
 * A batch of already-resolved requests is handed to a small, fixed set of
 * worker threads that claim contiguous chunks of it, each running the active
 * engine's submit_batch on its own thread_local ring. Multiple independent
 * rings can submit/reap concurrently, which is what lets one top-level call
 * reach an array's aggregate bandwidth (a single ring saturates at most one
 * device); it also lets a single call cross the Python GIL once instead of the
 * caller managing threads.
 *
 * Workers pull from all live jobs round-robin and never synchronise on a job
 * boundary, so back-to-back batches keep the device queue full and a small
 * batch is not stuck behind a large one. See io_pool.cpp for the details.
 *
 * There is deliberately no NUMA-based routing here: the actual P2P transfer
 * is device-to-device DMA (NVMe controller -> PCIe -> GPU BAR) that never
 * touches host RAM, so which CPU/node issues io_uring_submit() has no bearing
 * on that data path. Pinning workers to "the GPU's NUMA node" would, if
 * anything, be pinning to the wrong device's node whenever the GPU and the
 * NVMe backing the fd are not on the same node.
 * ------------------------------------------------------------------ */

#define PHXFS_POOL_THREADS 4   /* worker/ring count; tune to the storage array's parallelism */

/*
 * Run a whole batch synchronously via the pool. Fills each req->result.
 * Returns the number of failed requests (>=0), or negative on a pool-level
 * error. Falls back to inline execution on the calling thread if the pool is
 * unavailable.
 */
int phxfs_pool_run(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op);

/*
 * Async pool primitives (compute-I/O overlap). submit enqueues on a bounded
 * FIFO and returns immediately; `reqs` must stay alive until wait() returns.
 * blocking=true waits for a free queue slot; blocking=false fails with
 * -EBUSY (via NULL + errno) when the queue is full instead of waiting.
 */
struct phxfs_pool_async;  /* opaque */
struct phxfs_pool_async *phxfs_pool_submit(struct phxfs_io_op_req *reqs, int n,
                                           enum phxfs_io_op op, bool blocking);
/* Cancel unclaimed ops without waiting or consuming the handle. */
int phxfs_pool_cancel(struct phxfs_pool_async *h);
int phxfs_pool_wait(struct phxfs_pool_async *h);

#endif /* __PHXFS_IO_ENGINE_H__ */
