/*
 * phx_stream.cpp — stream-ordered asynchronous I/O (host-function model).
 *
 * Implements phxfs_read_stream / phxfs_write_stream (see phoenix.h for
 * the public contract). There is NO stream registration: like
 * cuFileReadAsync / cuFileWriteAsync, every submission carries the
 * stream handle and needs no prior per-stream state.
 *
 * WHY A HOST FUNCTION, NOT AN EVENT BRIDGE
 * ----------------------------------------
 * libphoenix's DMA is host-driven (io_uring P2P or pread) and unknown to
 * CUDA, so the transfer can never be a stream op the way cuFileReadAsync's
 * is. The earlier event-bridge design ran the DMA on a background thread
 * and enqueued cudaStreamWaitEvent AFTER it — sound, but the wait is
 * enqueued from another thread, which races the caller's own consumer
 * kernels unless the library is drained first (half-synchronous, fragile
 * contract). The semaphore alternative (cuStreamWaitValue32) was rejected
 * by CUDA's own documentation: synchronization established via stream
 * memory operations is not visible to the CUDA scheduler and must not be
 * the only ordering between CUDA tasks.
 *
 * The host-function model instead enqueues ONE host callback on the user
 * stream (devconn->launch_host_func). CUDA guarantees the callback is
 * "called after currently enqueued work and will block work added after
 * it" — so read-consumers (MAR) and write-gathers (WAR) are ordered
 * correctly BY CONSTRUCTION, and a bare cudaStreamSynchronize is always
 * safe. Consecutive host functions on one stream execute in order
 * (officially supported).
 *
 * THE ONE HARD RULE
 * -----------------
 * A CUDA host callback must not make CUDA API calls — and this is
 * stronger than it reads: while a callback executes, libcuda holds
 * internal locks, so ANY thread the callback blocks on deadlocks the
 * moment that thread enters a CUDA call (verified on H20 + CUDA 13.0:
 * callback waiting on an executor that called cudaStreamCreate froze
 * both threads on a driver rwlock). The callback here therefore runs
 * only the pure pread/pwrite loop (xfer) against the P2P host address
 * of the registered buffer (or a plain CPU address), plus free().
 *
 * Staging-mode devices (map_mode=staging) are NOT supported: their
 * two-hop path (SSD -> staging pool -> D2D -> user) would need the D2D
 * leg inside or around the callback, which the rule above forbids.
 * Submissions whose buffer device is staging fail with -EOPNOTSUPP;
 * staging devices keep using the synchronous phxfs_read / phxfs_write
 * (and the batch API).
 *
 * The stream API requires a connector with the launch_host_func
 * primitive; without it submissions fail with -EOPNOTSUPP (there is no
 * synchronous fallback).
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <unistd.h>

#include "phoenix.h"
#include "phx_internal.h"

/* ------------------------------------------------------------------ */
/* The stream callback (runs on a CUDA callback thread)               */
/* ------------------------------------------------------------------ */

/*
 * One submitted job. Parameters are snapshotted at submit time; only
 * bytes_done still points into caller storage (contract: valid until
 * the stream is synchronized). The callback runs xfer against `host` —
 * the FULL-mode P2P host address of the registered buffer, or the plain
 * CPU address for CPU buffers.
 */
enum phxfs_op {
    PHXFS_OP_READ = 0,
    PHXFS_OP_WRITE = 1,
};

struct phx_stream_job {
    int          fd;
    enum phxfs_op op;
    void        *host;           /* pread/pwrite target host address */
    size_t       nbytes;
    off_t        f_offset;
    ssize_t     *bytes_done;     /* caller storage, written on completion */
};

/*
 * Runs when every op previously enqueued on the user stream completed,
 * and blocks everything enqueued after it until it returns.
 *
 * NO CUDA API CALLS, DIRECTLY OR INDIRECTLY, IN HERE — including waiting
 * on any thread that might be inside a CUDA call (driver rwlock
 * deadlock, see the file header). Everything below is pread/pwrite plus
 * free().
 */
static void phx_stream_callback(void *arg) {
    struct phx_stream_job *j = (struct phx_stream_job *)arg;

    ssize_t rc = xfer(j->fd, j->host, (ssize_t)j->nbytes, j->f_offset,
                      j->op == PHXFS_OP_WRITE);
    if (j->bytes_done)
        *j->bytes_done = rc;
    free(j);
}

/* ------------------------------------------------------------------ */
/* Submission                                                         */
/* ------------------------------------------------------------------ */

static int phxfs_stream_io(int fd, int device_id, void *buf, size_t *nbytes,
                           off_t *buf_offset, off_t *f_offset,
                           ssize_t *bytes_done, void *stream,
                           enum phxfs_op op) {
    if (!nbytes || !buf_offset || !f_offset || !bytes_done || !stream || !buf)
        return -EINVAL;
    if (*nbytes == 0 || *buf_offset < 0 || *f_offset < 0)
        return -EINVAL;
    if ((uint64_t)*nbytes > (uint64_t)INT64_MAX - (uint64_t)*f_offset)
        return -EINVAL;
    /* Staging-mode buffer devices are not supported by the stream API
     * (the D2D leg of their two-hop path cannot live inside or around
     * the callback). */
    if (device_id >= 0 && device_id < g_device_count &&
        mbuffer[device_id].map_mode == PHX_MAP_MODE_STAGING)
        return -EOPNOTSUPP;

    /* The stream API requires the host-function primitive. */
    if (!devconn || !devconn->launch_host_func)
        return -EOPNOTSUPP;

    const size_t nb = *nbytes;
    const off_t  bo = *buf_offset, fo = *f_offset;

    *bytes_done = 0;   /* in-flight marker (written again on completion) */

    struct phx_stream_job *j =
        (struct phx_stream_job *)calloc(1, sizeof(*j));
    if (!j) {
        *bytes_done = -ENOMEM;
        return -ENOMEM;
    }
    j->fd = fd;
    j->op = op;
    j->nbytes = nb;
    j->f_offset = fo;
    j->bytes_done = bytes_done;

    /*
     * Resolve the host leg. resolve_registered() takes a mapping
     * reference, making the address resolution itself atomic w.r.t. a
     * concurrent phxfs_deregmem(); the reference is released immediately
     * — the buffer-lifetime contract (see phoenix.h: fd, buffer and its
     * registration stay valid until the stream is synchronized) covers
     * the callback's later use of the address, exactly as with
     * cuFileReadAsync / cuFileWriteAsync.
     */
    if (device_id < 0) {
        if (resolve_cpu_buf(buf, bo, nb, &j->host) != 0) {
            free(j);
            *bytes_done = -EFAULT;
            return -EFAULT;
        }
    } else {
        phxfs_p2p_map_t *node = NULL;
        if (resolve_registered(device_id, buf, bo, nb, &j->host, &node) != 1) {
            free(j);
            *bytes_done = -EFAULT;
            return -EFAULT;   /* buf not inside a registration on this device */
        }
        map_release(&mbuffer[device_id], node);
    }

    /* Enqueue the callback — it IS the whole DMA. Ordering with the
     * caller's preceding (WAR: write gathers) and following (MAR: read
     * consumers) ops is guaranteed by the host-function semantics. */
    int rc = devconn->launch_host_func(device_id, stream,
                                       phx_stream_callback, j);
    if (rc != 0) {
        free(j);
        *bytes_done = rc;
        return rc;
    }

    return 0;   /* accepted; outcome in *bytes_done after stream sync */
}

int phxfs_read_stream(int fd, int device_id, void *buf, size_t *nbytes,
                      off_t *buf_offset, off_t *f_offset,
                      ssize_t *bytes_done, void *stream) {
    return phxfs_stream_io(fd, device_id, buf, nbytes, buf_offset, f_offset,
                           bytes_done, stream, PHXFS_OP_READ);
}

int phxfs_write_stream(int fd, int device_id, void *buf, size_t *nbytes,
                       off_t *buf_offset, off_t *f_offset,
                       ssize_t *bytes_done, void *stream) {
    return phxfs_stream_io(fd, device_id, buf, nbytes, buf_offset, f_offset,
                           bytes_done, stream, PHXFS_OP_WRITE);
}
