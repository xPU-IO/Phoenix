/*
 * phx_stream.cpp — stream-ordered async I/O via a CUDA host function.
 *
 * Implements phxfs_read_stream / phxfs_write_stream (contract in
 * phoenix.h). No stream registration: every submission carries the
 * stream, like cuFileReadAsync / cuFileWriteAsync.
 *
 * libphoenix's DMA is host-driven and invisible to CUDA, so it cannot be
 * a stream op. Instead a single host callback (launch_host_func ==
 * cudaLaunchHostFunc) IS the DMA: CUDA guarantees it runs after the ops
 * enqueued before it and blocks the ops enqueued after it, so read
 * consumers (MAR) and write gathers (WAR) are ordered by construction,
 * and a bare stream sync is always safe. The alternatives were rejected:
 * event bridges enqueue their wait from another thread (racy without
 * drains); cuStreamWaitValue-style memops are documented as invisible to
 * the CUDA scheduler.
 *
 * HARD RULE: the callback must not call CUDA APIs, not even indirectly —
 * libcuda holds internal locks while it runs, so any thread the callback
 * waits on deadlocks the moment that thread enters CUDA. The callback is
 * therefore only xfer() (pread/pwrite) + free().
 *
 * Staging-mode devices and connectors without launch_host_func are
 * rejected with -EOPNOTSUPP — there is no synchronous fallback.
 */

#include <cerrno>
#include <cstdint>
#include <cstdlib>

#include "phoenix.h"
#include "phx_internal.h"

enum phxfs_op {
    PHXFS_OP_READ = 0,
    PHXFS_OP_WRITE = 1,
};

/* One submitted job: parameters snapshotted at submit time; only
 * bytes_done still points into caller storage (valid until the stream
 * is synchronized). host is the registered buffer's P2P host address,
 * or the plain CPU address for CPU buffers. */
struct phx_stream_job {
    int          fd;
    enum phxfs_op op;
    void        *host;           /* pread/pwrite target host address */
    size_t       nbytes;
    off_t        f_offset;
    ssize_t     *bytes_done;     /* caller storage, written on completion */
};

/* Runs when every op previously enqueued on the stream completed, and
 * blocks everything enqueued after it until it returns. See the HARD
 * RULE above: xfer() and free() only. */
static void phx_stream_callback(void *arg) {
    struct phx_stream_job *j = (struct phx_stream_job *)arg;

    *j->bytes_done = xfer(j->fd, j->host, (ssize_t)j->nbytes, j->f_offset,
                          j->op == PHXFS_OP_WRITE);
    free(j);
}

static int phxfs_stream_io(int fd, void *buf, size_t *nbytes,
                           off_t *buf_offset, off_t *f_offset,
                           ssize_t *bytes_done, void *stream,
                           enum phxfs_op op) {
    if (!nbytes || !buf_offset || !f_offset || !bytes_done || !stream || !buf)
        return -EINVAL;
    if (*nbytes == 0 || *buf_offset < 0 || *f_offset < 0)
        return -EINVAL;
    if ((uint64_t)*nbytes > (uint64_t)INT64_MAX - (uint64_t)*f_offset)
        return -EINVAL;

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

    /* Resolve the buffer across every opened device (the caller does not
     * name one): a registration hit yields the P2P host address and its
     * device; no hit anywhere means a plain CPU address. The reference
     * taken by resolve_registered() is released immediately — the
     * callback's use of the address is covered by the buffer-lifetime
     * contract in phoenix.h. */
    {
        phxfs_p2p_map_t *node = NULL;
        int device_id = -1;
        for (int d = 0; d < g_device_count; d++) {
            if (!mbuffer[d].init_stat)
                continue;
            int r = resolve_registered(d, buf, bo, nb, &j->host, &node);
            if (r == 1) {
                device_id = d;
                break;
            }
            if (r < 0) {
                /* inside a registration but the extent is invalid */
                free(j);
                *bytes_done = -EFAULT;
                return -EFAULT;
            }
        }

        if (device_id >= 0) {
            const bool staging =
                mbuffer[device_id].map_mode == PHX_MAP_MODE_STAGING;
            map_release(&mbuffer[device_id], node);
            if (staging) {
                free(j);
                *bytes_done = -EOPNOTSUPP;
                return -EOPNOTSUPP;
            }
        } else if (resolve_cpu_buf(buf, bo, nb, &j->host) != 0) {
            free(j);
            *bytes_done = -EFAULT;
            return -EFAULT;
        }
    }

    /* The callback IS the whole DMA; the host-function semantics order it
     * after the caller's gathers and before the caller's consumers. */
    int rc = devconn->launch_host_func(stream, phx_stream_callback, j);
    if (rc != 0) {
        free(j);
        *bytes_done = rc;
        return rc;
    }

    return 0;   /* accepted; outcome in *bytes_done after stream sync */
}

int phxfs_read_stream(int fd, void *buf, size_t *nbytes,
                      off_t *buf_offset, off_t *f_offset,
                      ssize_t *bytes_done, void *stream) {
    return phxfs_stream_io(fd, buf, nbytes, buf_offset, f_offset,
                           bytes_done, stream, PHXFS_OP_READ);
}

int phxfs_write_stream(int fd, void *buf, size_t *nbytes,
                       off_t *buf_offset, off_t *f_offset,
                       ssize_t *bytes_done, void *stream) {
    return phxfs_stream_io(fd, buf, nbytes, buf_offset, f_offset,
                           bytes_done, stream, PHXFS_OP_WRITE);
}
