// SPDX-License-Identifier: Apache-2.0
/*
 * io_uring batch engine.
 *
 * Submits a whole batch of reads/writes with a single io_uring_submit(),
 * then reaps all completions — eliminating the per-request syscall of the
 * sync engine and letting the storage layer service requests concurrently.
 *
 * Concurrency & runtime-cost:
 *   - Per-thread ring + scratch pool (thread_local). A single submitting
 *     thread saturates only ~10 GiB/s on a 4-disk RAID0; multiple worker
 *     threads (e.g. one per GPU) each get their own ring and submit
 *     concurrently with no lock/contention, reaching the array's limit.
 *   - The ring (QD=1024) and pool are created lazily on a thread's first
 *     batch and reused for its lifetime; no per-call setup/teardown.
 *   - slice/iovec scratch is a fixed per-thread pool, so the hot path
 *     performs no malloc.
 *
 * Buffer handling: requests carry already-resolved host addresses (GPU
 * P2P VMA or CPU buffer). We deliberately use the *non-fixed* OP_READ /
 * OP_READV path (not registered buffers / READ_FIXED): it goes through the
 * same O_DIRECT DIO path as pread — which already works for phxfs GPU
 * VMAs — and avoids taking a second FOLL_LONGTERM pin on device pages.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <liburing.h>

#include "io_engine.h"

/* Ring depth. KV-cache retrieve issues many concurrent small reads. */
#define PHXFS_URING_QD 1024

/* Pre-allocated scratch pool. One slot per in-flight slice; batches with
 * more slices than this are processed in multiple waves reusing the pool. */
#define PHXFS_SLICE_POOL 16384

/*
 * A "slice" is one SQE covering [off, off+len) of request `idx`.
 * Each request is fanned out into <=chunk slices so no single op exceeds
 * PHXFS_IO_CHUNK.
 */
struct slice {
    int    idx;
    size_t off;
    size_t len;
};

/*
 * Per-thread ring + scratch (thread_local => lock-free concurrent submit).
 * The ring lives in a small RAII holder so it is torn down with
 * io_uring_queue_exit() when the owning thread exits — worker threads are
 * joined at library unload (see io_pool shutdown), so their rings are not
 * leaked and no unloaded code runs later.
 */
namespace {
struct RingHolder {
    struct io_uring ring;
    bool            ready = false;
    ~RingHolder() { if (ready) io_uring_queue_exit(&ring); }
};
}  // namespace
static thread_local RingHolder   t_ringh;
static thread_local struct slice t_slices[PHXFS_SLICE_POOL];
static thread_local struct iovec t_iovs[PHXFS_SLICE_POOL];
/* Indices of slices needing resubmission after a short read/write. */
static thread_local int          t_reissue[PHXFS_SLICE_POOL];

/* OP_READ vs OP_READV capability — process-wide, resolved once. */
static bool g_have_rw = false;
static bool g_uring_avail = false;

/*
 * Determine whether OP_READ (single-buffer) is accepted, using a harmless
 * zero-length read. IORING_REGISTER_PROBE is unreliable on some backported
 * kernels, so we test the opcode directly. Falls back to OP_READV. Also
 * confirms io_uring is usable at all (g_uring_avail).
 */
static void detect_caps(void) {
    struct io_uring ring;
    if (io_uring_queue_init(2, &ring, 0) < 0)
        return;  /* g_uring_avail stays false */
    g_uring_avail = true;

    /* Use a dedicated fd rather than stdin (fd 0): a daemonised or
     * containerised process may have closed fd 0, in which case a 0-length
     * read on it completes with -EBADF, which is neither -EINVAL nor
     * -EOPNOTSUPP and would be misread as "OP_READ is supported". */
    int probe_fd = open("/dev/null", O_RDONLY);
    if (probe_fd >= 0) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        char c;
        io_uring_prep_read(sqe, probe_fd, &c, 0, 0);  /* 0-length: no data moved */
        if (io_uring_submit(&ring) == 1) {
            struct io_uring_cqe *cqe;
            if (io_uring_wait_cqe(&ring, &cqe) == 0) {
                if (cqe->res != -EINVAL && cqe->res != -EOPNOTSUPP)
                    g_have_rw = true;
                io_uring_cqe_seen(&ring, cqe);
            }
        }
        close(probe_fd);
    }
    io_uring_queue_exit(&ring);
}

/* Lazily create this thread's ring on first use. */
static bool thread_ring(void) {
    if (t_ringh.ready)
        return true;
    if (!g_uring_avail)
        return false;
    if (io_uring_queue_init(PHXFS_URING_QD, &t_ringh.ring, 0) < 0)
        return false;
    t_ringh.ready = true;
    return true;
}

static int uring_probe(void) {
    detect_caps();
    return g_uring_avail ? 0 : -1;
}

/* Prep a single slice into an sqe (OP_READ or OP_READV per capability). */
static void prep_op(struct io_uring_sqe *sqe, struct phxfs_io_op_req *r,
                    size_t off, size_t len, enum phxfs_io_op op,
                    struct iovec *iov) {
    char *addr = (char *)r->host_addr + off;
    off_t foff = r->f_offset + (off_t)off;
    if (g_have_rw) {
        if (op == PHXFS_IO_READ)
            io_uring_prep_read(sqe, r->fd, addr, (unsigned)len, foff);
        else
            io_uring_prep_write(sqe, r->fd, addr, (unsigned)len, foff);
    } else {
        iov->iov_base = addr;
        iov->iov_len = len;
        if (op == PHXFS_IO_READ)
            io_uring_prep_readv(sqe, r->fd, iov, 1, foff);
        else
            io_uring_prep_writev(sqe, r->fd, iov, 1, foff);
    }
}

/*
 * Build slices starting at request *cur_req, byte offset *cur_off, filling up
 * to PHXFS_SLICE_POOL slots. On return, the cursor (cur_req, cur_off) points
 * at the first not-yet-sliced byte, so a single request larger than the pool
 * simply spans multiple waves and is never truncated. Slice length uses
 * subtraction (total - off), so it cannot overflow near SIZE_MAX. Returns the
 * number of slices built this wave.
 */
static size_t build_slices(struct phxfs_io_op_req *reqs, int n,
                           int *cur_req, size_t *cur_off) {
    size_t s = 0;
    int    i = *cur_req;
    size_t off = *cur_off;
    while (i < n && s < PHXFS_SLICE_POOL) {
        size_t total = reqs[i].nbytes;
        if (total == 0) {              /* zero-length request: one empty slice */
            t_slices[s++] = {i, 0, 0};
            i++; off = 0;
            continue;
        }
        while (off < total && s < PHXFS_SLICE_POOL) {
            size_t rem = total - off;  /* subtraction avoids overflow near SIZE_MAX */
            size_t len = rem > PHXFS_IO_CHUNK ? PHXFS_IO_CHUNK : rem;
            t_slices[s++] = {i, off, len};
            off += len;
        }
        if (off >= total) {            /* finished this request */
            i++; off = 0;
        }
        /* else: pool full mid-request; resume here on the next wave */
    }
    *cur_req = i;
    *cur_off = off;
    return s;
}

/*
 * Account one completion into the owning request's result.
 *
 * io_uring completions for slices of the same request can arrive in any
 * order, including "error first, success later". We must never lose
 * completed bytes regardless of ordering, and must match the sync engine's
 * semantics: "partial progress if any, else the negative errno".
 *
 * Strategy: result accumulates completed bytes (always >= 0 once any
 * progress is made). A negative errno is stashed in a side array; it is
 * only applied to result if, at the end of the batch, result is still 0
 * (no slice ever succeeded).
 */
static thread_local int t_req_err[PHXFS_SLICE_POOL];   /* per-slice errno stash (indexed by slice, not by req) */

static inline void account(struct phxfs_io_op_req *reqs, struct slice *sl,
                           int res) {
    int idx = sl->idx;
    if (res < 0) {
        /* Record the error but do NOT touch result: a later-arriving
         * success for another slice of this same request must still be
         * able to accumulate. */
        t_req_err[sl - t_slices] = res;
    } else {
        if (reqs[idx].result >= 0)
            reqs[idx].result += res;
    }
}

/*
 * Destroy this thread's ring so the next batch re-creates a clean one. Called
 * after any unrecoverable submit/wait error: it discards every pending SQE
 * and un-reaped CQE, so no stale state can be mis-attributed to a later batch.
 * thread_ring() lazily recreates on the next call.
 */
static void ring_reset(void) {
    if (t_ringh.ready) {
        io_uring_queue_exit(&t_ringh.ring);
        t_ringh.ready = false;
    }
}

/*
 * Flush all prepared-but-unsubmitted SQEs, tolerating EINTR and partial
 * submits (io_uring_submit returns the count actually consumed; the rest stay
 * queued and are flushed on the next call). Returns 0, or a negative errno
 * that the caller must treat as unrecoverable (-> ring_reset).
 */
static int submit_all(size_t *pending) {
    while (*pending > 0) {
        int s = io_uring_submit(&t_ringh.ring);
        if (s < 0) {
            if (s == -EINTR)
                continue;
            return s;              /* real submit error */
        }
        if (s == 0)
            return -EIO;           /* nothing accepted while work pending */
        *pending -= (size_t)s;
    }
    return 0;
}

/*
 * Submit `nslices` from this thread's pool as a pipeline keeping up to QD
 * slices in flight; reap completions in batches. Handles:
 *   - partial submit: keep submitting until all prepared SQEs are accepted;
 *   - EINTR on submit/wait: retry;
 *   - short read/write: advance the slice and resubmit the remainder;
 *   - unrecoverable submit/wait error: destroy+rebuild the ring, so no stale
 *     SQE/CQE leaks into the next batch.
 */
static int run_slices(struct phxfs_io_op_req *reqs, size_t nslices,
                      enum phxfs_io_op op) {
    size_t inflight = 0;    /* prepared SQEs awaiting completion */
    size_t pending  = 0;    /* prepared SQEs not yet accepted by submit */
    size_t terminal = 0;    /* slices that reached a terminal state */
    size_t next_new = 0;    /* next fresh slice to enqueue */
    int    nreissue = 0;    /* slices queued for resubmission (short I/O) */

    struct io_uring_cqe *cqes[256];

    while (terminal < nslices) {
        /* Fill: resubmissions first, then fresh slices, up to QD. */
        while (inflight < PHXFS_URING_QD &&
               (nreissue > 0 || next_new < nslices)) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&t_ringh.ring);
            if (!sqe)
                break;  /* SQ ring full for now */
            size_t si = (nreissue > 0) ? (size_t)t_reissue[--nreissue]
                                       : next_new++;
            struct slice *sl = &t_slices[si];
            prep_op(sqe, &reqs[sl->idx], sl->off, sl->len, op, &t_iovs[si]);
            io_uring_sqe_set_data(sqe, sl);
            inflight++;
            pending++;
        }

        if (pending > 0) {
            int rc = submit_all(&pending);
            if (rc < 0) { ring_reset(); return rc; }
        }

        if (inflight == 0)
            break;  /* all slices terminal */

        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&t_ringh.ring, &cqe);
        if (ret == -EINTR)
            continue;
        if (ret < 0) { ring_reset(); return ret; }

        unsigned got = io_uring_peek_batch_cqe(&t_ringh.ring, cqes,
                                               (unsigned)(inflight < 256 ? inflight : 256));
        for (unsigned c = 0; c < got; c++) {
            struct slice *sl = (struct slice *)io_uring_cqe_get_data(cqes[c]);
            int res = cqes[c]->res;
            account(reqs, sl, res);
            if (res > 0 && (size_t)res < sl->len) {
                /* Short I/O: advance and resubmit the remaining range. */
                sl->off += (size_t)res;
                sl->len -= (size_t)res;
                t_reissue[nreissue++] = (int)(sl - t_slices);
            } else {
                terminal++;   /* fully done, EOF (res==0), or error (res<0) */
            }
        }
        io_uring_cq_advance(&t_ringh.ring, got);
        inflight -= got;
    }

    /* After all slices in this wave have terminal-completed, apply any
     * stashed errnos to requests that still have result==0 (no slice ever
     * succeeded for them). Requests that did make progress keep their
     * positive byte count — the caller detects the failure via
     * result != nbytes. */
    for (size_t i = 0; i < nslices; i++) {
        if (t_req_err[i] != 0) {
            int idx = t_slices[i].idx;
            if (reqs[idx].result == 0)
                reqs[idx].result = t_req_err[i];
            t_req_err[i] = 0;   /* clear for the next wave */
        }
    }

    return 0;
}

static int uring_submit_batch(struct phxfs_io_op_req *reqs, int n,
                              enum phxfs_io_op op) {
    if (n <= 0)
        return 0;
    /*
     * The probe only creates a depth-2 ring; the real per-thread ring is
     * QD=1024 and may still fail here (memlock, RLIMIT, container policy).
     * Rather than fail the whole batch, fall back to the always-available
     * sync engine for this thread.
     */
    if (!thread_ring())
        return phxfs_io_engine_sync.submit_batch(reqs, n, op);

    for (int i = 0; i < n; i++)
        reqs[i].result = 0;

    int rc = 0;
    /*
     * Process the batch in waves of at most PHXFS_SLICE_POOL slices. The
     * (cur_req, cur_off) cursor lets a single request span multiple waves, so
     * an arbitrarily large request is fully issued rather than truncated.
     * Common case (n <= pool, each req <= chunk) is a single wave.
     */
    int    cur_req = 0;
    size_t cur_off = 0;
    while (cur_req < n) {
        size_t ns = build_slices(reqs, n, &cur_req, &cur_off);
        if (ns == 0)
            break;   /* defensive: no progress possible */
        /* Clear only this wave's slice range of the errno stash. run_slices()
         * leaves it clean on the normal path, but an early ring_reset() return
         * can leave entries set, so clearing here (O(ns), not O(pool)) is what
         * keeps waves independent. */
        for (size_t i = 0; i < ns; i++)
            t_req_err[i] = 0;
        rc = run_slices(reqs, ns, op);
        if (rc < 0)
            break;
    }

    if (rc < 0)
        return rc;

    /* A CQE may report that this fd/filesystem does not support the io_uring
     * op (-EOPNOTSUPP) even though pread/pwrite works. A request whose only
     * slice(s) all returned -EOPNOTSUPP will have result == -EOPNOTSUPP (zero
     * progress, so the stashed errno was applied); retrying it whole via the
     * always-available sync engine is side-effect-safe (no partial write). */
    for (int k = 0; k < n; k++)
        if (reqs[k].result == -EOPNOTSUPP)
            phxfs_io_engine_sync.submit_batch(&reqs[k], 1, op);

    int failed = 0;
    for (int k = 0; k < n; k++)
        if (reqs[k].result != (ssize_t)reqs[k].nbytes)
            failed++;
    return failed;
}

const struct phxfs_io_engine phxfs_io_engine_uring = {
    "io_uring",
    uring_probe,
    uring_submit_batch,
};
