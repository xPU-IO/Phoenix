// SPDX-License-Identifier: Apache-2.0
/*
 * Batch I/O thread pool.
 *
 * A fixed set of PHXFS_POOL_THREADS worker threads, each running the active
 * engine's submit_batch on its own thread_local ring. Workers pull chunks of
 * work from the live jobs, so N rings issue I/O concurrently and reach the
 * storage array's aggregate bandwidth ceiling, while the caller makes one
 * (blocking or async) call and crosses the Python GIL once.
 *
 * Workers are NOT pinned to any NUMA node: the P2P transfer this pool exists
 * to drive is device-to-device DMA that never touches host RAM, so the CPU a
 * worker runs on has no effect on that data path (see io_engine.h). The only
 * job of this pool is fanning a batch out across enough independent rings.
 *
 * Scheduling model (shared work queue, no per-job barrier):
 *   - Every submitted job joins a bounded FIFO of *live* jobs. A worker claims
 *     a chunk of a job's ops (ceil(n / workers), capped at
 *     PHXFS_POOL_CHUNK_MAX), runs that chunk to completion on
 *     its own ring, accounts the results, and claims again.
 *   - There is no barrier between jobs: a worker that exhausts one job's ops
 *     immediately claims from the next, so the device queue does not drain at
 *     a job boundary. (The previous model ran one job at a time across all
 *     workers and waited for every worker before promoting the next job, which
 *     emptied the device queue once per job — visible as a PCIe bubble per
 *     staging group.)
 *   - Claiming is round-robin over the live jobs, so a small batch submitted
 *     behind a huge one (e.g. another GPU's worker) makes progress right away
 *     instead of waiting behind it (no head-of-line blocking).
 *   - A chunk is a contiguous index range, so the worker hands the engine a
 *     slice of the caller's array directly — no copy — and the offsets it
 *     issues are sequential, which suits merging in the block layer.
 *   - Each job carries its own result state (claimed/completed/failed/err);
 *     results are read by that job's wait(), decoupled from scheduling.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <pthread.h>

#include "io_engine.h"

/* Bounded in-flight jobs (queued + running). Enough to pipeline a few batches
 * for compute/I/O overlap without letting a runaway producer queue unbounded
 * work. */
#define PHXFS_POOL_QUEUE_CAP 16

/*
 * Upper bound on the ops one worker claims per round. Sized so a worker keeps
 * a deep ring (hundreds of concurrent ops) while still returning often enough
 * to pick up another job's work. The per-job claim size is derived from this
 * and the job size (see job_enqueue) so that a small batch is still spread
 * across every worker instead of landing on one ring.
 */
#define PHXFS_POOL_CHUNK_MAX 256

/* One unit of work. Independent, heap- or stack-allocated by the submitter; its
 * result fields are filled by the workers and read by the matching wait(). */
struct pool_job {
    struct phxfs_io_op_req *reqs;
    int              n;
    enum phxfs_io_op op;
    int              chunk;    /* ops per claim (fixed for the job) */
    int              claimed;  /* ops handed out to workers */
    int              completed;/* ops accounted (== n -> done) */
    int              failed;   /* OUT: accumulated failure count (under mtx) */
    int              err;      /* OUT: first engine-level (<0) error, else 0 */
    bool             has_callbacks;
    bool             done;     /* set once every op has been accounted */
    struct pool_job *next;     /* FIFO link */
};

struct worker {
    pthread_t tid;
    struct phxfs_io_worker_ctx ctx;
};

struct pool_state {
    struct worker   workers[PHXFS_POOL_THREADS];
    int             nthreads;

    pthread_mutex_t mtx;
    pthread_cond_t  work_cv;   /* new work / stop -> workers */
    pthread_cond_t  done_cv;   /* a job completed -> waiters */
    pthread_cond_t  free_cv;   /* a queue slot freed -> blocked submitters */

    struct pool_job *q_head, *q_tail;  /* live jobs, in submit order */
    struct pool_job *rr;              /* round-robin claim cursor */
    int              inflight;         /* live jobs (queued + running) */
    bool             stop;
};

static struct pool_state g_pool;
static bool              g_pool_ready = false;
static pthread_once_t    g_once = PTHREAD_ONCE_INIT;

/*
 * Run reqs[base .. base+cnt) of `job` on this thread's ring. The range is
 * contiguous, so the engine writes straight into the caller's array.
 */
static void worker_run_chunk(struct pool_job *job, int base, int cnt,
                             struct phxfs_io_worker_ctx *worker,
                             int *out_failed, int *out_err) {
    const struct phxfs_io_engine *eng = phxfs_io_engine_get();
    struct phxfs_io_op_req *ops = &job->reqs[base];

    int prep_failed = 0;
    int callback_slots = 0;
    for (int k = 0; k < cnt; k++) {
        ops[k].worker_slot = -1;
        if (!ops[k].prepare)
            continue;
        if (callback_slots >= PHXFS_HOST_STAGE_SLOTS ||
            ops[k].prepare(&ops[k], job->op, worker, callback_slots) != 0) {
            prep_failed = -ENOMEM;
            break;
        }
        ops[k].worker_slot = callback_slots++;
    }

    if (prep_failed) {
        for (int k = 0; k < cnt; k++)
            ops[k].result = prep_failed;
        *out_failed = cnt;
        *out_err = prep_failed;
        return;
    }

    int rc = eng->submit_batch(ops, cnt, job->op);
    if (rc < 0) {
        /* Engine-level failure: record it and mark this chunk's requests, but
         * do NOT fold into a positive failure count. */
        for (int k = 0; k < cnt; k++)
            ops[k].result = rc;
        *out_failed = cnt;
        *out_err = rc;
    } else {
        *out_failed = rc;   /* engine contract: >=0 is the failed-request count */
        *out_err = 0;
    }

    for (int k = 0; k < cnt; k++) {
        if (ops[k].complete)
            ops[k].complete(&ops[k], job->op, worker, ops[k].worker_slot);
    }
}

/*
 * Claim up to this job's chunk size from some live job, round-robin across
 * jobs. Returns the job, with the claimed range written to base/cnt, or NULL
 * if no live job has unclaimed ops. Caller holds mtx.
 */
static struct pool_job *claim_locked(struct pool_state *p, int *base, int *cnt) {
    struct pool_job *start = p->rr ? p->rr : p->q_head;
    struct pool_job *j = start;

    while (j) {
        if (j->claimed < j->n) {
            int take = j->n - j->claimed;
            if (take > j->chunk)
                take = j->chunk;
            *base = j->claimed;
            *cnt  = take;
            j->claimed += take;
            /* Next claim starts at the following job -> round robin. */
            p->rr = j->next ? j->next : p->q_head;
            return j;
        }
        j = j->next ? j->next : p->q_head;   /* wrap */
        if (j == start)
            break;                            /* full circle: nothing to claim */
    }
    return NULL;
}

/* Unlink a completed job, keeping q_tail and the round-robin cursor valid.
 * Caller holds mtx. */
static void job_remove_locked(struct pool_state *p, struct pool_job *job) {
    struct pool_job **pp = &p->q_head;
    struct pool_job  *prev = NULL;

    while (*pp && *pp != job) {
        prev = *pp;
        pp = &(*pp)->next;
    }
    if (!*pp)
        return;                     /* not linked (defensive) */
    *pp = job->next;
    if (p->q_tail == job)
        p->q_tail = prev;
    if (p->rr == job)
        p->rr = job->next ? job->next : p->q_head;
    job->next = NULL;
}

static void job_complete_locked(struct pool_state *p, struct pool_job *job) {
    if (job->done || job->completed != job->n)
        return;
    job->done = true;
    job_remove_locked(p, job);
    p->inflight--;
    pthread_cond_broadcast(&p->done_cv);
    pthread_cond_signal(&p->free_cv);
    if (p->stop && p->inflight == 0)
        pthread_cond_broadcast(&p->work_cv);
}

static void *worker_main(void *arg) {
    struct pool_state *p = &g_pool;
    struct worker *worker = (struct worker *)arg;
    struct phxfs_io_worker_ctx *worker_ctx = &worker->ctx;

    pthread_mutex_lock(&p->mtx);
    for (;;) {
        int base = 0, cnt = 0;
        struct pool_job *job = claim_locked(p, &base, &cnt);

        while (!job) {
            /* Exit only once every live job has been fully accounted, so no
             * waiter is ever left hanging by a shutdown. */
            if (p->stop && p->inflight == 0) {
                pthread_mutex_unlock(&p->mtx);
                if (worker_ctx->cleanup && worker_ctx->opaque)
                    worker_ctx->cleanup(worker_ctx->opaque);
                return NULL;
            }
            pthread_cond_wait(&p->work_cv, &p->mtx);
            job = claim_locked(p, &base, &cnt);
        }
        pthread_mutex_unlock(&p->mtx);

        /*
         * `job` stays valid here without the lock: done (and therefore the
         * waiter's return, which may free stack-allocated jobs) can only be
         * set once completed == n, and this chunk is counted below.
         */
        int lfailed = 0, lerr = 0;
        worker_run_chunk(job, base, cnt, worker_ctx, &lfailed, &lerr);

        pthread_mutex_lock(&p->mtx);
        job->failed += lfailed;
        if (job->err == 0 && lerr < 0)
            job->err = lerr;
        job->completed += cnt;
        job_complete_locked(p, job);
    }
}

/*
 * Called in the child after fork(): the parent's worker threads did not
 * survive, so the pool's mutex/condvars and g_once are stale. Reset everything
 * to "never initialised" so the child lazily re-creates the pool on first use.
 */
static void pool_atfork_child(void) {
    g_pool_ready = false;
    g_once = PTHREAD_ONCE_INIT;
    g_pool.nthreads = 0;
    g_pool.q_head = g_pool.q_tail = NULL;
    g_pool.rr = NULL;
    g_pool.inflight = 0;
    g_pool.stop = false;
}

static void pool_init(void) {
    struct pool_state *p = &g_pool;
    pthread_mutex_init(&p->mtx, NULL);
    pthread_cond_init(&p->work_cv, NULL);
    pthread_cond_init(&p->done_cv, NULL);
    pthread_cond_init(&p->free_cv, NULL);
    p->q_head = p->q_tail = NULL;
    p->rr = NULL;
    p->inflight = 0;
    p->stop = false;
    p->nthreads = PHXFS_POOL_THREADS;
    for (int t = 0; t < p->nthreads; t++) {
        p->workers[t].ctx.index = t;
        p->workers[t].ctx.opaque = NULL;
        p->workers[t].ctx.cleanup = NULL;
        if (pthread_create(&p->workers[t].tid, NULL, worker_main,
                           &p->workers[t]) != 0) {
            p->nthreads = t;
            break;
        }
    }
    g_pool_ready = true;

    /*
     * fork() only copies the calling thread; the worker threads vanish but
     * g_pool_ready / g_once survive. Without a reset the child would enqueue
     * a job, broadcast work_cv, and block forever (no workers to process it).
     * pthread_atfork(child) resets the pool to its pre-init state so the
     * child lazily re-creates it on first use.
     */
    pthread_atfork(NULL, NULL, pool_atfork_child);
}

/* Validate the pool. Returns NULL if inline fallback needed. */
static struct pool_state *pool_get(void) {
    pthread_once(&g_once, pool_init);
    if (!g_pool_ready || g_pool.nthreads <= 0)
        return NULL;
    return &g_pool;
}

/*
 * Enqueue `job` on `p` (bounded FIFO). blocking=false returns -EBUSY when the
 * queue is full (async submit); blocking=true waits for a free slot (sync).
 * Returns 0 on success, -1 with errno set (EBUSY / ESHUTDOWN) otherwise.
 */
static int job_enqueue(struct pool_state *p, struct pool_job *job, bool blocking) {
    pthread_mutex_lock(&p->mtx);
    if (p->stop) {
        pthread_mutex_unlock(&p->mtx);
        errno = ESHUTDOWN;
        return -1;
    }
    while (p->inflight >= PHXFS_POOL_QUEUE_CAP) {
        if (!blocking) {
            pthread_mutex_unlock(&p->mtx);
            errno = EBUSY;
            return -1;
        }
        pthread_cond_wait(&p->free_cv, &p->mtx);
        if (p->stop) {
            pthread_mutex_unlock(&p->mtx);
            errno = ESHUTDOWN;
            return -1;
        }
    }
    job->done = false;
    job->claimed = 0;
    job->completed = 0;
    job->failed = 0;
    job->err = 0;
    job->next = NULL;
    job->has_callbacks = false;
    for (int i = 0; i < job->n; i++) {
        if (job->reqs[i].prepare || job->reqs[i].complete) {
            job->has_callbacks = true;
            break;
        }
    }
    /*
     * Claim size: spread the job over every worker (so a small batch still
     * uses all rings, as the previous striped model did), capped so a huge
     * batch is handed out in bounded pieces and workers can interleave jobs.
     * Fixed for the job's lifetime, so the number of engine calls is
     * ceil(n / chunk) — no shrinking tail chunks.
     */
    job->chunk = (job->n + p->nthreads - 1) / p->nthreads;
    if (job->chunk > PHXFS_POOL_CHUNK_MAX)
        job->chunk = PHXFS_POOL_CHUNK_MAX;
    if (job->chunk < 1)
        job->chunk = 1;
    if (job->has_callbacks && job->chunk > PHXFS_HOST_STAGE_SLOTS)
        job->chunk = PHXFS_HOST_STAGE_SLOTS;
    if (p->q_tail) p->q_tail->next = job; else p->q_head = job;
    p->q_tail = job;
    if (!p->rr)
        p->rr = job;
    p->inflight++;
    pthread_cond_broadcast(&p->work_cv);   /* several workers may take chunks */
    pthread_mutex_unlock(&p->mtx);
    return 0;
}

/* Wait for `job` to complete; returns failure count or first engine error. */
static int job_wait(struct pool_state *p, struct pool_job *job) {
    pthread_mutex_lock(&p->mtx);
    while (!job->done)
        pthread_cond_wait(&p->done_cv, &p->mtx);
    int failed = job->failed, err = job->err;
    pthread_mutex_unlock(&p->mtx);
    return err < 0 ? err : failed;
}

/*
 * Run a job on the calling thread (no pool, or pool shutting down). Goes
 * through worker_run_chunk() like a worker does, so per-op prepare/complete
 * hooks still run, and keeps chunks within PHXFS_HOST_STAGE_SLOTS when hooks
 * are present (each prepared op needs its own worker slot).
 */
static int run_inline(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op) {
    struct pool_job job{};
    struct phxfs_io_worker_ctx worker{};
    int failed = 0, err = 0;
    job.reqs = reqs;
    job.n = n;
    job.op = op;
    worker.index = -1;

    int step = n;
    for (int i = 0; i < n; i++) {
        if (reqs[i].prepare || reqs[i].complete) {
            step = PHXFS_HOST_STAGE_SLOTS;
            break;
        }
    }
    for (int base = 0; base < n; base += step) {
        int cnt = n - base < step ? n - base : step;
        int lfailed = 0, lerr = 0;
        worker_run_chunk(&job, base, cnt, &worker, &lfailed, &lerr);
        failed += lfailed;
        if (err == 0 && lerr < 0)
            err = lerr;
    }
    if (worker.cleanup && worker.opaque)
        worker.cleanup(worker.opaque);
    return err < 0 ? err : failed;
}

int phxfs_pool_run(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op) {
    if (n <= 0)
        return 0;

    struct pool_state *p = pool_get();
    if (!p)
        return run_inline(reqs, n, op);

    /* Synchronous: stack job (valid until we wait it), blocking enqueue. */
    struct pool_job job;
    memset(&job, 0, sizeof(job));
    job.reqs = reqs;
    job.n = n;
    job.op = op;
    if (job_enqueue(p, &job, /*blocking=*/true) != 0)
        return run_inline(reqs, n, op);   /* shutting down */
    return job_wait(p, &job);
}

/* ------------------------------------------------------------------ *
 * Async: submit enqueues and returns immediately; wait blocks for that job.
 * Multiple jobs may be queued (bounded), enabling pipelining.
 * ------------------------------------------------------------------ */
struct phxfs_pool_async {
    struct pool_state *p;
    struct pool_job     job;      /* embedded; enqueued by address */
    bool  inline_done;
    int   inline_failed;
};

struct phxfs_pool_async *phxfs_pool_submit(struct phxfs_io_op_req *reqs, int n,
                                           enum phxfs_io_op op, bool blocking) {
    struct phxfs_pool_async *h =
        (struct phxfs_pool_async *)calloc(1, sizeof(*h));
    if (!h)
        return NULL;

    if (n <= 0) {
        h->inline_done = true;
        h->inline_failed = 0;
        return h;
    }

    struct pool_state *p = pool_get();
    if (!p) {
        /* Async needs the worker pool. Do NOT run the batch inline here: that
         * would make "submit returns immediately" a lie. Fail so the caller
         * can fall back to the synchronous batch API instead. */
        free(h);
        errno = ENOTSUP;
        return NULL;
    }

    h->p = p;
    h->job.reqs = reqs;
    h->job.n = n;
    h->job.op = op;
    /* Enqueue on the bounded FIFO. blocking=false (async) returns EBUSY only
     * when the queue is FULL (not merely non-empty), so pipelining several
     * batches succeeds. blocking=true is used by the sync path when it wants
     * to reserve a slot. */
    if (job_enqueue(p, &h->job, blocking) != 0) {
        free(h);            /* errno set by job_enqueue (EBUSY / ESHUTDOWN) */
        return NULL;
    }
    return h;
}

int phxfs_pool_cancel(struct phxfs_pool_async *h) {
    if (!h)
        return -EINVAL;
    if (h->inline_done)
        return 0;

    struct pool_state *p = h->p;
    struct pool_job *job = &h->job;
    pthread_mutex_lock(&p->mtx);
    int pending = job->n - job->claimed;
    for (int i = job->claimed; i < job->n; i++)
        job->reqs[i].result = -ECANCELED;
    job->claimed = job->n;
    job->completed += pending;
    job->failed += pending;
    job_complete_locked(p, job);
    pthread_mutex_unlock(&p->mtx);
    return 0;
}

int phxfs_pool_wait(struct phxfs_pool_async *h) {
    if (!h)
        return -EINVAL;
    int rc;
    if (h->inline_done)
        rc = h->inline_failed;
    else
        rc = job_wait(h->p, &h->job);
    free(h);
    return rc;
}

/*
 * Stop and join every worker at library unload. New submits are refused
 * (ESHUTDOWN); already-queued jobs are drained so their waiters never hang.
 * Joining lets each worker return from worker_main(), running its
 * thread_local ring's destructor (io_uring_queue_exit), so no ring leaks and
 * no unloaded code runs later. Safe if the pool was never initialised.
 */
__attribute__((destructor))
static void phxfs_pool_shutdown(void) {
    if (!g_pool_ready)
        return;
    struct pool_state *p = &g_pool;
    pthread_mutex_lock(&p->mtx);
    p->stop = true;
    pthread_cond_broadcast(&p->work_cv);   /* wake idle workers to drain/exit */
    pthread_cond_broadcast(&p->free_cv);   /* wake blocked submitters */
    pthread_mutex_unlock(&p->mtx);
    for (int t = 0; t < p->nthreads; t++)
        pthread_join(p->workers[t].tid, NULL);
    g_pool_ready = false;
}
