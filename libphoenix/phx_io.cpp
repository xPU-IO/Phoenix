/*
 * phx_io.cpp — single-request read/write, batch I/O, and async batch.
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
#include "io_engine/io_engine.h"

/* ------------------------------------------------------------------ */
/* Single-request pread/pwrite loop                                   */
/* ------------------------------------------------------------------ */

/*
 * True if the batch/request references a GPU device operating in staging
 * mode. The mapping mode is a global kernel module parameter, so all phxfs
 * devices share it; checking the first referenced device is sufficient.
 */
static bool device_is_staging(int device_id) {
    return device_id >= 0 && device_id < g_device_count &&
           mbuffer[device_id].map_mode == PHX_MAP_MODE_STAGING;
}

static bool batch_uses_staging(phxfs_io_req_t *reqs, int n) {
    for (int i = 0; i < n; i++)
        if (device_is_staging(reqs[i].device_id))
            return true;
    return false;
}

/*
 * Resolve a plain CPU (host) buffer address: buf + buf_offset, with the
 * pointer/length arithmetic checked against address-space overflow. Shared
 * by the single-request read/write path, the batch prepare path, and the
 * stream path (phx_stream.cpp) so a CPU-buffer (device_id < 0) request is
 * validated identically in all three. Returns 0 with *host set, or -EINVAL
 * (and *host left NULL).
 */
int resolve_cpu_buf(const void *buf, off_t buf_offset, size_t nbyte,
                    void **host) {
    *host = NULL;
    if (buf_offset < 0)
        return -EINVAL;
    uintptr_t b = (uintptr_t)buf;
    uintptr_t o = (uintptr_t)buf_offset;
    if (b > UINTPTR_MAX - o || nbyte > (size_t)(UINTPTR_MAX - (b + o)))
        return -EINVAL;
    *host = (void *)(b + o);
    return 0;
}

/*
 * Shared pread/pwrite loop for the single-request phxfs_read/phxfs_write
 * API and the stream path (phx_stream.cpp). Chunks at PHXFS_IO_CHUNK
 * because a single read()/write() syscall (or io_uring op — same
 * underlying VFS path) transfers at most MAX_RW_COUNT (~2GiB, a Linux
 * kernel constant, not something phxfs controls) per call. Retries EINTR;
 * on a genuine error returns whatever progress was already made instead
 * of discarding it.
 *
 * Pure pread/pwrite — no CUDA API — so it is legal to run inside a
 * cudaLaunchHostFunc callback.
 */
ssize_t xfer(int fd, void *host, ssize_t nbyte, off_t f_offset, bool is_write) {
    char *base = (char *)host;
    ssize_t done = 0;
    while (done < nbyte) {
        size_t chunk = (size_t)(nbyte - done);
        if (chunk > PHXFS_IO_CHUNK)
            chunk = PHXFS_IO_CHUNK;
        ssize_t ret = is_write
            ? pwrite(fd, base + done, chunk, f_offset + done)
            : pread(fd, base + done, chunk, f_offset + done);
        if (ret < 0) {
            if (errno == EINTR)
                continue;   /* retry the same chunk */
            fprintf(stderr, "%s: %s error: %s\n", __func__,
                    is_write ? "pwrite" : "pread", strerror(errno));
            return done > 0 ? done : -errno;   /* keep partial progress if any */
        }
        if (ret == 0)
            break;   /* EOF */
        done += ret;
    }
    return done;
}

/*
 * device_id >= 0: `buf` must lie inside a registration on that phxfs device.
 * device_id <  0: `buf` is a plain CPU (host) address — same convention as
 * the batch API, so a single request behaves the same as a 1-element batch.
 */
ssize_t phxfs_read(int fd, int device_id, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    if (nbyte < 0 || f_offset < 0 ||
        (uint64_t)nbyte > (uint64_t)INT64_MAX - (uint64_t)f_offset)
        return -EINVAL;

    if (device_id < 0) {
        void *host = NULL;
        int r = resolve_cpu_buf(buf, buf_offset, (size_t)nbyte, &host);
        if (r != 0)
            return r;
        return xfer(fd, host, nbyte, f_offset, /*is_write=*/false);
    }

    if (device_is_staging(device_id)) {
        phxfs_io_req_t r{};
        r.fd = fd; r.device_id = device_id; r.buf = buf;
        r.buf_offset = buf_offset; r.nbytes = (size_t)nbyte; r.f_offset = f_offset;
        phx_staging_batch(&r, 1, /*is_write=*/0);
        return r.result;
    }

    phxfs_mmap_buffer_t *pb = dev_get(device_id);
    if (!pb)
        return -1;
    void *host = NULL;
    phxfs_p2p_map_t *node = NULL;
    if (resolve_registered(device_id, buf, buf_offset, (size_t)nbyte, &host, &node) != 1) {
        dev_put(pb);
        return -1;  /* not inside a registration on this device */
    }
    ssize_t rc = xfer(fd, host, nbyte, f_offset, /*is_write=*/false);
    map_release(&mbuffer[device_id], node);
    dev_put(pb);
    return rc;
}

ssize_t phxfs_write(int fd, int device_id, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    if (nbyte < 0 || f_offset < 0 ||
        (uint64_t)nbyte > (uint64_t)INT64_MAX - (uint64_t)f_offset)
        return -EINVAL;

    if (device_id < 0) {
        void *host = NULL;
        int r = resolve_cpu_buf(buf, buf_offset, (size_t)nbyte, &host);
        if (r != 0)
            return r;
        return xfer(fd, host, nbyte, f_offset, /*is_write=*/true);
    }

    if (device_is_staging(device_id)) {
        phxfs_io_req_t r{};
        r.fd = fd; r.device_id = device_id; r.buf = buf;
        r.buf_offset = buf_offset; r.nbytes = (size_t)nbyte; r.f_offset = f_offset;
        phx_staging_batch(&r, 1, /*is_write=*/1);
        return r.result;
    }

    phxfs_mmap_buffer_t *pb = dev_get(device_id);
    if (!pb)
        return -1;
    void *host = NULL;
    phxfs_p2p_map_t *node = NULL;
    if (resolve_registered(device_id, buf, buf_offset, (size_t)nbyte, &host, &node) != 1) {
        dev_put(pb);
        return -1;  /* not inside a registration on this device */
    }
    ssize_t rc = xfer(fd, host, nbyte, f_offset, /*is_write=*/true);
    map_release(&mbuffer[device_id], node);
    dev_put(pb);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Engine selection                                                   */
/* ------------------------------------------------------------------ */

/*
 * The batch engine is chosen once at library load (constructor below),
 * not per call, so the hot path just reads a pointer. Preference order:
 * io_uring, falling back to sync. Never NULL: the sync engine always
 * probes successfully.
 */
static const struct phxfs_io_engine *g_engine = nullptr;

static void phxfs_io_engine_select(void) {
    const struct phxfs_io_engine *candidates[] = {
#ifdef PHXFS_HAVE_LIBURING
        &phxfs_io_engine_uring,
#endif
        &phxfs_io_engine_sync,
    };
    for (const auto *e : candidates) {
        if (e->probe && e->probe() == 0) {
            g_engine = e;
            return;
        }
    }
    g_engine = &phxfs_io_engine_sync;  /* guaranteed fallback */
}

__attribute__((constructor))
static void phxfs_io_engine_ctor(void) {
    phxfs_io_engine_select();
}

const struct phxfs_io_engine *phxfs_io_engine_get(void) {
    if (!g_engine)          /* defensive: constructor should have run */
        phxfs_io_engine_select();
    return g_engine;
}

const char *phxfs_io_engine_name(void) {
    return phxfs_io_engine_get()->name;
}

/* ------------------------------------------------------------------ */
/* Synchronous batch                                                  */
/* ------------------------------------------------------------------ */

/*
 * Bundled state for one in-flight batch (sync = stack, async = in handle).
 * A batch holds a dev_get() operation ref on every distinct GPU device it
 * touches for its whole duration, so a concurrent close() waits.
 * CPU-buffer requests use device_id < 0; a request with device_id >= 0 MUST
 * resolve to a registered GPU buffer or it fails — no silent CPU fallthrough,
 * which also closes the closing-device race.
 */
struct batch_ctx {
    struct phxfs_io_op_req *ops;      /* compact resolved requests */
    int                    *map;      /* orig index per compact op */
    phxfs_p2p_map_t        **nodes;   /* mapping ref per op (NULL for CPU) */
    int                    *devs;     /* device per op (-1 for CPU); used to
                                         release without trusting user data
                                         after submit */
    int                     cnt;      /* resolved op count */
    int                     resolve_fail;
    bool                    dev_held[PHXFS_MAX_DEVICES]; /* dev_get()'d */
};

static void batch_ctx_free(struct batch_ctx *bc) {
    free(bc->ops);   bc->ops = NULL;
    free(bc->map);   bc->map = NULL;
    free(bc->nodes); bc->nodes = NULL;
    free(bc->devs);  bc->devs = NULL;
}

/* Release mapping refs, then per-device operation refs. Order matters:
 * mapping refs must drop before dev_put lets a draining close() proceed. */
static void batch_ctx_release(struct batch_ctx *bc) {
    /* One lock acquisition per device for the whole batch (nodes of CPU
     * requests are NULL and skipped). */
    batch_release_mappings(bc->devs, bc->nodes, bc->cnt);
    for (int d = 0; d < g_device_count; d++)
        if (bc->dev_held[d])
            dev_put(&mbuffer[d]);
}

/*
 * Resolve all requests into a compact op array (unresolved requests get
 * result = -EFAULT and are dropped, never handed to the engine). Every
 * distinct GPU device referenced is dev_get()'d up front and tracked in
 * bc->dev_held, so a concurrent close() waits until batch completion.
 * Returns 0, or -ENOMEM.
 */
static int phxfs_batch_prepare(phxfs_io_req_t *reqs, int n, struct batch_ctx *bc) {
    memset(bc, 0, sizeof(*bc));
    bc->ops   = (struct phxfs_io_op_req *)malloc((size_t)n * sizeof(*bc->ops));
    bc->map   = (int *)malloc((size_t)n * sizeof(*bc->map));
    bc->nodes = (phxfs_p2p_map_t **)malloc((size_t)n * sizeof(*bc->nodes));
    bc->devs  = (int *)malloc((size_t)n * sizeof(*bc->devs));
    void **hosts = (void **)malloc((size_t)n * sizeof(*hosts));
    if (!bc->ops || !bc->map || !bc->nodes || !bc->devs || !hosts) {
        free(hosts);
        batch_ctx_free(bc);
        return -ENOMEM;
    }

    /* Hold an operation ref on every distinct GPU device referenced. */
    for (int i = 0; i < n; i++) {
        int d = reqs[i].device_id;
        if (d >= 0 && d < g_device_count && !bc->dev_held[d])
            if (dev_get(d))
                bc->dev_held[d] = true;
    }

    /* Resolve every GPU request against its device's registration table in a
     * single pass: one mapping-lock acquisition per device for the whole
     * batch instead of one lock/unlock pair per request. Skipped requests
     * (CPU buffers, invalid ids/offsets, un-held devices, unregistered
     * buffers) keep hosts[i] == NULL and fail below. */
    batch_resolve_registered(reqs, n, bc->dev_held, hosts, bc->nodes);

    int cnt = 0, fail = 0;
    for (int i = 0; i < n; i++) {
        int d = reqs[i].device_id;
        void *host = hosts[i];
        phxfs_p2p_map_t *node = bc->nodes[i];

        /* Reject a negative file offset and an f_offset+nbytes that would
         * overflow off_t before doing any address resolution. */
        bool valid = reqs[i].f_offset >= 0 &&
                     (uint64_t)reqs[i].nbytes <=
                         (uint64_t)INT64_MAX - (uint64_t)reqs[i].f_offset;

        if (!valid) {
            host = NULL;   /* batch_resolve skipped it: node is NULL */
        } else if (d < 0) {
            /* CPU buffer (caller-owned host memory). */
            resolve_cpu_buf(reqs[i].buf, reqs[i].buf_offset, reqs[i].nbytes, &host);
            node = NULL;
        }
        /* else GPU buffer: host/node already resolved above. */

        if (!host) {
            reqs[i].result = -EFAULT;
            fail++;
            continue;
        }
        bc->ops[cnt].fd = reqs[i].fd;
        bc->ops[cnt].host_addr = host;
        bc->ops[cnt].nbytes = reqs[i].nbytes;
        bc->ops[cnt].f_offset = reqs[i].f_offset;
        bc->ops[cnt].result = -EFAULT;
        bc->map[cnt] = i;
        bc->nodes[cnt] = node;
        bc->devs[cnt] = d;
        cnt++;
    }
    free(hosts);
    bc->cnt = cnt;
    bc->resolve_fail = fail;
    return 0;
}

static int phxfs_batch(phxfs_io_req_t *reqs, int n, enum phxfs_io_op op) {
    if (n <= 0)
        return 0;

    PHX_RANGE(op == PHXFS_IO_WRITE ? "phx.batch.write" : "phx.batch.read");

    /* Staging mode: the whole batch goes through the staging pool + D2D. The
     * mode is global, so a single staging device implies all are staging. */
    if (batch_uses_staging(reqs, n))
        return phx_staging_batch(reqs, n, op == PHXFS_IO_WRITE);

    struct batch_ctx bc;
    if (phxfs_batch_prepare(reqs, n, &bc) < 0)
        return -ENOMEM;

    int ret = 0;
    if (bc.cnt > 0) {
        phx_range_push("phx.io.pool_run");
        ret = phxfs_pool_run(bc.ops, bc.cnt, op);
        phx_range_pop();
        for (int k = 0; k < bc.cnt; k++)
            reqs[bc.map[k]].result = bc.ops[k].result;
    }

    int resolve_fail = bc.resolve_fail;
    batch_ctx_release(&bc);
    batch_ctx_free(&bc);

    if (ret < 0)               /* engine-level error dominates */
        return ret;
    return ret + resolve_fail; /* per-request failures + unresolved */
}

int phxfs_read_batch(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch(reqs, n, PHXFS_IO_READ);
}

int phxfs_write_batch(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch(reqs, n, PHXFS_IO_WRITE);
}

/* ------------------------------------------------------------------ */
/* Async batch (compute/I/O overlap)                                  */
/* ------------------------------------------------------------------ */
struct phxfs_batch {
    struct phxfs_pool_async *pool_h;   /* internal pool handle (may be NULL) */
    struct batch_ctx         bc;       /* resolved ops + held device/mapping refs */
    phxfs_io_req_t          *reqs;     /* user array, for result copy-back */
    bool                     joined;   /* wait()/destroy() already consumed this handle */
    bool                     staging;  /* staging-mode batch: run at wait() time */
    int                      staging_n;
    int                      staging_is_write;
};

static phxfs_batch_t *phxfs_batch_submit(phxfs_io_req_t *reqs, int n,
                                         enum phxfs_io_op op) {
    PHX_RANGE(op == PHXFS_IO_WRITE ? "phx.batch.submit.write"
                : "phx.batch.submit.read");
    phxfs_batch_t *h = (phxfs_batch_t *)calloc(1, sizeof(*h));
    if (!h)
        return NULL;
    h->reqs = reqs;
    h->joined = false;

    if (n <= 0)
        return h;   /* empty batch: wait() returns 0, matching sync semantics */

    /* Staging mode: defer the whole transfer (SSD->staging->D2D->user) to
     * wait(). The internal read pipeline still overlaps NVMe DMA with the D2D
     * copy; cross-call compute overlap is not offered on this path. */
    if (batch_uses_staging(reqs, n)) {
        h->staging = true;
        h->staging_n = n;
        h->staging_is_write = (op == PHXFS_IO_WRITE);
        return h;
    }

    if (phxfs_batch_prepare(reqs, n, &h->bc) < 0) {
        free(h);
        return NULL;
    }

    if (h->bc.cnt > 0) {
        /* Non-blocking: NULL means the pool queue is full (EBUSY) or OOM.
         * Release refs and fail. */
        h->pool_h = phxfs_pool_submit(h->bc.ops, h->bc.cnt, op, /*blocking=*/false);
        if (!h->pool_h) {
            batch_ctx_release(&h->bc);
            batch_ctx_free(&h->bc);
            free(h);
            return NULL;
        }
    }
    return h;
}

phxfs_batch_t *phxfs_batch_submit_read(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch_submit(reqs, n, PHXFS_IO_READ);
}

phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch_submit(reqs, n, PHXFS_IO_WRITE);
}

int phxfs_batch_wait(phxfs_batch_t *h) {
    if (!h)
        return -EINVAL;
    if (h->joined) {
        fprintf(stderr, "%s: handle already waited/destroyed\n", __func__);
        return -EINVAL;
    }
    h->joined = true;

    PHX_RANGE("phx.batch.wait");

    if (h->staging) {
        int fails = phx_staging_batch(h->reqs, h->staging_n, h->staging_is_write);
        free(h);
        return fails;
    }

    int ret = 0;
    if (h->pool_h) {
        phx_range_push("phx.io.pool_wait");
        ret = phxfs_pool_wait(h->pool_h);
        phx_range_pop();
    }

    for (int k = 0; k < h->bc.cnt; k++)
        h->reqs[h->bc.map[k]].result = h->bc.ops[k].result;

    int resolve_fail = h->bc.resolve_fail;
    batch_ctx_release(&h->bc);
    batch_ctx_free(&h->bc);
    int rc = (ret < 0) ? ret : ret + resolve_fail;
    free(h);
    return rc;
}

/*
 * Abandon an async batch: wait for in-flight I/O to quiesce (submitted I/O
 * cannot be cancelled), then release the held device/mapping refs and free
 * the handle WITHOUT copying results back. Gives a caller that drops a batch
 * a defined way to release the pool and all references instead of leaking
 * them until process exit. Returns 0, or -EINVAL for NULL/already-joined.
 */
int phxfs_batch_destroy(phxfs_batch_t *h) {
    if (!h)
        return -EINVAL;
    if (h->joined) {
        fprintf(stderr, "%s: handle already waited/destroyed\n", __func__);
        return -EINVAL;
    }
    h->joined = true;

    if (h->staging) {
        /* Nothing was submitted at submit() time; just drop the handle. */
        free(h);
        return 0;
    }

    if (h->pool_h)
        (void)phxfs_pool_wait(h->pool_h);
    batch_ctx_release(&h->bc);
    batch_ctx_free(&h->bc);
    free(h);
    return 0;
}
