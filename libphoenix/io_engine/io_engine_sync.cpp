// SPDX-License-Identifier: Apache-2.0
/*
 * Synchronous batch engine — always-available fallback.
 *
 * Just loops pread/pwrite over the resolved host addresses, chunking each
 * request under MAX_RW_COUNT (~2GiB). No concurrency, but correct on any
 * kernel and used when io_uring / libaio are unavailable.
 */
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include "io_engine.h"

static int sync_probe(void) {
    return 0;  /* always available */
}

static ssize_t do_one(struct phxfs_io_op_req *r, enum phxfs_io_op op) {
    char *base = (char *)r->host_addr;
    size_t done = 0;
    while (done < r->nbytes) {
        size_t chunk = r->nbytes - done;
        if (chunk > PHXFS_IO_CHUNK)
            chunk = PHXFS_IO_CHUNK;
        ssize_t ret = (op == PHXFS_IO_READ)
            ? pread(r->fd, base + done, chunk, r->f_offset + (off_t)done)
            : pwrite(r->fd, base + done, chunk, r->f_offset + (off_t)done);
        if (ret < 0) {
            if (errno == EINTR)
                continue;                        /* interrupted: retry the same chunk */
            /* Return partial progress if any, else the negative errno, to
             * match phxfs_read/write's unified semantics. */
            return done > 0 ? (ssize_t)done : -errno;
        }
        if (ret == 0)
            break;  /* EOF */
        done += (size_t)ret;
    }
    return (ssize_t)done;
}

static int sync_submit_batch(struct phxfs_io_op_req *reqs, int n,
                             enum phxfs_io_op op) {
    int failed = 0;
    for (int i = 0; i < n; i++) {
        reqs[i].result = do_one(&reqs[i], op);
        if (reqs[i].result != (ssize_t)reqs[i].nbytes)
            failed++;
    }
    return failed;
}

const struct phxfs_io_engine phxfs_io_engine_sync = {
    "sync",
    sync_probe,
    sync_submit_batch,
};
