// SPDX-License-Identifier: Apache-2.0
/* phxfile.c — frozen-ABI shim implementation over libphoenix.
 *
 * libphoenix's phxfs_regmem/phxfs_deregmem still require a phxfs device
 * index, but the frozen phxFile* surface is device-free: this shim owns
 * everything the old Python wrapper (_phx_async.py) used to juggle.
 *
 * Device resolution is PROBE-BASED — mirroring how libphoenix's own
 * stream path resolves buffers at IO time (phx_stream.cpp): on first use
 * the shim opens every FULL-mode phxfs device, and a registration is
 * attempted on each until one accepts it. The kernel P2P mapping only
 * succeeds on the GPU whose BAR covers the buffer, and libphoenix rolls
 * failed attempts back cleanly (munmap + node free), so probing is
 * side-effect-free. Registration is a low-frequency control operation,
 * so the probe loop's cost is irrelevant. On staging-mode devices regmem
 * pins nothing (it accepts any buffer); the stream path then resolves the
 * owning GPU from the pointer itself, so the recorded device is unused.
 *
 * Additional shim-side state: the registration bookkeeping (base addr ->
 * aligned length + device, needed to play back the aligned length on
 * deregister) and the device page-size cache.
 *
 * When libphoenix drops the device parameter, phxFileBufRegister switches
 * from probe-based to a direct forward — an internal change behind the
 * same frozen signature. No consumer change either way.
 */

#include "phxfile.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <phoenix.h> /* libphoenix public C API (/usr/local/include) */

/* Probe scan ceiling — NOT the device count. The real set of phxfs
 * devices is discovered via /dev/phxfs_devN existence at runtime, so a
 * libphoenix built for more devices than today's 8 needs no shim change.
 * The only hard requirement is that this bound be >= libphoenix's
 * PHXFS_MAX_DEVICES: a device index beyond phxfs_open()'s internal
 * bounds is rejected there anyway, and an absent node is skipped by the
 * access() check, so over-scanning is always safe. */
#define PHXFILE_MAX_DEVS 64

/* ------------------------------------------------------------------ *
 * Internal state (all guarded by g_lock)
 * ------------------------------------------------------------------ */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static int g_opened[PHXFILE_MAX_DEVS]; /* 1 = opened by this shim */
static int g_nopened;
static int g_probed;   /* one-time enumeration done */
static int g_probe_rc; /* cached enumeration result: 0 / -ENODEV / -EOPNOTSUPP */

/* Registration table: sorted by addr, duplicates allowed (libphoenix
 * reference-counts exact duplicates; every successful registration is
 * recorded so register/deregister stay symmetric). */
struct reg_entry {
    uintptr_t addr;
    size_t len; /* aligned length actually passed to phxfs_regmem */
    int dev;    /* phxfs device index that accepted the registration */
};
static struct reg_entry *g_regs;
static size_t g_nregs;
static size_t g_regcap;

static uint64_t g_page_size; /* 0 = not yet queried */

/* ------------------------------------------------------------------ *
 * Helpers (caller holds g_lock)
 * ------------------------------------------------------------------ */

/* Cached device page size. phxfs_get_page_size() cannot fail (a 0 return
 * is treated as -EIO by the caller). */
static uint64_t page_size_locked(void)
{
    if (g_page_size == 0)
        g_page_size = phxfs_get_page_size();
    return g_page_size;
}

/* First index i with g_regs[i].addr >= key (lower bound). */
static size_t reg_lower_bound(uintptr_t key)
{
    size_t lo = 0, hi = g_nregs;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (g_regs[mid].addr < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

static int reg_insert(uintptr_t addr, size_t len, int dev)
{
    if (g_nregs == g_regcap) {
        size_t cap = g_regcap ? g_regcap * 2 : 16;
        struct reg_entry *p = realloc(g_regs, cap * sizeof(*p));
        if (!p)
            return -ENOMEM;
        g_regs = p;
        g_regcap = cap;
    }
    size_t i = reg_lower_bound(addr);
    memmove(&g_regs[i + 1], &g_regs[i], (g_nregs - i) * sizeof(*g_regs));
    g_regs[i].addr = addr;
    g_regs[i].len = len;
    g_regs[i].dev = dev;
    g_nregs++;
    return 0;
}

static void reg_remove(size_t i)
{
    memmove(&g_regs[i], &g_regs[i + 1],
            (g_nregs - i - 1) * sizeof(*g_regs));
    g_nregs--;
}

/* ------------------------------------------------------------------ *
 * Device discovery (probe)
 * ------------------------------------------------------------------ */

/*
 * One-time discovery: probe accelerator ordinals through the unified node.
 * libphoenix resolves each ordinal to its sysfs GPU index for mmap/ioctl.
 * Returns 0, -ENODEV (node absent) or -EOPNOTSUPP (no GPU could be opened).
 * Caller holds g_lock.
 */
static int probe_devices_locked(void)
{
    if (g_probed)
        return g_probe_rc;

    if (access("/dev/phxfs_dev", F_OK) != 0) {
        g_probe_rc = -ENODEV;
        g_probed = 1;
        return g_probe_rc;
    }
    for (int d = 0; d < PHXFILE_MAX_DEVS; d++) {
        if (phxfs_open(d) == 0) {
            g_opened[d] = 1;
            g_nopened++;
        }
    }

    if (g_nopened == 0)
        g_probe_rc = -EOPNOTSUPP;
    /* else g_probe_rc stays 0 */
    g_probed = 1;
    return g_probe_rc;
}

/* ------------------------------------------------------------------ *
 * Driver lifecycle
 * ------------------------------------------------------------------ */

int phxFileDriverOpen(void)
{
    /* Lazy model: devices are discovered on first buffer registration.
     * Kept as a frozen no-op for hipFile-surface parity. */
    return 0;
}

int phxFileDriverClose(void)
{
    pthread_mutex_lock(&g_lock);

    for (size_t i = 0; i < g_nregs; i++) {
        int rc = phxfs_deregmem(g_regs[i].dev,
                                (const void *)g_regs[i].addr,
                                g_regs[i].len);
        if (rc < 0) {
            fprintf(stderr,
                    "phxfile: phxfs_deregmem(0x%lx) failed with %d\n",
                    (unsigned long)g_regs[i].addr, rc);
        }
    }
    g_nregs = 0;

    for (int d = 0; d < PHXFILE_MAX_DEVS; d++) {
        if (!g_opened[d])
            continue;
        int rc = phxfs_close(d);
        if (rc < 0) {
            fprintf(stderr, "phxfile: phxfs_close(%d) failed with %d\n",
                    d, rc);
        }
        g_opened[d] = 0;
    }
    g_nopened = 0;
    g_probed = 0;
    g_probe_rc = 0;
    g_page_size = 0;

    pthread_mutex_unlock(&g_lock);
    return 0;
}

/* ------------------------------------------------------------------ *
 * Buffer registration
 * ------------------------------------------------------------------ */

int phxFileBufRegister(const void *addr, size_t length)
{
    if (!addr || length == 0)
        return -EINVAL;

    pthread_mutex_lock(&g_lock);

    int rc = probe_devices_locked();
    if (rc != 0)
        goto out;

    uint64_t ps = page_size_locked();
    if (ps == 0) {
        rc = -EIO;
        goto out;
    }

    /* Round up without assuming the page size is a power of two. */
    size_t aligned = (size_t)(((length + ps - 1) / ps) * ps);

    /* Probe loop: register on the first device whose BAR covers the
     * buffer. A failed attempt rolls back cleanly inside libphoenix
     * (mmap/ioctl unwind), so trying the next device is safe. The loop
     * runs only on this low-frequency control path, never on IO. */
    int last = -ENODEV;
    for (int d = 0; d < PHXFILE_MAX_DEVS; d++) {
        if (!g_opened[d])
            continue;
        void *target = NULL;
        int r = phxfs_regmem(d, addr, aligned, &target);
        if (r == 0) {
            /* target_addr is an internal host-mapped handle; the IO path
             * keys on the ORIGINAL device address (phoenix.h), so it is
             * dropped here and never exposed through the frozen surface.
             * Edge case: a bookkeeping failure after a successful
             * registration leaks that registration until DriverClose. */
            rc = reg_insert((uintptr_t)addr, aligned, d);
            goto out;
        }
        last = r;
    }
    rc = last;

out:
    pthread_mutex_unlock(&g_lock);
    return rc;
}

int phxFileBufDeregister(const void *addr)
{
    if (!addr)
        return -EINVAL;

    pthread_mutex_lock(&g_lock);
    int rc = 0;
    size_t i = reg_lower_bound((uintptr_t)addr);
    if (i < g_nregs && g_regs[i].addr == (uintptr_t)addr) {
        rc = phxfs_deregmem(g_regs[i].dev, addr, g_regs[i].len);
        if (rc == 0)
            reg_remove(i);
        /* On failure the entry is kept so a retry can re-attempt. */
    }
    /* Unregistered addresses are silently tolerated. */
    pthread_mutex_unlock(&g_lock);
    return rc;
}

/* ------------------------------------------------------------------ *
 * File handle
 * ------------------------------------------------------------------ */

int phxFileHandleRegister(void **fh, int fd)
{
    if (!fh || fd < 0)
        return -EINVAL;
    /* Identity boxing: phxfs performs IO on plain POSIX fds. */
    *fh = (void *)(intptr_t)fd;
    return 0;
}

int phxFileHandleDeregister(void *fh)
{
    (void)fh; /* nothing to release behind an identity boxing */
    return 0;
}

/* ------------------------------------------------------------------ *
 * Stream registration
 * ------------------------------------------------------------------ */

int phxFileStreamRegister(void *stream)
{
    (void)stream; /* no per-stream state today; frozen for parity */
    return 0;
}

int phxFileStreamDeregister(void *stream)
{
    (void)stream;
    return 0;
}

/* ------------------------------------------------------------------ *
 * Stream-ordered IO submission
 * ------------------------------------------------------------------ */

/* libphoenix's stream API takes (fd, buf, *nbytes, *buf_offset, *f_offset,
 * *bytes_done, stream) — buf_offset BEFORE f_offset — while the frozen ABI
 * is hipFile-ordered (f_offset before buf_offset), so the two offset
 * pointers swap on the way through. off_t / ssize_t are 64-bit on LP64 and
 * binary-layout compatible with int64_t; the pointers must be forwarded
 * as-is (never copied) because the library dereferences them lazily when
 * the stream executes the op (late-binding contract). */
int phxFileReadAsync(void *fh, void *buf_base,
                     size_t *nbytes, int64_t *file_offset,
                     int64_t *buf_offset, int64_t *bytes_done,
                     void *stream)
{
    if (!fh || !buf_base || !nbytes || !file_offset || !buf_offset ||
        !bytes_done)
        return -EINVAL;

    return phxfs_read_stream((int)(intptr_t)fh, buf_base, nbytes,
                             (off_t *)buf_offset, (off_t *)file_offset,
                             (ssize_t *)bytes_done, stream);
}

int phxFileWriteAsync(void *fh, void *buf_base,
                      size_t *nbytes, int64_t *file_offset,
                      int64_t *buf_offset, int64_t *bytes_done,
                      void *stream)
{
    if (!fh || !buf_base || !nbytes || !file_offset || !buf_offset ||
        !bytes_done)
        return -EINVAL;

    return phxfs_write_stream((int)(intptr_t)fh, buf_base, nbytes,
                              (off_t *)buf_offset, (off_t *)file_offset,
                              (ssize_t *)bytes_done, stream);
}
