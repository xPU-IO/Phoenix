#ifndef __PHX_INTERNAL_H__
#define __PHX_INTERNAL_H__

/*
 * Internal header shared by phx_device.cpp / phx_mem.cpp / phx_io.cpp /
 * phx_stream.cpp. NOT part of the public API — do not include from
 * outside libphoenix.
 */

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include "common.h"   /* u64 */
#include "connectors/devconnector.h"

/* ---- Profiler ranges----
 * Thin wrappers over the connector's optional rangeops (NVTX on NVIDIA), so
 * core files stay vendor-agnostic. A vendor without a profiler API leaves the
 * ops NULL and these become a single NULL test.
 */
static inline void phx_range_push(const char *name) {
    if (devconn && devconn->range_push)
        devconn->range_push(name);
}

static inline void phx_range_pop(void) {
    if (devconn && devconn->range_pop)
        devconn->range_pop();
}

#ifdef __cplusplus
/* Scope guard for whole-function/scope ranges: PHX_RANGE("phx.batch.read"); */
class PhxRange {
public:
    explicit PhxRange(const char *name) { phx_range_push(name); }
    ~PhxRange() { phx_range_pop(); }
    PhxRange(const PhxRange &) = delete;
    PhxRange &operator=(const PhxRange &) = delete;
};
#define PHX_RANGE_CAT2(a, b) a##b
#define PHX_RANGE_CAT(a, b)  PHX_RANGE_CAT2(a, b)
#define PHX_RANGE(name)      PhxRange PHX_RANGE_CAT(phx_range_scope_, __LINE__)(name)
#endif

#define HUGE_PAGE_SIZE   (64 * 1024)   /* NVIDIA GPU page size */
#define PHXFS_MAX_DEVICES 8

/* BAR mapping mode, read from the kernel (sysfs map_mode) at device open.
 * Mirrors PHXFS_MAP_MODE_* in the kernel module's phxfs.h. */
#define PHX_MAP_MODE_FULL     0   /* direct SSD -> user GPU DMA */
#define PHX_MAP_MODE_STAGING  1   /* SSD -> Phoenix staging pool -> D2D -> user */

/* ---- Registration node (one per phxfs_regmem call) ---- */
typedef struct phxfs_mmap_node_s {
    void *vaddr;        /* single contiguous host-mapped P2P address */
    uint64_t dev_addr;  /* registered GPU/device address (lookup key) */
    size_t length;      /* registered length */
    bool has_reg;       /* kernel P2P mapping active (ioctl MAP done, not UNMAP) */
    bool mapped;        /* host VMA active (mmap done, munmap not yet) */
    int  user_refs;     /* outstanding regmem() references (dup reuse) */
    int  refcount;      /* in-flight resolutions/I/O referencing this node */
    struct phxfs_mmap_node_s *next;
} phxfs_p2p_map_t;

/* ---- Per-device state ---- */
typedef struct phxfs_mmap_buffer_s {
    int device_id;
    int bdev_fd;
    bool init_stat;     /* device is OPEN (fd valid, usable) */
    bool closing;       /* close in progress: reject new dev_get() */
    int  open_count;    /* client open refcount */
    int  active_ops;    /* in-flight device operations */
    struct phxfs_mmap_node_s *head;
    struct phxfs_mmap_node_s *last_hit;  /* MRU lookup cache */
    pthread_mutex_t lock;
    pthread_cond_t  drain_cv;   /* mapping refcount==0 or active_ops==0 */

    /* ---- Staging mode (PHX_MAP_MODE_STAGING) ---- */
    int    map_mode;        /* PHX_MAP_MODE_* (read from kernel at open) */
    void  *staging_raw;     /* raw device allocation backing the pool (freed) */
    void  *staging_dptr;    /* Phoenix-owned device staging pool (or NULL) */
    void  *staging_host;    /* host-mapped P2P vaddr of the staging pool */
    size_t staging_size;    /* staging pool byte size */
} phxfs_mmap_buffer_t;

extern int g_device_count;
extern phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];

/* ---- phx_device.cpp ---- */
phxfs_mmap_buffer_t *dev_get(int device_id);
void dev_put(phxfs_mmap_buffer_t *pb);

/* ---- phx_io.cpp ---- */
/* Shared pread/pwrite loop (chunks at PHXFS_IO_CHUNK, retries EINTR,
 * keeps partial progress). Used by the single-request path and the
 * stream path (phx_stream.cpp). Pure host I/O — safe inside a
 * cudaLaunchHostFunc callback. */
ssize_t xfer(int fd, void *host, ssize_t nbyte, off_t f_offset,
             bool is_write);
/* Validate + resolve a plain CPU buffer address (buf + buf_offset, with
 * overflow checks). Returns 0 with *host set, or -EINVAL. */
int resolve_cpu_buf(const void *buf, off_t buf_offset, size_t nbyte,
                    void **host);

/* ---- phx_mem.cpp ---- */
void free_phxfs_p2p_map(phxfs_mmap_buffer_t *buffer);
void map_release(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *m);
int resolve_registered(int device_id, const void *buf, off_t buf_offset,
                       size_t nbyte, void **host, phxfs_p2p_map_t **out_node);
/* Batch variants used by phx_io.cpp's batch path: same per-request semantics
 * as resolve_registered()/map_release(), but take each device's mapping lock
 * once per device for the whole batch instead of once per request.
 * dev_held: caller's per-device dev_get() success array (PHXFS_MAX_DEVICES).
 * Skipped requests (NULL outputs): CPU buffers, invalid device ids, un-held
 * devices, negative buf_offset, f_offset+nbytes overflow. */
void batch_resolve_registered(const phxfs_io_req_t *reqs, int n,
                              const bool *dev_held, void **hosts,
                              phxfs_p2p_map_t **nodes);
void batch_release_mappings(const int *devs, phxfs_p2p_map_t *const *nodes,
                            int n);
/* Real (non-no-op) registration used by the staging pool. Caller holds a
 * dev_get() reference on pb. Returns 0, or a negative errno. */
int phx_regmem_internal(phxfs_mmap_buffer_t *pb, const void *addr, size_t len,
                        void **target_addr);

/* ---- phx_staging.cpp ---- */
/* Allocate + register the per-device staging pool (staging mode only). */
int  phx_staging_setup(int device_id);
/* Free the staging pool. Called during close teardown AFTER the staging
 * registration has been torn down; pb ref is not required. */
void phx_staging_teardown(phxfs_mmap_buffer_t *pb);
/* Run a batch entirely through the staging path (SSD<->staging<->D2D<->user).
 * is_write selects the direction. Returns failed-request count (>=0) or a
 * negative errno, matching phxfs_read_batch/phxfs_write_batch. */
int  phx_staging_batch(phxfs_io_req_t *reqs, int n, int is_write);

#endif /* __PHX_INTERNAL_H__ */
