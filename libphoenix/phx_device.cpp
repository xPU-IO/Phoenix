/*
 * phx_device.cpp — device lifecycle: open / close / operation refcounting.
 */

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <cstdlib>

#include "phoenix.h"
#include "phx_internal.h"
#include "connectors/devconnector.h"

int g_device_count = PHXFS_MAX_DEVICES;
phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];

static std::vector<std::string> phxfs_dev_path = {
    "/dev/phxfs_dev0", "/dev/phxfs_dev1",
    "/dev/phxfs_dev2", "/dev/phxfs_dev3",
    "/dev/phxfs_dev4", "/dev/phxfs_dev5",
    "/dev/phxfs_dev6", "/dev/phxfs_dev7"
};

static std::vector<bool> phxfs_initialized(PHXFS_MAX_DEVICES, false);

bool phx_host_mode_requested(void) {
    const char *mode = std::getenv("PHXFS_MODE");
    return mode && (strcmp(mode, "host") == 0 ||
                    strcmp(mode, "host_staging") == 0 ||
                    strcmp(mode, "auto") == 0);
}

bool phx_host_mode_auto(void) {
    const char *mode = std::getenv("PHXFS_MODE");
    return mode && strcmp(mode, "auto") == 0;
}

/*
 * Read the kernel BAR mapping mode for a phxfs device from sysfs. Absent node
 * (older module without the attribute) defaults to FULL — the historical
 * direct-DMA behaviour.
 */
static int phxfs_read_map_mode(int device_id) {
    std::string path = "/sys/class/phxfs-generic/phxfs_dev"
                       + std::to_string(device_id) + "/map_mode";
    std::ifstream ifs(path);
    int mode = PHX_MAP_MODE_FULL;
    if (ifs.is_open()) {
        int m;
        if (ifs >> m)
            mode = m;
    }
    return mode;
}

int phx_probe_direct_device(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return -EINVAL;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    if (!pb->init_stat || pb->bdev_fd < 0)
        return -ENODEV;
    if (pb->phxfs_device_id < 0)
        return -ENODEV;
    return phx_direct_probe(device_id);
}

/*
 * Serialises open()/close() across devices so concurrent phxfs_open() from
 * multiple threads (e.g. one worker per GPU) and open-vs-close are safe.
 * reg/dereg/batch synchronise per-device on mbuffer[].lock instead.
 */
static pthread_mutex_t g_open_lock = PTHREAD_MUTEX_INITIALIZER;
/* Signalled (under g_open_lock) when a device's `opening` phase ends. */
static pthread_cond_t g_open_cv = PTHREAD_COND_INITIALIZER;

/* Initialise per-device sync primitives + state once, before any open. */
__attribute__((constructor))
static void phxfs_dev_ctor(void) {
    for (int i = 0; i < PHXFS_MAX_DEVICES; i++) {
        pthread_mutex_init(&mbuffer[i].lock, NULL);
        pthread_cond_init(&mbuffer[i].drain_cv, NULL);
        mbuffer[i].init_stat  = false;
        mbuffer[i].phxfs_device_id = -1;
        mbuffer[i].closing    = false;
        mbuffer[i].opening    = false;
        mbuffer[i].open_count = 0;
        mbuffer[i].active_ops = 0;
        mbuffer[i].head       = NULL;
        mbuffer[i].last_hit   = NULL;
        mbuffer[i].map_mode     = PHX_MAP_MODE_FULL;
        mbuffer[i].staging_raw  = NULL;
        mbuffer[i].staging_dptr = NULL;
        mbuffer[i].staging_host = NULL;
        mbuffer[i].staging_size = 0;
        pthread_mutex_init(&mbuffer[i].staging_lock, NULL);
        pthread_cond_init(&mbuffer[i].staging_cv, NULL);
        pthread_mutex_init(&mbuffer[i].staging_submit_lock, NULL);
        mbuffer[i].staging_owner = NULL;
        mbuffer[i].staging_users = 0;
        for (int s = 0; s < PHX_HOST_STAGING_SLOTS; s++) {
            mbuffer[i].host_staging[s] = NULL;
            mbuffer[i].host_staging_busy[s] = false;
        }
        mbuffer[i].host_staging_size = 0;
        pthread_cond_init(&mbuffer[i].host_staging_cv, NULL);
    }
    /* Honour the connector init contract once at load. */
    devconn_init();
}

/*
 * Acquire a device-operation reference. Returns the device buffer if it is
 * OPEN and not closing, else NULL. Holding a ref keeps close() waiting, so
 * an in-flight regmem/dereg/read/write/batch can never race teardown.
 */
phxfs_mmap_buffer_t *dev_get(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return NULL;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    pthread_mutex_lock(&pb->lock);
    if (!pb->init_stat || pb->closing) {
        pthread_mutex_unlock(&pb->lock);
        return NULL;
    }
    pb->active_ops++;
    pthread_mutex_unlock(&pb->lock);
    return pb;
}

/* Release a dev_get() reference; wakes a draining close() at zero. */
void dev_put(phxfs_mmap_buffer_t *pb) {
    if (!pb)
        return;
    pthread_mutex_lock(&pb->lock);
    if (--pb->active_ops == 0 && pb->closing)
        pthread_cond_broadcast(&pb->drain_cv);
    pthread_mutex_unlock(&pb->lock);
}

/*
 * Tear down an OPEN device after all in-flight operations have drained.
 * Caller must have set closing=true and waited active_ops==0 under pb->lock.
 * The process-lifetime lock/drain_cv are NOT destroyed here.
 */
static void __phxfs_teardown(phxfs_mmap_buffer_t *pb) {
    if (pb->map_mode == PHX_MAP_MODE_HOST) {
        pthread_mutex_lock(&pb->lock);
        phxfs_p2p_map_t *m = pb->head;
        pb->head = NULL;
        pb->last_hit = NULL;
        pthread_mutex_unlock(&pb->lock);
        while (m) {
            phxfs_p2p_map_t *next = m->next;
            free(m);
            m = next;
        }
    } else {
        free_phxfs_p2p_map(pb);          /* UNMAP + munmap each registration */
    }
    phx_staging_teardown(pb);            /* free the staging pool (staging mode) */
    phx_host_staging_teardown(pb);   /* free pinned host fallback pool */
    if (pb->bdev_fd >= 0)                /* fd 0 is a valid fd and must be closed too */
        close(pb->bdev_fd);
    pb->bdev_fd  = -1;
    pb->init_stat = false;
    pb->closing   = false;
}

/*
 * Drain in-flight operations on a device whose closing=true and init_stat=false
 * have already been set by the caller, then tear down. Does NOT hold
 * g_open_lock, so a drain on one device does not block open/close on another.
 */
static void __phxfs_close_drain(phxfs_mmap_buffer_t *pb) {
    pthread_mutex_lock(&pb->lock);
    while (pb->active_ops > 0)
        pthread_cond_wait(&pb->drain_cv, &pb->lock);
    pthread_mutex_unlock(&pb->lock);
    __phxfs_teardown(pb);
}

/*
 * Drop the last client reference of an OPEN device: flip it to closing under
 * g_open_lock (which the caller holds and this releases), then drain and tear
 * down outside it.
 */
static void phx_close_last_unlock(phxfs_mmap_buffer_t *pb, int device_id) {
    pb->open_count = 0;
    phxfs_initialized[device_id] = false;
    /* Set closing under pb->lock before flipping init_stat, so open()'s
     * EBUSY check is watertight (no window where init_stat==false but
     * closing==false). Lock order g_open_lock -> pb->lock is safe: dev ops
     * never take g_open_lock. */
    pthread_mutex_lock(&pb->lock);
    pb->closing = true;
    pb->init_stat = false;
    pthread_mutex_unlock(&pb->lock);
    pthread_mutex_unlock(&g_open_lock);

    __phxfs_close_drain(pb);  /* drain (wait active_ops==0) + teardown, outside g_open_lock */
}

int phxfs_close(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return -1;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];

    /*
     * g_open_lock protects only the open_count check/decrement and the
     * init_stat/closing flip — NOT the drain. This way a long drain on
     * device A (waiting for in-flight I/O) does not block open/close on
     * device B.
     *
     * Race safety: closing=true is set under pb->lock BEFORE g_open_lock is
     * released, so a concurrent phxfs_open() for this same device that
     * acquires g_open_lock after us sees init_stat==false && closing==true
     * and returns -EBUSY. There is no window where open sees closing==false
     * after close has committed to teardown.
     */
    pthread_mutex_lock(&g_open_lock);
    /* A fresh open of this device is still setting up; let it settle. */
    while (pb->opening)
        pthread_cond_wait(&g_open_cv, &g_open_lock);
    if (!pb->init_stat) {
        pthread_mutex_unlock(&g_open_lock);
        return -1;
    }
    /* Only the last client's close actually tears the device down. */
    if (pb->open_count > 1) {
        pb->open_count--;
        pthread_mutex_unlock(&g_open_lock);
        return 0;
    }
    phx_close_last_unlock(pb, device_id);
    return 0;
}

/* Undo a fresh open whose post-open setup failed. Runs in the opener while
 * pb->opening is set, so it still owns the one and only client reference. */
static void phx_open_abort(phxfs_mmap_buffer_t *pb, int device_id) {
    pthread_mutex_lock(&g_open_lock);
    if (pb->init_stat)
        phx_close_last_unlock(pb, device_id);
    else
        pthread_mutex_unlock(&g_open_lock);
}

/* Open the char device. lock/drain_cv are process-lifetime (see ctor). */
static int __phxfs_open(const char *dev_path, phxfs_mmap_buffer_t *mbuffer,
                        int device_id, int phxfs_device_id) {
    mbuffer->bdev_fd = open(dev_path, O_RDWR);

    if (mbuffer->bdev_fd == -1) {
        printf("failed to open file %s\n", dev_path);
        return -1;
    }
    mbuffer->head = NULL;
    mbuffer->device_id = device_id;
    mbuffer->phxfs_device_id = phxfs_device_id;
    mbuffer->closing = false;
    mbuffer->active_ops = 0;
    mbuffer->map_mode = phxfs_read_map_mode(phxfs_device_id);
    mbuffer->staging_raw = NULL;
    mbuffer->staging_dptr = NULL;
    mbuffer->staging_host = NULL;
    mbuffer->staging_size = 0;
    mbuffer->init_stat = true;
    return 0;
}

static int phx_open_finish(int deviceID, phxfs_mmap_buffer_t *pb);

/*
 * Open a phxfs device, or add a client reference if another caller already
 * has it open. g_open_lock makes concurrent phxfs_open() from multiple
 * threads safe.
 */
int phxfs_open(int deviceID) {
    if (deviceID < 0 || deviceID >= g_device_count)
        return -1;
    if (phx_host_mode_requested()) {
        if (!devconn || !devconn->validate_device)
            return -EOPNOTSUPP;
        int rc = devconn->validate_device(deviceID);
        if (rc != 0)
            return rc;
    }
    phxfs_mmap_buffer_t *pb = &mbuffer[deviceID];

    PHX_RANGE("phx.open");

    pthread_mutex_lock(&g_open_lock);
    /* Another thread is finishing a fresh open of this device: wait for its
     * outcome rather than piggy-backing on a half-initialised device. */
    while (pb->opening)
        pthread_cond_wait(&g_open_cv, &g_open_lock);
    int ret = 0;
    bool fresh_open = false;
    if (pb->init_stat) {
        pb->open_count++;
    } else if (pb->closing) {
        ret = -EBUSY;
    } else {
        const bool force_host = phx_host_mode_requested() && !phx_host_mode_auto();
        int phxfs_device_id = -1;
        if (!force_host && devconn && devconn->find_device)
            phxfs_device_id = devconn->find_device(deviceID);

        if (phxfs_device_id >= 0 &&
            phxfs_device_id < (int)phxfs_dev_path.size()) {
            ret = __phxfs_open(phxfs_dev_path[phxfs_device_id].c_str(), pb,
                               deviceID, phxfs_device_id);
            if (ret == 0) {
                pb->open_count = 1;
                phxfs_initialized[deviceID] = true;
                fresh_open = true;
            }
        } else if (phx_host_mode_requested()) {
            pb->bdev_fd = -1;
            pb->device_id = deviceID;
            pb->phxfs_device_id = -1;
            pb->closing = false;
            pb->active_ops = 0;
            pb->map_mode = PHX_MAP_MODE_HOST;
            pb->staging_raw = NULL;
            pb->staging_dptr = NULL;
            pb->staging_host = NULL;
            pb->staging_size = 0;
            pb->init_stat = true;
            pb->open_count = 1;
            phxfs_initialized[deviceID] = true;
            fresh_open = true;
            fprintf(stderr,
                    "phxfs_open: CUDA device %d using host staging mode (PHXFS_MODE=%s)\n",
                    deviceID, std::getenv("PHXFS_MODE"));
        } else {
            ret = -ENODEV;
        }
    }
    /* The post-open steps below run outside g_open_lock (they call into the
     * device runtime), so fence this device off until they are done: no other
     * open() can join it and no close() can tear it down underneath. */
    if (ret == 0 && fresh_open)
        pb->opening = true;
    pthread_mutex_unlock(&g_open_lock);

    if (ret == 0 && fresh_open) {
        ret = phx_open_finish(deviceID, pb);
        pthread_mutex_lock(&g_open_lock);
        pb->opening = false;
        pthread_cond_broadcast(&g_open_cv);
        pthread_mutex_unlock(&g_open_lock);
    }
    return ret;
}

/*
 * Post-open setup of a fresh open: auto-mode direct probe (with host-staging
 * fallback) and the staging pools. Called with pb->opening set; on failure
 * the device is torn down again and a negative errno returned.
 */
static int phx_open_finish(int deviceID, phxfs_mmap_buffer_t *pb) {
    if (phx_host_mode_auto() && pb->map_mode == PHX_MAP_MODE_FULL) {
        int probe = phx_probe_direct_device(deviceID);
        if (probe != 0) {
            fprintf(stderr,
                    "phxfs_open: direct probe failed on CUDA device %d (%d); "
                    "using host staging fallback\n", deviceID, probe);
            phx_open_abort(pb, deviceID);
            pthread_mutex_lock(&g_open_lock);
            pb->bdev_fd = -1;
            pb->device_id = deviceID;
            pb->phxfs_device_id = -1;
            pb->closing = false;
            pb->active_ops = 0;
            pb->map_mode = PHX_MAP_MODE_HOST;
            pb->staging_raw = NULL;
            pb->staging_dptr = NULL;
            pb->staging_host = NULL;
            pb->staging_size = 0;
            pb->init_stat = true;
            pb->open_count = 1;
            phxfs_initialized[deviceID] = true;
            pthread_mutex_unlock(&g_open_lock);
        }
    }

    if (pb->map_mode == PHX_MAP_MODE_HOST) {
        int src = phx_host_staging_setup(deviceID);
        if (src != 0) {
            fprintf(stderr, "phxfs_open: host staging setup failed on CUDA device %d: %d\n",
                    deviceID, src);
            phx_open_abort(pb, deviceID);
            return src;
        }
    }

    if (pb->map_mode == PHX_MAP_MODE_STAGING) {
        int src = phx_staging_setup(deviceID);
        if (src != 0) {
            fprintf(stderr, "phxfs_open: staging setup failed on CUDA device %d: %d\n",
                    deviceID, src);
            phx_open_abort(pb, deviceID);
            return src;
        }
    }

    return 0;
}

int phxfs_find_dev(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return -1;
    if (phx_host_mode_requested())
        return devconn && devconn->validate_device &&
            devconn->validate_device(device_id) == 0 ? device_id : -1;
    if (!devconn || !devconn->find_device ||
        devconn->find_device(device_id) < 0)
        return -1;
    /* Compatibility helper: public device ids are accelerator ordinals. */
    return device_id;
}

uint64_t phxfs_get_page_size(void) {
    return devconn->page_size;
}

int phxfs_get_map_mode(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return -EINVAL;
    if (mbuffer[device_id].init_stat)
        return mbuffer[device_id].map_mode;
    if (phx_host_mode_requested())
        return PHX_MAP_MODE_HOST;
    if (!devconn || !devconn->find_device)
        return -ENODEV;
    int phxfs_device_id = devconn->find_device(device_id);
    return phxfs_device_id < 0 ? -ENODEV : phxfs_read_map_mode(phxfs_device_id);
}
