// SPDX-License-Identifier: Apache-2.0
/* Host-staging fallback pool. */
#include <cerrno>
#include <cstdio>
#include <pthread.h>

#include "phoenix.h"
#include "phx_internal.h"
#include "connectors/devconnector.h"

int phx_host_staging_setup(int device_id)
{
    phxfs_mmap_buffer_t *pb = dev_get(device_id);
    if (!pb)
        return -ENODEV;
    if (!devconn || !devconn->host_alloc || !devconn->host_free ||
        !devconn->memcpy_h2d || !devconn->memcpy_d2h) {
        dev_put(pb);
        return -ENOTSUP;
    }

    pthread_mutex_lock(&pb->lock);
    bool ready = pb->host_staging_size != 0;
    pthread_mutex_unlock(&pb->lock);
    if (ready) {
        dev_put(pb);
        return 0;
    }

    void *slots[PHX_HOST_STAGING_SLOTS] = {};
    for (int i = 0; i < PHX_HOST_STAGING_SLOTS; i++) {
        if (devconn->host_alloc(PHX_HOST_STAGING_BYTES, &slots[i]) != 0) {
            for (int j = 0; j < i; j++)
                devconn->host_free(slots[j]);
            dev_put(pb);
            return -ENOMEM;
        }
    }

    pthread_mutex_lock(&pb->lock);
    for (int i = 0; i < PHX_HOST_STAGING_SLOTS; i++) {
        pb->host_staging[i] = slots[i];
        pb->host_staging_busy[i] = false;
    }
    pb->host_staging_size = PHX_HOST_STAGING_BYTES;
    pthread_mutex_unlock(&pb->lock);
    dev_put(pb);
    return 0;
}

void phx_host_staging_teardown(phxfs_mmap_buffer_t *pb)
{
    void *slots[PHX_HOST_STAGING_SLOTS] = {};
    pthread_mutex_lock(&pb->lock);
    for (int i = 0; i < PHX_HOST_STAGING_SLOTS; i++) {
        slots[i] = pb->host_staging[i];
        pb->host_staging[i] = NULL;
        pb->host_staging_busy[i] = false;
    }
    pb->host_staging_size = 0;
    pthread_cond_broadcast(&pb->host_staging_cv);
    pthread_mutex_unlock(&pb->lock);

    if (devconn && devconn->host_free)
        for (void *slot : slots)
            if (slot)
                devconn->host_free(slot);
}

void *phx_host_staging_acquire(phxfs_mmap_buffer_t *pb, int *slot)
{
    if (!slot)
        return NULL;
    pthread_mutex_lock(&pb->lock);
    for (;;) {
        for (int i = 0; i < PHX_HOST_STAGING_SLOTS; i++) {
            if (pb->host_staging[i] && !pb->host_staging_busy[i]) {
                pb->host_staging_busy[i] = true;
                *slot = i;
                void *ptr = pb->host_staging[i];
                pthread_mutex_unlock(&pb->lock);
                return ptr;
            }
        }
        if (pb->closing || !pb->init_stat || pb->host_staging_size == 0) {
            pthread_mutex_unlock(&pb->lock);
            return NULL;
        }
        pthread_cond_wait(&pb->host_staging_cv, &pb->lock);
    }
}

void phx_host_staging_release(phxfs_mmap_buffer_t *pb, int slot)
{
    if (slot < 0 || slot >= PHX_HOST_STAGING_SLOTS)
        return;
    pthread_mutex_lock(&pb->lock);
    pb->host_staging_busy[slot] = false;
    pthread_cond_signal(&pb->host_staging_cv);
    pthread_mutex_unlock(&pb->lock);
}
