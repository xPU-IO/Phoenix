/*
 * AMD DevConnector (HIP/ROCm)
 *
 * All AMD/HIP specific code in the user library lives here, mirroring
 * nvidia_connector.cpp. Core libphoenix never includes HIP headers or
 * calls HIP APIs directly.
 *
 * Notes:
 *  - page_size is NOT hard-coded. The kernel backend queries the real GPU
 *    page granularity from the amdgpu driver (KFD get_page_size) and
 *    exposes it via /sys/class/phxfs-generic/phxfs_devN/page_size; we read
 *    it at init time so user space and the kernel always agree.
 *  - launch_host_func maps to hipLaunchHostFunc (ROCm >= 5.1): same
 *    signature and ordering contract as cudaLaunchHostFunc — the callback
 *    runs after all previously enqueued work and blocks everything
 *    enqueued after it.
 *  - Like the callback rule on CUDA, the callback must not call HIP APIs;
 *    the core only runs pure host I/O inside it.
 */

/* Plain g++ (no hipcc) must tell the HIP headers which platform it is
 * compiling for; hipcc would define this itself. */
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <pthread.h>

#include "devconnector.h"

#define AMD_MAX_GPUS 64        /* HIP-GPU cache size */
/* sysfs scan bound: must match the core's PHXFS_MAX_DEVICES, as find_device
 * returns a phxfs index the core can actually open. */
#define PHXFS_DEV_SCAN_MAX 8

/* Fallback when the kernel module is not loaded yet at init time: the
 * KFD get_page_size implementation reports PAGE_SIZE. */
#define AMD_PAGE_SIZE_FALLBACK 4096

/* ------------------------------------------------------------------ */
/* Device discovery: HIP device ID -> phxfs index via PCI BDF match   */
/* ------------------------------------------------------------------ */

static int             g_dev_cache[AMD_MAX_GPUS];
static bool            g_dev_cache_init = false;
static pthread_mutex_t g_dev_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static bool bdf_equal(const char *a, const std::string &b)
{
    size_t i = 0;
    for (; a[i] && i < b.size(); i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
            return false;
    return a[i] == '\0' && i == b.size();
}

static int amd_find_device_uncached(int hip_gpu_id)
{
    char hip_bdf[32];
    if (hipDeviceGetPCIBusId(hip_bdf, sizeof(hip_bdf), hip_gpu_id) != hipSuccess) {
        fprintf(stderr, "amd_find_device: hipDeviceGetPCIBusId failed for GPU %d\n",
                hip_gpu_id);
        return -1;
    }

    for (int i = 0; i < PHXFS_DEV_SCAN_MAX; i++) {
        std::string sysfs_path = "/sys/class/phxfs-generic/phxfs_dev"
                                 + std::to_string(i) + "/pci_bdf";
        std::ifstream ifs(sysfs_path);
        if (!ifs.is_open())
            continue;
        std::string sysfs_bdf;
        if (!std::getline(ifs, sysfs_bdf))
            continue;
        while (!sysfs_bdf.empty() &&
               (sysfs_bdf.back() == '\n' || sysfs_bdf.back() == '\r' ||
                sysfs_bdf.back() == ' '  || sysfs_bdf.back() == '\t'))
            sysfs_bdf.pop_back();
        if (bdf_equal(hip_bdf, sysfs_bdf))
            return i;
    }
    return -1;
}

static int amd_find_device(int hip_gpu_id)
{
    if (hip_gpu_id < 0 || hip_gpu_id >= AMD_MAX_GPUS)
        return amd_find_device_uncached(hip_gpu_id);

    pthread_mutex_lock(&g_dev_cache_lock);
    if (!g_dev_cache_init) {
        for (int i = 0; i < AMD_MAX_GPUS; i++)
            g_dev_cache[i] = -2;   /* unresolved */
        g_dev_cache_init = true;
    }
    int cached = g_dev_cache[hip_gpu_id];
    pthread_mutex_unlock(&g_dev_cache_lock);
    if (cached >= 0)
        return cached;             /* only successful lookups are cached */

    int idx = amd_find_device_uncached(hip_gpu_id);
    if (idx >= 0) {
        pthread_mutex_lock(&g_dev_cache_lock);
        g_dev_cache[hip_gpu_id] = idx;
        pthread_mutex_unlock(&g_dev_cache_lock);
    }
    return idx;
}

/* ------------------------------------------------------------------ */
/* Staging-mode device memory operations                              */
/* ------------------------------------------------------------------ */

/* Reverse of find_device: phxfs index -> HIP device id. */
static int amd_phxfs_to_hip(int phxfs_dev)
{
    std::string sysfs_path = "/sys/class/phxfs-generic/phxfs_dev"
                             + std::to_string(phxfs_dev) + "/pci_bdf";
    std::ifstream ifs(sysfs_path);
    if (!ifs.is_open())
        return -1;
    std::string sysfs_bdf;
    if (!std::getline(ifs, sysfs_bdf))
        return -1;
    while (!sysfs_bdf.empty() &&
           (sysfs_bdf.back() == '\n' || sysfs_bdf.back() == '\r' ||
            sysfs_bdf.back() == ' '  || sysfs_bdf.back() == '\t'))
        sysfs_bdf.pop_back();

    int n_gpus = 0;
    if (hipGetDeviceCount(&n_gpus) != hipSuccess)
        return -1;
    for (int g = 0; g < n_gpus; g++) {
        char hip_bdf[32];
        if (hipDeviceGetPCIBusId(hip_bdf, sizeof(hip_bdf), g) != hipSuccess)
            continue;
        if (bdf_equal(hip_bdf, sysfs_bdf))
            return g;
    }
    return -1;
}

static int amd_mem_alloc(int phxfs_dev, size_t size, void **dptr)
{
    if (!dptr)
        return -EINVAL;
    int hip_id = amd_phxfs_to_hip(phxfs_dev);
    if (hip_id < 0) {
        fprintf(stderr, "amd_mem_alloc: no HIP device for phxfs dev %d\n",
                phxfs_dev);
        return -ENODEV;
    }
    /* Allocate on the phxfs device's accelerator, but restore the caller's
     * current device afterwards so we don't disturb the application's HIP
     * context state. */
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(hip_id) != hipSuccess)
        return -EIO;
    void *p = nullptr;
    hipError_t rc = hipMalloc(&p, size);
    if (prev >= 0)
        (void)hipSetDevice(prev);
    if (rc != hipSuccess) {
        fprintf(stderr, "amd_mem_alloc: hipMalloc(%zu) failed: %s\n",
                size, hipGetErrorString(rc));
        return -ENOMEM;
    }
    *dptr = p;
    return 0;
}

static void amd_mem_free(void *dptr)
{
    if (dptr)
        (void)hipFree(dptr);
}

/* Synchronous device-to-device copy: returns only after completion, which
 * is the completion contract the staging path relies on. */
static int amd_memcpy_dtod(void *dst, const void *src, size_t n)
{
    hipError_t rc = hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice);
    if (rc != hipSuccess) {
        fprintf(stderr, "amd_memcpy_dtod: %s\n", hipGetErrorString(rc));
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Asynchronous D2D: one stream per (phxfs device, staging slot)       */
/* ------------------------------------------------------------------ */

#define AMD_MAX_QUEUES 4   /* per device; >= PHX_STAGING_SLOTS */

static hipStream_t     g_streams[PHXFS_DEV_SCAN_MAX][AMD_MAX_QUEUES];
static pthread_mutex_t g_stream_lock = PTHREAD_MUTEX_INITIALIZER;

static int amd_stream_get(int phxfs_dev, int slot, hipStream_t *out)
{
    if (phxfs_dev < 0 || phxfs_dev >= PHXFS_DEV_SCAN_MAX ||
        slot < 0 || slot >= AMD_MAX_QUEUES)
        return -EINVAL;

    pthread_mutex_lock(&g_stream_lock);
    if (g_streams[phxfs_dev][slot] == nullptr) {
        int hip_id = amd_phxfs_to_hip(phxfs_dev);
        if (hip_id < 0) {
            pthread_mutex_unlock(&g_stream_lock);
            return -ENODEV;
        }
        int prev = -1;
        (void)hipGetDevice(&prev);
        if (hipSetDevice(hip_id) != hipSuccess) {
            pthread_mutex_unlock(&g_stream_lock);
            return -EIO;
        }
        hipStream_t s = nullptr;
        hipError_t rc = hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
        if (prev >= 0)
            (void)hipSetDevice(prev);
        if (rc != hipSuccess) {
            fprintf(stderr, "amd_stream_get: hipStreamCreate: %s\n",
                    hipGetErrorString(rc));
            pthread_mutex_unlock(&g_stream_lock);
            return -EIO;
        }
        g_streams[phxfs_dev][slot] = s;
    }
    *out = g_streams[phxfs_dev][slot];
    pthread_mutex_unlock(&g_stream_lock);
    return 0;
}

static int amd_memcpy_dtod_async(int phxfs_dev, int slot, void *dst,
                                 const void *src, size_t n)
{
    hipStream_t s = nullptr;
    int rc = amd_stream_get(phxfs_dev, slot, &s);
    if (rc != 0)
        return rc;

    hipError_t err = hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToDevice, s);
    if (err != hipSuccess) {
        fprintf(stderr, "amd_memcpy_dtod_async: %s\n",
                hipGetErrorString(err));
        return -EIO;
    }
    return 0;
}

static int amd_queue_sync(int phxfs_dev, int slot)
{
    if (phxfs_dev < 0 || phxfs_dev >= PHXFS_DEV_SCAN_MAX ||
        slot < 0 || slot >= AMD_MAX_QUEUES)
        return -EINVAL;

    pthread_mutex_lock(&g_stream_lock);
    hipStream_t s = g_streams[phxfs_dev][slot];
    pthread_mutex_unlock(&g_stream_lock);
    if (!s)
        return 0;   /* nothing was ever enqueued on it */

    hipError_t err = hipStreamSynchronize(s);
    if (err != hipSuccess) {
        fprintf(stderr, "amd_queue_sync: %s\n", hipGetErrorString(err));
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Stream-ordered I/O primitive (host-function model)                 */
/* ------------------------------------------------------------------ */

/* hipLaunchHostFunc has the exact cudaLaunchHostFunc semantics: the
 * callback runs after all previously enqueued work on the stream and
 * blocks all work enqueued after it — the ordering contract the core's
 * stream path rests on. The callback must not call HIP APIs; the core
 * only runs pure host I/O inside it. */
static int amd_launch_host_func(void *stream,
                                void (*fn)(void *), void *arg)
{
    if (!stream || !fn)
        return -EINVAL;
    hipError_t rc = hipLaunchHostFunc((hipStream_t)stream,
                                      (hipHostFn_t)fn, arg);
    if (rc != hipSuccess) {
        fprintf(stderr, "amd_launch_host_func: %s\n", hipGetErrorString(rc));
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Connector registration                                             */
/* ------------------------------------------------------------------ */

/*
 * Read the device page size the kernel backend queried from the amdgpu
 * driver (see module/amd-backend.c: it calls the KFD get_page_size()
 * instead of hard-coding a value). Falls back to the KFD-documented
 * default (PAGE_SIZE) when the module is not loaded yet, so the library
 * can be linked/loaded before insmod.
 */
static uint64_t amd_query_page_size(void)
{
    for (int i = 0; i < PHXFS_DEV_SCAN_MAX; i++) {
        std::string sysfs_path = "/sys/class/phxfs-generic/phxfs_dev"
                                 + std::to_string(i) + "/page_size";
        std::ifstream ifs(sysfs_path);
        if (!ifs.is_open())
            continue;
        uint64_t ps = 0;
        if (ifs >> ps && ps > 0)
            return ps;
    }
    fprintf(stderr, "amd_connector: page_size sysfs node not found, "
                    "assuming %d\n", AMD_PAGE_SIZE_FALLBACK);
    return AMD_PAGE_SIZE_FALLBACK;
}

static int amd_init(void);

static struct devconn_ops amd_devconn = {
    .name         = "amd",
    .page_size    = AMD_PAGE_SIZE_FALLBACK,  /* refreshed in init() */
    .init         = amd_init,
    .find_device  = amd_find_device,
    .mem_alloc    = amd_mem_alloc,
    .mem_free     = amd_mem_free,
    .memcpy_dtod  = amd_memcpy_dtod,
    .memcpy_dtod_async = amd_memcpy_dtod_async,
    .queue_sync   = amd_queue_sync,
    .launch_host_func = amd_launch_host_func,
    .range_push   = NULL,
    .range_pop    = NULL,
};

/* The global connector — referenced by core code via extern */
struct devconn_ops *devconn = &amd_devconn;

static int amd_init(void)
{
    /* The HIP runtime lazily initializes itself on first API use. Pick up
     * the kernel-reported page size so both sides always agree. */
    amd_devconn.page_size = amd_query_page_size();
    return 0;
}

int devconn_init(void)
{
    if (devconn && devconn->init)
        return devconn->init();
    return 0;
}
