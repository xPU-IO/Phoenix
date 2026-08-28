/*
 * NVIDIA DevConnector
 *
 * All NVIDIA/CUDA specific code in the user library lives here.
 * Core libphoenix (phoenix.cpp) never includes CUDA headers or calls
 * CUDA APIs directly.
 *
 * Note: MetaX (沐曦) GPUs also reuse this connector. MetaX's MACA SDK ships
 * a CUDA-compatible runtime (cudaMalloc, cudaMemcpyAsync, cudaLaunchHostFunc,
 * BDF queries via cudaDeviceGetPCIBusId, ... all work), so the NVIDIA
 * connector handles MetaX hardware without any MetaX-specific code path.
 * To run on MetaX, install the MACA driver + MACA SDK and set:
 *     export MACA_PATH=/opt/maca
 *     export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
 * — see doc/install.md for the full setup. Only cmake flag change is required
 * (-DPHXFS_VENDOR=METAX).
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <string>
#include <pthread.h>

#include "devconnector.h"

#define NV_MAX_GPUS 64        /* CUDA-GPU cache size */
/* sysfs scan bound: must match the core's PHXFS_MAX_DEVICES (phoenix.cpp), as
 * find_device returns a phxfs index the core can actually open. */
#define PHXFS_DEV_SCAN_MAX 8

/* ------------------------------------------------------------------ */
/* Device discovery: CUDA device ID → phxfs index via PCI BDF match  */
/* ------------------------------------------------------------------ */

/* Cache CUDA-GPU -> phxfs-index so repeated find() calls don't re-query CUDA
 * and re-scan sysfs every time. -2 = unresolved; only successful (>=0) results
 * are cached, so a device that appears later can still be found. */
static int             g_dev_cache[NV_MAX_GPUS];
static bool            g_dev_cache_init = false;
static pthread_mutex_t g_dev_cache_lock = PTHREAD_MUTEX_INITIALIZER;

/* Case-insensitive compare of a C string and std::string (BDFs differ only in
 * hex case between CUDA's "0000:65:00.0" and sysfs). */
static bool bdf_equal(const char *a, const std::string &b)
{
    size_t i = 0;
    for (; a[i] && i < b.size(); i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
            return false;
    return a[i] == '\0' && i == b.size();
}

static int nvidia_find_device_uncached(int cuda_gpu_id)
{
    /* Full PCI bus id "domain:bus:device.function" straight from CUDA — no
     * assumption that the function is 0, and no fragile separator parsing. */
    char cuda_bdf[32];
    if (cudaDeviceGetPCIBusId(cuda_bdf, sizeof(cuda_bdf), cuda_gpu_id) != cudaSuccess) {
        fprintf(stderr, "nvidia_find_device: cudaDeviceGetPCIBusId failed for GPU %d\n",
                cuda_gpu_id);
        return -1;
    }

    /* Match against sysfs pci_bdf entries. A missing phxfs_devN does not stop
     * the scan (indices need not be contiguous). */
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
        if (bdf_equal(cuda_bdf, sysfs_bdf))
            return i;
    }
    return -1;
}

static int nvidia_find_device(int cuda_gpu_id)
{
    if (cuda_gpu_id < 0 || cuda_gpu_id >= NV_MAX_GPUS)
        return nvidia_find_device_uncached(cuda_gpu_id);

    pthread_mutex_lock(&g_dev_cache_lock);
    if (!g_dev_cache_init) {
        for (int i = 0; i < NV_MAX_GPUS; i++)
            g_dev_cache[i] = -2;   /* unresolved */
        g_dev_cache_init = true;
    }
    int cached = g_dev_cache[cuda_gpu_id];
    pthread_mutex_unlock(&g_dev_cache_lock);
    if (cached >= 0)
        return cached;             /* only successful lookups are cached */

    int idx = nvidia_find_device_uncached(cuda_gpu_id);
    if (idx >= 0) {
        /* Cache success only. A negative result is deliberately NOT cached, so
         * a device that appears after the first query (module inserted / sysfs
         * node created later) is found on a later call without restarting the
         * process. */
        pthread_mutex_lock(&g_dev_cache_lock);
        g_dev_cache[cuda_gpu_id] = idx;
        pthread_mutex_unlock(&g_dev_cache_lock);
    }
    return idx;
}

/* ------------------------------------------------------------------ */
/* Staging-mode device memory operations                              */
/* ------------------------------------------------------------------ */

/* Reverse of find_device: phxfs index -> CUDA device id, by matching the
 * phxfs sysfs pci_bdf against each CUDA device's PCI bus id. Returns the CUDA
 * device id (>=0) or -1. */
static int nvidia_phxfs_to_cuda(int phxfs_dev)
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
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess)
        return -1;
    for (int g = 0; g < n_gpus; g++) {
        char cuda_bdf[32];
        if (cudaDeviceGetPCIBusId(cuda_bdf, sizeof(cuda_bdf), g) != cudaSuccess)
            continue;
        if (bdf_equal(cuda_bdf, sysfs_bdf))
            return g;
    }
    return -1;
}

/*
 * Staging pool allocation.
 *
 * The kernel remaps the pool's BAR range in 2MiB units and requires every
 * fresh unit to be wholly owned by the pool, so ideally we would allocate in
 * 2MiB physical chunks. The CUDA VMM API (cuMemCreate) offers exactly that,
 * but VMM mappings cannot be pinned by the legacy nvidia_p2p_get_pages API
 * the kernel module uses (EINVAL), and the DMA-BUF P2P path that supports
 * VMM requires kernel >= 5.12 -- so on kernel 5.4 the pool must stay on
 * cudaMalloc. Large cudaMalloc allocations are backed by 2MiB large pages in
 * practice, which tiles the kernel's units; if the driver ever falls back to
 * 64KiB small pages (fragmentation), the kernel's exclusive-unit check
 * rejects the registration loudly instead of silently breaking RDMA/peermem
 * registration of unrelated GPU memory.
 */
static int nvidia_mem_alloc(int phxfs_dev, size_t size, void **dptr)
{
    if (!dptr)
        return -EINVAL;
    int cuda_id = nvidia_phxfs_to_cuda(phxfs_dev);
    if (cuda_id < 0) {
        fprintf(stderr, "nvidia_mem_alloc: no CUDA device for phxfs dev %d\n",
                phxfs_dev);
        return -ENODEV;
    }
    /* Allocate on the phxfs device's accelerator, but restore the caller's
     * current device afterwards so we don't disturb the application's CUDA
     * context state. */
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(cuda_id) != cudaSuccess)
        return -EIO;
    void *p = nullptr;
    cudaError_t rc = cudaMalloc(&p, size);
    if (prev >= 0)
        cudaSetDevice(prev);
    if (rc != cudaSuccess) {
        fprintf(stderr, "nvidia_mem_alloc: cudaMalloc(%zu) failed: %s\n",
                size, cudaGetErrorString(rc));
        return -ENOMEM;
    }
    *dptr = p;
    return 0;
}

static void nvidia_mem_free(void *dptr)
{
    if (dptr)
        cudaFree(dptr);
}

/* Synchronous device-to-device copy. cudaMemcpy(...DeviceToDevice) does not
 * return to the host until the copy has completed, which is the completion
 * contract the staging path relies on. */
static int nvidia_memcpy_dtod(void *dst, const void *src, size_t n)
{
    cudaError_t rc = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice);
    if (rc != cudaSuccess) {
        fprintf(stderr, "nvidia_memcpy_dtod: %s\n", cudaGetErrorString(rc));
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Asynchronous D2D: one stream per (phxfs device, staging slot)       */
/* ------------------------------------------------------------------ */

/*
 * Non-blocking streams so the staging path's copies never serialise against
 * the application's work on the legacy default stream. Created lazily on first
 * use and kept for the process lifetime (a handful of streams per device).
 */
#define NV_MAX_QUEUES 4   /* per device; >= PHX_STAGING_SLOTS */

static cudaStream_t    g_streams[PHXFS_DEV_SCAN_MAX][NV_MAX_QUEUES];
static pthread_mutex_t g_stream_lock = PTHREAD_MUTEX_INITIALIZER;

/* Fetch (creating on first use) the stream for (phxfs_dev, slot). */
static int nvidia_stream_get(int phxfs_dev, int slot, cudaStream_t *out)
{
    if (phxfs_dev < 0 || phxfs_dev >= PHXFS_DEV_SCAN_MAX ||
        slot < 0 || slot >= NV_MAX_QUEUES)
        return -EINVAL;

    pthread_mutex_lock(&g_stream_lock);
    if (g_streams[phxfs_dev][slot] == nullptr) {
        int cuda_id = nvidia_phxfs_to_cuda(phxfs_dev);
        if (cuda_id < 0) {
            pthread_mutex_unlock(&g_stream_lock);
            return -ENODEV;
        }
        /* A stream belongs to a device, so create it with that device current,
         * then restore the caller's. */
        int prev = -1;
        cudaGetDevice(&prev);
        if (cudaSetDevice(cuda_id) != cudaSuccess) {
            pthread_mutex_unlock(&g_stream_lock);
            return -EIO;
        }
        cudaStream_t s = nullptr;
        cudaError_t rc = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        if (prev >= 0)
            cudaSetDevice(prev);
        if (rc != cudaSuccess) {
            fprintf(stderr, "nvidia_stream_get: cudaStreamCreate: %s\n",
                    cudaGetErrorString(rc));
            pthread_mutex_unlock(&g_stream_lock);
            return -EIO;
        }
        g_streams[phxfs_dev][slot] = s;
    }
    *out = g_streams[phxfs_dev][slot];
    pthread_mutex_unlock(&g_stream_lock);
    return 0;
}

static int nvidia_memcpy_dtod_async(int phxfs_dev, int slot, void *dst,
                                    const void *src, size_t n)
{
    cudaStream_t s = nullptr;
    int rc = nvidia_stream_get(phxfs_dev, slot, &s);
    if (rc != 0)
        return rc;

    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) {
        fprintf(stderr, "nvidia_memcpy_dtod_async: %s\n",
                cudaGetErrorString(err));
        return -EIO;
    }
    return 0;
}

static int nvidia_queue_sync(int phxfs_dev, int slot)
{
    if (phxfs_dev < 0 || phxfs_dev >= PHXFS_DEV_SCAN_MAX ||
        slot < 0 || slot >= NV_MAX_QUEUES)
        return -EINVAL;

    /* No stream created => nothing was ever enqueued on it. */
    pthread_mutex_lock(&g_stream_lock);
    cudaStream_t s = g_streams[phxfs_dev][slot];
    pthread_mutex_unlock(&g_stream_lock);
    if (!s)
        return 0;

    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        fprintf(stderr, "nvidia_queue_sync: %s\n", cudaGetErrorString(err));
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Stream-ordered I/O primitive (host-function model)                 */
/* ------------------------------------------------------------------ */

/*
 * Enqueue a host callback on the user stream. CUDA guarantees the callback
 * runs after all previously enqueued work on the stream and blocks all work
 * enqueued after it — the entire ordering contract of phx_stream.cpp rests
 * on this. Consecutive host functions on one stream are officially
 * supported ("the stream will remain idle across consecutive host
 * functions"). The callback itself must not make CUDA API calls; the core
 * only runs pure-host I/O inside it.
 */
static int nvidia_launch_host_func(void *stream,
                                   void (*fn)(void *), void *arg)
{
    if (!stream || !fn)
        return -EINVAL;
    cudaError_t rc = cudaLaunchHostFunc((cudaStream_t)stream,
                                        (cudaHostFn_t)fn, arg);
    if (rc != cudaSuccess) {
        fprintf(stderr, "nvidia_launch_host_func: %s\n",
                cudaGetErrorString(rc));
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Profiler ranges (NVTX)*/
/* ------------------------------------------------------------------ */

/*
 * NVTX v3 is header-only, so this adds no link dependency. When no profiler is
 * attached the calls collapse into a NULL-injection-pointer check inside NVTX
 * itself. Compile out entirely with cmake -Dphx_nvtx=false.
 */
#ifdef PHX_NVTX
#include <nvtx3/nvToolsExt.h>

static void nvidia_range_push(const char *name)
{
    nvtxRangePushA(name);
}

static void nvidia_range_pop(void)
{
    nvtxRangePop();
}
#endif

/* ------------------------------------------------------------------ */
/* Connector registration                                             */
/* ------------------------------------------------------------------ */

static int nvidia_init(void)
{
    /* CUDA runtime auto-initializes on first API call — nothing to do */
    return 0;
}

static struct devconn_ops nvidia_devconn = {
    .name         = "nvidia",
    .page_size    = 64 * 1024,
    .init         = nvidia_init,
    .find_device  = nvidia_find_device,
    .mem_alloc    = nvidia_mem_alloc,
    .mem_free     = nvidia_mem_free,
    .memcpy_dtod  = nvidia_memcpy_dtod,
    .memcpy_dtod_async = nvidia_memcpy_dtod_async,
    .queue_sync   = nvidia_queue_sync,
    .launch_host_func = nvidia_launch_host_func,
#ifdef PHX_NVTX
    .range_push   = nvidia_range_push,
    .range_pop    = nvidia_range_pop,
#else
    .range_push   = NULL,
    .range_pop    = NULL,
#endif
};

/* The global connector — referenced by core code via extern */
struct devconn_ops *devconn = &nvidia_devconn;

int devconn_init(void)
{
    if (devconn && devconn->init)
        return devconn->init();
    return 0;
}
