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
#include <cstring>
#include <fstream>
#include <string>
#include <pthread.h>
#include <unistd.h>

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
static int nvidia_mem_alloc(int device_id, size_t size, void **dptr)
{
    if (!dptr)
        return -EINVAL;
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess ||
        device_id < 0 || device_id >= n_gpus)
        return -ENODEV;
    int cuda_id = device_id;
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

extern "C" int phx_connector_probe_alloc(int device_id, size_t size, void **ptr)
{
    return nvidia_mem_alloc(device_id, size, ptr);
}

static void nvidia_mem_free(void *dptr)
{
    if (dptr)
        cudaFree(dptr);
}

extern "C" void phx_connector_probe_free(void *ptr)
{
    nvidia_mem_free(ptr);
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

static int nvidia_host_alloc(size_t size, void **ptr)
{
    if (!ptr || size == 0)
        return -EINVAL;
    cudaError_t rc = cudaMallocHost(ptr, size);
    if (rc != cudaSuccess) {
        fprintf(stderr, "nvidia_host_alloc: %s\n", cudaGetErrorString(rc));
        *ptr = nullptr;
        return -ENOMEM;
    }
    return 0;
}

static void nvidia_host_free(void *ptr)
{
    if (ptr)
        cudaFreeHost(ptr);
}

static int nvidia_set_device(int device_id)
{
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess ||
        device_id < 0 || device_id >= n_gpus ||
        cudaSetDevice(device_id) != cudaSuccess)
        return -ENODEV;
    return 0;
}

static int nvidia_memcpy_h2d(void *dst, const void *src, size_t n)
{
    cudaError_t rc = cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
    return rc == cudaSuccess ? 0 : -EIO;
}

static int nvidia_memcpy_d2h(void *dst, const void *src, size_t n)
{
    cudaError_t rc = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
    return rc == cudaSuccess ? 0 : -EIO;
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
static int nvidia_stream_get(int device_id, int slot, cudaStream_t *out)
{
    if (device_id < 0 || device_id >= PHXFS_DEV_SCAN_MAX ||
        slot < 0 || slot >= NV_MAX_QUEUES)
        return -EINVAL;

    pthread_mutex_lock(&g_stream_lock);
    if (g_streams[device_id][slot] == nullptr) {
        int cuda_id = device_id;
        int n_gpus = 0;
        if (cudaGetDeviceCount(&n_gpus) != cudaSuccess ||
            cuda_id < 0 || cuda_id >= n_gpus) {
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
        g_streams[device_id][slot] = s;
    }
    *out = g_streams[device_id][slot];
    pthread_mutex_unlock(&g_stream_lock);
    return 0;
}

static int nvidia_memcpy_dtod_async(int device_id, int slot, void *dst,
                                    const void *src, size_t n)
{
    cudaStream_t s = nullptr;
    int rc = nvidia_stream_get(device_id, slot, &s);
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

static int nvidia_queue_sync(int device_id, int slot)
{
    if (device_id < 0 || device_id >= PHXFS_DEV_SCAN_MAX ||
        slot < 0 || slot >= NV_MAX_QUEUES)
        return -EINVAL;

    /* No stream created => nothing was ever enqueued on it. */
    pthread_mutex_lock(&g_stream_lock);
    cudaStream_t s = g_streams[device_id][slot];
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

/* Owning ordinal of device/managed memory; -ENOENT for host memory. */
static int nvidia_pointer_device(const void *ptr)
{
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess) {
        cudaGetLastError();   /* older runtimes report host ptrs as errors */
        return -ENOENT;
    }
    if (attrs.type == cudaMemoryTypeDevice ||
        attrs.type == cudaMemoryTypeManaged)
        return attrs.device;
    return -ENOENT;
}

static int nvidia_memcpy_dtod_on_stream(int device_id, void *dst,
                                        const void *src, size_t n,
                                        void *stream)
{
    int prev = -1;
    if (cudaGetDevice(&prev) != cudaSuccess)
        return -EIO;
    if (prev != device_id && cudaSetDevice(device_id) != cudaSuccess)
        return -EIO;
    cudaError_t rc = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
                                     (cudaStream_t)stream);
    if (prev != device_id)
        cudaSetDevice(prev);
    if (rc != cudaSuccess) {
        fprintf(stderr, "nvidia_memcpy_dtod_on_stream: %s\n",
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

#ifdef PHXFS_ILUVATAR
extern "C" int iluvatar_cuda_bind(void);
#endif

static int nvidia_init(void)
{
#ifdef PHXFS_ILUVATAR
    return iluvatar_cuda_bind() == 0 ? 0 : -ENODEV;
#else
    return 0;
#endif
}

static struct devconn_ops nvidia_devconn = {
#ifdef PHXFS_ILUVATAR
    .name         = "iluvatar",
    .page_size    = 4096,
#else
    .name         = "nvidia",
    .page_size    = 64 * 1024,
#endif
    .init         = nvidia_init,
    .find_device  = nvidia_find_device,
    .mem_alloc    = nvidia_mem_alloc,
    .mem_free     = nvidia_mem_free,
    .memcpy_dtod  = nvidia_memcpy_dtod,
    .host_alloc   = nvidia_host_alloc,
    .host_free    = nvidia_host_free,
    .set_device   = nvidia_set_device,
    .memcpy_h2d   = nvidia_memcpy_h2d,
    .memcpy_d2h   = nvidia_memcpy_d2h,
    .memcpy_dtod_async = nvidia_memcpy_dtod_async,
    .queue_sync   = nvidia_queue_sync,
    .launch_host_func = nvidia_launch_host_func,
    .pointer_device = nvidia_pointer_device,
    .memcpy_dtod_on_stream = nvidia_memcpy_dtod_on_stream,
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
