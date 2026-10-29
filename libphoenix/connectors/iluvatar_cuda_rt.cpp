/*
 * Load the libcuda that matches the running iluvatar.ko, then the
 * libcudart that sits in the same directory.
 *
 * CoreX headers rename cudaMalloc to ixMalloc. This file defines those
 * ix* symbols. libphoenix does not DT_NEEDED libcudart, so a process that
 * only links libphoenix has not loaded any libcuda yet. bind() dlopens
 * the matching pair by absolute path. glibc caches LD_LIBRARY_PATH at
 * process start, so setenv here cannot change a later dlopen("libcuda.so.1").
 *
 * If some libcuda is already mapped and its UMD version string is not the
 * KMD version, bind fails and names the directory that would have matched.
 * Unloading the wrong SONAME is not possible from here.
 */

#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <pthread.h>
#include <string>
#include <vector>

#include "iluvatar_cuda_rt.h"

struct iluvatar_cuda_api {
	cudaError_t (*DeviceGetPCIBusId)(char *, int, int);
	cudaError_t (*GetDeviceCount)(int *);
	cudaError_t (*GetDevice)(int *);
	cudaError_t (*SetDevice)(int);
	cudaError_t (*Malloc)(void **, size_t);
	cudaError_t (*Free)(void *);
	cudaError_t (*Memcpy)(void *, const void *, size_t, cudaMemcpyKind);
	cudaError_t (*Memset)(void *, int, size_t);
	cudaError_t (*DeviceSynchronize)(void);
	cudaError_t (*MallocHost)(void **, size_t);
	cudaError_t (*FreeHost)(void *);
	cudaError_t (*StreamCreateWithFlags)(cudaStream_t *, unsigned int);
	cudaError_t (*StreamSynchronize)(cudaStream_t);
	cudaError_t (*MemcpyAsync)(void *, const void *, size_t, cudaMemcpyKind,
				    cudaStream_t);
	cudaError_t (*LaunchHostFunc)(cudaStream_t, cudaHostFn_t, void *);
	cudaError_t (*PointerGetAttributes)(cudaPointerAttributes *, const void *);
	const char *(*GetErrorString)(cudaError_t);
	cudaError_t (*GetLastError)(void);
};

static iluvatar_cuda_api g_api;
static int g_bind_rc = -1;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static std::string trim_copy(std::string s)
{
	while (!s.empty() && std::isspace((unsigned char)s.back()))
		s.pop_back();
	size_t i = 0;
	while (i < s.size() && std::isspace((unsigned char)s[i]))
		i++;
	return s.substr(i);
}

static std::string read_kmd_version(void)
{
	std::ifstream in("/sys/module/iluvatar/version");
	std::string s;
	if (!in)
		return {};
	std::getline(in, s);
	return trim_copy(s);
}

/* UMD_VERSION_STRING is a NUL-terminated rodata token, next to the log
 * literal "Iluvatar Corex UMD version:". Both must be present so a random
 * "4.4.5" inside an unrelated .so does not match. */
static bool umd_matches(const std::string &path, const std::string &ver)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
		return false;
	std::string data((std::istreambuf_iterator<char>(in)),
			 std::istreambuf_iterator<char>());
	if (data.find("Iluvatar Corex UMD version:") == std::string::npos)
		return false;
	/* The version is a C string. The byte in front is not always NUL:
	 * this 4.4.5 libcuda places "4.4.5" directly after another object. */
	size_t pos = 0;
	while ((pos = data.find(ver, pos)) != std::string::npos) {
		bool end = pos + ver.size() < data.size() &&
			   data[pos + ver.size()] == '\0';
		unsigned char prev = pos == 0 ? 0 : (unsigned char)data[pos - 1];
		bool longer = pos > 0 && (std::isdigit(prev) || prev == '.');
		if (end && !longer)
			return true;
		pos += 1;
	}
	return false;
}

static std::string mapped_path(const char *needle)
{
	std::ifstream in("/proc/self/maps");
	std::string line;
	while (std::getline(in, line)) {
		auto slash = line.rfind(" /");
		if (slash == std::string::npos)
			continue;
		std::string path = line.substr(slash + 1);
		if (path.find(needle) != std::string::npos)
			return path;
	}
	return {};
}

static void push_dir(std::vector<std::string> &dirs, const std::string &dir)
{
	if (dir.empty() || dir == ".")
		return;
	for (const auto &have : dirs)
		if (have == dir)
			return;
	dirs.push_back(dir);
}

static void push_env_path(std::vector<std::string> &dirs, const char *env)
{
	if (!env || !*env)
		return;
	std::string all(env);
	size_t start = 0;
	while (start <= all.size()) {
		size_t colon = all.find(':', start);
		if (colon == std::string::npos)
			colon = all.size();
		push_dir(dirs, all.substr(start, colon - start));
		if (colon == all.size())
			break;
		start = colon + 1;
	}
}

static std::string find_umd_dir(const std::string &ver)
{
	std::vector<std::string> dirs;
	push_env_path(dirs, std::getenv("PHX_UMD_LIB"));
	push_env_path(dirs, std::getenv("LD_LIBRARY_PATH"));

	for (const auto &dir : dirs) {
		if (umd_matches(dir + "/libcuda.so.1", ver))
			return dir;
	}
	return {};
}

/* POD, not std::string: this is filled from another translation unit's
 * static constructor. A std::string at file scope can be constructed
 * after that call and wipe the path. */
static char g_libcuda_path[4096];
static pthread_once_t g_path_once = PTHREAD_ONCE_INIT;

static void resolve_libcuda_path(void)
{
	std::string ver = read_kmd_version();
	if (ver.empty())
		return;
	std::string dir = find_umd_dir(ver);
	if (dir.empty())
		return;
	std::string path = dir + "/libcuda.so.1";
	if (path.size() >= sizeof(g_libcuda_path))
		return;
	memcpy(g_libcuda_path, path.c_str(), path.size() + 1);
}

extern "C" const char *iluvatar_libcuda_path(void)
{
	pthread_once(&g_path_once, resolve_libcuda_path);
	if (g_libcuda_path[0] == '\0')
		return nullptr;
	return g_libcuda_path;
}

static void *open_cudart(const std::string &dir)
{
	const char *names[] = {"/libcudart.so.10.2", "/libcudart.so", nullptr};
	for (int i = 0; names[i]; i++) {
		std::string path = dir + names[i];
		void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
		if (h)
			return h;
	}
	return nullptr;
}

#define LOAD(field, symbol)                                                    \
	do {                                                                   \
		g_api.field = (decltype(g_api.field))dlsym(cudart, symbol);   \
		if (!g_api.field) {                                            \
			fprintf(stderr, "iluvatar: %s has no symbol %s\n",     \
				cudart_path.c_str(), symbol);                  \
			return -1;                                             \
		}                                                              \
	} while (0)

static int bind_impl(void)
{
	std::string ver = read_kmd_version();
	if (ver.empty()) {
		fprintf(stderr,
			"iluvatar: cannot read /sys/module/iluvatar/version\n");
		return -1;
	}

	const char *libcuda = iluvatar_libcuda_path();
	if (!libcuda) {
		fprintf(stderr,
			"iluvatar: no libcuda.so.1 matches KMD %s. Set PHX_UMD_LIB to the directory that contains libcuda.so.1 and libcudart\n",
			ver.c_str());
		return -1;
	}
	std::string cuda_path = libcuda;
	std::string dir = cuda_path.substr(0, cuda_path.find_last_of('/'));

	std::string loaded = mapped_path("libcuda.so");
	if (!loaded.empty() && !umd_matches(loaded, ver)) {
		fprintf(stderr,
			"iluvatar: process already mapped %s, which does not match KMD %s. The dynamic linker fixed the search path at startup.\n"
			"  export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n",
			loaded.c_str(), ver.c_str(), dir.c_str());
		return -1;
	}

	std::string loaded_rt = mapped_path("libcudart.so");
	if (!loaded_rt.empty() && loaded_rt.find(dir + "/") != 0) {
		fprintf(stderr,
			"iluvatar: process already mapped %s, not under %s\n",
			loaded_rt.c_str(), dir.c_str());
		return -1;
	}

	void *libcuda_handle = dlopen(cuda_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
	if (!libcuda_handle) {
		fprintf(stderr, "iluvatar: dlopen %s failed: %s\n",
			cuda_path.c_str(), dlerror());
		return -1;
	}

	void *cudart = open_cudart(dir);
	if (!cudart) {
		fprintf(stderr, "iluvatar: no libcudart in %s: %s\n",
			dir.c_str(), dlerror());
		return -1;
	}
	std::string cudart_path = dir + "/libcudart.so.10.2";

	LOAD(DeviceGetPCIBusId, "cudaDeviceGetPCIBusId");
	LOAD(GetDeviceCount, "cudaGetDeviceCount");
	LOAD(GetDevice, "cudaGetDevice");
	LOAD(SetDevice, "cudaSetDevice");
	LOAD(Malloc, "cudaMalloc");
	LOAD(Free, "cudaFree");
	LOAD(Memcpy, "cudaMemcpy");
	LOAD(Memset, "cudaMemset");
	LOAD(DeviceSynchronize, "cudaDeviceSynchronize");
	LOAD(MallocHost, "cudaMallocHost");
	LOAD(FreeHost, "cudaFreeHost");
	LOAD(StreamCreateWithFlags, "cudaStreamCreateWithFlags");
	LOAD(StreamSynchronize, "cudaStreamSynchronize");
	LOAD(MemcpyAsync, "cudaMemcpyAsync");
	LOAD(LaunchHostFunc, "cudaLaunchHostFunc");
	LOAD(PointerGetAttributes, "cudaPointerGetAttributes");
	LOAD(GetErrorString, "cudaGetErrorString");
	LOAD(GetLastError, "cudaGetLastError");

	fprintf(stderr, "iluvatar: libcuda %s (KMD %s)\n", cuda_path.c_str(),
		ver.c_str());
	return 0;
}

#undef LOAD

static void bind_once(void)
{
	g_bind_rc = bind_impl();
}

int iluvatar_cuda_bind(void)
{
	pthread_once(&g_once, bind_once);
	return g_bind_rc;
}

static cudaError_t need(void)
{
	if (iluvatar_cuda_bind() != 0)
		return cudaErrorInitializationError;
	return cudaSuccess;
}

extern "C" cudaError_t ixDeviceGetPCIBusId(char *pci, int len, int dev)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.DeviceGetPCIBusId(pci, len, dev);
}

extern "C" cudaError_t ixGetDeviceCount(int *n)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.GetDeviceCount(n);
}

extern "C" cudaError_t ixGetDevice(int *dev)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.GetDevice(dev);
}

extern "C" cudaError_t ixSetDevice(int dev)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.SetDevice(dev);
}

extern "C" cudaError_t ixMalloc(void **p, size_t n)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.Malloc(p, n);
}

extern "C" cudaError_t ixFree(void *p)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.Free(p);
}

extern "C" cudaError_t ixMemcpy(void *dst, const void *src, size_t n,
				cudaMemcpyKind kind)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.Memcpy(dst, src, n, kind);
}

extern "C" cudaError_t ixMemset(void *p, int v, size_t n)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.Memset(p, v, n);
}

extern "C" cudaError_t ixDeviceSynchronize(void)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.DeviceSynchronize();
}

extern "C" cudaError_t ixMallocHost(void **p, size_t n)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.MallocHost(p, n);
}

extern "C" cudaError_t ixFreeHost(void *p)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.FreeHost(p);
}

extern "C" cudaError_t ixStreamCreateWithFlags(cudaStream_t *s, unsigned int flags)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.StreamCreateWithFlags(s, flags);
}

extern "C" cudaError_t ixStreamSynchronize(cudaStream_t s)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.StreamSynchronize(s);
}

extern "C" cudaError_t ixMemcpyAsync(void *dst, const void *src, size_t n,
				     cudaMemcpyKind kind, cudaStream_t stream)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.MemcpyAsync(dst, src, n, kind, stream);
}

extern "C" cudaError_t ixLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
					void *arg)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.LaunchHostFunc(stream, fn, arg);
}

extern "C" cudaError_t ixPointerGetAttributes(cudaPointerAttributes *attrs,
					      const void *ptr)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.PointerGetAttributes(attrs, ptr);
}

extern "C" const char *ixGetErrorString(cudaError_t err)
{
	if (iluvatar_cuda_bind() != 0)
		return "iluvatar libcuda bind failed";
	return g_api.GetErrorString(err);
}

extern "C" cudaError_t ixGetLastError(void)
{
	if (need() != cudaSuccess)
		return cudaErrorInitializationError;
	return g_api.GetLastError();
}
