#ifndef ILUVATAR_CUDA_RT_H
#define ILUVATAR_CUDA_RT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Absolute path of the libcuda.so.1 whose UMD version matches the
 * running KMD. Does not dlopen. NULL if none was found. */
const char *iluvatar_libcuda_path(void);

/* 0 when libcuda and libcudart matching /sys/module/iluvatar/version
 * are loaded. Negative errno after the attempt. Safe to call again. */
int iluvatar_cuda_bind(void);

#ifdef __cplusplus
}
#endif

#endif
