/*
 * libcudart dlopens "libcuda.so.1" by SONAME. Opening the matching
 * libcuda by absolute path first does not stop that search from loading
 * /usr/lib/x86_64-linux-gnu/libcuda.so.1. This definition has to live in
 * the executable: the dynamic loader consults the executable before libc.
 * Link with -Wl,--export-dynamic so a libcudart that was itself dlopened
 * still sees it.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

extern "C" const char *iluvatar_libcuda_path(void);

extern "C" void *dlopen(const char *name, int flags)
{
	static void *(*real_dlopen)(const char *, int) = nullptr;
	if (!real_dlopen)
		real_dlopen = (void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen");
	const char *use = name;
	if (name && (strcmp(name, "libcuda.so.1") == 0 ||
		     strcmp(name, "libcuda.so") == 0)) {
		const char *picked = iluvatar_libcuda_path();
		if (picked)
			use = picked;
	}
	return real_dlopen(use, flags);
}
