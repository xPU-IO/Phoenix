// FULL check: one 4KiB write and one 4KiB read of a cudaMalloc buffer.
// Run only when phxfs_map_mode=0. The file path is required.
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "phoenix.h"

static const size_t k_bytes = 4096;

static void die(const char *what, int rc)
{
	fprintf(stderr, "%s failed: %d (%s)\n", what, rc, strerror(rc < 0 ? -rc : errno));
	exit(1);
}

extern "C" const char *iluvatar_libcuda_path(void);

static int libcuda_is_match(void)
{
	const char *want = iluvatar_libcuda_path();
	FILE *maps = fopen("/proc/self/maps", "r");
	char line[512];
	int saw = 0;

	if (!want || !maps) {
		fprintf(stderr, "libcuda path missing\n");
		return 0;
	}
	while (fgets(line, sizeof line, maps)) {
		if (!strstr(line, "libcuda.so"))
			continue;
		saw = 1;
		if (strstr(line, want)) {
			printf("libcuda %s\n", want);
			fclose(maps);
			return 1;
		}
		fprintf(stderr, "loaded libcuda is not %s\n%s", want, line);
		fclose(maps);
		return 0;
	}
	fclose(maps);
	if (!saw)
		fprintf(stderr, "libcuda not mapped\n");
	return 0;
}

static int same(const unsigned char *a, unsigned char expect, size_t n)
{
	size_t i;
	for (i = 0; i < n; i++) {
		if (a[i] != expect) {
			fprintf(stderr, "mismatch at %zu got %02x expect %02x\n",
				i, a[i], expect);
			return 0;
		}
	}
	return 1;
}

int main(int argc, char **argv)
{
	const char *path;
	int device;

	if (argc < 2) {
		fprintf(stderr, "usage: %s <file> [device]\n", argv[0]);
		return 2;
	}
	path = argv[1];
	device = argc > 2 ? atoi(argv[2]) : 0;
	void *gpu = NULL;
	void *target = NULL;
	unsigned char *host = NULL;
	unsigned char *filebuf = NULL;
	int fd = -1;
	int mode;
	ssize_t n;

	fd = open(path, O_CREAT | O_RDWR, 0644);
	if (fd < 0)
		die("open", -errno);
	if (posix_fallocate(fd, 0, k_bytes) != 0)
		die("posix_fallocate", -errno);
	close(fd);
	fd = open(path, O_RDWR | O_DIRECT);
	if (fd < 0)
		die("open O_DIRECT", -errno);

	if (phxfs_open(device) != 0)
		die("phxfs_open", -1);
	mode = phxfs_get_map_mode(device);
	printf("map_mode=%d page_size=%llu\n", mode,
	       (unsigned long long)phxfs_get_page_size());
	if (mode != 0)
		die("need FULL map_mode 0", mode);

	host = (unsigned char *)aligned_alloc(4096, k_bytes);
	filebuf = (unsigned char *)aligned_alloc(4096, k_bytes);
	if (!host || !filebuf)
		die("aligned_alloc", -ENOMEM);
	memset(host, 0xAB, k_bytes);

	if (cudaMalloc(&gpu, k_bytes) != cudaSuccess)
		die("cudaMalloc", -1);
	if (!libcuda_is_match())
		return 1;
	if (cudaMemcpy(gpu, host, k_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
		die("H2D", -1);
	if (cudaDeviceSynchronize() != cudaSuccess)
		die("sync after H2D", -1);
	if (phxfs_regmem(device, gpu, k_bytes, &target) != 0)
		die("phxfs_regmem", -1);

	n = phxfs_write(fd, device, gpu, 0, k_bytes, 0);
	if (n != (ssize_t)k_bytes)
		die("phxfs_write", n < 0 ? (int)n : -EIO);
	memset(filebuf, 0, k_bytes);
	if (pread(fd, filebuf, k_bytes, 0) != (ssize_t)k_bytes)
		die("pread", -errno);
	if (!same(filebuf, 0xAB, k_bytes))
		return 1;
	printf("write file %02x %02x %02x %02x\n",
	       filebuf[0], filebuf[1], filebuf[2], filebuf[3]);

	memset(host, 0xCD, k_bytes);
	if (pwrite(fd, host, k_bytes, 0) != (ssize_t)k_bytes)
		die("pwrite", -errno);
	if (cudaMemset(gpu, 0, k_bytes) != cudaSuccess)
		die("cudaMemset", -1);
	if (cudaDeviceSynchronize() != cudaSuccess)
		die("sync after memset", -1);
	n = phxfs_read(fd, device, gpu, 0, k_bytes, 0);
	if (n != (ssize_t)k_bytes)
		die("phxfs_read", n < 0 ? (int)n : -EIO);
	memset(host, 0, k_bytes);
	if (cudaMemcpy(host, gpu, k_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
		die("D2H", -1);
	if (!same(host, 0xCD, k_bytes))
		return 1;
	printf("read gpu %02x %02x %02x %02x\n", host[0], host[1], host[2], host[3]);

	phxfs_deregmem(device, gpu, k_bytes);
	cudaFree(gpu);
	phxfs_close(device);
	close(fd);
	printf("FULL roundtrip ok\n");
	return 0;
}
