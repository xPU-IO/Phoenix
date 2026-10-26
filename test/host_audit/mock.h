#pragma once
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include "phoenix.h"
#include "phx_internal.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
namespace mock {
extern int find_result[PHXFS_MAX_DEVICES];
extern int device_count, kernel_mode;
extern std::atomic<int> open_error, map_error, unmap_error, munmap_error, host_error;
extern std::atomic<int> maps, unmaps, host_live, device_live, launches;
extern std::atomic<bool> fail_threads;
extern std::atomic<int> fail_fd;
extern std::atomic<off_t> fail_from, fail_until;
extern void (*alloc_gate)();
extern void (*map_gate)();
extern int (*map_check)(uint64_t, size_t);
int file(size_t size);
phxfs_io_req_t req(int fd, int d, void *buf, size_t n, off_t fo = 0, off_t bo = 0);
void host_open(int d = 0);
void check_idle(int d = 0);
}
