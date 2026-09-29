// CPU-only connector for testing production HOST lifetime code, not GPU DMA.
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include "connectors/devconnector.h"

static int allocate(size_t n, void **p) {
    return posix_memalign(p, 4096, n) == 0 ? 0 : -ENOMEM;
}
static int device_alloc(int, size_t n, void **p) { return allocate(n, p); }
static int copy(void *d, const void *s, size_t n) { memcpy(d, s, n); return 0; }
static int set_device(int d) { return d == 0 ? 0 : -ENODEV; }
static devconn_ops ops{};
extern "C" {
devconn_ops *devconn = &ops;
int devconn_init() {
    ops.name = "test-cpu";
    ops.page_size = 65536;
    ops.mem_alloc = device_alloc;
    ops.mem_free = free;
    ops.memcpy_dtod = copy;
    ops.host_alloc = allocate;
    ops.host_free = free;
    ops.set_device = set_device;
    ops.memcpy_h2d = copy;
    ops.memcpy_d2h = copy;
    return 0;
}
int phx_connector_probe_alloc(int d, size_t n, void **p) { return device_alloc(d, n, p); }
void phx_connector_probe_free(void *p) { free(p); }
}
