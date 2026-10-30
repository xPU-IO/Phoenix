#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GFP_KERNEL 0
#define GPU_PAGE_SIZE 65536UL
#define KERN_ERR ""
#define KERN_CRIT ""
#define KERN_WARNING ""
#define printk(...) do { if (0) fprintf(stderr, __VA_ARGS__); } while (0)
#define kmalloc(n, flags) malloc(n)
#define kfree(p) free(p)
#define container_of(p, type, member) ((type *)((char *)(p) - offsetof(type, member)))
struct list_node { int unused; };
struct list { struct list_node head; };
struct ctrl { struct list_node list; void *pdev; };
struct peer_page_table { int unused; };
struct peer_dma_mapping { int unused; };
struct phxfs_p2p_handle;
struct gpu_region { struct peer_page_table *pages; struct peer_dma_mapping **mappings; void *phx_handle; };
struct map { uint64_t vaddr; unsigned long n_addrs, page_size; void *pdev, *data; struct list *ctrl_list;
             void (*release)(struct map *); uint64_t addrs[4]; };
static uint32_t max_num_ctrls = 1;
static int native_gets, native_maps, native_puts, native_unmaps, deregisters;
static int response_error, with_handle = 1, with_addrs = 1;
static uint32_t response_pages = 3;
static uint64_t addresses[4] = {65536, 131072, 196608, 262144};
static struct ctrl controller;
static struct peer_page_table native_pt;
static struct peer_dma_mapping native_mapping;
static const struct list_node *list_next(const struct list_node *n)
{
    return n == &controller.list ? NULL : &controller.list;
}
static int mock_native_get(uint64_t token, uint32_t space, uint64_t va, uint64_t len,
                           struct peer_page_table **pt, void (*cb)(void *), void *data)
{
    (void)token; (void)space; (void)va; (void)len; (void)cb; (void)data;
    ++native_gets; *pt = &native_pt; return 0;
}
static int mock_native_map(void *dev, struct peer_page_table *pt, struct peer_dma_mapping **mapping,
                           unsigned long page_size)
{
    (void)dev; assert(pt == &native_pt && page_size == GPU_PAGE_SIZE);
    ++native_maps; *mapping = &native_mapping; return 0;
}
static void mock_native_unmap(void *dev, struct peer_page_table *pt, struct peer_dma_mapping *mapping)
{
    (void)dev; assert(pt == &native_pt && mapping == &native_mapping); ++native_unmaps;
}
static void mock_native_put(uint64_t token, uint32_t space, uint64_t va, struct peer_page_table *pt)
{
    (void)token; (void)space; (void)va; assert(pt == &native_pt); ++native_puts;
}
static struct {
    int (*get_pages)(uint64_t, uint32_t, uint64_t, uint64_t, struct peer_page_table **, void (*)(void *), void *);
    int (*dma_map_pages)(void *, struct peer_page_table *, struct peer_dma_mapping **, unsigned long);
    void (*dma_unmap_pages)(void *, struct peer_page_table *, struct peer_dma_mapping *);
    void (*put_pages)(uint64_t, uint32_t, uint64_t, struct peer_page_table *);
} peer_memory_ops = {mock_native_get, mock_native_map, mock_native_unmap, mock_native_put};
static uint64_t *peer_memory_dm_addresses(struct peer_dma_mapping *m) { assert(m == &native_mapping); return addresses; }
static uint32_t peer_memory_pt_entries(struct peer_page_table *pt) { assert(pt == &native_pt); return 3; }
static void force_release_gpu_memory(struct map *m) { (void)m; abort(); }
static int mock_register(uint64_t va, uint64_t length, struct phxfs_p2p_handle **h,
                         const uint64_t **a, uint32_t *n)
{
    assert(va == 65536 && length == 3 * GPU_PAGE_SIZE);
    *h = with_handle ? (void *)1 : NULL;
    *a = with_addrs ? addresses : NULL;
    *n = response_pages;
    return response_error;
}
static void mock_deregister(struct phxfs_p2p_handle *h) { assert(h == (void *)1); ++deregisters; }
static int (*phx_register_fn)(uint64_t, uint64_t, struct phxfs_p2p_handle **, const uint64_t **, uint32_t *) = mock_register;
static void (*phx_deregister_fn)(struct phxfs_p2p_handle *) = mock_deregister;
#include "map-under-test.c"

static void run(int expected)
{
    struct list list = {};
    struct map m = {.vaddr = 65536, .n_addrs = 3, .ctrl_list = &list,
                    .addrs = {7, 7, 7, 7}};
    int before = deregisters;
    assert(map_gpu_memory(&m, &list) == expected);
    assert(!native_gets && !native_maps && m.n_addrs == 3);
    if (!expected) {
        for (int i = 0; i < 3; ++i) assert(m.addrs[i] == addresses[i]);
    } else {
        for (int i = 0; i < 3; ++i) assert(m.addrs[i] == 7);
    }
    assert(m.addrs[3] == 7);
    m.release(&m);
    assert(!m.data && deregisters == before + with_handle);
    m.release(&m);
    assert(deregisters == before + with_handle && !native_puts && !native_unmaps);
}
int main(void)
{
    run(0);
    int errors[] = {-ENOMEM, -EINVAL, -EIO, -ENODEV};
    for (unsigned i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i) {
        response_error = errors[i]; run(errors[i]);
    }
    with_handle = 0; run(-ENODEV);
    response_error = 0; run(-EIO); with_handle = 1;
    with_addrs = 0; run(-EIO); with_addrs = 1;
    uint32_t invalid[] = {0, 2, 4};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        response_pages = invalid[i]; run(-EIO);
    }
    phx_register_fn = NULL;
    struct list list = {};
    struct map m = {.vaddr = 65536, .n_addrs = 3, .ctrl_list = &list};
    assert(map_gpu_memory(&m, &list) == 0);
    assert(native_gets == 1 && native_maps == 1 && m.addrs[2] == addresses[2]);
    m.release(&m);
    assert(!m.data && native_puts == 1 && native_unmaps == 1);
    puts("PASS snvme fail-closed, exact page count, cleanup and absent-service native path");
    return 0;
}
