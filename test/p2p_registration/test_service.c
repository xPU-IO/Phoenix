#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define GFP_KERNEL 0
#define THIS_MODULE NULL
#define EXPORT_SYMBOL_GPL(name)
#define phxfs_err(...) do { if (0) fprintf(stderr, __VA_ARGS__); } while (0)
#define phxfs_warn phxfs_err
#define phxfs_info phxfs_err
#define READ_ONCE(x) __atomic_load_n(&(x), __ATOMIC_ACQUIRE)
#define WRITE_ONCE(x, v) __atomic_store_n(&(x), (v), __ATOMIC_RELEASE)
#define xchg(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
typedef atomic_int atomic_t;
typedef atomic_int refcount_t;
#define atomic_set(p, v) atomic_store(p, v)
static int atomic_cmpxchg(atomic_t *p, int old, int value)
{
    atomic_compare_exchange_strong(p, &old, value);
    return old;
}
#define refcount_set(p, v) atomic_store(p, v)
#define refcount_dec_and_test(p) (atomic_fetch_sub(p, 1) == 1)
struct completion { pthread_mutex_t lock; pthread_cond_t cond; int done; };
static void init_completion(struct completion *c)
{
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->cond, NULL);
    c->done = 0;
}
static void complete_all(struct completion *c)
{
    pthread_mutex_lock(&c->lock);
    c->done = 1;
    pthread_cond_broadcast(&c->cond);
    pthread_mutex_unlock(&c->lock);
}
static void wait_for_completion(struct completion *c)
{
    pthread_mutex_lock(&c->lock);
    while (!c->done) pthread_cond_wait(&c->cond, &c->lock);
    pthread_mutex_unlock(&c->lock);
}
static int module_refs, handles, arrays, pins, put_count, frees;
static int fail_handle, fail_array, pin_error, fill_error, null_pt, module_available = 1;
static uint32_t pages = 852640;
static size_t array_bytes;
static void *array_ptr;
static int try_module_get(void *m) { (void)m; if (!module_available) return 0; ++module_refs; return 1; }
static void module_put(void *m) { (void)m; assert(--module_refs >= 0); }
static void *kzalloc(size_t n, int flags)
{
    (void)flags;
    if (fail_handle) return NULL;
    void *p = calloc(1, n);
    assert(p); ++handles; return p;
}
static void kfree(void *p) { if (p) { assert(--handles >= 0); free(p); } }
static void *kvmalloc_array(size_t n, size_t size, int flags)
{
    (void)flags;
    if (fail_array || n > SIZE_MAX / size) return NULL;
    array_bytes = n * size;
    array_ptr = mmap(NULL, array_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array_ptr != MAP_FAILED); ++arrays;
    return array_ptr;
}
static void kvfree(void *p)
{
    if (!p) return;
    assert(p == array_ptr && arrays == 1);
    assert(munmap(p, array_bytes) == 0); --arrays;
}
struct phxfs_page_table { int unused; };
static struct phxfs_page_table table;
static void (*reclaim_cb)(void *);
static void *reclaim_data;
static int mock_get_pages(uint64_t va, uint64_t len, struct phxfs_page_table **pt,
                          void (*cb)(void *), void *data)
{
    (void)va; (void)len;
    if (pin_error) return pin_error;
    if (null_pt) { *pt = NULL; return 0; }
    *pt = &table; reclaim_cb = cb; reclaim_data = data; ++pins;
    return 0;
}
static void mock_put_pages(uint64_t va, struct phxfs_page_table *pt)
{
    (void)va; assert(pt == &table); assert(--pins == 0); ++put_count;
}
static uint32_t mock_n_pages(struct phxfs_page_table *pt) { assert(pt == &table); return pages; }
static int mock_addrs(struct phxfs_page_table *pt, uint64_t *addrs, uint32_t n)
{
    assert(pt == &table);
    if (fill_error) return fill_error;
    for (uint32_t i = 0; i < n; ++i) addrs[i] = 0x100000000ULL + (uint64_t)i * 65536;
    return 0;
}
static void mock_free_table(struct phxfs_page_table *pt)
{
    assert(pt == &table); assert(--pins == 0); ++frees;
}
struct phxfs_p2p_ops {
    int (*get_pages)(uint64_t, uint64_t, struct phxfs_page_table **, void (*)(void *), void *);
    void (*put_pages)(uint64_t, struct phxfs_page_table *);
    uint32_t (*get_n_pages)(struct phxfs_page_table *);
    int (*get_phys_addrs)(struct phxfs_page_table *, uint64_t *, uint32_t);
    void (*free_page_table)(struct phxfs_page_table *);
};
static struct phxfs_p2p_ops ops = {mock_get_pages, mock_put_pages, mock_n_pages, mock_addrs, mock_free_table};
static struct phxfs_p2p_ops *phxfs_p2p = &ops;
#include "service-under-test.c"

static void balanced(void) { assert(!module_refs && !handles && !arrays && !pins); }
static void failure(int expected)
{
    struct phxfs_p2p_handle *h = (void *)1;
    const uint64_t *a = (void *)1;
    uint32_t n = 1;
    assert(phxfs_p2p_register(65536, 55878615040ULL, &h, &a, &n) == expected);
    assert(!h && !a && !n); balanced();
}
int main(void)
{
    struct phxfs_p2p_handle *h;
    const uint64_t *a;
    uint32_t n;
    assert(phxfs_p2p_register(1, 1, NULL, &a, &n) == -EINVAL);
    h = (void *)1; a = (void *)1; n = 1;
    assert(phxfs_p2p_register(0, 1, &h, &a, &n) == -EINVAL);
    assert(!h && !a && !n);
    phxfs_p2p = NULL; failure(-ENODEV); phxfs_p2p = &ops;
    module_available = 0; failure(-ENODEV); module_available = 1;
    fail_handle = 1; failure(-ENOMEM); fail_handle = 0;
    pin_error = -EINVAL; failure(-EINVAL); pin_error = 0;
    null_pt = 1; failure(-ENOMEM); null_pt = 0;
    pages = 0; failure(-EIO); pages = 852640;
    fail_array = 1; failure(-ENOMEM); fail_array = 0;
    fill_error = -EIO; failure(-EIO); fill_error = 0;
    int before_put_count = put_count;
    assert(phxfs_p2p_register(65536, 55878615040ULL, &h, &a, &n) == 0);
    assert(n == 852640 && array_bytes == 6821120 && module_refs == 1);
    for (uint32_t i = 0; i < n; ++i) assert(a[i] == 0x100000000ULL + (uint64_t)i * 65536);
    phxfs_p2p_deregister(h); assert(put_count == before_put_count + 1); balanced();
    assert(phxfs_p2p_register(65536, 55878615040ULL, &h, &a, &n) == 0);
    before_put_count = put_count;
    reclaim_cb(reclaim_data);
    assert(frees == 1 && handles == 1 && arrays == 1 && module_refs == 1);
    assert(a[n - 1] == 0x100000000ULL + (uint64_t)(n - 1) * 65536);
    phxfs_p2p_deregister(h); assert(put_count == before_put_count); balanced();
    phxfs_p2p_deregister(NULL);
    puts("PASS Phoenix service allocation, errors, address order, owner/callback cleanup");
    return 0;
}
