// test_stream.cu
//
// phxfs_read_stream / phxfs_write_stream — stream-ordered asynchronous
// I/O correctness (host-function model, no stream registration).
//
// THE discriminating test is the "no-drain consume" pattern: submit reads
// on a stream and IMMEDIATELY enqueue a verify kernel on the very same
// stream — no host sync of any kind in between.
// With the host-function model the callback (which runs the DMA) blocks
// everything enqueued after it, so the kernel must see the DMA'd data by
// construction. The earlier event-bridge design failed exactly this
// pattern (the wait-event was enqueued from a background thread and
// could lag the consumer kernel), which is why the model was replaced.
//
// Build: via CMake (make test_stream)
// Run:   ./test_stream [file]     (default TEST_FILE, O_DIRECT-capable fs)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include "phoenix.h"

#define KiB (1024ULL)
#define MiB (1024ULL * 1024)

#define CHUNK   (64 * KiB)   // per-submission transfer
#define NCHUNK  128          // submissions per test round
#define BUF_SZ  (CHUNK * NCHUNK)

static int tests_run = 0, tests_passed = 0, tests_failed = 0, tests_skipped = 0;
#define CHECK(cond, msg, ...) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [PASS] " msg "\n", ##__VA_ARGS__); } \
    else { tests_failed++; printf("  [FAIL] " msg "\n", ##__VA_ARGS__); } \
} while (0)
#define SKIP(msg, ...) do { \
    tests_skipped++; \
    printf("  [SKIP] " msg "\n", ##__VA_ARGS__); \
} while (0)

// Offset-encoding pattern (same idea as test_batch): every 8-byte word at
// file offset o = (chunk << 40) | (o & 0xffffffffff). Swapped/duplicated/
// missing data is detectable.
static inline uint64_t pattern_word(uint64_t off) {
    return ((off / CHUNK) << 40) | (off & 0xffffffffffULL);
}
// Write-round pattern: bit 39 set, so read-round and write-round data are
// distinguishable if a stale buffer or file region leaks through.
static inline uint64_t pattern2_word(uint64_t off) {
    return pattern_word(off) | (1ULL << 39);
}

// Device-side verifier, enqueued on the SAME stream as the reads.
__global__ void verify_words(const uint64_t *buf, uint64_t base_foff,
                             size_t nwords, unsigned int *bad) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) {
        uint64_t off = base_foff + (uint64_t)i * 8;
        uint64_t expect = ((off / CHUNK) << 40) | (off & 0xffffffffffULL);
        if (buf[i] != expect) atomicAdd(bad, 1u);
    }
}

// Device-side fill kernel for the write round (gathers pattern2 into the
// buffer ON THE STREAM, right before the write submissions — the WAR
// case: the write DMA must not read the buffer before this finished).
__global__ void fill_words2(uint64_t *buf, uint64_t base_foff, size_t nwords) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) {
        uint64_t off = base_foff + (uint64_t)i * 8;
        buf[i] = ((off / CHUNK) << 40) | (off & 0xffffffffffULL) | (1ULL << 39);
    }
}

// Host helpers ---------------------------------------------------------

static int ensure_pattern_file(const char *path, size_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) { printf("open(write) %s: %s\n", path, strerror(errno)); return -1; }
    const size_t BUF = 8 * MiB;
    void *tmp = NULL;
    if (posix_memalign(&tmp, 4096, BUF) != 0) { close(fd); return -1; }
    size_t done = 0;
    while (done < size) {
        size_t w = (size - done < BUF) ? (size - done) : BUF;
        uint64_t *wp = (uint64_t *)tmp;
        for (size_t j = 0; j < w / 8; j++) wp[j] = pattern_word(done + j * 8);
        if (pwrite(fd, tmp, w, done) != (ssize_t)w) {
            printf("pwrite: %s\n", strerror(errno)); free(tmp); close(fd); return -1;
        }
        done += w;
    }
    free(tmp); fsync(fd); close(fd);
    return 0;
}

// Verify [0, size) of `path` against pattern2 (write round). Returns
// mismatch count, or SIZE_MAX on I/O error.
static size_t verify_file_pattern2(const char *path, size_t size) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return SIZE_MAX;
    void *tmp = NULL;
    if (posix_memalign(&tmp, 4096, 8 * MiB) != 0) { close(fd); return SIZE_MAX; }
    size_t bad = 0, done = 0;
    while (done < size) {
        size_t w = (size - done < 8 * MiB) ? (size - done) : 8 * MiB;
        if (pread(fd, tmp, w, done) != (ssize_t)w) { bad = SIZE_MAX; break; }
        uint64_t *wp = (uint64_t *)tmp;
        for (size_t j = 0; j < w / 8; j++)
            if (wp[j] != pattern2_word(done + j * 8)) bad++;
        done += w;
    }
    free(tmp); close(fd);
    return bad;
}

static uint64_t *h_pinned(size_t bytes) {
    void *p = NULL;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) return NULL;
    return (uint64_t *)p;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : TEST_FILE;

    printf("== phx stream-ordered I/O test, host-function model (file=%s) ==\n", path);
    printf("io engine: %s\n", phxfs_io_engine_name());

    // ---- environment ----
    int gpu = 0;
    if (cudaGetDeviceCount(&gpu) != cudaSuccess || gpu == 0) {
        SKIP("no CUDA device visible");
        printf("%d run, %d passed, %d failed, %d skipped\n",
               tests_run, tests_passed, tests_failed, tests_skipped + 1);
        return 0;
    }
    int dev = phxfs_find_dev(0);
    if (dev < 0) {
        SKIP("phxfs device not found for CUDA device 0 (module loaded?)");
        printf("%d run, %d passed, %d failed, %d skipped\n",
               tests_run, tests_passed, tests_failed, tests_skipped + 1);
        return 0;
    }
    if (phxfs_open(dev) != 0) {
        SKIP("phxfs_open(%d) failed", dev);
        printf("%d run, %d passed, %d failed, %d skipped\n",
               tests_run, tests_passed, tests_failed, tests_skipped + 1);
        return 0;
    }
    printf("phxfs dev=%d, page=%llu, map_mode=%s\n", dev,
           (unsigned long long)phxfs_get_page_size(),
           phxfs_get_map_mode(dev) == 0 ? "FULL" : "STAGING");
    if (phxfs_get_map_mode(dev) != 0) {
        SKIP("device is in STAGING mode; the stream API supports FULL only "
             "(stream-ordered staging is planned; staging devices use the "
             "synchronous phxfs_read/phxfs_write)");
        printf("%d run, %d passed, %d failed, %d skipped\n",
               tests_run, tests_passed, tests_failed, tests_skipped + 1);
        return 0;
    }

    if (ensure_pattern_file(path, BUF_SZ) != 0) {
        printf("FATAL: cannot prepare %s\n", path);
        return 1;
    }

    // GPU staging buffer + registration (whole 8MB at once; phxfs aligns
    // up to 64KB internally — same as the LMCache GDS tier's region reg).
    uint64_t *gbuf = NULL;
    if (cudaMalloc(&gbuf, BUF_SZ) != cudaSuccess) {
        printf("FATAL: cudaMalloc(%llu) failed\n", (unsigned long long)BUF_SZ);
        return 1;
    }
    void *reg_target = NULL;
    if (phxfs_regmem(dev, gbuf, BUF_SZ, &reg_target) != 0) {
        printf("FATAL: phxfs_regmem failed\n");
        return 1;
    }
    unsigned int *g_bad = NULL;
    cudaMalloc(&g_bad, sizeof(unsigned int));

    // Pinned submission parameter storage (the API takes pointers).
    size_t  *h_nbyte  = h_pinned(sizeof(size_t) * NCHUNK);
    off_t   *h_foff   = (off_t   *)h_pinned(sizeof(off_t) * NCHUNK);
    off_t   *h_boff   = (off_t   *)h_pinned(sizeof(off_t) * NCHUNK);
    ssize_t *h_bdone  = (ssize_t *)h_pinned(sizeof(ssize_t) * NCHUNK);
    if (!h_nbyte || !h_foff || !h_boff || !h_bdone) {
        printf("FATAL: pinned alloc failed\n");
        return 1;
    }

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Shared read fd for the read rounds (O_DIRECT as everywhere).
    int rfd = open(path, O_RDONLY | O_DIRECT);

    // T1: THE no-drain discriminating test -----------------------------
    // Submit 128 reads on s1, then enqueue verify kernels on s1 right
    // away — no drain, no host sync between the submissions and the
    // kernels. Data must be correct in ALL chunks.
    {
        for (int i = 0; i < NCHUNK; i++) {
            h_nbyte[i] = CHUNK;
            h_foff[i]  = (off_t)i * CHUNK;
            h_boff[i]  = (off_t)i * CHUNK;
        }
        cudaMemset(g_bad, 0, sizeof(unsigned int));
        int submitted = 0;
        for (int i = 0; i < NCHUNK; i++) {
            int rc = phxfs_read_stream(rfd,
                                       gbuf, &h_nbyte[i], &h_boff[i],
                                       &h_foff[i], &h_bdone[i], s1);
            if (rc != 0) break;
            submitted++;
        }
        CHECK(submitted == NCHUNK, "T1 submitted %d/%d reads", submitted, NCHUNK);
        // Immediately enqueue consumers — the whole point.
        for (int i = 0; i < NCHUNK; i++) {
            verify_words<<<(CHUNK / 8) / 256, 256, 0, s1>>>(
                gbuf + (size_t)i * CHUNK / 8, (uint64_t)i * CHUNK,
                CHUNK / 8, g_bad);
        }
        cudaError_t sync_rc = cudaStreamSynchronize(s1);
        CHECK(sync_rc == cudaSuccess, "T1 stream synchronize clean");
        unsigned int bad = 1;
        cudaMemcpy(&bad, g_bad, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        CHECK(bad == 0, "T1 NO-DRAIN read->kernel consume: %u mismatches "
                        "(event-bridge design failed here)", bad);
        bool all_ok = true;
        for (int i = 0; i < submitted; i++)
            if (h_bdone[i] != (ssize_t)CHUNK) all_ok = false;
        CHECK(all_ok, "T1 all bytes_done == %llu", (unsigned long long)CHUNK);
    }

    // T2: interleaved submit/consume (read_i, verify_i, read_i+1, ...) --
    // Even tighter: each verify kernel is enqueued between read
    // submissions, so the ordering must hold mid-batch too.
    {
        cudaMemset(gbuf, 0xAB, BUF_SZ);   // poison
        cudaMemset(g_bad, 0, sizeof(unsigned int));
        int submitted = 0;
        for (int i = 0; i < NCHUNK; i++) {
            h_nbyte[i] = CHUNK;
            h_foff[i]  = (off_t)i * CHUNK;
            h_boff[i]  = (off_t)i * CHUNK;
            if (phxfs_read_stream(rfd,
                                  gbuf, &h_nbyte[i], &h_boff[i], &h_foff[i],
                                  &h_bdone[i], s1) == 0)
                submitted++;
            verify_words<<<(CHUNK / 8) / 256, 256, 0, s1>>>(
                gbuf + (size_t)i * CHUNK / 8, (uint64_t)i * CHUNK,
                CHUNK / 8, g_bad);
        }
        CHECK(submitted == NCHUNK, "T2 submitted %d interleaved reads", submitted);
        CHECK(cudaStreamSynchronize(s1) == cudaSuccess, "T2 sync clean");
        unsigned int bad = 1;
        cudaMemcpy(&bad, g_bad, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        CHECK(bad == 0, "T2 interleaved read+verify: %u mismatches", bad);
    }

    // T3: write round with on-stream gather (WAR) ----------------------
    // fill kernel writes pattern2 into the buffer ON s1, then write
    // submissions follow on s1. The DMA must read the filled data.
    {
        for (int i = 0; i < NCHUNK; i++) {
            h_nbyte[i] = CHUNK;
            h_foff[i]  = (off_t)i * CHUNK;
            h_boff[i]  = (off_t)i * CHUNK;
        }
        int wfd = open(path, O_WRONLY | O_DIRECT);
        for (int i = 0; i < NCHUNK; i++)
            fill_words2<<<(CHUNK / 8) / 256, 256, 0, s1>>>(
                gbuf + (size_t)i * CHUNK / 8, (uint64_t)i * CHUNK, CHUNK / 8);
        int submitted = 0;
        for (int i = 0; i < NCHUNK; i++) {
            if (phxfs_write_stream(wfd, gbuf, &h_nbyte[i], &h_boff[i],
                                   &h_foff[i], &h_bdone[i], s1) == 0)
                submitted++;
        }
        CHECK(submitted == NCHUNK, "T3 submitted %d writes", submitted);
        CHECK(cudaStreamSynchronize(s1) == cudaSuccess, "T3 sync clean");
        close(wfd);
        size_t bad = verify_file_pattern2(path, BUF_SZ);
        CHECK(bad == 0, "T3 write-after-gather file content: %zu mismatches", bad);
        bool all_ok = true;
        for (int i = 0; i < submitted; i++)
            if (h_bdone[i] != (ssize_t)CHUNK) all_ok = false;
        CHECK(all_ok, "T3 all bytes_done == CHUNK");
    }

    // T4: bad-fd error injection (read) ---------------------------------
    // A failed submission must report a negative bytes_done and must NOT
    // stall the stream (later work on s1 still runs).
    {
        size_t n = CHUNK; off_t f = 0, b = 0; ssize_t bd = 0;
        int badfd = open("/nonexistent-phx-test", O_RDONLY);
        int rc = phxfs_read_stream(badfd, gbuf, &n, &b, &f, &bd, s1);
        CHECK(rc == 0 || rc < 0, "T4 submission returned (%d)", rc);
        if (rc == 0) {
            // Accepted: callback ran the DMA; it must have failed and the
            // stream must still be alive.
            verify_words<<<1, 1, 0, s1>>>(gbuf, 0, 1, g_bad);   // canary op
            CHECK(cudaStreamSynchronize(s1) == cudaSuccess,
                  "T4 stream not stalled after failed DMA");
            CHECK(bd < 0, "T4 failed read reported bytes_done=%zd", bd);
        } else {
            SKIP("T4 submission-level rejection (rc=%d)", rc);
        }
        if (badfd >= 0) close(badfd);
    }

    // T5: CPU buffer roundtrip ------------------------------------------
    {
        static uint64_t cpu_buf[CHUNK / 8] __attribute__((aligned(4096)));
        size_t n = CHUNK; off_t f = 0, b = 0; ssize_t bd = 0;
        // Re-seed the first TWO chunks with the read pattern (T3 rewrote
        // the whole file with pattern2; T6 reads chunk 1 as well).
        int fd = open(path, O_WRONLY | O_DIRECT);
        void *tmp = NULL; posix_memalign(&tmp, 4096, 2 * CHUNK);
        for (size_t j = 0; j < 2 * CHUNK / 8; j++)
            ((uint64_t *)tmp)[j] = pattern_word(j * 8);
        pwrite(fd, tmp, 2 * CHUNK, 0); close(fd); free(tmp);

        CHECK(phxfs_read_stream(rfd, cpu_buf,
                                &n, &b, &f, &bd, s1) == 0,
              "T5 CPU-buffer read submitted");
        CHECK(cudaStreamSynchronize(s1) == cudaSuccess, "T5 sync clean");
        size_t bad = 0;
        for (size_t j = 0; j < CHUNK / 8; j++)
            if (cpu_buf[j] != pattern_word(j * 8)) bad++;
        CHECK(bd == (ssize_t)CHUNK && bad == 0,
              "T5 CPU-buffer data correct (bd=%zd, bad=%zu)", bd, bad);
    }

    // T6: two streams, concurrent submissions ---------------------------
    // (No registration model: an unregistered stream works on first use.)
    {
        cudaMemset(g_bad, 0, sizeof(unsigned int));
        size_t n1 = CHUNK, n2 = CHUNK;
        off_t f1 = 0, f2 = CHUNK, b1 = 0, b2 = 2 * CHUNK;
        ssize_t bd1 = 0, bd2 = 0;
        int r1 = phxfs_read_stream(rfd,
                                   gbuf, &n1, &b1, &f1, &bd1, s1);
        int r2 = phxfs_read_stream(rfd,
                                   gbuf, &n2, &b2, &f2, &bd2, s2);
        CHECK(r1 == 0 && r2 == 0, "T6 both streams submitted");
        verify_words<<<(CHUNK / 8) / 256, 256, 0, s1>>>(
            gbuf, 0, CHUNK / 8, g_bad);
        verify_words<<<(CHUNK / 8) / 256, 256, 0, s2>>>(
            gbuf + 2 * CHUNK / 8, CHUNK, CHUNK / 8, g_bad);
        CHECK(cudaStreamSynchronize(s1) == cudaSuccess &&
              cudaStreamSynchronize(s2) == cudaSuccess,
              "T6 both streams sync clean");
        unsigned int bad = 1;
        cudaMemcpy(&bad, g_bad, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        CHECK(bad == 0 && bd1 == (ssize_t)CHUNK && bd2 == (ssize_t)CHUNK,
              "T6 concurrent streams correct (bad=%u)", bad);
    }

    // Teardown -----------------------------------------------------------
    phxfs_deregmem(dev, gbuf, BUF_SZ);
    cudaFree(gbuf); cudaFree(g_bad);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFreeHost(h_nbyte); cudaFreeHost(h_foff);
    cudaFreeHost(h_boff); cudaFreeHost(h_bdone);
    close(rfd);
    phxfs_close(dev);

    printf("%d run, %d passed, %d failed, %d skipped\n",
           tests_run, tests_passed, tests_failed, tests_skipped);
    return tests_failed ? 1 : 0;
}
