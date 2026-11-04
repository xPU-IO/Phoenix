// SPDX-License-Identifier: Apache-2.0
// bar_holder — hold a write-combining userspace mapping over a GPU's PCI BAR.
//
// Reproduces, without any second kernel module, the field condition behind
// the metax load failures: some tool (or driver path) mmaps the BAR via
// sysfs resource*_wc, which registers a write-combining PAT entry over the
// mapped range. phoenixfs must react to that cleanly: refuse to load
// (whole-BAR coverage), route around it (partial coverage), and never WARN
// or corrupt state.
//
// Usage: bar_holder <bdf> full   <seconds>
//        bar_holder <bdf> window <offMiB> <lenMiB> <seconds>
//   bdf    PCI address as in sysfs, e.g. 0000:06:00.0
//   full   mmap the whole (largest) BAR
//   window mmap [offMiB, offMiB+lenMiB) of it
// Prints "HOLDING ..." once the mapping exists, "RELEASED" on clean exit.
// SIGTERM/SIGINT release the mapping and exit 0. Never touches the mapped
// memory: the PAT entry is created by mmap() itself.
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static volatile sig_atomic_t g_run = 1;
static void on_sig(int sig) { (void)sig; g_run = 0; }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <bdf> full <seconds> | <bdf> window <offMiB> <lenMiB> <seconds>\n",
                argv[0]);
        return 2;
    }
    const char *bdf = argv[1];
    const char *mode = argv[2];

    uint64_t off = 0, len = 0;
    long secs;
    if (strcmp(mode, "full") == 0) {
        if (argc != 4) { fprintf(stderr, "full needs <seconds>\n"); return 2; }
        secs = atol(argv[3]);
    } else if (strcmp(mode, "window") == 0) {
        if (argc != 6) { fprintf(stderr, "window needs <offMiB> <lenMiB> <seconds>\n"); return 2; }
        off = strtoull(argv[3], NULL, 10) * 1024ULL * 1024;
        len = strtoull(argv[4], NULL, 10) * 1024ULL * 1024;
        secs = atol(argv[5]);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 2;
    }

    /* Find the largest BAR and its index from the sysfs resource file. */
    char path[128];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", bdf);
    FILE *r = fopen(path, "r");
    if (!r) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 2; }
    uint64_t best_start = 0, best_size = 0;
    int best_idx = -1, idx = 0;
    unsigned long long s, e;
    unsigned long fl;
    while (fscanf(r, "%llx %llx %lx", &s, &e, &fl) == 3) {
        if (s != 0 && e >= s && e - s + 1 > best_size) {
            best_size = e - s + 1;   /* resource end is inclusive */
            best_start = s;
            best_idx = idx;
        }
        idx++;
    }
    fclose(r);
    if (best_idx < 0) { fprintf(stderr, "no BAR found for %s\n", bdf); return 2; }
    if (len == 0)
        len = best_size;   /* "full": whole BAR, trimmed below if needed */
    if (off >= best_size || len > best_size - off) {
        fprintf(stderr, "window [0x%llx+0x%llx) outside BAR size 0x%llx\n",
                (unsigned long long)off, (unsigned long long)len,
                (unsigned long long)best_size);
        return 2;
    }

    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource%d_wc", bdf, best_idx);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }

    void *p = MAP_FAILED;
    for (;;) {
        p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)off);
        if (p != MAP_FAILED)
            break;
        /*
         * "full" only: some machines already hold a small non-WB PAT entry
         * at the BAR's very end (e.g. the GPU driver's own uncached range);
         * a WC mmap overlapping it fails with EINVAL, so back off 1MiB at a
         * time until the mapping fits. A few MiB of slack still leave every
         * 16MiB probe block conflicted.
         */
        if (strcmp(mode, "full") != 0 || errno != EINVAL ||
            len <= best_size / 2) {
            fprintf(stderr, "mmap %s [0x%llx+0x%llx): %s\n", path,
                    (unsigned long long)off, (unsigned long long)len,
                    strerror(errno));
            close(fd);
            return 1;
        }
        len -= 1ULL << 20;
    }

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    printf("HOLDING bdf=%s bar=%d bar_start=0x%llx off=%lluMiB len=%lluMiB\n",
           bdf, best_idx, (unsigned long long)best_start,
           (unsigned long long)(off >> 20), (unsigned long long)(len >> 20));
    fflush(stdout);

    for (long t = 0; t < secs && g_run; t++)
        sleep(1);

    munmap(p, len);
    close(fd);
    printf("RELEASED\n");
    return 0;
}
