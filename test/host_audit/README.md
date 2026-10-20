# HOST and staging correctness checks

These tests compile the production userspace core with CPU allocations, memfd
files and controlled connector/syscall substitutes. They do not validate CUDA,
HIP, BAR/P2P, storage DMA, Spark or GPU stream ordering. No module, accelerator
or block device is used. Run CTest serially to bound RAM.

```sh
cmake -S test/host_audit -B build/host-audit -DAUDIT_ASAN=ON
cmake --build build/host-audit -j2
ctest --test-dir build/host-audit --output-on-failure
```

CTest discovers the case_*.cpp files present in this checkout. Each focused
patch includes its own correctness checks and relevant adjacent controls.
They use the sync engine. AddressSanitizer and UndefinedBehaviorSanitizer
instrument both production code and the substitutes.

Use -DPHX_ROOT=/absolute/path/to/another/checkout to run the identical checks
against a different production revision. AUDIT_LEGACY_PROBE=OFF disables the
old NVIDIA-only global helper symbols, checking that core linking needs only
the connector operation table. Older revisions need the default ON setting
for behavior checks; their OFF build is expected to fail at link time.

Lifecycle checks use condition-variable barriers. Stream checks, when present,
use an ordered CPU callback/copy queue and do not model CUDA enqueue failures.
Worker-local HOST buffers remain cached until pool shutdown; device-owned pool
buffers and registration references follow the device/request lifetime.

UndefinedBehaviorSanitizer diagnostics are fatal when AUDIT_ASAN is enabled:
-fno-sanitize-recover=undefined is a PUBLIC compile option of audit_core and
is inherited by the production core, CPU substitute and every test executable.
