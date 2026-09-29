# HOST deregistration lifetime regression

Builds the production device, mapping, HOST transfer and batch code with ASan.
The default CPU connector replaces only device allocation/copies; the optional
NVIDIA connector runs the same scenarios with real CUDA allocation and copies.
No phxfs module is needed. A 4 KiB `memfd` supplies the data, so these tests do
not write to a filesystem or benchmark a disk.

```sh
cmake -S test/host_dereg -B build-host-dereg
cmake --build build-host-dereg -j
ctest --test-dir build-host-dereg --output-on-failure
```

For a CUDA machine, add `-DPHX_TEST_NVIDIA=ON` and, if necessary,
`-DCUDAToolkit_ROOT=/path/to/cuda`. Add `-DPHX_TEST_URING=ON` to compile and test
both engines (requires liburing). Reserve a GPU using the site's scheduler
before running CUDA tests; preserve its `CUDA_VISIBLE_DEVICES`. Tests use
visible ordinal 0 and reject silent engine fallback.

On the tested L40S/CUDA 13.4 system, ASan required the per-process setting
`ASAN_OPTIONS=protect_shadow_gap=0` for CUDA initialization. Without it, even
a standalone `cudaGetDeviceCount` probe returned `cudaErrorMemoryAllocation`;
the non-ASan probe and the adjusted ASan probe succeeded. This setting was
used for both baseline and fixed CUDA runs; heap-use-after-free detection
remained active (and caught the baseline bug).

Each of six scenarios registers the same region twice, drops the non-final
registration reference, and runs final deregistration on another thread:

- async read, released by wait;
- async read, released by destroy;
- async write, released by wait;
- async read whose device copy finished but handle has not been consumed;
- failed async read (`fd = -1`), released by wait;
- synchronous read, released by transfer completion.

A connector copy gate keeps transfers in flight where needed. A test-only
linker wrapper observes the actual `pthread_cond_wait` in deregistration,
without modifying production code or relying on a sleep. Deregistration must
wait until the reference is released. The suite also checks duplicate reference
handling, results, payload preservation, record removal, re-registration after
completed deregistration and clean close. It has a bounded timeout.

To reproduce the old bug, build the same test directory against commit
`79eac265ac3f02d29673059b48563ffcb916f486`. The first scenario reports an early
deregistration and ASan reports a heap-use-after-free in
`batch_release_mappings`, called by `phxfs_batch_wait`. The registration node
was freed by `phxfs_deregmem` while the handle still held it. ASan stops at the
first failure, so the baseline does not execute all six scenarios.

This is a lifetime regression, not a Spark, BAR/P2P, O_DIRECT, zero-copy or
storage-performance test. It does not test concurrent registration lifecycle
calls for the same region: callers must serialize those calls, just as with
the current FULL implementation. No DRAINING state or rejection of new I/O is
introduced. Async handles must be consumed before same-thread final
deregistration, or by another thread while deregistration waits.
