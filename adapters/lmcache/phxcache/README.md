# lmcache adapter

This directory hosts the Phoenix adapter for [lmcache](https://github.com/LMCache/LMCache), enabling KV-cache offload/loading acceleration via direct storage→GPU DMA.

Environment variable `PHXFS_VENDOR` should be set as NVIDIA (default) or METAX ... for specific xPU vendor backend.

**Status: planned — not yet implemented.** See [docs/phoenix_lmcache_adapter_plan.md](../../../docs/phoenix_lmcache_adapter_plan.md) for the full design.

## Design: Asymmetric Phoenix GDS

The adapter uses an **asymmetric** I/O model (not symmetric GPU direct read/write):

| Operation | Path | I/O Mechanism |
|-----------|------|---------------|
| **Store** | CPU MemoryObj → Phoenix storage | POSIX write (data already in CPU via D2H) |
| **Retrieve** | Phoenix storage → GPU MemoryObj | `phxfs_read_async` DMA (bypasses CPU) |

Store reuses the CPU MemoryObj allocated by `LocalCPUBackend` (via `get_allocator_backend()` returning `local_cpu_backend`), avoiding redundant H2D + GPU DMA. Retrieve uses `phxfs_read_async` on a dedicated CUDA stream for batched DMA directly to GPU memory.

This mirrors the optimized direction for `GdsBackend` described in `docs/gds_backend_update.md`.

## Components

- `phxcache/` — pybind11 C++ extension wrapping the Phoenix C API (`phxfs_read`/`phxfs_read_async`/`phxfs_regmem`/etc.), following the same build pattern as `adapters/vllm/phxloader`
- `LMCache/lmcache/v1/storage_backend/phx_backend.py` — `PhxBackend` (asymmetric store/retrieve)
- `LMCache/lmcache/v1/memory_allocators/phx_file_memory_allocator.py` — `PhxFileMemoryAllocator` (GPU memory + `phxfs_regmem`)
