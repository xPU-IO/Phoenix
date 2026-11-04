<!-- SPDX-License-Identifier: Apache-2.0 -->
# Adapters

Adapters connect Phoenix to AI applications through `libphoenix`: vLLM uses `pybind11` bindings, while the official LMCache GDS backend uses the `libphxfile.so` C shim via ctypes.

## vLLM — `phxloader` (available)

`adapters/vLLM/phxloader` provides GPU Direct Storage loading of `safetensors` model weights via Phoenix's DMA engine. It parses the safetensors header, plans reads into read-groups, registers a shared GPU buffer, and DMAs weights straight into GPU memory — bypassing CPU host memory.

**Current release: V2.2**

### Version evolution

- **V1 — contiguous synchronous DMA**: single contiguous DMA of the whole data section; one GPU buffer per file; synchronous I/O. (Historical; code preserved on the `tencent-backup` branch.)
- **V2 — batch DMA + shared buffer**: splits each file into read-groups (adjacent tensors with gap < 64 KiB merged), reuses one shared GPU buffer via `regmem`/`deregmem`, one `phxfs_read` per group. (Historical; code preserved on the `tencent-backup` branch.)
- **V2.1 — double buffer + async DMA**: dual GPU buffers (bufA/bufB) with background DMA on a C++ thread, overlapping `yield(copy_)` to hide yield latency; adds a DMA timer.
- **V2.2 (current) — official release + API rename**: functionally identical to V2.1; API renamed for clarity (`read_into_registered` → `load_tensors_into_buffer`, `read_into_registered_async` → `load_tensors_into_buffer_async`, class `PhxLoaderV2` → `PhxLoader`); hardened safetensors header parsing. Published as the `phxloader` package (no version suffix).

### Install

```bash
# Prerequisites: conda env active, libphoenix built (build/libphoenix.so)
cd adapters/vLLM/phxloader
bash install.sh
```

### Use

```bash
# vLLM launch flag
--load-format phxsafetensors
```
```python
from vllm import LLM
llm = LLM(model="...", load_format="phxsafetensors")
```

### API

```python
from phxloader import PhxLoader

loader = PhxLoader(device_id=0)
loader.regmem(gpu_ptr, size)
loader.load_tensors_into_buffer(path, gpu_ptr, batch)        # synchronous
loader.load_tensors_into_buffer_async(path, gpu_ptr, batch)  # background DMA
loader.wait_dma()
loader.reset_dma_timer(); loader.get_dma_seconds()
loader.deregmem(gpu_ptr, size)
loader.close()
```

### Dependencies

`libphoenix`, `liburing`, CUDA, `pybind11`, PyTorch.

## LMCache — official GDS `phx` backend

The official LMCache backend in `lmcache/v1/gpu_connector/gds_backends/phx.py` calls `libphxfile.so` through ctypes. The shim forwards `phxFileReadAsync` / `phxFileWriteAsync` to `phxfs_read_stream` / `phxfs_write_stream`, preserving stream ordering without a Python extension.

Build and ABI details: [`adapters/lmcache/phxfile/README.md`](../adapters/lmcache/phxfile/README.md). GPU I/O requires FULL mapping mode; STAGING and HOST fallback are not supported by the current stream path.
