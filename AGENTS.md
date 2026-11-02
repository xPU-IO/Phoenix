# AGENTS.md — Phoenix developer orientation (for AI coding agents)

This file is the **single entry point** for AI agents working in this repository. Read it first; do **not** read the whole tree unless a task requires it. Detailed docs live in `doc/`.

## What this project is

Phoenix is middleware for **direct I/O from storage to xPU (GPU/NPU)** via DMA, bypassing CPU host memory ("phony buffers"). Three layers: a Linux kernel module (`phxfs`), a user-space library (`libphoenix`), and application **adapters** (vLLM via pybind11; official LMCache GDS via the `phxfile` C shim). Multi-vendor support (NVIDIA/AMD/Huawei) is selected at build time via `PHXFS_VENDOR`; only NVIDIA is implemented today, but the backend interfaces are vendor-agnostic.

## Directory map (responsibilities)

| Path | Role |
| --- | --- |
| `module/` | `phxfs` kernel module — GPU BAR remap, single `/dev/phxfs_dev` node with GPU-coded ioctl numbers and GPU-neutral mmap reservations and per-GPU sysfs metadata, `mmap`/`ioctl` P2P mapping; `phxfs-backend.*` + `nvidia-backend.c` select the P2P backend |
| `libphoenix/` | User C/C++ lib — `phxfs_open/close`, `regmem/deregmem`, `read/write`, batch + async batch (`phx_device.cpp` / `phx_mem.cpp` / `phx_io.cpp`), pluggable I/O engines under `io_engine/` (`io_engine_*.cpp`, `io_pool.cpp`); `connectors/` holds the vendor `DevConnector` (`nvidia_connector.cpp`) |
| `adapters/vLLM/phxloader/` | vLLM weight loader (safetensors → GPU DMA) via pybind11, published `phxloader` pkg |
| `adapters/lmcache/phxfile/` | C shim for the official LMCache GDS `phx` backend; stream-ordered I/O |
| `test/` | Correctness + performance tests (`test_regmem`, `test_io`, `test_batch`) |
| `doc/` | All documentation (index: the Documentation table in `README.md`) |

## Build / test / install (reference environment)

```shell
# prerequisites: NVIDIA GDS + MLNX_OFED, kernel source for running kernel, CUDA 12.4, liburing
mkdir -p build && cd build && cmake ../ && make -j    # user library + tests; default vendor: NVIDIA
make modules              # explicitly build kernel module
sudo make insmod          # insert kernel module (run `nvidia-smi` first)
sudo make rmmod           # remove
./bin/test_regmem 0       # memory registration lifecycle tests
./bin/test_io 0           # I/O correctness + performance tests
```
Default `make` builds user space; `make install` installs only the library and headers. Use `make install with_module` to also build and install the kernel module.
Target a different vendor: `cmake -DPHXFS_VENDOR=AMD ../` (requires implementing `module/amd-backend.c` + `libphoenix/connectors/amd_connector.cpp` first).

## Common tasks

**Kernel module work** — edit under `module/`; rebuild with `make modules` in `build/`; `sudo make insmod`. Watch `dmesg` for `phxfs*` messages. If `insmod` fails with "Operation not permitted", the GPU BAR is held by another process/driver (see `doc/troubleshooting.md`). Vendor-specific P2P calls live only in `<vendor>-backend.c`; core files (`phxfs.c`, `phxfs-mem.c`, `phxfs-p2p-service.c`) call through the `phxfs_p2p` function-pointer table and must stay vendor-agnostic.

**Library / API work** — edit `libphoenix/`; adapters consume `libphoenix.so` via pybind11 or the `phxfile` C shim. Vendor-specific calls (CUDA, HIP, ...) live only in `libphoenix/connectors/<vendor>_connector.cpp`; core files (`phx_device.cpp`, `phx_mem.cpp`, `phx_io.cpp`) call through the `devconn` function-pointer table and must stay vendor-agnostic.

**App integration** — vLLM: `adapters/vLLM/phxloader` exposes `PhxLoader` and a `phxsafetensors` load_format. LMCache: its official GDS `phx` backend calls the shim in `adapters/lmcache/phxfile/`, which forwards to `phxfs_read_stream`/`phxfs_write_stream`; GPU stream I/O requires FULL mode.

**Bug reporting** — gather environment info (`uname -r`, `nvidia-smi`, `lsmod | grep phxfs`, relevant `dmesg`), then open an issue using `.github/ISSUE_TEMPLATE/bug_report.md`.

## Gotchas (read before changing core code)

- **BAR exclusivity**: only one driver may own the GPU PCIe BAR. Phoenix must be inserted when no other process/driver uses it.
- **Registration size limit**: `regmem` > 32 GiB still fails (mapping descriptor uses `kmalloc`). Large transfers are chunked at `PHXFS_IO_CHUNK` (1 GiB) for `read`/`write` to stay under `MAX_RW_COUNT`.
- **Async I/O**: batch submit/wait uses the worker pool and the `io_uring` engine (sync fallback). The official LMCache shim uses `phxfs_read_stream/write_stream`, which orders host I/O callbacks on the caller's stream and supports FULL-mode GPU buffers, not STAGING/HOST fallback. The old `phxfs_read_async/write_async` names were removed.
- **Single vendor per machine**: only one `PHXFS_VENDOR` backend is compiled in; mixed-vendor machines are not supported.

## Where to look

- Architecture / data path: `doc/architecture.md`
- Kernel interfaces: `doc/kernel-module.md`
- Library API: `doc/libphoenix.md`
- Adapter guide: `doc/adapters.md`
- Roadmap (eng + research + MCP): `doc/roadmap.md`
