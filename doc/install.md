# Install & Quick Start

## Prerequisites

- **OS**: Linux x86_64
- **Kernel**: a kernel source / module tree matching your running kernel (to build `phxfs`)
- **CUDA**: a CUDA toolkit (its installer provides the matching NVIDIA driver; used by the NVIDIA vendor backend). No `nvidia-fs` / GDS needed
- **Build tools**: CMake ≥ 3.18, a CUDA-capable toolchain, `liburing`

Phoenix is self-contained: it does **not** require GPUDirect Storage (`nvidia-fs`), a specific
filesystem, or any RDMA stack. It issues `O_DIRECT` I/O on whatever fd the application opens.

### Supported accelerators

Phoenix currently builds against the **NVIDIA** vendor backend. The same connector
(`libphoenix/connectors/nvidia_connector.cpp`) is reused on **MetaX (沐曦)** GPUs, because
MetaX's MACA SDK ships a CUDA-compatible runtime — `cudaMalloc`, `cudaMemcpyAsync`,
`cudaLaunchHostFunc`, BDF queries via `cudaDeviceGetPCIBusId`, etc. all work as-is, so
the connector needs no MetaX-specific code path.

To run on a MetaX card:

1. Install the MACA driver and MACA SDK from
   <https://developer.metax-tech.com/softnova> (download the **Driver** and **SDK** packages).
2. Set the environment before building / running:
   ```shell
   export MACA_PATH=/opt/maca
   export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
   ```
3. Build Phoenix with the default `NVIDIA` vendor (no `-DPHXFS_VENDOR` change needed):
   ```shell
   mkdir -p build && cd build && cmake ../ && make -j
   ```

The rest of the install steps (kernel module load, tests) are unchanged.

## 1. Storage backend

Phoenix is storage-agnostic — it works on any storage device / filesystem that supports
direct I/O: a local NVMe SSD (ext4/xfs, e.g. mounted at `/mnt/nvme0`), an NVMe-oF or
RDMA-backed target, a parallel filesystem, etc. The only unsupported case is **FUSE**
(its direct-I/O path cannot carry the P2P DMA). Just point Phoenix at the data file;
nothing else to set up.

## 2. Build Phoenix

```shell
mkdir -p build && cd build
cmake ../
make -j
```
This compiles the user library, the `phxfs` kernel module, and the tests.

- Skip the kernel module: `cmake -Dno_module=true ../`
- Target a different accelerator vendor: `cmake -DPHXFS_VENDOR=AMD ../` (default `NVIDIA`).

## 3. Install

### Install to system paths (`make install`)

Installs `libphoenix.so` → `<prefix>/lib`, the headers → `<prefix>/include` (default prefix
`/usr/local`), and the kernel module → `/lib/modules/$(uname -r)/extra` + `depmod`:

```shell
cd build
sudo make install
```

This installs (copies) the kernel module; it does **not** load it. If the module install fails
(e.g. the running kernel has no writable module tree), `make install` prints the reason and
still installs the library.

### Load the kernel module

Run `nvidia-smi` first (loads the NVIDIA driver), then:

```shell
cd build
sudo make insmod          # loads phoenixfs.ko with the build's default mode (STAGING)
```

If installation fails, see [troubleshooting.md](troubleshooting.md).

### BAR mapping mode (STAGING default vs FULL)

`phxfs` has two BAR mapping modes. The **default is STAGING**; FULL must be opted into.

- **STAGING (`phxfs_map_mode=1`, default)** — remaps only a small Phoenix-owned staging pool;
  user GPU memory is left unmapped so it stays registerable by RDMA/peermem (resolves the
  BAR-remap / `dma_map_resource` conflict). Data flows SSD → staging pool → D2D copy → user
  buffer. The safe, coexistence-friendly default.
- **FULL (`phxfs_map_mode=0`)** — remaps the whole GPU BAR at load; registered user GPU
  buffers DMA directly (no D2D hop). Maximum performance, but prevents peermem/RDMA from
  registering GPU memory on the same GPU. Use on nodes without GPUDirect-RDMA.

The default mode is chosen at build time and baked into the `.ko`; `make insmod` loads it as-is:

```shell
cmake -DPHXFS_MAP_MODE=full ../      # build a FULL-default module (default is staging)
```

You can also override at load time without rebuilding, via named (TAB-completable) targets:

```shell
sudo make insmod-full          # force FULL for this load
sudo make insmod-staging       # force STAGING (the default anyway)
```

To switch modes, stop GPU processes using `phxfs`, `sudo make rmmod`, then load the other
mode. Check the active mode (`0` = FULL, `1` = STAGING):

```shell
cat /sys/module/phoenixfs/parameters/phxfs_map_mode
```

The user library reads the mode from the kernel at `phxfs_open()` and adapts automatically —
the same `libphoenix.so` and the same application code work with either mode.

## 4. Run tests

The test binaries double as end-to-end examples of the library API:

```shell
cd build
./bin/test_regmem 0    # memory registration lifecycle
./bin/test_io 0        # I/O correctness + performance
./bin/test_batch       # batch I/O correctness + bandwidth
```

For vLLM weight loading, install the adapter and set `--load-format phxsafetensors` (see
[adapters.md](adapters.md)).
