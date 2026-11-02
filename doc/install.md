# Install & Quick Start

## Prerequisites

- **OS**: Linux x86_64
- **Kernel**: a kernel source / module tree matching your running kernel (to build `phxfs`);
- **Accelerator runtime**: a vendor runtime supported by Phoenix (see
  [Supported accelerators](#supported-accelerators) below). Phoenix does **not** require any
  vendor-specific direct-storage plugin (e.g. NVIDIA `nvidia-fs` / GPUDirect Storage), a specific
  filesystem, or any RDMA stack.
- **Build tools**: CMake ≥ 3.18, the compiler toolchain for your chosen vendor, `liburing`

### Kernel boot parameters

The host must boot with the kernel command-line parameters below — the same requirements as
NVIDIA GDS. The IOMMU parameters all serve one contract: P2P DMA assumes `phys == bus address`
(see [vendor-porting-guide.md §2](vendor-porting-guide.md)), i.e. an identity-mapped or disabled
IOMMU. Missing parameters can make the module fail to load or the DMA path misbehave, and the
failure shows up in `dmesg` as a low-level remap / DMA error that is hard to trace back to boot
parameters. Configure them before deployment.

| Parameter | Required on | Reason |
| --- | --- | --- |
| `iommu=pt` | Intel CPUs (or disable VT-d in BIOS) | Puts the Intel IOMMU into passthrough (identity) mode, so `phys == bus address` |
| `amd_iommu=off` | AMD CPUs, e.g. EPYC (or disable AMD-Vi in BIOS) | Unlike Intel, the AMD IOMMU is enabled by default, and `iommu=pt` is not reliably honored by `amd_iommu` — disable the IOMMU entirely, as GDS recommends for AMD platforms |
| `nokaslr` | Kernel ≥ 6.8 | KASLR conflicts with the ZONE_DEVICE BAR remap in `phxfs` (`devm_memremap_pages` builds struct pages over the GPU BAR); kernel address randomization must be disabled |


Phoenix is self-contained: it works at the VFS / block layer and only needs a valid `fd`. It
issues `O_DIRECT` I/O on whatever fd the application opens — local ext4 / xfs or a parallel
filesystem (BeeGFS / Lustre) are transparent, no direct-storage support-list required.

### Supported accelerators

Phoenix is vendor-agnostic by construction: a single compile-time switch, `PHXFS_VENDOR`,
selects the accelerator backend, and the same upper-layer / adapter code runs on any supported
vendor. Core code never references vendor APIs directly — it calls through a function-pointer
table (`phxfs_p2p` in the kernel, `devconn` in the user library).

| Vendor | `PHXFS_VENDOR` | Runtime requirement | Status |
| --- | --- | --- | --- |
| NVIDIA | `NVIDIA` (default) | CUDA Toolkit 12.4+ + NVIDIA driver | ✅ Shipped |
| MetaX (沐曦) | `METAX` | MACA SDK (CUDA-compatible) | ✅ Shipped (reuses the CUDA connector) |

 AMD/Huawei NPU/Moore Threads (摩尔线程) is comming soon.

Vendor-specific setup steps for each shipped backend are below; the common build procedure is
in [§2 Build Phoenix](#2-build-phoenix).

#### Running on NVIDIA

1. Install the NVIDIA driver and the CUDA Toolkit 12.4 or newer
   (the CUDA installer ships the matching driver).
2. Confirm the driver is loaded:
   ```shell
   nvidia-smi
   ```
3. Build Phoenix with the default `NVIDIA` vendor (no `-DPHXFS_VENDOR` flag needed) — see
   [§2 Build Phoenix](#2-build-phoenix).

#### Running on MetaX (沐曦)

MetaX's MACA SDK ships a CUDA-compatible runtime — `cudaMalloc`, `cudaMemcpyAsync`,
`cudaLaunchHostFunc`, BDF queries via `cudaDeviceGetPCIBusId`, etc. all work as-is, so
Phoenix reuses the same CUDA connector for MetaX with no MetaX-specific code path.

To build / run on a MetaX card:

1. Install the MACA driver and MACA SDK from
   <https://developer.metax-tech.com/softnova> (download the **Driver** and **SDK** packages).
2. Set the environment before building / running:
   ```shell
   export MACA_PATH=/opt/maca
   export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
   ```
3. Build Phoenix with `-DPHXFS_VENDOR=METAX` — see [§2 Build Phoenix](#2-build-phoenix).

## 1. Storage backend

Phoenix is storage-agnostic — it works on any storage device / filesystem that supports
direct I/O: a local NVMe SSD (ext4/xfs and anyother storage type, e.g. Raid/Loop), an NVMe-oF or
RDMA-backed target, a parallel filesystem, etc. The only unsupported case is **FUSE**
(its direct-I/O path cannot carry the P2P DMA). Just point Phoenix at the data file;
nothing else to set up.

NVMe-oF/RDMA-backed parallel filesystem is comming soon.

## 2. Build Phoenix

```shell
mkdir -p build && cd build
cmake -DPHXFS_VENDOR=<vendor> ../      # see the options below; default is NVIDIA
make -j
```
This compiles the user library and tests. The kernel module is built only when
explicitly requested with `make modules` or `make install with_module`.

Vendor selection (`-DPHXFS_VENDOR=<value>`):

```shell
cmake -DPHXFS_VENDOR=NVIDIA ../       # default; CUDA Toolkit 12.4+ + NVIDIA driver
cmake -DPHXFS_VENDOR=METAX ../        # MetaX MACA (CUDA-compatible, reuses NVIDIA connector)
cmake -DPHXFS_VENDOR=AMD ../          # requires module/amd-backend.c + libphoenix/connectors/amd_connector.cpp
cmake -DPHXFS_VENDOR=MTHREADS ../     # requires module/mthreads-backend.c + libphoenix/connectors/mthreads_connector.cpp
cmake -DPHXFS_VENDOR=HUAWEI ../        # requires module/huawei-backend.c + libphoenix/connectors/huawei_connector.cpp
```

Other build options:

- Switch the default BAR mapping mode: `cmake -DPHXFS_MAP_MODE=full ../` (default `staging`; see [below](#bar-mapping-mode-staging-default-vs-full)).

## 3. Install

### Library installation (`make install`)

```shell
cd build
sudo make install
```

This builds user space and installs only `libphoenix` into `<prefix>/lib` and
headers into `<prefix>/include` (default prefix `/usr/local`). It never builds,
installs, or loads the kernel module, runs `depmod`, or refreshes the PAT snapshot.
No module-related configuration option is needed, including inside containers.

### Including the kernel module (`make install with_module`)

```shell
cd build
sudo make install with_module
```

The additional `with_module` target builds and installs the kernel module into
`/lib/modules/$(uname -r)/extra`, refreshes `/run/phxfs/pat_memtype_list` when
available, and runs `depmod`. It does **not** load the module. A requested
module build/install failure makes the command fail; the library installation
is not rolled back.

The existing module targets remain available: `make modules` builds only the
module; `make clean_modules` cleans it; `make insmod`, `make insmod-full`,
`make insmod-staging`, and `make rmmod` control loading and unloading.

### Load the kernel module

Load the vendor accelerator driver first (e.g. `nvidia-smi` on NVIDIA, or the equivalent
command from your vendor's runtime), then:

```shell
cd build
make modules             # explicit build; unnecessary if with_module already built it
sudo make insmod          # loads phoenixfs.ko with the build's default mode (STAGING)
```

All `make insmod*` targets refresh `/run/phxfs/pat_memtype_list` immediately before loading.
A direct module load without that snapshot remains fail-open and logs a warning directing the
user to reload through `make insmod*`. If debugfs is unavailable during refresh, any stale
snapshot is removed.

If installation fails, see [troubleshooting.md](troubleshooting.md).

### BAR mapping mode (STAGING default vs FULL)

`phxfs` has two BAR mapping modes. The **default is STAGING**; FULL must be opted into.

- **STAGING (`phxfs_map_mode=1`, default)** — remaps only a small Phoenix-owned staging pool;
  user GPU memory is left unmapped so it stays registerable by RDMA / peer-memory frameworks
  (e.g. NVIDIA `nvidia-peermem`, `ibv_reg_mr`) on the same GPU. Data flows SSD → staging pool
  → D2D copy → user buffer. The safe, coexistence-friendly default.
- **FULL (`phxfs_map_mode=0`)** — remaps the whole GPU BAR at load; registered user GPU
  buffers DMA directly (no D2D hop). Maximum performance, but prevents peer-memory / RDMA
  frameworks from registering GPU memory on the same GPU. Use on nodes without peer-memory / RDMA.

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

### Containers (Docker)

Load `phoenixfs` on the **host**; install only `libphoenix` and the application's
adapter inside the image. NVIDIA containers also need NVIDIA Container Toolkit.
All GPUs use the single `/dev/phxfs_dev` node (add storage mounts as needed):

```shell
docker run --rm -it --gpus all \
  --device /dev/phxfs_dev:/dev/phxfs_dev:rw --ulimit memlock=-1 \
  --mount type=bind,src=/sys/class/phxfs-generic,dst=/sys/class/phxfs-generic,readonly \
  --mount type=bind,src=/sys/devices/virtual/phxfs-generic,dst=/sys/devices/virtual/phxfs-generic,readonly \
  IMAGE COMMAND
```

`/sys/class/phxfs-generic` exposes GPU metadata links; the read-only
`/sys/devices/virtual/phxfs-generic` mount makes their targets accessible. These
are current metadata paths, not additional character devices. Verify `pci_bdf`,
`map_mode`, and `page_size` are readable inside the container.
The library encodes GPU indices internally; vLLM/LMCache APIs and configuration
do not change. Deploy the updated module, library and LMCache shim together.
Granting the shared node permits GPU selection across the module's device table;
`--gpus`/`CUDA_VISIBLE_DEVICES` are not a per-GPU Phoenix authorization boundary.

Keep default capabilities, seccomp, and SELinux policies; do not add broad
capabilities or disable isolation unless the vendor/application/host policy
requires it. If seccomp denies `io_uring`, Phoenix falls back to sync (or set
`-e PHXFS_IO_ENGINE=sync`). `--ipc=host` is application-specific; use
`--shm-size` if private shared memory suffices. `/sys/module/phoenixfs` is only
needed for diagnostics. Rootless/user-namespace setups need host-specific device
permissions.

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
