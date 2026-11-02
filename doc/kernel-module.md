# Kernel Module — `phxfs`

The `phxfs` kernel module is the core of Phoenix. It remaps GPU PCIe BAR memory via `ZONE_DEVICE` and exposes a single character device, `/dev/phxfs_dev`, for all GPUs. GPU routing is stateless at the file level: `mmap` reserves an unbound CPU address range, and MAP/UNMAP encode the GPU in the ioctl number. The first MAP attempt passing basic argument checks fixes that VMA's GPU until teardown. Numbered `/sys/class/phxfs-generic/phxfs_devN` directories contain discovery metadata only; no numbered character nodes are created.

## Multi-vendor P2P backend

Vendor-specific P2P calls (pin/unpin device pages, get physical addresses) are abstracted behind `struct phxfs_p2p_ops` (`phxfs-backend.h`). The active backend is selected at **compile time** via `PHXFS_VENDOR` (default `NVIDIA`) and exposed through the global `phxfs_p2p` pointer. Core files (`phxfs.c`, `phxfs-mem.c`, `phxfs-p2p-service.c`) call only through `phxfs_p2p->get_pages()` / `put_pages()` / `page_size` and never reference vendor APIs directly.

`module/nvidia-backend.c` implements the NVIDIA backend (loads `nvidia_p2p_*` kernel symbols via `__symbol_get`). Adding a new vendor means writing `<vendor>-backend.c` and registering it in `phxfs-backend.c`; no other kernel file needs to change.

Only one vendor backend is compiled in per build — mixed-vendor machines are not supported.

## Initialization

On `insmod`, `phxfs_init` performs:

1. Initialize the compiled-in P2P backend (`phxfs_p2p_backend_init`).
2. Discover accelerator devices by PCI vendor ID and build the device table (`phxfs_discover_devices`, populates `gpu_info_table`, `npu_num`).
3. For each device, discover its PCIe BAR and `devm_memremap` the BAR into kernel space (`phxfs_ctrl_init`).
4. Create sysfs-only GPU metadata devices, then register the single misc character device (`phxfs_devices_register`).

## Uninitialization

On `rmmod`, `phxfs_exit` deregisters the unified node, unregisters GPU metadata devices, unmaps BAR memory, and releases the P2P backend. Open files and their VMAs keep the module referenced until they are closed.

## Character device interface

Three base operations are exposed to user space:

### `open`
Opens `/dev/phxfs_dev` without GPU state. One fd, including its dup/fork aliases, can serve multiple GPUs.

### `mmap`
Requires offset 0 (`-EINVAL` otherwise). It reserves a CPU virtual address range and creates an unbound descriptor with the existing `VM_MIXEDMAP`, non-cached flags. It neither selects a GPU nor pins GPU pages or allocates payload RAM.

### `ioctl`
The standard 8-bit ioctl number is `(gpu_index << 4) | operation`: GPU index in the high 4 bits, MAP=1/UNMAP=2 in the low 4. Direction, type and argument-size fields are unchanged. This supports 16 GPU indices; malformed commands return `-ENOTTY`, missing GPUs `-ENODEV`, and bad argument storage `-EFAULT`.

- `PHXFS_IOCTL_MAP_GPU(index)`: the first attempt passing basic address/length checks binds the descriptor under its existing lock. This binding survives clean failures; same-GPU retries are allowed until pages have been inserted. Successful or partially inserted mappings remain consumed, so another MAP requires a new VMA. A different GPU command returns `-EINVAL`.
- `PHXFS_IOCTL_UNMAP_GPU(index)`: release the pin only when the descriptor is bound to this GPU; unbound or wrong-GPU requests are no-ops. It never clears the binding or drops the staging span while BAR PTEs remain. Final VMA teardown releases the remaining resources.
- `PHXFS_IOCTL_MAP` / `PHXFS_IOCTL_UNMAP` are the GPU 0 encodings. Parameter structures retain their layout; there is no separate selection ioctl.

## Build & install

See [install.md](install.md) and [troubleshooting.md](troubleshooting.md).
