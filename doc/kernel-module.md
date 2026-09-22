# Kernel Module — `phxfs`

The `phxfs` kernel module is the core of Phoenix. It remaps each GPU device's PCIe BAR memory into kernel space via `ZONE_DEVICE`, creates one character device per GPU (`/dev/phxfs_devN`), and serves `mmap` / `ioctl` requests that establish peer-to-peer (P2P) mappings between GPU memory and a user-space VMA.

## Multi-vendor P2P backend

Vendor-specific P2P calls (pin/unpin device pages, get physical addresses) are abstracted behind `struct phxfs_p2p_ops` (`phxfs-backend.h`). The active backend is selected at **compile time** via `PHXFS_VENDOR` (default `NVIDIA`) and exposed through the global `phxfs_p2p` pointer. Core files (`phxfs.c`, `phxfs-mem.c`, `phxfs-p2p-service.c`) call only through `phxfs_p2p->get_pages()` / `put_pages()` / `page_size` and never reference vendor APIs directly.

`module/nvidia-backend.c` implements the NVIDIA backend (loads `nvidia_p2p_*` kernel symbols via `__symbol_get`). Adding a new vendor means writing `<vendor>-backend.c` and registering it in `phxfs-backend.c`; no other kernel file needs to change.

Only one vendor backend is compiled in per build — mixed-vendor machines are not supported.

## Initialization

On `insmod`, `phxfs_init` performs:

1. Initialize the compiled-in P2P backend (`phxfs_p2p_backend_init`).
2. Discover accelerator devices by PCI vendor ID and build the device table (`phxfs_discover_devices`, populates `gpu_info_table`, `npu_num`).
3. For each device, discover its PCIe BAR and `devm_memremap` the BAR into kernel space (`phxfs_ctrl_init`).
4. Create a character device per device (`phxfs_cdev_init`).

## Uninitialization

On `rmmod`, `phxfs_exit` deletes the char devices, unmaps the BAR memory, releases the P2P backend (`phxfs_p2p_backend_exit`), and destroys the device class / char-device region.

## Character device interface

Three base operations are exposed to user space:

### `open`
Saves the device metadata for `deviceID` into `file->private_data`.

### `mmap`
Sets VMA flags (`VM_MIXEDMAP`, non-cached) and creates the buffer descriptor owned by the VMA via `phxfs_add_phony_buffer`. The mapping is lazy — it does not pin GPU memory yet.

### `ioctl`
- `PHXFS_IOCTL_MAP`: map a device (GPU) address into the user-space VMA. Internally `phxfs_map_dev_addr` → `phxfs_map_dev_addr_inner` calls `phxfs_p2p->get_pages()` (vendor backend) to pin device pages, then `vm_insert_page` to insert them into the VMA.
- `PHXFS_IOCTL_UNMAP`: release the mapping (`phxfs_map_dev_release`).

## Build & install

See [install.md](install.md) and [troubleshooting.md](troubleshooting.md).
