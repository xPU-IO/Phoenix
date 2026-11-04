<!-- SPDX-License-Identifier: Apache-2.0 -->
# Troubleshooting — Kernel Module Installation

Because Phoenix uses `ZONE_DEVICE` to provide memory-mapping services, no other application or kernel driver may be using the GPU's PCIe BAR resources when `phxfs.ko` is inserted.

If you see:

```shell
insmod: ERROR: could not insert module phoenixfs.ko: Operation not permitted
```

the BAR resources are likely already in use. Resolve as follows.

## 1. Check for user-space processes using the GPU

```shell
sudo lsof /dev/nvidia*
```
Terminate any listed processes, then retry `insmod`.

## 2. Check for kernel drivers using the GPU

```shell
sudo lsmod | grep nvidia
```
Typically `nvidia_drm` is held by other DRM modules, partially occupying GPU memory. Either:

- Add `nvidia_drm` to the system blacklist to prevent auto-loading, or
- Manually `rmmod nvidia_drm` and its dependents, then retry.
