<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing to Phoenix

Thanks for your interest in Phoenix! This guide covers how to report issues and how to propose changes.

## Reporting bugs

1. Gather your environment: kernel version (`uname -r`), NVIDIA driver (`nvidia-smi`), loaded modules (`lsmod | grep phxfs`), and any relevant `dmesg` output.
2. Open an issue using the **Bug report** template (`.github/ISSUE_TEMPLATE/bug_report.md`) and paste the environment info plus clear reproduction steps.

> A future MCP server will aggregate these reports across users to build a shared bug knowledge base (see `doc/roadmap.md`).

## Development setup

Follow [doc/install.md](doc/install.md) to build the kernel module, library, and tests. Use the `tencent-backup` branch only as a historical reference for the SC'25 artifact; active development happens on `main`.

## API & ABI stability

Phoenix is pre-1.0: **there is no stable ABI yet.** Exported symbols, struct layouts, and the `libphoenix.so` SONAME may change between commits without a major-version bump, and symbols may be added or removed (for example, the legacy stream-based `phxfs_read_async`/`phxfs_write_async` were removed in favour of the batch API). Pin to a specific commit and rebuild dependents against it until a 1.0 release defines a stable ABI.

## Pull requests

- Keep changes focused; describe the motivation and the testing you performed.
- For kernel-module changes, include `dmesg` output showing a successful `insmod` and any new test results from `test/`.
- Run the relevant benchmark/example to confirm no regression before opening a PR.

## Code style

- C: follow the existing style in `module/` and `libphoenix/` (see `CMakeLists.txt` warning flags).
- Python: 4-space indent, typed where reasonable.
- Document new public APIs in `doc/`.


## Quick install & demo

### 1. Build kernel module + user library

```shell
# Default: NVIDIA GPU (no extra flags needed)
mkdir -p build && cd build && cmake ../ && make -j
make modules

# For other vendors (not yet implemented, interface ready):
# cmake -DPHXFS_VENDOR=AMD ../
# cmake -DPHXFS_VENDOR=HUAWEI ../
```

This produces:
- `build/libphoenix.so` — user-space library
- `build/module/phoenixfs.ko` — kernel module
- `build/bin/test_regmem`, `build/bin/test_io` — test programs

### 2. Insert kernel module

```shell
# Run nvidia-smi first to wake up the GPU BAR, then insert
nvidia-smi
sudo make insmod

# Verify: one character node, plus one metadata directory per GPU
ls /dev/phxfs_dev
ls /sys/class/phxfs-generic/
# Check dmesg for BAR remap messages
dmesg | tail -20
```

### 3. Run tests

```shell
# Memory registration lifecycle (32 checks)
./bin/test_regmem 0          # GPU 0

# I/O correctness + performance (19 checks + throughput)
./bin/test_io 0              # GPU 0
```