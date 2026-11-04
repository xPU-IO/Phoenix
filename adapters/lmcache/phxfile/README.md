<!-- SPDX-License-Identifier: Apache-2.0 -->
# libphxfile — frozen-ABI shim between LMCache and libphoenix

`libphxfile.so` freezes the file-IO surface the official LMCache GDS `phx`
backend (`lmcache/v1/gpu_connector/gds_backends/phx.py`, a ctypes wrapper)
programs against. libphoenix is free to evolve underneath (e.g. the planned
removal of the `phxdev` parameter from `phxfs_regmem`/`phxfs_deregmem`); the
change is absorbed inside this library. Consumers only reinstall the `.so`
— zero code change, zero re-binding.

This is the supported LMCache integration in this repository. It is a pure
C library with no Python, pybind11 or torch build dependency — buildable
with any C11 compiler.

## Frozen ABI

Naming and semantics mirror AMD hipFile so LMCache's four GDS backend
wrappers (cuFile / hipFile / uGDS / phx) stay line-by-line analogous:

| Symbol | Semantics |
|---|---|
| `phxFileDriverOpen` / `phxFileDriverClose` | process lifecycle; Close sweeps leftover registrations and closes devices |
| `phxFileBufRegister(addr, len)` | register device memory — **no device, no alignment requirement** |
| `phxFileBufDeregister(addr)` | deregister — only the base address needed |
| `phxFileHandleRegister(fh**, fd)` / `...Deregister` | fd → opaque handle (identity boxing today) |
| `phxFileStreamRegister` / `...Deregister` | frozen no-ops (submissions carry the stream) |
| `phxFileReadAsync` / `phxFileWriteAsync` | stream-ordered IO, hipFile parameter order |

Error model: `int` return, `0` success, `< 0` `-errno`.

### IO semantic contract

1. Submission returns immediately, never blocks.
2. The DMA is stream-ordered (after prior ops, before later ops on the stream).
3. Transfer failures land in `*bytes_done` (valid after the stream sync), not
   in the return value.
4. `nbytes` / `file_offset` / `buf_offset` / `bytes_done` are late-binding:
   the library dereferences them when the stream executes the op; keep the
   storage alive until then.
5. The `[buf_offset, buf_offset + *nbytes)` extent must stay inside the
   buffer's registration.

Note the parameter order: the frozen ABI is hipFile-ordered
(`file_offset` before `buf_offset`); libphoenix's own stream API is the
reverse, and the swap happens inside the shim.

## Device resolution (probe-based)

libphoenix's `phxfs_regmem`/`phxfs_deregmem` take an accelerator ordinal,
but the frozen surface is device-free. The shim resolves the device by probing:

- on first use it checks `/dev/phxfs_dev`, then calls `phxfs_open` for accelerator
  ordinals; libphoenix resolves the sysfs GPU index and encodes it in subsequent
  MAP/UNMAP ioctl numbers; mmap only reserves an unbound address range;
- a registration is attempted on each opened device until one accepts it —
  the kernel P2P mapping only succeeds on the GPU whose BAR covers the
  buffer, and libphoenix rolls failed attempts back cleanly, so probing is
  side-effect-free;
- registration probes **skip non-FULL devices**: this shim requires FULL-mode
  stream buffers; non-FULL devices opened during discovery are not registration
  candidates.

Registration is a low-frequency control operation, so the probe loop's
cost is irrelevant. The shim also keeps the registration bookkeeping
(base address → aligned length + device, so deregister plays back the
aligned length) and the page-size cache.

Once libphoenix drops its device parameter, `phxFileBufRegister` switches
from probe-based to a direct forward — an internal change behind the same
frozen signature.

## Build & install

```bash
bash install.sh          # build + sudo install to /usr/local + ldconfig
```

or manually:

```bash
make
sudo make install PREFIX=/usr/local
sudo ldconfig
```

Requires libphoenix headers/library in `/usr/local`. After reinstalling
libphoenix, rebuild this shim too — it is compile-time bound to libphoenix's
C ABI.

## Freeze discipline

- Symbol names, signatures and the semantic contract above never change.
- Evolution (new capabilities) happens via NEW symbols; existing ones keep
  their behavior.
- Any libphoenix change is absorbed by editing `phxfile.c` only.
