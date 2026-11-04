<!-- SPDX-License-Identifier: Apache-2.0 -->
# Roadmap

Phoenix is evolving from a research artifact (SC'25) into a long-term open-source middleware for storage→xPU I/O. The roadmap below is split into **engineering** (hardening for production) and **research** (new capabilities) tracks.

## Engineering

- **Batch / async I/O engines**: the batch API (`phxfs_batch_submit_read/write` + `phxfs_batch_wait`) now runs on an `io_uring` engine behind a shared worker pool (sync fallback). Remaining work: GUP-compatibility verification for `ZONE_DEVICE` / `PCI_P2PDMA` pages across target kernels, and error-injection / lifecycle test coverage.
- **Large-region registration**: `>32 GiB` single `regmem` still fails because the mapping descriptor uses `kmalloc`; convert it to `kvalloc` (verify `kvfree` is safe in the vendor driver callback context).
- **Broader kernel support**: validate and document supported kernel versions beyond the tested 6.1; provide DKMS packaging for the `phxfs` module.
- **Packaging & distribution**: produce wheel / deb artifacts for `libphoenix` and `phxloader`; CI to build and smoke-test on reference kernels.
- **Correctness regression suite**: expand `test/` (e.g. `>2 GiB` registration + offset verification) and wire into CI.
- **Bug intelligence (MCP)**: a lightweight local collector for structured bug reports, plus a future **MCP server** to aggregate reports across users into a shared knowledge base for faster triage. *Planned, not yet implemented.*

## Research

- **Multi-vendor P2P backends**: the kernel `phxfs_p2p_ops` and user-library `DevConnector` interfaces are in place (`PHXFS_VENDOR=NVIDIA|AMD|HUAWEI`); implementing the AMD (`amd-backend.c`/`amd_connector.cpp`) and Huawei NPU (`huawei-backend.c`/`huawei_connector.cpp`) backends is the next step — direct storage→NPU transfer has been experimentally proven feasible.
- **Multi-framework adapters**: further AI-data-loader offloads; LMCache already integrates through its official GDS `phx` backend and the shim in `adapters/lmcache/phxfile/`.
- **Policy & scheduling**: adaptive chunking, multi-GPU striping, and QoS-aware DMA scheduling.
- **Filesystem breadth**: broader backend coverage (more distributed/storage backends) with unchanged application APIs.

## How to contribute

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) and the bug-report template under `.github/ISSUE_TEMPLATE/`.
