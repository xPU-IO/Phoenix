# Phoenix

**An open-source, refactored GPU Direct Storage (GDS) I/O stack — without phony buffers.**

Phoenix is a rebuilt version of GPU Direct Storage (GDS) that lets data flow straight from storage into GPU/NPU memory — faster, and easier to deploy than existing GDS. AI applications plug in through simple adapters.

> 📄 Built on the SC'25 paper *"Phoenix: A Refactored I/O Stack for GPU Direct Storage without Phony Buffers"*. Phoenix is now a long-term open-source middleware for storage→xPU I/O, with adapters for AI data (vLLM done; lmcache planned).

> 🚀 **To build and run Phoenix, see [doc/install.md](doc/install.md).**

## 📰 News

- **2026.9.9** — AMD (MI300 series) support: new `amd` P2P backend (amdkfd peer-direct) and HIP connector, validated on 8× Instinct MI308X 
- **2026.8.27** — [LMCache Phoenix backend released and merged into upstream](https://github.com/LMCache/LMCache/pull/4673),
- **2026.8.20 Stream-Ordered I/O** - Phoenix now supports stream-ordered asynchronous I/O. The new `phxfs_read_stream` / `phxfs_write_stream` APIs take the stream with each submission, no registration required. Since the DMA is host-driven and invisible to the accelerator runtime, the transfer runs inside a host callback enqueued on the stream via the vendor connector's `launch_host_func` primitive — stream semantics then order read consumers and write gathers by construction, a bare stream synchronize is always sufficient.
- **2026.8.6 staging buffer mode** — new `phxfs_map_mode=1` remaps only the BAR units each registered buffer touches (on demand, refcounted, released when idle) instead of the whole GPU BAR, so the rest of the BAR stays `pfn_valid=false` and GPUDirect RDMA / peer-memory (`ibv_reg_mr`) can coexist with Phoenix on the same GPU — the full-BAR mapping previously made peer MMIO unmappable by `dma_map_resource`.
- **2026.7.21 batch I/O API** — `phxfs_read_batch`/`phxfs_write_batch` and async `phxfs_batch_submit_*`/`phxfs_batch_wait`, backed by an `io_uring` engine and a NUMA-aware thread pool; removes per-request syscall overhead for KV-cache / weight-loading workloads.
- **2026.7.10 phxloader released** — adapter for **vLLM**: safetensors weight loading via DMA straight into GPU memory (`--load-format phxsafetensors`).
- **2025.12** — Phoenix paper accepted at [SC'25](https://doi.org/10.1145/3712285.3759862).

## What is Phoenix

By removing the phony buffer, Phoenix becomes a direct hub between storage and xPU — accelerators (GPU/NPU) and AI apps plug in, and data streams straight through:

<p align="center">
  <img src="doc/phoenix-architecture.png" alt="Phoenix architecture: AI applications plug into the Phoenix hub, connecting storage to xPU accelerators" width="900">
</p>

The kernel module (`phxfs`) maps accelerator memory via `ZONE_DEVICE` and serves P2P mappings; the user library (`libphoenix`) exposes simple registration/I/O APIs; adapters (via pybind11) plug Phoenix into AI frameworks.

## Why Phoenix (vs. NVIDIA cuFile / GDS)

Phoenix targets the same storage→GPU direct-I/O problem as NVIDIA GPUDirect Storage (cuFile), but with a different set of design priorities. It is a fit when you value control, portability, and deployment breadth over binding to a proprietary stack.

| Advantage | What it means in practice |
| --- | --- |
| **Fully open source, top to bottom** | The entire path — `phxfs` kernel module, `libphoenix`, and the `io_uring` engine — is open and auditable. When something breaks you can trace and fix it yourself, down to the kernel P2P mapping, instead of filing a ticket against a closed `libcufile.so`. Essential for self-controlled / compliance-audited environments. |
| **Vendor-agnostic by construction** | The DMA backend is selected at build time (CUDA / HIP / NPU) behind a `connector` function-pointer table; core code carries no vendor headers. The same upper-layer and adapter code runs on NVIDIA today, and on AMD / Huawei NPU once their backends land. cuFile is NVIDIA-only. |
| **No cuFile dependency, no filesystem gate** | `phxfs` works at the VFS / block layer and only needs a valid `fd`. Local ext4 / xfs or parallel filesystems (BeeGFS / Lustre) are transparent — no cuFile support-list, no `nvidia-fs`, no GDS-specific driver/hardware requirement. `fd` points somewhere, the DMA reads from there. |
| **`io_uring`-accelerated NVMe** | The batch / async engine (`phxfs_*_batch`, `phxfs_batch_submit_*`) drives parallel NVMe arrays at full bandwidth with minimal syscall and queueing overhead — all in open, modifiable code. |

On the direct path Phoenix delivers full storage→GPU DMA with no CPU bounce buffer ("phony buffer"), so the data never traverses host memory. Performance is competitive with cuFile's direct mode; the differentiators above are what set Phoenix apart.

## Features & supported matrix

| Area | Status | Notes |
| --- | --- | --- |
| Core I/O (storage→GPU DMA, no phony buffer) | ✅ Implemented | `phxfs` + `libphoenix` |
| Python bindings | ✅ Implemented | `python/phxfs` (ctypes) |
| vLLM model loading | ✅ Implemented | `adapters/vllm/phxloader` (V2.2) |
| Batch & async I/O (`io_uring`) | ✅ Implemented | sync `phxfs_*_batch` + async submit/wait; NUMA thread pool; `libaio` engine planned |
| lmcache KV-cache acceleration | 🚧 Roadmap | see [roadmap](doc/roadmap.md) |
| NPU (non-NVIDIA) support | 🔬 Research | experimentally proven, not yet in code |

 
**Environment (tested)**

- OS: Ubuntu 22.04 · Kernel: Linux 6.1 · NVIDIA driver 550.54 (open + `nvidia-fs`) · CUDA 12.4 · MLNX_OFED 24.10
- Storage backends: NVMe-of, NFS (local NVMe supported for the direct path)
- Accelerator: NVIDIA GPU (CUDA). NPU: not yet supported by current code.

## Roadmap

Engineering (production hardening) and research (new capabilities) tracks are detailed in [doc/roadmap.md](doc/roadmap.md) — including a `libaio` batch engine, >32 GiB registration, broader kernel/DKMS packaging, NPU support, and a future MCP-based shared bug knowledge base.



## Documentation

| Doc | Link |
| --- | --- |
| Architecture & data path | [doc/architecture.md](doc/architecture.md) |
| Build / install / demo | [doc/install.md](doc/install.md) |
| Kernel module (`phxfs`) | [doc/kernel-module.md](doc/kernel-module.md) |
| User library (`libphoenix`) | [doc/libphoenix.md](doc/libphoenix.md) |
| Python API | [doc/python-api.md](doc/python-api.md) |
| Adapters (vLLM / lmcache) | [doc/adapters.md](doc/adapters.md) |
| Troubleshooting | [doc/troubleshooting.md](doc/troubleshooting.md) |
| Roadmap | [doc/roadmap.md](doc/roadmap.md) |
| AI developer orientation | [AGENTS.md](AGENTS.md) |

## Partners

*Partnership section — to be announced. We welcome storage vendors, accelerator vendors, and AI-framework teams to collaborate. Contact us (below).*

## Contact & community

*Contact information to be added (maintainers / mailing list / Slack or WeChat group).* For bug reports, please use the issue template — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

If you use Phoenix in your research, please cite our SC'25 paper:

```bibtex
@inproceedings {sc25Phoenix,
  author = {Jianqin Yan, Shi Qiu, Yina Lv, Yifan Hu, Hao Chen, Zhirong Shen, Xin Yao, Renhai Chen, Jiwu Shu, Gong Zhang, and Yiming Zhang.},
  title = {Phoenix:A Refactored I/O Stack for GPU Direct Storage without Phony Buffers.},
  booktitle = {The International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)},
  year = {2025},
  address = {St Louis, MO, USA},
  publisher = {ACM}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
