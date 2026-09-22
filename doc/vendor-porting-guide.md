# GPU 厂商适配指南 / GPU Vendor Porting Guide

面向尚未适配 Phoenix 的 GPU/NPU 厂商，讲解如何把一块新卡接入 Phoenix 的"存储 → 加速器直通 DMA"数据通路。核心适配面只有两处：内核模块的 P2P 后端（`module/<vendor>-backend.c`）和用户态库的设备连接器（`libphoenix/connectors/<vendor>_connector.cpp`）。

This guide explains how to bring a new GPU/NPU into Phoenix's storage-to-accelerator direct-DMA data path. There are exactly two adaptation surfaces: the kernel module's P2P backend (`module/<vendor>-backend.c`) and the user-space library's device connector (`libphoenix/connectors/<vendor>_connector.cpp`).

---

## 1. 总览 | Overview

Phoenix 分三层：`phxfs` 内核模块（GPU BAR remap + 每卡一个字符设备 + mmap/ioctl P2P 映射）、`libphoenix` 用户库（regmem / read / write / batch）、应用适配器（vLLM 等）。多厂商支持在**编译期**由 `PHXFS_VENDOR`（NVIDIA / METAX / AMD / HUAWEI）选定，一次构建只编入一个厂商后端，不支持混插。

Phoenix has three layers: the `phxfs` kernel module (GPU BAR remap + one char device per GPU + mmap/ioctl P2P mapping), the `libphoenix` user library (regmem / read / write / batch), and application adapters (vLLM etc.). Multi-vendor support is selected at **compile time** via `PHXFS_VENDOR` (NVIDIA / METAX / AMD / HUAWEI); only one vendor backend is compiled in per build — mixed-vendor machines are not supported.

厂商相关的代码被刻意隔离在两个文件里，核心代码只通过两张函数指针表调用厂商能力：

Vendor-specific code is deliberately confined to two files; core code reaches vendor capabilities only through two function-pointer tables:

| 层 | 隔离接口 | 实现文件 | 核心调用方 |
| --- | --- | --- | --- |
| 内核 | `struct phxfs_p2p_ops`（`module/phxfs-backend.h`） | `module/nvidia-backend.c`、`module/metax-backend.c` | `phxfs.c`、`phxfs-mem.c`、`phxfs-p2p-service.c` |
| 用户态 | `struct devconn_ops`（`libphoenix/connectors/devconnector.h`） | `libphoenix/connectors/nvidia_connector.cpp` | `phx_device.cpp`、`phx_mem.cpp`、`phx_io.cpp`、`phx_staging.cpp`、`phx_stream.cpp` |

硬性规则：核心文件（`phxfs.c`、`phxfs-mem.c`、`phxfs-p2p-service.c`、`phx_device.cpp`、`phx_mem.cpp`、`phx_io.cpp` 等）**绝不** include 厂商头或直接调用厂商 API（`nv-p2p.h`、`cuda.h`、HIP、CANN……）。新增厂商 = 写两个新文件 + 三处注册，核心文件理论上零改动。

Hard rule: core files (`phxfs.c`, `phxfs-mem.c`, `phxfs-p2p-service.c`, `phx_device.cpp`, `phx_mem.cpp`, `phx_io.cpp`, ...) must **never** include vendor headers or call vendor APIs (`nv-p2p.h`, `cuda.h`, HIP, CANN, ...). Adding a vendor = two new files + three registration points, with zero core-file changes in theory.

### 适配清单 | Adaptation checklist

```
内核侧 kernel side
  1. module/<vendor>-backend.c        实现 struct phxfs_p2p_ops（§3）
  2. module/phxfs-backend.h           PCI vendor ID + CONFIG_PHXFS_VENDOR_<NAME>（§3.4）
  3. module/phxfs-backend.c           <vendor>_backend_register() 分发（§3.4）
  4. module/phxfs.c                   设备发现的 PCI class 扫描（按需，§3.4）
  5. CMakeLists.txt                   VENDOR_BACKEND / VENDOR_DEFINE 分支（§3.4）

用户态 user side
  6. libphoenix/connectors/<vendor>_connector.cpp   实现 struct devconn_ops（§4）
  7. CMakeLists.txt                   devconn_source 分支 + 厂商运行时链接（§4.3）
  8. test/                            测试为 CUDA 形态，按需移植
```

---

## 2. 厂商驱动的前置条件 | Prerequisites on the vendor driver stack

**EN follows.** 适配之前，先确认厂商内核驱动具备以下能力。NVIDIA 的对照物是 `nv-p2p.h` 导出的 `nvidia_p2p_*` 符号族。

Before porting, confirm the vendor kernel driver provides the following. The NVIDIA equivalents are the `nvidia_p2p_*` symbol family exported by `nv-p2p.h`.

1. **页钉住（pin）内核 API，按虚拟地址调用**：给定设备 VA + 长度，返回页表（页数、每页物理/总线地址）。必须以 `EXPORT_SYMBOL` 导出，供 phxfs 经 `__symbol_get()` 动态获取（松耦合，避免链接期依赖）。
   **A pin-by-VA kernel API**: given a device VA + length, return a page table (entry count, physical/bus address per page). Must be `EXPORT_SYMBOL`-ed so phxfs can fetch it via `__symbol_get()` (loose coupling, no link-time dependency).
2. **强制回收回调**：驱动强行回收页（如进程/context 销毁）时必须回调 phxfs 注册的 `free_cb`。回调语义见 §3.2；这是防止 use-after-free 的关键契约。
   **Force-reclaim callback**: when the driver reclaims pinned pages (context teardown etc.), it must invoke the `free_cb` phxfs registered. Semantics in §3.2; this is the key contract preventing use-after-free.
3. **物理地址落在 BAR 内**：`get_phys_addrs` 返回的地址必须位于该卡被 phxfs 选作 HBM 窗口的那个 PCIe BAR 内（phxfs 取**最大的 BAR** 作为窗口，见 `phxfs.c` `phxfs_ctrl_init`）。核心用 `addr - dev->paddr` 计算 BAR 偏移，落到 BAR 外的地址会导致映射失败。
   **Addresses inside the BAR**: the addresses from `get_phys_addrs` must lie inside the PCIe BAR that phxfs picks as the HBM window (phxfs picks the **largest BAR**, see `phxfs_ctrl_init` in `phxfs.c`). The core computes `addr - dev->paddr` as the BAR offset; an address outside the BAR fails the mapping.
4. **IOMMU 直通部署契约**：Phoenix 的 P2P DMA（含 `phxfs-p2p-service.c` 导出路径）假定 `phys == bus address`，即 `iommu=pt` 或关闭 IOMMU 的裸机部署。厂商平台必须满足该假设。
   **IOMMU passthrough contract**: Phoenix's P2P DMA (including the `phxfs-p2p-service.c` export path) assumes `phys == bus address` — bare-metal with `iommu=pt` or IOMMU off. The vendor platform must satisfy this.

以上 4 条是厂商侧的硬性前置。另有一项能力（staging 池需 pin 为物理连续的 2MiB 大页）为当前实现所依赖，但**不满足时的改进点在 Phoenix 自身而非厂商驱动**——应对路径见 §4.2。

The four items above are hard prerequisites on the vendor side. One further capability (pinning the staging pool as physically contiguous 2MiB huge pages) is relied on by the current implementation, but **when it is unavailable the fix belongs in Phoenix, not in the vendor driver** — see the mitigation path in §4.2.

---

## 3. 内核侧：`module/<vendor>-backend.c` | Kernel side: the P2P backend

### 3.1 `struct phxfs_p2p_ops` 契约 | The ops contract

定义在 `module/phxfs-backend.h`。每个字段的核心语义与错误契约：

Defined in `module/phxfs-backend.h`. Semantics and error contracts per field:

| 成员 | 语义 | 关键契约 |
| --- | --- | --- |
| `name` | 后端名（日志用） | — |
| `init` / `exit` | 加载/卸载厂商符号或后端初始化 | init 失败 → 模块加载失败（`-ENOSYS` 等） |
| `get_pages(vaddr, length, &pt, free_cb, data)` | 按设备 VA pin 页 | 成功返回 0 且 `*pt` 非空；**必须**把 `free_cb`+`data` 原样转交给厂商驱动的强制回收注册。返回的 `pt` 对核心是不透明指针 |
| `put_pages(vaddr, pt)` | 正常释放（非强制回收路径） | 内部先调厂商 unpin，再释放 `pt` 及其包装 |
| `get_n_pages(pt)` | 页表条目数 | **必须在 `get_pages` 返回后即可调用**（NVIDIA 由驱动填充 `entries`） |
| `get_phys_addrs(pt, addrs, n)` | 填物理（BAR 总线）地址数组 | `n` 必须等于 `get_n_pages()`；任一页条目为空返回 `-ENOMEM`；地址必须落在 BAR 窗口内（§2.3） |
| `free_page_table(pt)` | 强制回收后仅释放堆结构 | **绝不**调 `put_pages`——厂商驱动已把页收回去了，再 unpin 就是双重释放 |
| `page_size` | 设备页大小（字节） | NVIDIA/METAX = 64KiB。核心用它推页数和 sub-page 展开；当前全栈假定 64KiB |

调用顺序（核心侧，`phxfs-mem.c`）：`get_pages` → `get_phys_addrs`（页数由核心按 `dev_len / page_size` 计算）；`phxfs-p2p-service.c`（内核态导出服务）则是 `get_pages` → `get_n_pages` → `get_phys_addrs`。**两种顺序都必须工作。**

Call order on the core side (`phxfs-mem.c`): `get_pages` → `get_phys_addrs` (count computed by the core as `dev_len / page_size`); the in-kernel export service (`phxfs-p2p-service.c`) uses `get_pages` → `get_n_pages` → `get_phys_addrs`. **Both orders must work.**

### 3.2 强制回收回调契约 | The force-reclaim callback contract

核心把 `force_release_gpu_memory`（包在厂商回调里）和 `data = mbuffer->map` 传给 `get_pages`。厂商驱动强制回收页时必须调用它。回调内部会：

The core passes `force_release_gpu_memory` (wrapped) and `data = mbuffer->map` to `get_pages`. The vendor driver must invoke it when it force-reclaims the pages. Inside that callback the core:

1. 不再对设备侧做任何 unpin（页已被驱动收回）——调用的是 `free_page_table`（只清堆）或等价物；
   performs **no** device-side unpin (the driver already reclaimed) — it calls `free_page_table` (heap-only cleanup) or an equivalent;
2. 释放 vendor page-table wrapper，并保留由 phony-buffer VMA 持有的核心
   mapping descriptor，直到 VMA close；同时打 WARN 日志。
   frees the vendor page-table wrapper, while the core mapping descriptor
   remains owned by the phony-buffer VMA until VMA close; it also logs a WARN.

因此对厂商驱动侧的要求是：回调可在原子或可睡眠上下文中执行（核心仅执行 `kfree` + `printk`），但**必须保证回调返回后页确实已不可再用**，且每个 pin 注册至多触发一次。

So the driver-side requirements: the callback may run in atomic or sleeping context (the core only does `kfree` + `printk`), but it must guarantee the pages are truly gone once the callback returns, and each pin registration triggers it at most once.

### 3.3 符号加载模式 | Symbol-loading pattern

推荐沿用 NVIDIA 后端的 `__symbol_get` 模式（`module/nvidia-backend.c`）：在 `init()` 里逐个 `__symbol_get("vendor_p2p_xxx")`，任一失败回滚并返回 `-ENOSYS`；`exit()` 里对称 `__symbol_put`。好处：phxfs 不必与厂商驱动有编译期/链接期耦合，符号缺失时 insmod 给出明确错误而不是加载后崩溃。

Follow the NVIDIA backend's `__symbol_get` pattern (`module/nvidia-backend.c`): in `init()`, `__symbol_get("vendor_p2p_xxx")` each symbol, rolling back with `-ENOSYS` on any failure; symmetrically `__symbol_put` in `exit()`. Benefit: phxfs needs no compile/link-time coupling to the vendor driver, and a missing symbol produces a clear insmod error instead of a post-load crash.

函数原型直接在 backend 文件里本地 typedef（METAX 的做法），**无需**厂商头文件——如果你有可安装的厂商 UAPI 头（如 `nv-p2p.h`），也可以在 CMake 里加 include 路径（当前只有 NVIDIA 分支这么做，`CMakeLists.txt` 的 `nv_p2p_files` glob）。

Typedef the prototypes locally in the backend file (the METAX approach) — **no vendor header needed**; if you do ship an installable vendor UAPI header (like `nv-p2p.h`), add an include path in CMake instead (currently only the NVIDIA branch does this, see the `nv_p2p_files` glob in `CMakeLists.txt`).

### 3.4 注册与构建接线 | Registration and build wiring

三处 + 一处按需：

Three mandatory points + one conditional:

1. **`module/phxfs-backend.h`**：加 `CONFIG_PHXFS_VENDOR_<NAME>` 分支，定义 `PHXFS_PCI_VENDOR_ID`（厂商 PCI ID，如 AMD `0x1002`、Huawei `0x19E5`）。若你的卡不是标准 3D/VGA 显示类设备，这里同时定义你的 PCI class（参照 METAX 分支的 `PCI_CLASS_DISPLAY 0x0380`）。
   **`module/phxfs-backend.h`**: add a `CONFIG_PHXFS_VENDOR_<NAME>` branch defining `PHXFS_PCI_VENDOR_ID` (e.g. AMD `0x1002`, Huawei `0x19E5`). If your card is not a standard 3D/VGA display-class device, also define your PCI class here (see the METAX branch's `PCI_CLASS_DISPLAY 0x0380`).
2. **`module/phxfs-backend.c`**：`extern int <vendor>_backend_register(void);` 声明 + `phxfs_p2p_backend_init()` 里的 `#elif` 调用。
   **`module/phxfs-backend.c`**: declare `extern int <vendor>_backend_register(void);` and add the `#elif` call in `phxfs_p2p_backend_init()`.
3. **`CMakeLists.txt`**：`VENDOR_BACKEND "<vendor>-backend.o"` + `VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_<NAME>"` 分支。
   **`CMakeLists.txt`**: add the `VENDOR_BACKEND "<vendor>-backend.o"` + `VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_<NAME>"` branch.
4. **（按需）`module/phxfs.c` `phxfs_discover_devices()`**：设备发现按 PCI class 扫描，目前是厂商条件编译的（非 METAX 扫 `PCI_CLASS_DISPLAY_3D` + `PCI_CLASS_DISPLAY_VGA`，METAX 扫 `PCI_CLASS_DISPLAY`）。**如果你的设备 class 不在其中，一张卡也发现不了**——需要扩展这段 `#ifdef`。
   **(Conditional) `phxfs_discover_devices()` in `module/phxfs.c`**: discovery scans by PCI class and is currently vendor-`#ifdef`-ed (non-METAX scans `PCI_CLASS_DISPLAY_3D` + `PCI_CLASS_DISPLAY_VGA`; METAX scans `PCI_CLASS_DISPLAY`). **If your device class is not covered, zero cards will be found** — this `#ifdef` block must be extended.

`Makefile.in` 的 `phoenixfs-objs` 通过 `@VENDOR_BACKEND@` 占位符自动带上你的文件，无需改动。`insmod` 之前先加载厂商驱动（`__symbol_get` 依赖符号已存在）。

`Makefile.in`'s `phoenixfs-objs` picks your file up automatically via the `@VENDOR_BACKEND@` placeholder — no edit needed. Load the vendor driver before `insmod` (the `__symbol_get`s depend on the symbols existing).

### 3.5 骨架 | Skeleton

```c
/* module/<vendor>-backend.c — 最小骨架 */
#include <linux/module.h>
#include <linux/slab.h>
#include "phxfs-backend.h"
#include "phxfs.h"

struct <vendor>_page_table {       /* 厂商句柄 + 我们记的回调 */
        void *vendor_handle;
        void (*free_cb)(void *); void *cb_data;
        uint32_t entries;
};

/* 本地 typedef 厂商导出符号的原型，__symbol_get 取指针 */
static int (*p_pin)(u64 vaddr, size_t len, void **handle,
                    int (*cb)(void *), void *cb_arg);
static void (*p_unpin)(void *handle);
static u64 (*p_page_phys)(void *handle, u32 idx);   /* 示意 */
...

static int <vendor>_get_pages(u64 vaddr, u64 length,
                              struct phxfs_page_table **pt,
                              void (*free_cb)(void *), void *data)
{
        struct phxfs_page_table *h = kzalloc(sizeof(*h), GFP_KERNEL);
        struct <vendor>_page_table *v;
        if (!h) return -ENOMEM;
        v = kzalloc(sizeof(*v), GFP_KERNEL);
        if (!v) { kfree(h); return -ENOMEM; }
        v->free_cb = free_cb; v->cb_data = data;
        if (p_pin(vaddr, length, &v->vendor_handle,
                  <vendor>_reclaim_cb, v)) { kfree(v); kfree(h); return -EIO; }
        v->entries = DIV_ROUND_UP(length, <vendor>_page_size);
        h->priv = v; *pt = h;
        return 0;
}
/* put_pages → p_unpin + 释放 v/h；get_phys_addrs → 逐页 p_page_phys；
 * free_page_table → 只 kfree（驱动已回收）；reclaim_cb → v->free_cb(v->cb_data) */

static struct phxfs_p2p_ops <vendor>_ops = {
        .name = "<vendor>", .page_size = 64 * 1024,
        /* init/exit = 逐个 __symbol_get / __symbol_put */
        .get_pages = <vendor>_get_pages, .put_pages = <vendor>_put_pages,
        .get_n_pages = <vendor>_get_n_pages,
        .get_phys_addrs = <vendor>_get_phys_addrs,
        .free_page_table = <vendor>_free_page_table,
};
int <vendor>_backend_register(void)
{ int r = <vendor>_init(); return r ? r : phxfs_p2p_register_backend(&<vendor>_ops); }
```

参考实现：`nvidia-backend.c`（结构最规范，建议以其为模板）、`metax-backend.c`（第二家厂商的真实实现，可作对照）。

Reference implementations: `nvidia-backend.c` (the cleanest structure — recommended template) and `metax-backend.c` (a real second-vendor implementation, useful as a cross-reference).

---

## 4. 用户态：`libphoenix/connectors/<vendor>_connector.cpp` | User side: the DevConnector

### 4.1 `struct devconn_ops` 契约 | The ops contract

定义在 `libphoenix/connectors/devconnector.h`。全局指针 `devconn` 在链接期由编入的连接器文件赋值，`devconn_init()` 在库加载时调用一次 `init`。

Defined in `libphoenix/connectors/devconnector.h`. The global `devconn` pointer is assigned at link time by the compiled-in connector; `devconn_init()` invokes `init` once at library load.

| 成员 | 必需性 | 语义 / 契约 |
| --- | --- | --- |
| `name` / `page_size` | 必需 | `page_size` **必须与内核后端的 `page_size` 一致** |
| `init` | 可选 NULL | 厂商运行时初始化（CUDA 风格运行时通常首次调用自初始化，可返回 0） |
| `find_device(device_id)` | 必需 | 厂商设备号（CUDA id / HIP id / NPU id…）→ phxfs 设备号。失败返回 -1 |
| `mem_alloc` / `mem_free` | STAGING 模式必需，否则可 NULL | 在指定 phxfs 设备上分配/释放显存。缺省时 staging 模式报 `-ENOTSUP` |
| `memcpy_dtod` | 同上 | **同步** D2D 拷贝：返回即完成（`cudaMemcpy` D2D 语义）。staging 路径的完成契约依赖这一点 |
| `memcpy_dtod_async(phxfs_dev, slot, dst, src, n)` | 可选 NULL | 异步 D2D。`slot ∈ [0, 2)`（双缓冲）；连接器自备队列并保证同 slot FIFO |
| `queue_sync(phxfs_dev, slot)` | 可选 NULL | 等该 slot 上已入队拷贝全部完成并回报错误。两者齐备才启用异步路径，否则核心回退同步 `memcpy_dtod` |
| `launch_host_func(stream, fn, arg)` | stream API 必需，否则可 NULL | 把 `fn(arg)` 排到用户 stream 上（`cudaLaunchHostFunc` 语义：晚于之前所有排队工作、阻塞之后所有工作）。缺省时 `phxfs_read/write_stream` 返回 `-EOPNOTSUPP`。回调内不可调厂商 API |
| `range_push` / `range_pop` | 可选 NULL | profiler 区间标注（NVIDIA 为 NVTX）。每线程严格配对 |

### 4.2 STAGING 模式对连接器的要求 | What STAGING mode needs from the connector

libphoenix 在首次 `phxfs_open` 时：`mem_alloc(size + 2MiB)` → 虚拟地址 2MiB 对齐 → 执行注册（触发内核按需 remap）→ 将池划分为两个 slot 双缓冲。因此：

On the first `phxfs_open`, libphoenix: `mem_alloc(size + 2MiB)` → align the VA to 2MiB → really register it (triggering the kernel's on-demand remap) → split the pool into two double-buffered slots. Therefore:

- `mem_alloc` 分配的显存被 pin 后，其 BAR 物理页必须构成**一段连续大页区段：物理地址连续、页号单调递增、起始地址 2MiB 对齐、总长度为 2MiB 的整数倍**——内核 `phxfs_staging_ensure_span` 对任何其他布局一律返回 `-EINVAL` 显式拒绝（fail-fast 设计）。NVIDIA 侧因大块 `cudaMalloc` 由 2MiB 大页背书而天然满足该约束；若厂商分配器产生物理碎片化的页，staging 注册将失败——这是有意的快速失败，而非缺陷。
  Once pinned, the allocation's BAR pages must form **one contiguous huge-page span: physically contiguous, monotonically ascending, 2MiB-aligned, and a whole multiple of 2MiB in length** — the kernel's `phxfs_staging_ensure_span` rejects any other layout with `-EINVAL` (fail-fast by design). NVIDIA satisfies this naturally, as large `cudaMalloc`s are backed by 2MiB huge pages; an allocator returning physically fragmented pages fails staging registration — an intentional fail-fast, not a defect.
- 若厂商分配器存在物理碎片、pin 不出连续大页区段：改进点在 Phoenix 而非厂商驱动。可行方向：① 放宽 `phxfs_staging_ensure_span`——允许一个池对应多个 2MiB 对齐的连续子区段（稀疏 remap；每个子区段仍独立满足对齐与整倍长度约束；改动集中于 `phxfs.c` 的区段逻辑与 `phxfs-mem.c` 调用侧）；② 用户态改为分配多个 2MiB 整数倍的小池并逐个注册（改造 `phx_staging_setup`）。若考虑改用厂商的 VMM 类物理粒度分配 API，须先验证 pin 路径与其兼容——NVIDIA 的教训：legacy `nvidia_p2p_get_pages` 对 VMM VA 返回 EINVAL，而 5.4 内核又无 dma-buf P2P 可回退。
  If the vendor allocator fragments physically and cannot pin a contiguous huge-page span: the fix belongs in Phoenix, not in the vendor driver. Options: ① relax `phxfs_staging_ensure_span` so one pool may map to several 2MiB-aligned contiguous sub-spans (sparse remap; each sub-span independently aligned and a whole multiple of 2MiB; changes concentrate in `phxfs.c`'s span logic and the `phxfs-mem.c` call site); ② allocate several smaller 2MiB-multiple pools and register each (rework `phx_staging_setup`). If you consider a VMM-style physical-granularity allocation API instead, validate the pin path against it first — the NVIDIA lesson: legacy `nvidia_p2p_get_pages` returns EINVAL on VMM VAs, and kernel 5.4 offers no dma-buf P2P fallback.
- `memcpy_dtod` 返回即完成的语义不可弱化；异步对（`memcpy_dtod_async` + `queue_sync`）至少支持 2 个 slot（NVIDIA 实现 4）。
  Do not weaken the "returns = completed" semantics of `memcpy_dtod`; the async pair must support at least 2 slots (NVIDIA implements 4).
- stream API 在 staging 设备上整体不支持（`-EOPNOTSUPP`），无需为它做 staging 适配。
  The stream API is entirely unsupported on staging devices (`-EOPNOTSUPP`) — no staging work needed for it.

### 4.3 CMake 接线 | CMake wiring

`CMakeLists.txt` 两处：`devconn_source` 分支指向你的连接器文件；`target_link_libraries` 链接厂商运行时（NVIDIA 是 `CUDA::cudart`）。若厂商工具链特殊（如 METAX 走 MACA `cu-bridge`，需 `MACA_PATH` 环境变量并整体换 compiler），参照 `PHXFS_VENDOR=METAX` 分支的写法。

Two places in `CMakeLists.txt`: the `devconn_source` branch pointing at your connector file, and `target_link_libraries` for the vendor runtime (NVIDIA: `CUDA::cudart`). If the vendor toolchain is special (e.g. METAX uses MACA's `cu-bridge`, requiring the `MACA_PATH` env and a full compiler swap), follow the `PHXFS_VENDOR=METAX` branch.

### 4.4 CUDA 兼容运行时捷径 | The CUDA-compatible-runtime shortcut

若厂商 SDK 提供 CUDA 兼容运行时（`cudaMalloc`/`cudaMemcpyAsync`/`cudaLaunchHostFunc`/`cudaDeviceGetPCIBusId` 全可用），**用户态连接器可以原样复用 `nvidia_connector.cpp`，零改动**——METAX 走的就是这条路（CMake 里 METAX 分支的 `devconn_source` 就是 `nvidia_connector.cpp`，用 MACA 的 `cucc` 编译）。此时适配工作量只剩内核 backend + 设备发现。环境要求见 `doc/install.md`。

If the vendor SDK ships a CUDA-compatible runtime (all of `cudaMalloc`/`cudaMemcpyAsync`/`cudaLaunchHostFunc`/`cudaDeviceGetPCIBusId` work), **the user-space connector can be reused as-is (`nvidia_connector.cpp`, zero changes)** — the path MetaX takes (the METAX branch in CMake sets `devconn_source` to `nvidia_connector.cpp`, compiled with MACA's `cucc`). The porting effort then reduces to the kernel backend + device discovery. Environment setup: see `doc/install.md`.

---

## 5. FULL / STAGING 模式对新厂商意味着什么 | FULL vs STAGING for a new vendor

两种模式由模块参数 `phxfs_map_mode` 选择：`0` = FULL——整条 GPU BAR 在模块加载时 remap 为 P2P 映射，用户显存可直接 DMA；`1` = STAGING（默认）——仅 Phoenix 自有的 staging 池在注册时按需 remap，数据经 SSD → staging 池 → 显存两跳（第二跳由连接器的 D2D 原语完成）。对适配工作量的影响：

Two modes selected by the module parameter `phxfs_map_mode`: `0` = FULL — the whole GPU BAR is remapped for P2P at module load, user device memory is directly DMA-able; `1` = STAGING (default) — only Phoenix's own staging pool is remapped on demand at registration, data takes SSD → staging pool → device memory two hops (the second hop done by the connector's D2D primitives). Impact on porting effort:

- 只有 FULL 模式：连接器可以省掉全部 staging ops（`mem_alloc`/`mem_free`/`memcpy_dtod` 置 NULL），但**当前实现**整条 BAR 被 remap（struct page 化），与厂商自家 RDMA/peermem 栈互斥（NVIDIA 上已实证的冲突，见 `doc/troubleshooting.md` 与仓库内 `STORE_BANDWIDTH_ROOT_CAUSE.md` 的分析）。该互斥并非 FULL 模式的固有属性：通过 dma-buf 导出/导入构建直通内存（要求内核 ≥5.12）不建 struct page，直通 DMA 与 RDMA 可共存——厂商若为设备内存提供 dma-buf exporter，这是值得推动 Phoenix 迁移的路径（本机 5.4 内核不可用 dma-buf P2P，故当前未走此路）。
  FULL-only port: the connector may leave all staging ops NULL, but the **current implementation** remaps the whole BAR (struct-page backing), which is mutually exclusive with the vendor's own RDMA/peermem stack (a proven conflict on NVIDIA — see `doc/troubleshooting.md` and the analysis in `STORE_BANDWIDTH_ROOT_CAUSE.md`). That exclusivity is not inherent to FULL mode: building the direct mapping via dma-buf export/import (kernel ≥5.12 required) creates no struct pages, so direct DMA and RDMA can coexist — a vendor shipping a dma-buf exporter for device memory makes this a route worth pushing Phoenix onto (unavailable on this 5.4 kernel box, hence not taken today).
- 要两者都支持：backend 的 pin 必须能为 staging 池给出连续大页区段（§4.2，含不满足时的改进路径），连接器必须实现 D2D 原语。NVIDIA 的历史教训：当 remap 粒度与 GPU 物理页粒度不一致时，无关显存会被"误伤"地获得 struct page（pfn_valid=true），进而导致 peermem 注册失败——新厂商务必先行验证自家驱动 pin 出的物理布局假设。
  Supporting both: the backend's pin must yield a contiguous huge-page span for the staging pool (§4.2, including the mitigation path when it cannot) and the connector must implement the D2D primitives. NVIDIA's historical lesson: when the remap granularity differs from the GPU's physical-page granularity, unrelated device memory acquires collateral struct pages (pfn_valid=true), which then breaks peermem registration — a new vendor must first validate its driver's pinned-physical-layout assumptions.

---

## 6. 验证清单 | Validation checklist

```shell
# 1. 构建（换厂商名）
mkdir -p build && cd build && cmake ../ -DPHXFS_VENDOR=<NAME> && make -j8

# 2. 加载顺序：先厂商驱动，后 phxfs
nvidia-smi          # 厂商侧确认驱动在位
sudo make insmod    # 失败先看 dmesg：符号缺失 → 后端 init 返回 -ENOSYS
dmesg | grep phxfs  # 期望: "registered P2P backend: <name> (page_size=...)" + 设备发现日志

# 3. sysfs 检查
ls /sys/class/phxfs-generic/         # phxfs_dev0..N
cat /sys/class/phxfs-generic/phxfs_dev0/pci_bdf   # BDF 正确
cat /sys/class/phxfs-generic/phxfs_dev0/map_mode  # 0=full 1=staging

# 4. 功能测试（test/ 目前是 CUDA 形态，其他厂商需先移植）
export LD_LIBRARY_PATH=$PWD/lib
./bin/test_regmem 0    # 注册生命周期：mmap→MAP→UNMAP→munmap，含强制回收路径
./bin/test_io 0        # 单请求读写的正确性 + 性能
./bin/test_batch 1     # 批量/异步 I/O
```

验收标准：`test_regmem` 全部通过（覆盖 pin/unpin/force-reclaim 三条路径）；staging 模式下 `test_io`/`test_batch` 全部通过（覆盖 D2D 原语）；进程异常退出（kill -9）后 `rmmod` 干净卸载（验证强制回收回调与引用计数平衡）。

Acceptance: `test_regmem` passes in full (covers pin/unpin/force-reclaim); in staging mode `test_io`/`test_batch` pass (covers the D2D primitives); a clean `rmmod` after `kill -9`-ing a process with live registrations (validates the force-reclaim callback and reference-count balance).

---

## 7. 相关文档 | Related documents

- [architecture.md](architecture.md) — 整体架构与数据通路 / system architecture and data path
- [kernel-module.md](kernel-module.md) — 内核模块设计 / kernel module design
- [libphoenix.md](libphoenix.md) — 用户库 API / user library API
- [install.md](install.md) — 构建与安装（含 MetaX/MACA 环境）/ build & install (incl. MetaX/MACA setup)
- [troubleshooting.md](troubleshooting.md) — 模块加载排障 / module-load troubleshooting
