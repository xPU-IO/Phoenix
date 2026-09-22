# LibPhoenix

`libphoenix` is the user-space C/C++ library that simplifies interaction with the `phxfs` kernel module. It manages device metadata and GPU buffer registration/unregistration.

## Multi-vendor DevConnector

Vendor-specific calls (device discovery via CUDA/HIP/CANN) are abstracted behind `struct devconn_ops` (`libphoenix/connectors/devconnector.h`). The active connector is selected at **compile time** via `PHXFS_VENDOR` (default `NVIDIA`) and exposed through the global `devconn` pointer. Core files (`phx_device.cpp`, `phx_mem.cpp`, `phx_io.cpp`) call only through `devconn->find_device()` / `page_size` and never include vendor headers (e.g. `cuda.h`).

`libphoenix/connectors/nvidia_connector.cpp` implements the NVIDIA connector. Adding a new vendor means writing `<vendor>_connector.cpp` and pointing `devconn` at it; no other user-library file needs to change.

## Driver management

### `phxfs_open`
```c++
int phxfs_open(int deviceID);
```
Opens the character device for `deviceID`, initializes and stores the metadata required for later buffer registration. Opens are reference-counted: a second `phxfs_open` on the same device only adds a client reference.

### `phxfs_close`
```c++
int phxfs_close(int deviceID);
```
Drops one client reference on `deviceID`. The last close waits for in-flight operations to drain, then unmaps every registration and closes the device. A concurrent `phxfs_open` on a draining device fails with `-EBUSY` (retryable).

## Buffer management

### `phxfs_regmem`
```c++
int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
```
In FULL mode, registers a memory region (`addr`, `len`) for `device_id`: `mmap`s a VMA from the char device, then issues `ioctl(PHXFS_IOCTL_MAP)` to pin the GPU pages into it. Both `addr` and `len` must be non-zero and aligned to the device page size. In STAGING mode, user-buffer registration is a no-op; the internal staging pool is registered during `phxfs_open` and must satisfy the kernel's physical 2 MiB span contract. On success, `target_addr` receives the host-mapped address in FULL mode, or `addr` in STAGING mode — an **internal handle for reference only**; the I/O calls identify a buffer by its original device address `addr`, never by `target_addr`.

Registration semantics: an exact-duplicate registration (same `addr` + `len`, still live) is reference-counted and reused (deregister once per register); any other overlap with a live registration is rejected with `-EINVAL`.

### `phxfs_deregmem`
```c++
int phxfs_deregmem(int device_id, const void *addr, size_t len);
```
Drops one reference on the registration. The last reference waits for in-flight I/O on the region to drain, then removes the kernel mapping via `ioctl(PHXFS_IOCTL_UNMAP)` and `munmap`s the user-space VMA.

## Single-request I/O

`phxfs_read` / `phxfs_write` transfer data directly between a file descriptor and the registered (GPU-backed) VMA:

```c++
ssize_t phxfs_read (int fd, int device_id, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(int fd, int device_id, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
```

`device_id` selects the buffer the same way as the batch API below: `>= 0` means `buf` must lie inside a registration on that phxfs device; `< 0` means `buf` is a plain CPU (host) address. For a registered buffer, `buf` may point anywhere inside the region; the host DMA address is resolved as `vaddr + (buf - registered_base) + buf_offset`, and an internal reference on the mapping is held for the transfer's duration, so a concurrent `phxfs_deregmem` cannot unmap it mid-I/O. Large transfers are chunked at `PHXFS_IO_CHUNK` (1 GiB) to stay under the kernel's `MAX_RW_COUNT`.

## Batch I/O

For workloads that issue many independent transfers (e.g. KV-cache retrieve/store, weight loading), the batch API submits a whole set of requests in one call. This removes the per-request syscall overhead of looping over `phxfs_read`/`phxfs_write` and lets the storage layer service requests concurrently.

### Request descriptor

```c++
typedef struct phxfs_io_req {
    int      fd;          // open file descriptor (O_DIRECT recommended)
    int      device_id;   // >=0: phxfs device the buf is registered on;
                          //  <0: plain CPU buffer
    void    *buf;         // GPU addr (registered) or CPU addr
    off_t    buf_offset;  // byte offset within buf
    size_t   nbytes;      // transfer length
    off_t    f_offset;    // file offset
    ssize_t  result;      // OUT: bytes transferred, or negative errno
} phxfs_io_req_t;
```

Each request's `buf` is resolved independently to a DMA-able host address:

- **Registered GPU buffer (`device_id >= 0`)** — `buf` must lie inside a `phxfs_regmem` registration on that device, or the request fails with `-EFAULT` (there is no silent CPU fallback). The mapping is reference-held for the batch's duration.
- **Plain CPU buffer (`device_id < 0`)** — `buf` is used as an ordinary host address (e.g. pinned staging memory).

### Synchronous batch

```c++
int phxfs_read_batch (phxfs_io_req_t *reqs, int n);
int phxfs_write_batch(phxfs_io_req_t *reqs, int n);
```

Submits all `n` requests, blocks until every one completes, and fills each `reqs[i].result`. Returns `0` if every request transferred exactly `nbytes`; otherwise the number of failed requests (`>0`); or a negative errno on a submission-level engine error. Requests whose buffer cannot be resolved are marked `result = -EFAULT` and never handed to the engine.

### Asynchronous batch (compute / I/O overlap)

```c++
phxfs_batch_t *phxfs_batch_submit_read (phxfs_io_req_t *reqs, int n);
phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n);
int            phxfs_batch_wait(phxfs_batch_t *handle);
int            phxfs_batch_destroy(phxfs_batch_t *handle);
```

`submit` queues the batch on the internal worker pool and returns an opaque handle immediately; the caller can run GPU compute meanwhile, then `phxfs_batch_wait` blocks for completion, fills results, and frees the handle. `wait` returns the same value convention as the synchronous batch. `phxfs_batch_destroy` abandons a batch whose results are not needed: it waits for in-flight I/O to quiesce, then frees the handle without copying results back.

Submitted batches queue up on a bounded FIFO and run one at a time with the pool's full worker set, so several submits can be pipelined ahead (e.g. layerwise prefetch); if the queue is full, submit fails with `NULL` + `errno == EBUSY`. An empty batch (`n <= 0`) returns a valid handle whose `wait` returns `0`, matching the synchronous API.


### Lifetime & concurrency contract

Everything a request references must stay valid until the call (sync) or `phxfs_batch_wait` (async) returns:

- `reqs[i].fd` must remain open — fds are **not** `dup()`'d internally.
- CPU buffers must not be freed.
- Registered GPU buffers are protected by an internal reference; a concurrent `phxfs_deregmem` on them blocks until the batch completes rather than unmapping under an in-flight transfer.

A batch may freely mix requests targeting different devices and CPU buffers; all requests share the same worker pool.

### I/O engines & worker pool

The batch path is built on two internal, pluggable layers:

- **I/O engine** (`libphoenix/io_engine/io_engine.h`) — selected once at library load, in preference order `io_uring → sync`. The `io_uring` engine keeps a per-thread ring (QD 1024) with a pre-allocated slice/iovec scratch pool and a sliding-window submit/reap pipeline; it uses the non-fixed `OP_READ`/`OP_WRITE` path (same O_DIRECT DIO route as `pread`, no second long-term pin on device pages). The `sync` engine is a `pread`/`pwrite` loop and is the always-available fallback (also used per-thread if a worker's ring cannot be created). `phxfs_io_engine_name()` reports the active engine, for diagnostics/tests.
- **Worker pool** (`libphoenix/io_engine/io_pool.cpp`) — a single pool of worker threads, each running the engine on its own ring. A batch is striped round-robin across the workers, so a single blocking call can saturate the array while the caller crosses the Python GIL only once. Submits queue up (bounded FIFO) so a caller can pipeline several async batches ahead. There is deliberately no NUMA pinning: the P2P transfer is device-to-device DMA that never touches host RAM, so which CPU/node issues the I/O is irrelevant to the data path. The pool registers a `pthread_atfork` child handler — a forked child lazily re-creates the pool on first use (in-flight async handles do not survive fork).

## Stream-ordered asynchronous I/O

For callers that need a file transfer to be ordered against compute work on a device stream, a cuFile-style asynchronous API is provided:

```c++
int phxfs_read_stream (int fd, void *buf, size_t *nbytes,
                       off_t *buf_offset, off_t *f_offset,
                       ssize_t *bytes_done, void *stream);
int phxfs_write_stream(int fd, void *buf, size_t *nbytes,
                       off_t *buf_offset, off_t *f_offset,
                       ssize_t *bytes_done, void *stream);
```

There is no stream registration: every submission carries the stream handle. The DMA runs inside a host callback enqueued on the user stream (the connector's `launch_host_func` primitive), which the runtime guarantees to run after previously enqueued work and to block work enqueued after it — so a READ's later consumers and a WRITE's preceding kernels are ordered by construction, a bare stream synchronize is always safe, and consecutive submissions on one stream execute in submission order.

Contract highlights:

- The buffer is resolved from `buf` itself: inside any opened device's registration it is a GPU buffer (DMA to its P2P host address); in no registration it is a plain CPU address; an extent sticking out of its registration fails with `-EFAULT`.
- `nbytes` / `buf_offset` / `f_offset` / `bytes_done` are late-binding: the library dereferences them when the stream executes the op, so keep the storage valid and unmodified until the stream is synchronized past the submission. `*bytes_done` then holds the transferred byte count or a negative errno; a failed DMA never stalls the stream.
- Unlike the other I/O interfaces there are **no internal references**: `fd`, the buffer, and its registration must stay valid until the stream is synchronized past the submission — synchronize before deregistering (the same rule cuFile async users follow).
- Returns `0` if the submission was accepted, or a negative errno for submission-level failures.
- Staging-mode devices are not supported (`-EOPNOTSUPP`; the two-hop path cannot run inside the callback) — use the synchronous or batch APIs there. The connector must provide `launch_host_func`.

## Analysis: choosing the right interface

The four data-plane families — single-request, synchronous batch, asynchronous batch, and stream-ordered — share one buffer-registration mechanism and one fd convention (the caller opens the file and passes the fd). They differ in **submission model, concurrency resources, ordering guarantees, and lifetime contract**. This section summarizes the differences so an application developer can pick the right one; the per-interface details are in the sections above.

### Prerequisite: FULL vs STAGING map modes

A kernel module parameter, global to all phxfs devices; query it with `phxfs_get_map_mode()` (0 = FULL, 1 = STAGING). It constrains which interfaces are usable:

| | FULL | STAGING |
|---|---|---|
| Kernel BAR handling | entire BAR remapped at probe | only a Phoenix-owned staging pool is remapped |
| `phxfs_regmem` | real registration (pins GPU pages into a VMA; device-page aligned) | no-op for user buffers (nonzero/range-valid only); internal staging pool is physically 2MiB-aligned |
| Data path | SSD →(P2P DMA)→ user GPU buffer | SSD →(P2P DMA)→ staging pool →(D2D copy)→ user buffer |
| Extra GPU memory | none | staging pool, default 256 MB (`PHX_STAGING_SIZE_MB`) |
| Cost | GPU memory loses RDMA/peermem registerability | an extra D2D hop; stream API unsupported (`-EOPNOTSUPP`) |

Pick STAGING when the same GPU memory must also be RDMA-registered (NCCL, GPUDirect RDMA, …); pick FULL for single-hop DMA bandwidth when RDMA is not needed. Probe at runtime — do not hard-code an assumption.

### Interface comparison

| Dimension | Single-request | Sync batch | Async batch | Stream-ordered |
|---|---|---|---|---|
| Call shape | one transfer per call, blocks | N requests per call, blocks until all complete | N requests, returns a handle immediately | one transfer attached to the caller's device compute stream |
| I/O concurrency | none | internal worker pool | internal worker pool | none |
| Measured throughput (4-disk RAID0, reference) | ~10–15 GiB/s per request; ~2.6 GiB/s in a per-chunk loop | ~25 GiB/s for one large batch (saturates the array) | ~25 GiB/s for one large batch (saturates the array) | comparable to single-request |
| Compute/I/O overlap | none | none (blocks; a pybind layer may release the GIL) | yes: submit, compute, then wait | yes: guaranteed by compute-stream ordering |
| Ordering | program order | requests within a batch are independent | multiple submits execute in submission order | strictly ordered against other ops on the stream |
| Registration protection during transfer | internal reference held; a concurrent dereg blocks until the transfer completes | internal reference held; a concurrent dereg blocks until the batch completes | internal reference held; a concurrent dereg blocks until the batch completes | none: synchronize the stream before deregistering |
| CPU buffers | `device_id < 0` marks a plain host address | supported; may mix with GPU requests in one batch | supported; may mix with GPU requests in one batch | auto-detected: a buf in no registration table is a CPU address |
| STAGING mode | supported (transparently routed through the two-hop path) | supported | supported | unsupported (`-EOPNOTSUPP`) |

### Which interface for which I/O pattern

| I/O pattern | Recommendation |
|---|---|
| Few large sequential transfers (GiB-scale, low frequency) | single-request `phxfs_read` / `phxfs_write` on a registered buffer |
| Many small independent transfers, throughput-bound | sync batch — one call saturates the array |
| Transfers that must overlap with compute, or be pipelined ahead of it | async batch: submit → compute → wait |
| Transfers ordered against compute on a device stream | stream-ordered API |
| GPU memory that must also be RDMA-registered | not an interface choice — run in STAGING mode, which then excludes the stream API |

Decision tree:

```
Need the I/O ordered inside a device compute stream (a same-stream kernel consumes the data)?
├─ yes → do you need co-exsit with GDR stack?(Phoenix in STAGING mode)
│        ├─ yes → unsupported yet
│        └─ no  → phxfs_read_stream / phxfs_write_stream
└─ no  → many independent I/Os per step?
         ├─ no (one large block, low frequency) → phxfs_read / phxfs_write
         └─ yes → anything else to do on this thread while I/O runs (GPU compute)?
                  ├─ no  → phxfs_read_batch / phxfs_write_batch
                  └─ yes → submit → work → wait (async batched io)
```

The four families may be mixed freely: one process, and even one batch, can combine requests for different GPU devices and plain CPU buffers, all sharing the same worker pool.

### Resource consumption

| Resource | Single-request | Sync batch | Async batch | Stream-ordered |
|---|---|---|---|---|
| Worker threads | 0 | 4 | 4 | 0 |
| io_uring rings | 0 | 1 per worker (QD 1024; falls back to a pread loop if the ring cannot be created) | 1 per worker (QD 1024; falls back to a pread loop if the ring cannot be created) | 0 |
| Queue slots | 0 | 1 (held for the call's duration) | 1 per batch, capacity 16 | 0 |
| Heap allocation per call | 0 | 0 (workers write straight into the caller's array, no copy) | 1 handle + job | 1 small job |
| GPU memory (FULL) | 0 | 0 | 0 | 0 |
| GPU memory (STAGING) | staging pool, default 256 MB per device | staging pool, default 256 MB per device | staging pool, default 256 MB per device | n/a |
| fds | caller-owned, never dup'd | caller-owned, never dup'd | caller-owned, never dup'd | caller-owned, never dup'd |

- The worker pool is created lazily on the process's first batch use, shared process-wide, and reused by both batch APIs.
- The I/O engine is also selected lazily. Set `PHXFS_IO_ENGINE=sync` to bypass io_uring at runtime (useful for vendor-driver isolation); `PHXFS_IO_ENGINE=io_uring` requests it explicitly and falls back to sync if unavailable.
- No NUMA pinning: P2P DMA never touches host RAM, so pinning would only pick the wrong node.
- fork(): the pool is lazily re-created in the child, but the parent's P2P mappings and device fds are **not** inherited across fork. Use spawn for multi-process (re-open and re-register in the child), and wait/destroy all outstanding async handles before forking.
- Only one driver may own the GPU PCIe BAR at a time.

### Lifetime contracts — the differences most likely to bite

| | single-request / sync batch / async batch | stream-ordered |
|---|---|---|
| Registration protection during the transfer | internal reference held; a concurrent `phxfs_deregmem` blocks until the call/wait/destroy returns rather than unmapping | none; a dereg racing an in-flight DMA is a caller error |
| Safe dereg point | after the call returns (sync) / after wait·destroy returns (async) | after the stream is synchronized past the submission |
| fd must stay valid until | the call / wait / destroy returns | the stream is synchronized past the submission |
| Parameter storage | copied by value at submission time | late-binding: `nbytes/buf_offset/f_offset/bytes_done` are dereferenced when the stream executes the op |

General rules for all interfaces: `O_DIRECT` is recommended and requires block alignment (4 KiB padding is a good practice); pair every `phxfs_regmem` with exactly one `phxfs_deregmem`.
