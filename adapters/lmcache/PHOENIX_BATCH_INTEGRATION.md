# Phoenix batch I/O — LMCache adapter integration guide (for `phxcache` + `phx_l2_adapter`)

> Audience: the author of
> [phoenix#7 (`phxcache`)](https://github.com/xPU-IO/phoenix/pull/7) and
> [LMCache#1 (`phx_l2_adapter`)](https://github.com/xPU-IO/LMCache/pull/1).
>
> This document describes **how to adapt the LMCache side** to the new
> Phoenix batch API. The Phoenix core changes (the batch API + internal
> worker pool) are already merged in `libphoenix`; **no further Phoenix core
> change is required for the basics below.** The changes here are to
> `phxcache` (pybind layer) and the two LMCache files.

---

## 0. TL;DR — what to change and why

The current PR loads KV chunks like this (`phx_l2_adapter.py::_process_load`):

```python
for i, (key, obj) in enumerate(zip(keys, objects)):   # one chunk at a time
    with phxcache.PhxFile(self._phx_cache, path, flags) as f:  # open + close per chunk
        ret = f.read(...)                              # one SYNC read per chunk
```

Per chunk this is: **1 `open` + 1 synchronous `phxfs_read` + 1 `close`**. For a
retrieve of N chunks that is N `open`/`close` (metadata + context switches) and
N **serialised** reads with no I/O concurrency. Measured single-request /
single-ring throughput tops out around **10–15 GiB/s**; the disks can do ~25.

**Fix (three layers):**

1. **`phxcache`**: expose the new Phoenix **batch** API (`phxfs_read_batch` and
   the async `phxfs_batch_submit_read` / `phxfs_batch_wait`) as one Python call
   that takes *all* chunks of a task at once.
2. **`phx_l2_adapter.py`**: replace the per-chunk `for` loop in `_process_load`
   with **one batch call** for the whole task. Do **not** add Python worker
   threads — the concurrency now lives *below* the GIL, inside Phoenix.
3. **File management**: the "one file per chunk" scheme is the main remaining
   cost (open/stat per chunk, O_DIRECT can't coalesce). Move to **packed
   files** so a batch reads many chunks from *few* fds.

Key performance fact from Phoenix-side benchmarking (4-disk RAID0):

| path | throughput |
| --- | --- |
| per-chunk sync read loop (current PR) | ~2.6 GiB/s |
| single big `phxfs_read_batch` (this guide) | **~25 GiB/s** |

A **single** `phxfs_read_batch(reqs, N)` call, with N large (thousands of
chunks), already saturates the array — Phoenix internally fans the batch across
a shared pool of worker threads, each with its own io_uring ring. **The adapter
stays single threaded** and crosses the GIL once.

---

## 1. Phoenix batch API (already in `libphoenix`, `phoenix.h`)

```c
typedef struct phxfs_io_req {
    int      fd;          // open file descriptor (O_DIRECT recommended)
    int      device_id;   // phxfs device the GPU buf is registered on;
                          //   ignored for plain CPU buffers
    void    *buf;         // GPU addr (registered via phxfs_regmem) OR CPU addr
    off_t    buf_offset;  // byte offset within buf
    size_t   nbytes;      // transfer length
    off_t    f_offset;    // file offset
    ssize_t  result;      // OUT: bytes transferred, or negative errno
} phxfs_io_req_t;

// Synchronous: submit all N, block until done, fill each req->result.
// Returns 0 if every request moved exactly nbytes, else the failure count.
int phxfs_read_batch (phxfs_io_req_t *reqs, int n);
int phxfs_write_batch(phxfs_io_req_t *reqs, int n);

// Asynchronous (compute/I/O overlap):
typedef struct phxfs_batch phxfs_batch_t;
phxfs_batch_t *phxfs_batch_submit_read (phxfs_io_req_t *reqs, int n);
phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n);
int            phxfs_batch_wait(phxfs_batch_t *handle);   // reqs must stay alive until this returns

const char *phxfs_io_engine_name(void);   // "io_uring" | "sync"
```

Notes that matter for the adapter:

- **One call, many requests.** `n` can be very large (tested at 16 384). Each
  request is independent `(fd, buf+buf_offset, nbytes, f_offset)`.
- **Different fds per request are fine.** You can point each request at a
  different file, or many requests at the *same* fd with different `f_offset`
  (this is what packed files below exploit).
- **GPU vs CPU buffer is selected by the sign of `device_id`.**
  `device_id >= 0` requires `buf` to lie inside a `phxfs_regmem` registration
  on that phxfs device — otherwise the request fails with `-EFAULT` (there is
  no silent CPU fallback). `device_id < 0` (e.g. `-1`) means `buf` is a plain
  CPU address. So **store (CPU) and retrieve (GPU) use the same batch call** —
  see §5.
- **No NUMA routing — deliberately.** The transfer is device-to-device DMA
  (NVMe controller → PCIe → GPU BAR) that never touches host RAM, so which CPU
  issues the I/O has no bearing on the data path. One shared pool of worker
  threads (4 by default), each with its own io_uring ring, drives every batch.
- **No `open` in the batch.** Files must already be open (fds passed in). This
  is where the fd-cache / packed-file work (§3) pays off.

---

## 2. `phxcache` — expose batch to Python

Add a `PhxBatch` type (or module-level functions) to `phx_cache.h/.cpp` +
`bindings.cpp`. The batch takes a list of per-request tuples and returns the
per-request byte counts.

### 2.1 `phx_cache.h` — add a batch builder

```cpp
// One resolved request as seen from Python.
struct PhxBatchReq {
    int      fd;
    int      device_id;
    uintptr_t buf;        // GPU base ptr (registered) or CPU ptr
    off_t    buf_offset;
    size_t   nbytes;
    off_t    f_offset;
};

class PhxBatch {
public:
    // reqs: list of (fd, device_id, buf, buf_offset, nbytes, f_offset)
    // Returns per-request bytes moved (negative == errno) after completion.
    static std::vector<ssize_t> read(const std::vector<PhxBatchReq> &reqs);
    static std::vector<ssize_t> write(const std::vector<PhxBatchReq> &reqs);

    // Async: submit + wait, so Python can overlap GPU compute in between.
    // submit returns an opaque token; wait fills & returns results.
    static PhxBatch *submit_read(const std::vector<PhxBatchReq> &reqs);
    static PhxBatch *submit_write(const std::vector<PhxBatchReq> &reqs);
    std::vector<ssize_t> wait();
private:
    phxfs_batch_t *h_ = nullptr;
    std::vector<phxfs_io_req_t> reqs_;   // MUST outlive the async batch
};
```

### 2.2 `phx_cache.cpp` — thin wrappers over the C API

```cpp
static std::vector<phxfs_io_req_t> to_reqs(const std::vector<PhxBatchReq> &in) {
    std::vector<phxfs_io_req_t> r(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        r[i] = phxfs_io_req_t{};
        r[i].fd = in[i].fd;
        r[i].device_id = in[i].device_id;
        r[i].buf = reinterpret_cast<void*>(in[i].buf);
        r[i].buf_offset = in[i].buf_offset;
        r[i].nbytes = in[i].nbytes;
        r[i].f_offset = in[i].f_offset;
    }
    return r;
}

std::vector<ssize_t> PhxBatch::read(const std::vector<PhxBatchReq> &in) {
    auto reqs = to_reqs(in);
    phxfs_read_batch(reqs.data(), (int)reqs.size());   // release GIL (see bindings)
    std::vector<ssize_t> out(reqs.size());
    for (size_t i = 0; i < reqs.size(); i++) out[i] = reqs[i].result;
    return out;
}
// write(): same but phxfs_write_batch.

PhxBatch *PhxBatch::submit_read(const std::vector<PhxBatchReq> &in) {
    auto *b = new PhxBatch();
    b->reqs_ = to_reqs(in);                 // kept alive in the object
    b->h_ = phxfs_batch_submit_read(b->reqs_.data(), (int)b->reqs_.size());
    return b;
}
std::vector<ssize_t> PhxBatch::wait() {
    phxfs_batch_wait(h_);                    // release GIL (see bindings)
    std::vector<ssize_t> out(reqs_.size());
    for (size_t i = 0; i < reqs_.size(); i++) out[i] = reqs_[i].result;
    return out;
}
```

### 2.3 `bindings.cpp` — pybind, **release the GIL** on the blocking calls

```cpp
py::class_<PhxBatchReq>(m, "PhxBatchReq")
    .def(py::init<>())
    .def_readwrite("fd", &PhxBatchReq::fd)
    .def_readwrite("device_id", &PhxBatchReq::device_id)
    .def_readwrite("buf", &PhxBatchReq::buf)
    .def_readwrite("buf_offset", &PhxBatchReq::buf_offset)
    .def_readwrite("nbytes", &PhxBatchReq::nbytes)
    .def_readwrite("f_offset", &PhxBatchReq::f_offset);

py::class_<PhxBatch>(m, "PhxBatch")
    .def_static("read",  &PhxBatch::read,  py::call_guard<py::gil_scoped_release>())
    .def_static("write", &PhxBatch::write, py::call_guard<py::gil_scoped_release>())
    .def_static("submit_read",  &PhxBatch::submit_read)
    .def_static("submit_write", &PhxBatch::submit_write)
    .def("wait", &PhxBatch::wait, py::call_guard<py::gil_scoped_release>());
```

> **Why `gil_scoped_release` matters:** the batch call blocks while Phoenix's
> internal threads do the I/O. Releasing the GIL lets the rest of Python run
> during that time. This is the *whole point* of pushing concurrency below the
> GIL — the adapter must NOT spawn Python threads to get parallel I/O.

`setup.py`: keep linking `phoenix uring cuda cudart` (already done).

---

## 3. File management — the biggest remaining win

### 3.1 Problem with the current scheme

`_key_to_path` / `_object_key_to_filename` map **one KV chunk → one file** under
a two-level dir. A retrieve of N chunks therefore needs N `open`+`close`, N
inode/dentry lookups, and (with O_DIRECT) N independent I/O submissions that the
block layer can't coalesce. This is the dominant cost once the per-request
syscall loop is removed.

For reference, LMCache's own local disk backends behave similarly (one blob per
key), but they are not on the GPU-DMA hot path; for Phoenix the packing matters.

### 3.2 Recommended: **packed files** (group many chunks per file)

Store chunks that are read together into **one file**, and keep an index
mapping `chunk -> (file_id, offset, len)`.

Good grouping keys (pick per workload):

- **Per (request/sequence, layer)** for layerwise (see §6), or
- **Per store batch** (all chunks of one `submit_store_task` → one file), or
- **Per (model, kv_rank, object_group)** bucket.

On **retrieve**, all chunks that live in the same packed file collapse into
**many `phxfs_io_req_t` sharing one `fd`, each with its own `f_offset`** — a
single `open` serves the whole group, and one `phxfs_read_batch` issues all of
them concurrently.

Sketch:

```python
# index entry per chunk (persist alongside the packed file, e.g. a sidecar
# .idx or an in-memory dict rebuilt on startup)
#   fname -> (pack_path, offset, length)

def _plan_load(self, keys, objects):
    reqs, plan = [], []          # plan[i] = (bitmap_index, expected_len)
    fd_cache = {}                # pack_path -> fd (see 3.3)
    for i, (key, obj) in enumerate(zip(keys, objects)):
        ent = self._index.get(_object_key_to_filename(key))
        if ent is None:
            continue
        pack_path, off, length = ent
        fd = fd_cache.get(pack_path)
        if fd is None:
            flags = os.O_RDONLY | (os.O_DIRECT if self._config.use_direct_io else 0)
            fd = os.open(pack_path, flags)
            fd_cache[pack_path] = fd
        t = obj.raw_tensor
        req = phxcache.PhxBatchReq()
        req.fd = fd
        req.device_id = self._phx_cache.device_id
        req.buf = self._phx_base_pointer
        req.buf_offset = t.data_ptr() - self._phx_base_pointer
        req.nbytes = length
        req.f_offset = off
        reqs.append(req); plan.append((i, length))
    return reqs, plan, fd_cache
```

**O_DIRECT alignment:** with packed files, each chunk's `offset` and `length`
must be 512-byte (block) aligned for O_DIRECT. Two options:

- **Pad** each chunk up to a 4 KiB (or 64 KiB) boundary when packing, so every
  `(offset, length)` is aligned; simplest, small space overhead.
- Or store an aligned header + aligned payload and read the aligned superset.

Padding to 4 KiB is recommended (KV chunks are typically ≥ tens of KiB, so the
overhead is small).

### 3.3 Cheaper interim step: **fd cache** (keep one-file-per-chunk)

If packed files are too big a change for a first cut, keep one file per chunk
but **stop re-`open`/`close` per read**: hold an LRU of open fds keyed by path
(store fds in `_hot_cache` instead of just paths). This removes the repeated
`open`/`close`/`stat` cost; batching then coalesces the reads. Packed files
still win more (fewer files, better readahead), so treat this as a stepping
stone.

---

## 4. Rewrite `_process_load` to one batch call

Replace the per-chunk loop (`phx_l2_adapter.py` lines ~402–425) with:

```python
def _process_load(self, task):
    task_id, keys, objects = task
    bitmap = Bitmap(len(keys))

    # GPU DMA path via batch; CPU objects still fall back to POSIX per-obj.
    gpu_reqs, gpu_plan, fd_cache = [], [], {}
    cpu_items = []
    for i, (key, obj) in enumerate(zip(keys, objects)):
        t = obj.raw_tensor
        if t is None:
            continue
        if t.is_cuda and self._phx_cache is not None:
            # build a PhxBatchReq (see _plan_load / packed-file mapping)
            ...  # append to gpu_reqs, gpu_plan
        else:
            cpu_items.append((i, key, obj))

    if gpu_reqs:
        results = phxcache.PhxBatch.read(gpu_reqs)   # ONE call, GIL released
        for (idx, expect), got in zip(gpu_plan, results):
            if got == expect:
                bitmap.set(idx)
                self._notify_keys_accessed([keys[idx]])
            else:
                logger.error("phx batch short read: %s/%s", got, expect)

    for i, key, obj in cpu_items:                    # unchanged POSIX fallback
        try:
            self._load_posix(self._path_for(key), obj.raw_tensor, obj.raw_tensor.nbytes)
            bitmap.set(i); self._notify_keys_accessed([key])
        except Exception as e:
            logger.error("phx posix load failed for %s: %s", key, e)

    for fd in fd_cache.values():      # if not using a persistent fd cache
        os.close(fd)

    with self._load_results_lock:
        self._load_results[task_id] = bitmap
    self._load_efd.notify()
```

**Do not** add Python threads around this. One `PhxBatch.read` already uses all
the I/O parallelism the disks can absorb (Phoenix does the fan-out internally).

The single background `_worker_loop` thread stays as-is; only the *body* of
`_process_load` changes.

---

## 5. Store path (CPU → disk) can also use batch

Today `_process_store` writes each tensor with buffered `open(..., "wb")` +
`numpy()` (CPU memory). You can optionally route store through the same batch
API, since a negative `device_id` marks the request as a plain CPU buffer:

- Build `phxfs_io_req_t` with `device_id = -1` (any negative value) and
  `buf = tensor.data_ptr()` (CPU pinned memory), then call
  `phxcache.PhxBatch.write(reqs)`. Phoenix treats it as a plain host buffer and
  issues the writes through the same pool.
- Requires O_DIRECT alignment on the write side too (pad chunk payload), and the
  CPU tensor should be **pinned** (`torch` pinned memory) for best throughput.

This is optional for a first cut — the read path is where the win is. If you
keep POSIX writes, still consider **packing** on write (append chunks into a
pack file + update the index) so reads can batch (§3.2).

---

## 6. vLLM + LMCache layerwise

Layerwise load/store streams KV **per layer**, which is naturally "many small
I/Os per step" — a perfect fit for batch. Recommended file layout:

- **Pack per (sequence/request, layer)**: one file (or one region) holds all KV
  blocks for a given layer of a given sequence. A layerwise step then issues a
  single `phxfs_read_batch` over that layer's blocks — all sharing one fd,
  distinct `f_offset` — while the model computes the previous layer.
- Use the **async** API for overlap:

  ```python
  b = phxcache.PhxBatch.submit_read(layer_reqs)   # issue layer L's KV load
  compute_layer(L-1)                              # overlap: GPU computes prev layer
  results = b.wait()                              # ensure layer L KV is resident
  ```

  Submitted batches queue on a bounded FIFO (16 slots) and the pool's workers
  service queued batches round-robin, so you can **prefetch several layers
  ahead**: submit L+1, L+2, … without waiting for the previous one, then
  `wait()` the handles in order. If the queue is full, submit blocks until a
  worker frees a slot (backpressure) — no EBUSY handling is needed.

- Index granularity: store `(layer, block) -> (pack_file, offset, len)`. On the
  store side, append a layer's blocks contiguously so the read offsets within a
  layer are sequential (best for readahead / large effective I/O).

> **Multi-process (vLLM MP) note.** Each process gets its own pool/rings and
> device fds — no cross-process coordination is needed. The pool registers a
> `pthread_atfork` child handler, so a forked child lazily re-creates the pool
> on first use and may use the batch API normally. One caveat: an async batch
> handle submitted *before* fork() does not survive into the child — always
> `phxfs_batch_wait()`/`phxfs_batch_destroy()` outstanding handles before
> forking.

---

## 7. Checklist for the PRs

`phxcache` (phoenix#7):

- [ ] Add `PhxBatchReq` + `PhxBatch` (read/write, submit/wait) in
      `phx_cache.h/.cpp`.
- [ ] Bind them in `bindings.cpp` with `py::call_guard<py::gil_scoped_release>()`
      on `read`/`write`/`wait`.
- [ ] `setup.py`: linkage already includes `phoenix uring cuda cudart` — no
      change needed.

`phx_l2_adapter.py` (LMCache#1):

- [ ] Introduce a packed-file store + a `chunk -> (file, offset, len)` index
      (or, interim, an fd cache; see §3.3), padding chunks for O_DIRECT.
- [ ] Rewrite `_process_load` to build one `PhxBatch.read` for all GPU chunks
      (§4); keep the POSIX fallback for CPU objects.
- [ ] (Optional) route store through `PhxBatch.write` with CPU (pinned) buffers
      (§5), or at least pack on write.
- [ ] (Layerwise) per-(seq,layer) packing + async `submit_read`/`wait` overlap
      (§6).
- [ ] Keep the single `_worker_loop` thread — **no extra Python worker
      threads**; concurrency lives inside Phoenix.

`phx_file_memory_allocator.py`: unchanged (still one `regmem` over the whole GPU
buffer; the batch `buf_offset` is computed from `tensor.data_ptr() - base`,
exactly as `_load_dma` already does).

---

## 8. Why not multi-thread the adapter instead?

Because retrieve runs under the Python GIL: N Python worker threads calling
`f.read()` would serialise on the GIL and add complexity to the L2 interface.
Pushing the fan-out **below** the C boundary (one `PhxBatch.read` that releases
the GIL and internally uses a shared pool of io_uring rings) gives full
disk bandwidth with a **single** adapter thread and a **single** GIL crossing.
This was verified on a 4-disk RAID0: one batch call ≈ 25 GB/s vs ≈ 2.6 GB/s for
the per-chunk loop.
