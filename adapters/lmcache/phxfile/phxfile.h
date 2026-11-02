#ifndef PHXFILE_H
#define PHXFILE_H
/* phxfile.h — Phoenix file-IO frozen ABI for LMCache (GDS L1 tier).
 *
 * libphxfile.so is the frozen layer between LMCache and libphoenix: the
 * symbols, signatures and semantics below NEVER change once published.
 * libphoenix evolution (e.g. dropping the phxfs device parameter from
 * phxfs_regmem/deregmem) is absorbed inside this library; consumers
 * (LMCache's ctypes wrapper _phx_async.py) only need the library
 * reinstalled, never re-coded.
 *
 * Naming and semantics deliberately mirror AMD hipFile (libhipfile.so) so
 * the four LMCache GDS backend wrappers (cuFile / hipFile / uGDS / phx)
 * stay line-by-line analogous. The IO parameter order here is hipFile's
 * (file_offset before buf_offset); libphoenix's own order differs and the
 * swap is performed inside this shim.
 *
 * Thread safety: all functions may be called concurrently. Registration
 * paths take an internal lock; the IO submission path is lock-free.
 *
 * Error model: every function returns int — 0 on success, a negative
 * -errno on failure. Transfer failures of async IO are NOT reported in
 * the return value: they land in *bytes_done after the stream is
 * synchronized past the submission.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * Driver lifecycle
 * ------------------------------------------------------------------ */

/* Process-level initialization. Idempotent. Currently a no-op placeholder
 * (phxfs devices are opened lazily on first buffer registration); kept in
 * the frozen surface for hipFile parity and future eager setup. */
int phxFileDriverOpen(void);

/* Release all shim-side state: deregisters every buffer still in the
 * registration table and closes every opened phxfs device. Idempotent.
 * Individual cleanup failures are reported on stderr and do not fail the
 * call (mirrors the tolerance of the cuFile-path teardown). */
int phxFileDriverClose(void);

/* ------------------------------------------------------------------ *
 * Buffer registration (frozen core: no device, no alignment, no
 * target_addr — all of that lives inside the shim / libphoenix)
 * ------------------------------------------------------------------ */

/* Register device memory for DMA. `length` is any positive size: the shim
 * rounds it up to the device page size for the underlying registration
 * (the IO size stays independent and must not exceed `length`).
 *
 * Device resolution is probe-based (mirroring how libphoenix's own stream
 * path resolves buffers at IO time): on first use the shim opens every
 * FULL-mode phxfs device, and the registration is attempted on each
 * until one accepts it — the underlying kernel P2P mapping only succeeds
 * on the GPU whose BAR covers the buffer, and failed attempts roll back
 * cleanly inside libphoenix, so probing is side-effect-free. Registration
 * is a low-frequency control operation, so the probe loop's cost is
 * irrelevant. Staging-mode devices are skipped (their stream path is
 * unsupported by libphoenix anyway; opening them would allocate their
 * staging pools for nothing). Exact duplicates are reference-counted
 * inside libphoenix; every successful call is recorded and must be
 * matched by one deregister.
 *
 * Returns 0, or -errno: -ENODEV (no phxfs device present),
 * -EOPNOTSUPP (all present devices are staging-mode), or the last
 * registration error. */
int phxFileBufRegister(const void *addr, size_t length);

/* Deregister a previously registered buffer. Only the base address is
 * needed: the aligned length and phxfs device are played back from the
 * shim's bookkeeping. An unregistered address is silently tolerated
 * (cuFile-path teardown parity). On failure the entry is kept so a retry
 * can re-attempt the deregistration. */
int phxFileBufDeregister(const void *addr);

/* ------------------------------------------------------------------ *
 * File handle
 * ------------------------------------------------------------------ */

/* Wrap an open POSIX fd into an opaque file handle. Currently an identity
 * boxing (fh == (void *)(intptr_t)fd; phxfs performs IO on plain fds);
 * should libphoenix ever grow library-side handle state, this evolves
 * into a real registration table behind the same frozen signature. */
int phxFileHandleRegister(void **fh, int fd);

/* Reverse of phxFileHandleRegister. Idempotent. */
int phxFileHandleDeregister(void *fh);

/* ------------------------------------------------------------------ *
 * Stream registration
 * ------------------------------------------------------------------ */

/* Register a CUDA/ROCm stream. Correctness does not depend on it (every
 * IO submission carries the stream, like cuFileReadAsync); the symbols
 * are frozen for wrapper-surface parity and reserved for future per-stream
 * resource pre-claiming (e.g. staging slots). Currently no-ops. */
int phxFileStreamRegister(void *stream);
int phxFileStreamDeregister(void *stream);

/* ------------------------------------------------------------------ *
 * Stream-ordered IO submission
 *
 * Semantic contract (identical to cuFile/hipFile/uGDS async — LMCache's
 * gds_context event-checkpoint machinery is built on it):
 *   1. Submission returns immediately and never blocks the caller.
 *   2. The DMA is stream-ordered: it runs after every op the caller
 *      enqueued on `stream` before the submission, and before every op
 *      enqueued after it.
 *   3. Transfer failures are NOT raised: they land in *bytes_done once
 *      the stream has been synchronized past the submission
 *      (< 0 means -errno; otherwise the transferred byte count).
 *   4. The four pointer arguments are late-binding: the library
 *      dereferences them when the stream executes the op, so the caller
 *      must keep the storage alive and unmodified until then.
 *   5. [buf_offset, buf_offset + *nbytes) must stay inside the
 *      registration of buf_base (failing extents surface as a negative
 *      *bytes_done, e.g. -EFAULT).
 * ------------------------------------------------------------------ */
int phxFileReadAsync(void *fh, void *buf_base,
                     size_t *nbytes, int64_t *file_offset,
                     int64_t *buf_offset, int64_t *bytes_done,
                     void *stream);

int phxFileWriteAsync(void *fh, void *buf_base,
                      size_t *nbytes, int64_t *file_offset,
                      int64_t *buf_offset, int64_t *bytes_done,
                      void *stream);

#ifdef __cplusplus
}
#endif

#endif /* PHXFILE_H */
