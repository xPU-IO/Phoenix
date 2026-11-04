# SPDX-License-Identifier: Apache-2.0
"""Hardware-dependent tests for PhxLoader (require device + phxfs device).

All tests are decorated with ``@skip_no_hardware`` and will be skipped
automatically when no phxfs device / device is available.

``PhxLoader.load_tensors_into_buffer`` uses the single-I/O ``phxfs_read``
API.  After recompiling against the latest libphoenix.so (Aug 2026),
phxfs_read works correctly — no xfail markers needed.
"""

import os

import pytest
import torch

from .conftest import skip_no_hardware


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_loader_init(phx_loader):
    """PhxLoader should instantiate without error."""
    assert phx_loader is not None


# ---------------------------------------------------------------------------
# regmem / deregmem
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_regmem_returns_valid_addr(phx_loader, gpu_buffer):
    """regmem should return a non-zero mapped address."""
    target = phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    assert target != 0
    phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_deregmem_unregistered_noop(phx_loader, gpu_buffer):
    """Deregistering an unregistered buffer should be a safe no-op."""
    phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_regmem_deregmem_multiple(phx_loader):
    """Register and deregister two distinct device buffers."""
    buf1 = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    buf2 = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    t1 = phx_loader.regmem(buf1.data_ptr(), buf1.nbytes)
    t2 = phx_loader.regmem(buf2.data_ptr(), buf2.nbytes)
    assert t1 != 0
    assert t2 != 0
    phx_loader.deregmem(buf1.data_ptr(), buf1.nbytes)
    phx_loader.deregmem(buf2.data_ptr(), buf2.nbytes)
    del buf1, buf2
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# load_tensors_into_buffer (sync)
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_load_tensors_into_buffer(phx_loader, gpu_buffer, make_aligned_file):
    """Sync DMA load: write data to file, DMA-read to device, verify content."""
    expected = torch.arange(4096, dtype=torch.uint8)
    path, aligned_info = make_aligned_file("load.safetensors", [
        (0, expected.numpy().tobytes()),
    ])
    aligned_offset, aligned_nbytes = aligned_info[0]

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_buffer.zero_()
        batch = [(0, aligned_offset, aligned_nbytes)]
        phx_loader.load_tensors_into_buffer(path, gpu_buffer.data_ptr(), batch)

        got = gpu_buffer[:4096].cpu()
        assert torch.equal(got, expected), "DMA load data mismatch"
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_load_tensors_into_buffer_empty_batch(phx_loader, gpu_buffer, tmp_path):
    """An empty batch should be a no-op (no file access, no exception)."""
    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        phx_loader.load_tensors_into_buffer(
            str(tmp_path / "nonexistent.safetensors"),
            gpu_buffer.data_ptr(),
            [],
        )
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_load_unregistered_ptr_raises(phx_loader, tmp_path):
    """Loading into an unregistered device pointer should raise RuntimeError."""
    buf = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
    path = str(tmp_path / "test.safetensors")
    with open(path, "wb") as f:
        f.truncate(4096)

    with pytest.raises(Exception):
        phx_loader.load_tensors_into_buffer(
            path, buf.data_ptr(), [(0, 0, 4096)]
        )
    del buf
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# load_tensors_into_buffer_async + wait_dma
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_load_async_and_wait(phx_loader, gpu_buffer, make_aligned_file):
    """Async DMA load followed by wait_dma should produce correct data."""
    expected = torch.arange(4096, dtype=torch.uint8)
    path, aligned_info = make_aligned_file("async.safetensors", [
        (0, expected.numpy().tobytes()),
    ])
    aligned_offset, aligned_nbytes = aligned_info[0]

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_buffer.zero_()
        batch = [(0, aligned_offset, aligned_nbytes)]
        phx_loader.load_tensors_into_buffer_async(
            path, gpu_buffer.data_ptr(), batch
        )
        phx_loader.wait_dma()

        got = gpu_buffer[:4096].cpu()
        assert torch.equal(got, expected), "Async DMA load data mismatch"
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_wait_dma_without_async_noop(phx_loader):
    """Calling wait_dma without a prior async call should not raise."""
    phx_loader.wait_dma()


@skip_no_hardware
def test_async_overlaps_previous(phx_loader, gpu_buffer, make_aligned_file):
    """Submitting a second async before wait_dma of the first is allowed."""
    expected1 = torch.arange(4096, dtype=torch.uint8)
    expected2 = torch.arange(4096, 8192, dtype=torch.uint8)

    path1, info1 = make_aligned_file("a1.safetensors", [
        (0, expected1.numpy().tobytes()),
    ])
    path2, info2 = make_aligned_file("a2.safetensors", [
        (0, expected2.numpy().tobytes()),
    ])

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        phx_loader.load_tensors_into_buffer_async(
            path1, gpu_buffer.data_ptr(),
            [(0, info1[0][0], info1[0][1])],
        )
        phx_loader.load_tensors_into_buffer_async(
            path2, gpu_buffer.data_ptr(),
            [(0, info2[0][0], info2[0][1])],
        )
        phx_loader.wait_dma()

        got = gpu_buffer[:4096].cpu()
        assert torch.equal(got, expected2), "Overlapping async data mismatch"
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


# ---------------------------------------------------------------------------
# DMA timer
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_dma_timer_reset_and_read(phx_loader):
    """reset → 0, get_dma_seconds returns a non-negative float."""
    phx_loader.reset_dma_timer()
    assert phx_loader.get_dma_seconds() == 0.0
    # After reset, timer should be 0 (even if no DMA has occurred)
    assert isinstance(phx_loader.get_dma_seconds(), float)


@skip_no_hardware
def test_dma_timer_after_dma(phx_loader, gpu_buffer, make_aligned_file):
    """reset → DMA → get_dma_seconds > 0 → reset → 0."""
    phx_loader.reset_dma_timer()
    assert phx_loader.get_dma_seconds() == 0.0

    expected = torch.arange(4096, dtype=torch.uint8)
    path, info = make_aligned_file("timer.safetensors", [
        (0, expected.numpy().tobytes()),
    ])

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        phx_loader.load_tensors_into_buffer(
            path, gpu_buffer.data_ptr(),
            [(0, info[0][0], info[0][1])],
        )
        elapsed = phx_loader.get_dma_seconds()
        assert elapsed > 0.0, "DMA timer should be > 0 after a DMA operation"

        phx_loader.reset_dma_timer()
        assert phx_loader.get_dma_seconds() == 0.0
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


# ---------------------------------------------------------------------------
# Multi-entry batch
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_multi_entry_batch(phx_loader, gpu_buffer, make_aligned_file):
    """Load multiple entries in a single batch call."""
    data1 = torch.arange(4096, dtype=torch.uint8)
    data2 = torch.arange(4096, 8192, dtype=torch.uint8)

    path, info = make_aligned_file("multi.safetensors", [
        (0, data1.numpy().tobytes()),
        (4096, data2.numpy().tobytes()),
    ])

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_buffer.zero_()
        batch = [
            (0, info[0][0], info[0][1]),
            (4096, info[1][0], info[1][1]),
        ]
        phx_loader.load_tensors_into_buffer(path, gpu_buffer.data_ptr(), batch)

        got1 = gpu_buffer[:4096].cpu()
        got2 = gpu_buffer[4096:8192].cpu()
        assert torch.equal(got1, data1), "Batch entry 1 mismatch"
        assert torch.equal(got2, data2), "Batch entry 2 mismatch"
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


# ---------------------------------------------------------------------------
# EOF boundary tests
#
# In production, offset and nbytes are always 4K-aligned (read_group.py
# guarantees this). File sizes are arbitrary (safetensors files are not
# 4K-aligned). When the aligned read reaches EOF, the old FULL-mode path
# (xfer + pread) returned a short read and phxloader handled it gracefully.
# The new staging-mode path returns EIO instead, which crashes vLLM.
#
# These tests verify that reading at EOF boundary works correctly —
# the read should either succeed with a short read (like pread) or
# be handled so that loading continues.
# ---------------------------------------------------------------------------


def _write_file(path, size):
    """Create a file of exactly `size` bytes with a repeating pattern."""
    data = bytes(range(256)) * (size // 256 + 1)
    with open(path, "wb") as f:
        f.write(data[:size])
        os.fsync(f.fileno())
    return data[:size]


@skip_no_hardware
def test_read_within_eof(phx_loader, gpu_buffer, tmp_path):
    """Aligned read fully within file bounds"""
    path = str(tmp_path / "test.bin")
    file_data = _write_file(path, 4196)  # not 4K-aligned, but read fits

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_buffer.zero_()
        phx_loader.load_tensors_into_buffer(path, gpu_buffer.data_ptr(), [(0, 0, 4096)])
        got = gpu_buffer[:4096].cpu()
        assert torch.equal(got, torch.tensor(list(file_data[:4096]), dtype=torch.uint8))
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_read_beyond_eof(phx_loader, gpu_buffer, tmp_path):
    """Aligned read that exceeds EOF

    File is 4196 bytes (not 4K-aligned). Read requests 8192 bytes
    (4K-aligned). The old FULL-mode path (pread) returned 4196 bytes
    (short read) and phxloader logged it and continued. The staging-mode
    path returns EIO, crashing vLLM.

    Expected behavior: phxfs_read or phx_loader should handle the EOF
    boundary so that the 4196 valid bytes are read and loading continues.
    """
    path = str(tmp_path / "test.bin")
    file_data = _write_file(path, 4196)

    phx_loader.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_buffer.zero_()
        phx_loader.load_tensors_into_buffer(path, gpu_buffer.data_ptr(), [(0, 0, 8192)])
        got = gpu_buffer[:4196].cpu()
        assert torch.equal(got, torch.tensor(list(file_data), dtype=torch.uint8))
    finally:
        phx_loader.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


# ---------------------------------------------------------------------------
# close semantics
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_close_idempotent():
    """Calling close() twice should not raise."""
    from phxloader import PhxLoader
    loader = PhxLoader(device_id=0)
    loader.close()
    loader.close()


# ---------------------------------------------------------------------------
# Invalid device handling (must run LAST — corrupts CUDA context)
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_invalid_device_raises():
    """An out-of-range device_id should raise.

    WARNING: phxfs_find_dev(invalid_id) calls cuDeviceGetPCIBusId with an
    invalid ordinal, which sets the CUDA error sticky bit.  We reset the
    device context afterwards so subsequent tests are not affected.

    This test is deliberately placed at the end of the module.
    """
    from phxloader import PhxLoader
    with pytest.raises(Exception):
        PhxLoader(device_id=99999)

    torch.cuda.synchronize()
    try:
        _ = torch.empty(1, device="cuda:0")
    except Exception:
        torch.cuda.synchronize()
