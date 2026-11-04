# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and hardware detection for phxloader unit tests."""

import json
import os
import struct

import pytest
import torch


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: tests that require a real device + phxfs device",
    )


# ---------------------------------------------------------------------------
# Hardware availability detection
# ---------------------------------------------------------------------------

def _check_hardware() -> bool:
    """Check that phxloader can be imported and CUDA is available.

    Does NOT instantiate PhxLoader (opening/closing the phxfs device at
    import time can interfere with subsequent CUDA operations).  The
    actual instantiation happens in the ``phx_loader`` fixture, which
    skips on failure.
    """
    try:
        import phxloader  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_HARDWARE = _check_hardware()

skip_no_hardware = pytest.mark.skipif(
    not HAS_HARDWARE,
    reason="no device or phxfs available (set up phxfs + compile phxloader)",
)


# ---------------------------------------------------------------------------
# Safetensors test-file factory
# ---------------------------------------------------------------------------

def _make_safetensors(path, tensors, metadata=None):
    """Build a minimal safetensors file on disk.

    Args:
        path: output file path
        tensors: dict of name -> dict with keys
            ``dtype`` (str, e.g. "F32"), ``shape`` (list[int]),
            ``data`` (bytes, raw tensor data).  ``data_offsets`` are
            auto-computed sequentially.
        metadata: optional dict for ``__metadata__``.

    The data section is padded to 4 KiB alignment (for O_DIRECT tests).
    """
    header = {}
    offset = 0
    data_chunks = []
    for name, info in tensors.items():
        raw = info["data"]
        start = offset
        end = offset + len(raw)
        header[name] = {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "data_offsets": [start, end],
        }
        data_chunks.append(raw)
        offset = end

    if metadata is not None:
        header["__metadata__"] = metadata

    header_json = json.dumps(header)
    header_bytes = header_json.encode("utf-8")
    header_len = len(header_bytes)

    # Pad data section to 4K for O_DIRECT compatibility
    total_data = b"".join(data_chunks)
    pad_len = (4096 - (len(total_data) % 4096)) % 4096
    total_data += b"\x00" * pad_len

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(total_data)

    return 8 + header_len  # header_size


@pytest.fixture
def make_safetensors(tmp_path):
    """Factory fixture: create safetensors files with arbitrary content.

    Usage::

        path, header_size = make_safetensors("model.safetensors", {
            "weight": {"dtype": "F32", "shape": [4], "data": b"..."},
        })
    """
    def _create(filename, tensors, metadata=None):
        path = str(tmp_path / filename)
        header_size = _make_safetensors(path, tensors, metadata)
        return path, header_size
    return _create


# ---------------------------------------------------------------------------
# Hardware fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def phx_loader():
    """Session-scoped PhxLoader instance on CUDA device 0.

    Skips if PhxLoader cannot be instantiated.
    """
    try:
        from phxloader import PhxLoader
    except ImportError:
        pytest.skip("phxloader not installed")
    try:
        loader = PhxLoader(device_id=0)
    except Exception as e:
        pytest.skip(f"PhxLoader init failed: {e}")
    yield loader
    loader.close()


@pytest.fixture
def gpu_buffer():
    """Function-scoped 8 MiB device buffer (uint8) for DMA tests."""
    buf = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    buf.zero_()
    yield buf
    del buf
    torch.cuda.empty_cache()


@pytest.fixture
def temp_file(tmp_path):
    """A temp file path (not yet created)."""
    return str(tmp_path / "phxloader_test.bin")


@pytest.fixture
def make_aligned_file(tmp_path):
    """Factory: create a file with 4K-aligned data at 4K-aligned offsets.

    Uses standard I/O to write data, then fsync to flush to disk so that
    phxfs_read (O_DIRECT) can read the data correctly.
    """
    def _create(filename, data_blocks):
        """data_blocks: list of (offset_in_file, bytes_data).
        Each offset and len(data) will be padded to 4096.
        Returns (path, list of (aligned_offset, aligned_nbytes)).
        """
        path = str(tmp_path / filename)
        # Determine file size
        max_end = 0
        for offset, data in data_blocks:
            aligned_end = ((offset + len(data) + 4095) // 4096) * 4096
            max_end = max(max_end, aligned_end)

        # Create file and write data blocks with standard I/O
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.ftruncate(fd, max_end)
            aligned_info = []
            for offset, data in data_blocks:
                aligned_offset = (offset // 4096) * 4096
                aligned_nbytes = (
                    ((offset + len(data) + 4095) // 4096) * 4096
                    - aligned_offset
                )
                padded = data + b"\x00" * (aligned_nbytes - len(data))
                os.pwrite(fd, padded, aligned_offset)
                aligned_info.append((aligned_offset, aligned_nbytes))
            os.fsync(fd)
        finally:
            os.close(fd)
        return path, aligned_info
    return _create
