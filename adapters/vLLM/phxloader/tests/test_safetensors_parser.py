# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``parse_safetensor_header`` (pure Python, no hardware).

Tests cover:
  - Basic parsing (dtype, shape, offsets, nbytes)
  - All supported dtype mappings (12 types)
  - ``__metadata__`` is skipped
  - Multiple tensors in one file
  - header_size computation
  - Error handling: truncated file, oversized header, truncated JSON,
    unsupported dtype
"""

import json
import os
import struct

import pytest
import torch

from phxloader.safetensors_parser import (
    SAFETENSORS_DTYPE_TO_TORCH,
    parse_safetensor_header,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path, header_dict, data=b""):
    """Write a safetensors file with a given header dict and data blob."""
    header_json = json.dumps(header_dict).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(data)
    return 8 + len(header_json)


def _write_raw(path, raw_bytes):
    """Write arbitrary bytes to a file (for malformed-file tests)."""
    with open(path, "wb") as f:
        f.write(raw_bytes)


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_parse_basic(make_safetensors):
    """Parse a single F32 tensor with shape [4]."""
    data = b"\x00\x00\x80\x3f" * 4  # 1.0f repeated
    path, _ = make_safetensors("basic.safetensors", {
        "weight": {"dtype": "F32", "shape": [4], "data": data},
    })

    meta, header_size = parse_safetensor_header(path)

    assert "weight" in meta
    dtype, shape, data_start, nbytes = meta["weight"]
    assert dtype == torch.float32
    assert shape == (4,)
    assert data_start == 0
    assert nbytes == 16  # 4 × 4 bytes


def test_parse_header_size(make_safetensors):
    """header_size = 8 + len(json_bytes)."""
    path, _ = make_safetensors("hs.safetensors", {
        "w": {"dtype": "F32", "shape": [2], "data": b"\x00" * 8},
    })
    _, header_size = parse_safetensor_header(path)
    # The factory returns header_size too; cross-check
    expected = 8 + len(json.dumps({
        "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }).encode())
    assert header_size == expected


# ---------------------------------------------------------------------------
# Dtype mapping (all 12 supported types)
# ---------------------------------------------------------------------------

# (safetensors_dtype_str, torch_dtype, element_size_bytes)
DTYPE_CASES = [
    ("BOOL", torch.bool, 1),
    ("U8", torch.uint8, 1),
    ("I8", torch.int8, 1),
    ("I16", torch.int16, 2),
    ("I32", torch.int32, 4),
    ("I64", torch.int64, 8),
    ("F16", torch.float16, 2),
    ("BF16", torch.bfloat16, 2),
    ("F32", torch.float32, 4),
    ("F64", torch.float64, 8),
    ("F8_E4M3", torch.float8_e4m3fn, 1),
]


@pytest.mark.parametrize("st_dtype, torch_dtype, elem_size", DTYPE_CASES)
def test_parse_all_dtypes(make_safetensors, st_dtype, torch_dtype, elem_size):
    """Each safetensors dtype string maps to the correct torch dtype."""
    data = b"\x00" * (elem_size * 3)  # 3 elements
    path, _ = make_safetensors(f"dt_{st_dtype}.safetensors", {
        "t": {"dtype": st_dtype, "shape": [3], "data": data},
    })
    meta, _ = parse_safetensor_header(path)
    dtype, shape, _, nbytes = meta["t"]
    assert dtype == torch_dtype
    assert shape == (3,)
    assert nbytes == elem_size * 3


def test_parse_f8_e8m0_dtype(make_safetensors):
    """F8_E8M0 maps to torch.float8_e8m0fnu (if the torch version supports it)."""
    f8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if f8_e8m0 is None:
        pytest.skip("torch.float8_e8m0fnu not available in this torch version")

    data = b"\x00" * 4
    path, _ = make_safetensors("f8e8m0.safetensors", {
        "t": {"dtype": "F8_E8M0", "shape": [4], "data": data},
    })
    meta, _ = parse_safetensor_header(path)
    dtype, _, _, nbytes = meta["t"]
    assert dtype == f8_e8m0
    assert nbytes == 4


def test_dtype_map_completeness():
    """SAFETENSORS_DTYPE_TO_TORCH should contain at least the core types."""
    expected_keys = {"BOOL", "U8", "I8", "I16", "I32", "I64",
                     "F16", "BF16", "F32", "F64", "F8_E4M3"}
    assert expected_keys.issubset(set(SAFETENSORS_DTYPE_TO_TORCH.keys()))


# ---------------------------------------------------------------------------
# __metadata__ handling
# ---------------------------------------------------------------------------

def test_parse_metadata_skipped(make_safetensors):
    """``__metadata__`` key must not appear in the tensor dict."""
    data = b"\x00" * 4
    path, _ = make_safetensors("meta.safetensors", {
        "w": {"dtype": "F32", "shape": [1], "data": data},
    }, metadata={"format": "pt", "version": "1.0"})

    meta, _ = parse_safetensor_header(path)
    assert "w" in meta
    assert "__metadata__" not in meta


# ---------------------------------------------------------------------------
# Multiple tensors
# ---------------------------------------------------------------------------

def test_parse_multiple_tensors(make_safetensors):
    """Parse a file with several tensors at different offsets."""
    t0_data = b"\x01" * 12   # 3 × F32
    t1_data = b"\x02" * 8    # 4 × I16
    t2_data = b"\x03" * 2    # 2 × U8

    path, _ = make_safetensors("multi.safetensors", {
        "t0": {"dtype": "F32", "shape": [3], "data": t0_data},
        "t1": {"dtype": "I16", "shape": [4], "data": t1_data},
        "t2": {"dtype": "U8",  "shape": [2], "data": t2_data},
    })

    meta, _ = parse_safetensor_header(path)
    assert len(meta) == 3

    # t0
    _, _, s0, n0 = meta["t0"]
    assert s0 == 0 and n0 == 12

    # t1 (offset right after t0)
    _, _, s1, n1 = meta["t1"]
    assert s1 == 12 and n1 == 8

    # t2
    _, _, s2, n2 = meta["t2"]
    assert s2 == 20 and n2 == 2


def test_parse_empty_tensors(make_safetensors):
    """A file with zero tensors (only __metadata__) should yield empty dict."""
    path, _ = make_safetensors("empty.safetensors", {}, metadata={"note": "empty"})
    meta, _ = parse_safetensor_header(path)
    assert meta == {}


def test_parse_scalar_shape(make_safetensors):
    """A scalar tensor has shape [] (empty tuple)."""
    data = b"\x00\x00\x80\x3f"
    path, _ = make_safetensors("scalar.safetensors", {
        "s": {"dtype": "F32", "shape": [], "data": data},
    })
    meta, _ = parse_safetensor_header(path)
    _, shape, _, _ = meta["s"]
    assert shape == ()


def test_parse_multidim_shape(make_safetensors):
    """Multi-dimensional shape is preserved as a tuple."""
    data = b"\x00" * 24  # 2×3×4 × 1 byte (U8)
    path, _ = make_safetensors("nd.safetensors", {
        "t": {"dtype": "U8", "shape": [2, 3, 4], "data": data},
    })
    meta, _ = parse_safetensor_header(path)
    _, shape, _, _ = meta["t"]
    assert shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_parse_truncated_file_too_small(tmp_path):
    """A file with fewer than 8 bytes should raise ValueError."""
    path = str(tmp_path / "trunc.safetensors")
    _write_raw(path, b"\x00\x01\x02")
    with pytest.raises(ValueError, match="too small"):
        parse_safetensor_header(path)


def test_parse_oversized_header(tmp_path):
    """header_len exceeding remaining file size should raise ValueError."""
    path = str(tmp_path / "oversize.safetensors")
    # Claim 1 MB header but file is tiny
    _write_raw(path, struct.pack("<Q", 1024 * 1024) + b"{}")
    with pytest.raises(ValueError, match="exceeds remaining file size"):
        parse_safetensor_header(path)


def test_parse_truncated_json(tmp_path):
    """File claims more header bytes than actually present should raise."""
    path = str(tmp_path / "truncjson.safetensors")
    header_len = 100
    _write_raw(path, struct.pack("<Q", header_len) + b'{"w": {')  # only 7 bytes
    with pytest.raises(ValueError, match="truncated"):
        parse_safetensor_header(path)


def test_parse_unsupported_dtype(make_safetensors):
    """An unknown dtype string should raise ValueError."""
    data = b"\x00" * 4
    path, _ = make_safetensors("baddtype.safetensors", {
        "w": {"dtype": "F128", "shape": [1], "data": data},
    })
    with pytest.raises(ValueError, match="Unsupported safetensors dtype"):
        parse_safetensor_header(path)


def test_parse_nonexistent_file(tmp_path):
    """Opening a non-existent file should raise."""
    path = str(tmp_path / "nope.safetensors")
    with pytest.raises((FileNotFoundError, OSError)):
        parse_safetensor_header(path)


def test_parse_exact_offset_boundaries(make_safetensors):
    """data_offsets [start, end] must yield nbytes = end - start."""
    data = b"\xAB" * 100
    path, _ = make_safetensors("offsets.safetensors", {
        "w": {"dtype": "U8", "shape": [100], "data": data},
    })
    meta, _ = parse_safetensor_header(path)
    _, _, start, nbytes = meta["w"]
    assert start == 0
    assert nbytes == 100
