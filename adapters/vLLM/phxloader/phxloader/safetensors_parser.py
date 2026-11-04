# SPDX-License-Identifier: Apache-2.0
"""Safetensors header parser for Phoenix weight loading.

Parses the JSON header to extract tensor metadata without reading
tensor data. Includes boundary validation and header_len upper-bound
protection.
"""

import json
import os

import torch

# Mapping from safetensors dtype strings to torch dtypes
SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def parse_safetensor_header(
    st_file: str,
) -> tuple[dict[str, tuple[torch.dtype, tuple[int, ...], int, int]], int]:
    """Parse safetensors file header.

    Returns:
        A tuple of (tensor_meta, header_size) where tensor_meta maps
        name -> (dtype, shape, start_offset_in_data, nbytes).
        header_size is the total header size (8 + json_len).
    """
    with open(st_file, "rb") as f:
        # First 8 bytes: little-endian uint64 = JSON header length
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(
                f"Safetensors file '{st_file}' is too small: expected 8 bytes "
                f"for header length field, got {len(length_bytes)}"
            )
        header_len = int.from_bytes(length_bytes, "little")
        file_size = os.path.getsize(st_file)
        if header_len > file_size - 8:
            raise ValueError(
                f"Safetensors file '{st_file}': declared header length "
                f"{header_len} exceeds remaining file size {file_size - 8}"
            )

        header_json = f.read(header_len)
        if len(header_json) != header_len:
            raise ValueError(
                f"Safetensors file '{st_file}' is truncated: expected "
                f"{header_len} bytes of header JSON, got {len(header_json)}"
            )

    header = json.loads(header_json)

    tensor_meta: dict[str, tuple[torch.dtype, tuple[int, ...], int, int]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        offsets = info["data_offsets"]  # [start, end] within data section
        nbytes = offsets[1] - offsets[0]

        dtype = SAFETENSORS_DTYPE_TO_TORCH.get(dtype_str)
        if dtype is None:
            raise ValueError(
                f"Unsupported safetensors dtype '{dtype_str}' for tensor "
                f"'{name}'"
            )

        tensor_meta[name] = (dtype, shape, offsets[0], nbytes)

    header_size = 8 + header_len
    return tensor_meta, header_size
