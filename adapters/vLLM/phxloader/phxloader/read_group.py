# SPDX-License-Identifier: Apache-2.0
"""Read group construction for Phoenix selective tensor loading.

A ReadGroup merges adjacent tensors (gap < merge_threshold) into a single
DMA read, reducing I/O count while respecting O_DIRECT 4K alignment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

# Alignment constants
PAGE_SIZE = 4096

# Type alias: name -> (dtype, shape, data_start_in_data_section, nbytes)
TensorMeta = Dict[str, Tuple[torch.dtype, Tuple[int, ...], int, int]]


def align_up(val: int, alignment: int) -> int:
    return (val + alignment - 1) & ~(alignment - 1)


def align_down(val: int, alignment: int) -> int:
    return val & ~(alignment - 1)


@dataclass
class ReadGroup:
    """A group of adjacent tensors read in a single DMA transfer.

    Attributes:
        aligned_f_offset: align_down(first_tensor_f_offset, PAGE_SIZE)
        read_size: aligned_end - aligned_f_offset (multiple of PAGE_SIZE)
        pre_padding: first_tensor_f_offset - aligned_f_offset (0..PAGE_SIZE-1)
        first_tensor_f_offset: header_size + first_tensor_data_start
        tensors: name -> (dtype, shape, data_start, nbytes)
    """
    aligned_f_offset: int
    read_size: int
    pre_padding: int
    first_tensor_f_offset: int
    tensors: TensorMeta = field(default_factory=dict)


@dataclass
class FilePlan:
    """Pre-scan result for a single safetensors file.

    Attributes:
        path: file path
        header_size: total header size (8 + json_len)
        groups: list of ReadGroup, sorted by file offset
        slots: buf_offset for each group (4K aligned)
        file_buf_size: total buffer bytes needed for this file
    """
    path: str
    header_size: int
    groups: List[ReadGroup]
    slots: List[int]
    file_buf_size: int


def build_read_groups(
    needed: TensorMeta,
    header_size: int,
    merge_threshold: int = 65536,
) -> List[ReadGroup]:
    """Sort needed tensors by file offset, merge adjacent (gap < threshold),
    compute aligned read parameters for each group.

    Args:
        needed: dict of name -> (dtype, shape, data_start, nbytes)
        header_size: total header size (8 + json_len)
        merge_threshold: max gap between adjacent tensors to merge (bytes)

    Returns:
        List of ReadGroup, sorted by file offset
    """
    if not needed:
        return []

    # Compute file offsets and sort by file offset
    sorted_tensors = sorted(
        ((header_size + data_start, name, dtype, shape, data_start, nbytes)
         for name, (dtype, shape, data_start, nbytes) in needed.items()),
        key=lambda x: x[0],
    )

    groups: List[ReadGroup] = []
    current_tensors: TensorMeta = {}
    current_first_foffset: int | None = None
    current_last_end: int = 0

    for f_offset, name, dtype, shape, data_start, nbytes in sorted_tensors:
        tensor_end = f_offset + nbytes

        if current_first_foffset is not None:
            gap = f_offset - current_last_end
            if gap < merge_threshold:
                # Merge into current group
                current_tensors[name] = (dtype, shape, data_start, nbytes)
                current_last_end = tensor_end
                continue

        # Start new group — finalize previous if exists
        if current_first_foffset is not None:
            groups.append(_make_group(
                current_first_foffset, current_last_end, current_tensors))

        current_first_foffset = f_offset
        current_last_end = tensor_end
        current_tensors = {name: (dtype, shape, data_start, nbytes)}

    # Finalize last group
    if current_first_foffset is not None:
        groups.append(_make_group(
            current_first_foffset, current_last_end, current_tensors))

    return groups


def _make_group(
    first_foffset: int,
    last_end: int,
    tensors: TensorMeta,
) -> ReadGroup:
    aligned_f_offset = align_down(first_foffset, PAGE_SIZE)
    aligned_end = align_up(last_end, PAGE_SIZE)
    read_size = aligned_end - aligned_f_offset
    pre_padding = first_foffset - aligned_f_offset

    return ReadGroup(
        aligned_f_offset=aligned_f_offset,
        read_size=read_size,
        pre_padding=pre_padding,
        first_tensor_f_offset=first_foffset,
        tensors=dict(tensors),
    )


def build_file_plan(
    path: str,
    header_size: int,
    groups: List[ReadGroup],
) -> FilePlan:
    """Assign buffer slots to groups and compute total buffer size.

    Each group gets a 4K-aligned slot in the buffer. Since read_size is
    already 4K-aligned and slots start at 0, all slots are naturally 4K-aligned.
    """
    slots: List[int] = []
    current_offset = 0
    for group in groups:
        slots.append(current_offset)
        current_offset += group.read_size

    return FilePlan(
        path=path,
        header_size=header_size,
        groups=groups,
        slots=slots,
        file_buf_size=current_offset,
    )
