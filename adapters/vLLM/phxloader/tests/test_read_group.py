# SPDX-License-Identifier: Apache-2.0
"""Unit tests for read_group.py (pure Python, no hardware).

Covers:
  - ``align_up`` / ``align_down`` boundary values
  - ``build_read_groups``: empty, single, merge, no-merge, sort, custom threshold
  - ``build_file_plan``: empty, single, multiple, slot assignment
  - End-to-end pipeline: parse → build_read_groups → build_file_plan
"""

import pytest
import torch

from phxloader.read_group import (
    PAGE_SIZE,
    ReadGroup,
    FilePlan,
    align_down,
    align_up,
    build_file_plan,
    build_read_groups,
)


# ---------------------------------------------------------------------------
# align_up / align_down
# ---------------------------------------------------------------------------

class TestAlignUp:
    def test_zero(self):
        assert align_up(0, 4096) == 0

    def test_exact_multiple(self):
        assert align_up(8192, 4096) == 8192

    def test_off_by_one(self):
        assert align_up(4097, 4096) == 8192

    def test_one_byte(self):
        assert align_up(1, 4096) == 4096

    def test_just_below_boundary(self):
        assert align_up(4095, 4096) == 4096

    def test_large_value(self):
        # 1_000_000 / 4096 = 244.14 → ceil = 245 → 245 * 4096 = 1_003_520
        assert align_up(1_000_000, 4096) == 1_003_520

    def test_alignment_one(self):
        """Alignment of 1 means no change."""
        assert align_up(42, 1) == 42


class TestAlignDown:
    def test_zero(self):
        assert align_down(0, 4096) == 0

    def test_exact_multiple(self):
        assert align_down(8192, 4096) == 8192

    def test_off_by_one(self):
        assert align_down(4097, 4096) == 4096

    def test_one_byte(self):
        assert align_down(1, 4096) == 0

    def test_just_below_boundary(self):
        assert align_down(4095, 4096) == 0

    def test_large_value(self):
        assert align_down(1_000_000, 4096) == 999_424

    def test_alignment_one(self):
        assert align_down(42, 1) == 42

    def test_idempotent(self):
        """align_down(align_up(x, a), a) == align_down(x, a) for aligned result."""
        x = 12345
        a = 4096
        assert align_down(align_up(x, a), a) == align_down(x, a) + (
            a if x % a != 0 else 0
        )


# ---------------------------------------------------------------------------
# build_read_groups
# ---------------------------------------------------------------------------

def _tensor_meta(dtype, shape, data_start, nbytes):
    """Convenience: build a single tensor meta entry."""
    return (dtype, tuple(shape), data_start, nbytes)


class TestBuildReadGroups:
    def test_empty(self):
        """Empty input → empty list."""
        assert build_read_groups({}, 8) == []

    def test_single_tensor(self):
        """One tensor → one group with correct alignment."""
        needed = {
            "w": _tensor_meta(torch.float32, [4], 0, 16),
        }
        groups = build_read_groups(needed, header_size=8)
        assert len(groups) == 1
        g = groups[0]
        # f_offset = header_size + data_start = 8 + 0 = 8
        assert g.first_tensor_f_offset == 8
        # aligned_down(8, 4096) = 0
        assert g.aligned_f_offset == 0
        # aligned_up(8 + 16, 4096) - 0 = 4096
        assert g.read_size == 4096
        # pre_padding = 8 - 0 = 8
        assert g.pre_padding == 8
        assert "w" in g.tensors

    def test_merge_adjacent_tensors(self):
        """Two tensors with gap < merge_threshold → one group."""
        needed = {
            "a": _tensor_meta(torch.float32, [4], 0, 16),
            "b": _tensor_meta(torch.float32, [4], 16, 16),  # immediately after
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=65536)
        assert len(groups) == 1
        g = groups[0]
        assert set(g.tensors.keys()) == {"a", "b"}

    def test_no_merge_large_gap(self):
        """Two tensors with gap >= merge_threshold → two groups."""
        needed = {
            "a": _tensor_meta(torch.float32, [4], 0, 16),
            "b": _tensor_meta(torch.float32, [4], 16 + 65536, 16),  # gap = 65536
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=65536)
        assert len(groups) == 2
        assert "a" in groups[0].tensors
        assert "b" in groups[1].tensors

    def test_custom_threshold(self):
        """merge_threshold=0 means never merge (gap of 0 is not < 0)."""
        needed = {
            "a": _tensor_meta(torch.float32, [4], 0, 16),
            "b": _tensor_meta(torch.float32, [4], 16, 16),
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=0)
        assert len(groups) == 2

    def test_sorted_by_file_offset(self):
        """Tensors given in non-sorted order should be sorted by file offset."""
        needed = {
            "late": _tensor_meta(torch.float32, [4], 1000, 16),
            "early": _tensor_meta(torch.float32, [4], 0, 16),
            "mid": _tensor_meta(torch.float32, [4], 500, 16),
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=0)
        assert len(groups) == 3
        # Groups must be sorted by file offset
        offsets = [g.first_tensor_f_offset for g in groups]
        assert offsets == sorted(offsets)
        # Verify the order matches the data_start order
        assert groups[0].tensors and "early" in groups[0].tensors
        assert "mid" in groups[1].tensors
        assert "late" in groups[2].tensors

    def test_group_alignment_properties(self):
        """read_size must be a multiple of PAGE_SIZE, pre_padding < PAGE_SIZE."""
        needed = {
            "w": _tensor_meta(torch.float32, [100], 10, 400),
        }
        groups = build_read_groups(needed, header_size=100)
        g = groups[0]
        assert g.read_size % PAGE_SIZE == 0
        assert 0 <= g.pre_padding < PAGE_SIZE
        assert g.aligned_f_offset % PAGE_SIZE == 0

    def test_three_tensors_partial_merge(self):
        """a-b merge, b-c no merge → 2 groups."""
        needed = {
            "a": _tensor_meta(torch.float32, [4], 0, 16),
            "b": _tensor_meta(torch.float32, [4], 16, 16),      # gap=0, merge with a
            "c": _tensor_meta(torch.float32, [4], 70000, 16),   # gap=70000-32, no merge
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=65536)
        assert len(groups) == 2
        assert set(groups[0].tensors.keys()) == {"a", "b"}
        assert set(groups[1].tensors.keys()) == {"c"}

    def test_tensors_dict_copied(self):
        """Modifying the returned tensors dict should not affect input."""
        needed = {
            "w": _tensor_meta(torch.float32, [4], 0, 16),
        }
        groups = build_read_groups(needed, header_size=8)
        groups[0].tensors["extra"] = _tensor_meta(torch.float32, [1], 0, 4)
        assert "extra" not in needed


# ---------------------------------------------------------------------------
# build_file_plan
# ---------------------------------------------------------------------------

class TestBuildFilePlan:
    def test_empty_groups(self):
        """No groups → empty plan with file_buf_size=0."""
        plan = build_file_plan("/path/to/file.safetensors", header_size=8, groups=[])
        assert plan.path == "/path/to/file.safetensors"
        assert plan.header_size == 8
        assert plan.groups == []
        assert plan.slots == []
        assert plan.file_buf_size == 0

    def test_single_group(self):
        """One group → slot at offset 0, file_buf_size = read_size."""
        needed = {"w": _tensor_meta(torch.float32, [4], 0, 16)}
        groups = build_read_groups(needed, header_size=8)
        plan = build_file_plan("/f.safetensors", 8, groups)
        assert len(plan.slots) == 1
        assert plan.slots[0] == 0
        assert plan.file_buf_size == groups[0].read_size

    def test_multiple_groups_slot_offsets(self):
        """Each group's slot = cumulative sum of previous read_sizes."""
        needed = {
            "a": _tensor_meta(torch.float32, [4], 0, 16),
            "b": _tensor_meta(torch.float32, [4], 70000, 16),
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=0)
        plan = build_file_plan("/f.safetensors", 8, groups)
        assert len(plan.slots) == 2
        assert plan.slots[0] == 0
        assert plan.slots[1] == groups[0].read_size
        assert plan.file_buf_size == groups[0].read_size + groups[1].read_size

    def test_slots_are_page_aligned(self):
        """All buffer slots must be 4K-aligned (since read_sizes are)."""
        needed = {
            "a": _tensor_meta(torch.float32, [4], 0, 16),
            "b": _tensor_meta(torch.float32, [4], 500, 16),
            "c": _tensor_meta(torch.float32, [4], 1000, 16),
        }
        groups = build_read_groups(needed, header_size=8, merge_threshold=0)
        plan = build_file_plan("/f.safetensors", 8, groups)
        for slot in plan.slots:
            assert slot % PAGE_SIZE == 0

    def test_plan_preserves_path_and_header(self):
        """FilePlan stores path and header_size from arguments."""
        plan = build_file_plan("/custom/path.safetensors", 42, [])
        assert plan.path == "/custom/path.safetensors"
        assert plan.header_size == 42


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """parse_safetensor_header → build_read_groups → build_file_plan."""

    def test_full_pipeline(self, make_safetensors):
        """Simulate a realistic safetensors file and plan its DMA read."""
        from phxloader import parse_safetensor_header, build_read_groups, build_file_plan

        # Create a file with 3 tensors: two adjacent (merge) + one far away
        t0 = b"\x01" * 64
        t1 = b"\x02" * 64
        t2 = b"\x03" * 64

        path, _ = make_safetensors("pipeline.safetensors", {
            "t0": {"dtype": "U8", "shape": [64], "data": t0},
            "t1": {"dtype": "U8", "shape": [64], "data": t1},
            "t2": {"dtype": "U8", "shape": [64], "data": t2},
        })

        # Step 1: parse header
        meta, header_size = parse_safetensor_header(path)
        assert len(meta) == 3

        # Step 2: build read groups (all adjacent → one group)
        needed = {
            name: (dtype, shape, data_start, nbytes)
            for name, (dtype, shape, data_start, nbytes) in meta.items()
        }
        groups = build_read_groups(needed, header_size)
        assert len(groups) == 1
        g = groups[0]
        assert set(g.tensors.keys()) == {"t0", "t1", "t2"}

        # Step 3: build file plan
        plan = build_file_plan(path, header_size, groups)
        assert plan.path == path
        assert len(plan.slots) == 1
        assert plan.file_buf_size > 0
        assert plan.file_buf_size % PAGE_SIZE == 0

    def test_pipeline_with_selective_loading(self, make_safetensors):
        """Only load a subset of tensors → groups reflect only selected."""
        from phxloader import parse_safetensor_header, build_read_groups, build_file_plan

        # 4 tensors, some far apart
        t0 = b"\x01" * 32
        t1 = b"\x02" * 32
        t2 = b"\x03" * 32
        t3 = b"\x04" * 32

        path, _ = make_safetensors("selective.safetensors", {
            "t0": {"dtype": "U8", "shape": [32], "data": t0},
            "t1": {"dtype": "U8", "shape": [32], "data": t1},
            "t2": {"dtype": "U8", "shape": [32], "data": t2},
            "t3": {"dtype": "U8", "shape": [32], "data": t3},
        })

        meta, header_size = parse_safetensor_header(path)

        # Select only t0 and t2 (skip t1 and t3)
        needed = {
            "t0": meta["t0"],
            "t2": meta["t2"],
        }
        groups = build_read_groups(needed, header_size, merge_threshold=0)
        # t0 and t2 are 32 bytes apart → with threshold=0, two groups
        assert len(groups) == 2

        plan = build_file_plan(path, header_size, groups)
        assert len(plan.slots) == 2
        assert plan.file_buf_size == groups[0].read_size + groups[1].read_size
