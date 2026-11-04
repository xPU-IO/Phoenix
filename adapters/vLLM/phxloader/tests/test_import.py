# SPDX-License-Identifier: Apache-2.0
"""Import and API-surface tests for phxloader (no hardware required).

Verifies that the compiled extension and Python modules can be imported
and that all public symbols expose the expected methods / functions.
"""

import pytest


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def test_import_phxloader():
    """``from phxloader import PhxLoader`` must succeed."""
    from phxloader import PhxLoader  # noqa: F401


def test_import_helper_functions():
    """All helper functions must be importable."""
    from phxloader import (
        parse_safetensor_header,
        build_read_groups,
        build_file_plan,
        ReadGroup,
        FilePlan,
    )
    # noqa: F401


def test_all_exports():
    from phxloader import __all__
    expected = {
        "PhxLoader",
        "parse_safetensor_header",
        "ReadGroup",
        "FilePlan",
        "build_read_groups",
        "build_file_plan",
    }
    assert expected.issubset(set(__all__))


# ---------------------------------------------------------------------------
# PhxLoader API surface
# ---------------------------------------------------------------------------

PHXLOADER_METHODS = [
    "regmem",
    "deregmem",
    "load_tensors_into_buffer",
    "load_tensors_into_buffer_async",
    "wait_dma",
    "get_dma_seconds",
    "reset_dma_timer",
    "close",
]


@pytest.mark.parametrize("method_name", PHXLOADER_METHODS)
def test_phxloader_has_method(method_name):
    from phxloader import PhxLoader
    assert hasattr(PhxLoader, method_name), (
        f"PhxLoader missing method '{method_name}'"
    )


# ---------------------------------------------------------------------------
# Python module API surface
# ---------------------------------------------------------------------------

def test_safetensors_parser_has_parse_function():
    from phxloader.safetensors_parser import parse_safetensor_header
    assert callable(parse_safetensor_header)


def test_read_group_has_build_functions():
    from phxloader.read_group import build_read_groups, build_file_plan
    assert callable(build_read_groups)
    assert callable(build_file_plan)


def test_read_group_has_align_functions():
    from phxloader.read_group import align_up, align_down
    assert callable(align_up)
    assert callable(align_down)


def test_read_group_dataclass_fields():
    from phxloader.read_group import ReadGroup
    # ReadGroup is a dataclass
    assert hasattr(ReadGroup, "__dataclass_fields__")
    fields = set(ReadGroup.__dataclass_fields__)
    assert {
        "aligned_f_offset",
        "read_size",
        "pre_padding",
        "first_tensor_f_offset",
        "tensors",
    }.issubset(fields)


def test_file_plan_dataclass_fields():
    from phxloader.read_group import FilePlan
    assert hasattr(FilePlan, "__dataclass_fields__")
    fields = set(FilePlan.__dataclass_fields__)
    assert {
        "path",
        "header_size",
        "groups",
        "slots",
        "file_buf_size",
    }.issubset(fields)


# ---------------------------------------------------------------------------
# Constructor error handling
# ---------------------------------------------------------------------------
# NOTE: PhxLoader(device_id=invalid) calls phxfs_find_dev which internally
# calls cuDeviceGetPCIBusId — an invalid device ordinal corrupts the CUDA
# context.  The invalid-device test lives in test_phxloader.py (hardware-
# dependent) where we can reset the CUDA context afterwards.
