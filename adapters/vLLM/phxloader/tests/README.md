# phxloader Unit Tests

## Overview

phxloader is the Phoenix DMA loader for vLLM. It wraps the `phxfs_read` (single I/O) API and pure-Python safetensors parsing / read-group construction logic to provide direct DMA loading of model weights from NVMe to device memory.

This test suite covers **88 tests** across three tiers:

| Tier | File | Tests | Hardware | Description |
|------|------|-------|----------|-------------|
| Tier 1 | `test_safetensors_parser.py` | 26 | None | Safetensors header parsing (pure Python) |
| Tier 1 | `test_read_group.py` | 32 | None | Alignment / grouping / file planning (pure Python) |
| Tier 2 | `test_import.py` | 16 | Compiled extension, no device | Import and API-surface validation |
| Tier 3 | `test_phxloader.py` | 15 | Device + phxfs | PhxLoader DMA loading |

## Requirements

| Dependency | Requirement |
|------------|-------------|
| Python environment | phxloader extension installed (`bash install.sh`) |
| libphoenix.so | `/usr/local/lib/libphoenix.so` (latest version with 6-param `phxfs_read`) |
| Device | Required for hardware tests (e.g. 8× H20) |
| phxfs device | `/dev/phxfs_dev` (required for hardware tests) |
| pytest | ≥ 7.0 |

## How to Run

```bash
# Run all tests (hardware tests auto-skip if no device)
bash run_tests.sh

# Run pure-Python tests only (no device needed, CI-friendly)
bash run_tests.sh --python

# Run hardware tests only
bash run_tests.sh --hardware

# Verbose output
bash run_tests.sh --verbose
```

Manual invocation:

```bash
# All tests
python -P -m pytest tests/ -v --import-mode=importlib

# Pure-Python tests only
python -P -m pytest tests/test_import.py tests/test_safetensors_parser.py tests/test_read_group.py \
    -v --import-mode=importlib

# Hardware tests only
python -P -m pytest tests/test_phxloader.py -v --import-mode=importlib
```

> **Note:** The `-P` flag (do not add cwd to `sys.path`) and `--import-mode=importlib` (avoid `conftest.py` conflicts) are required to prevent the source directory from shadowing the installed package.

## Test Files

### `conftest.py` — Shared Fixtures

| Component | Purpose |
|-----------|---------|
| `_check_hardware()` | Checks if `phxloader` is importable and a device is available (does not instantiate PhxLoader) |
| `skip_no_hardware` | pytest marker that auto-skips when no device/phxfs is available |
| `make_safetensors` | Factory fixture: builds real safetensors files on disk (8-byte header_len + JSON header + 4K-aligned data) |
| `phx_loader` | Session-scoped `PhxLoader(device_id=0)` instance |
| `gpu_buffer` | Function-scoped 8 MiB device buffer (`torch.empty(8M, uint8, cuda:0)`) |
| `make_aligned_file` | Factory fixture: creates 4K-aligned data files (standard I/O + fsync) |

### `test_import.py` — Import & API Surface (16 tests, no hardware)

**Objective**: Verify the compiled extension and Python modules import correctly, and all public symbols expose the expected methods / functions / dataclass fields.

**Method**: Pure `import` + `hasattr` assertions. No objects are instantiated.

| Test | Verifies |
|------|----------|
| `test_import_phxloader` | `from phxloader import PhxLoader` succeeds |
| `test_import_helper_functions` | `parse_safetensor_header`, `build_read_groups`, `build_file_plan`, `ReadGroup`, `FilePlan` are all importable |
| `test_all_exports` | `__all__` contains all 6 public symbols |
| `test_phxloader_has_method` ×8 (parametrized) | `PhxLoader` has `regmem`, `deregmem`, `load_tensors_into_buffer`, `load_tensors_into_buffer_async`, `wait_dma`, `get_dma_seconds`, `reset_dma_timer`, `close` |
| `test_safetensors_parser_has_parse_function` | `parse_safetensor_header` is callable |
| `test_read_group_has_build_functions` | `build_read_groups`, `build_file_plan` are callable |
| `test_read_group_has_align_functions` | `align_up`, `align_down` are callable |
| `test_read_group_dataclass_fields` | `ReadGroup` has `aligned_f_offset`, `read_size`, `pre_padding`, `first_tensor_f_offset`, `tensors` fields |
| `test_file_plan_dataclass_fields` | `FilePlan` has `path`, `header_size`, `groups`, `slots`, `file_buf_size` fields |

### `test_safetensors_parser.py` — Safetensors Header Parsing (26 tests, pure Python)

**Objective**: Verify `parse_safetensor_header()` correctly parses safetensors file headers, extracting tensor dtype / shape / offset / size.

**Method**: Uses the `make_safetensors` fixture or `_write_raw` helper to construct various safetensors files (valid / malformed) under `tmp_path`, then calls `parse_safetensor_header()` and asserts the return values.

| Test | Data | Verifies |
|------|------|----------|
| `test_parse_basic` | 1 F32 tensor, shape=[4], data=`1.0f × 4` (16 bytes) | dtype=torch.float32, shape=(4,), data_start=0, nbytes=16 |
| `test_parse_header_size` | 1 F32 tensor | header_size = 8 + len(json) |
| `test_parse_all_dtypes` ×11 (parametrized) | 11 dtypes × 3 elements each: BOOL/U8/I8/I16/I32/I64/F16/BF16/F32/F64/F8_E4M3 | Each dtype maps to the correct torch dtype |
| `test_parse_f8_e8m0_dtype` | F8_E8M0, 4 elements | Maps to torch.float8_e8m0fnu (skips if unsupported) |
| `test_dtype_map_completeness` | — | `SAFETENSORS_DTYPE_TO_TORCH` dict contains 11 core keys |
| `test_parse_metadata_skipped` | 1 tensor + `__metadata__` | `__metadata__` does not appear in returned tensor dict |
| `test_parse_multiple_tensors` | 3 tensors: t0(F32,12B), t1(I16,8B), t2(U8,2B) | 3 tensors, offsets 0/12/20 |
| `test_parse_empty_tensors` | 0 tensors + metadata | Returns empty dict |
| `test_parse_scalar_shape` | 1 F32 scalar, shape=[] | shape=() |
| `test_parse_multidim_shape` | 1 U8, shape=[2,3,4] | shape=(2,3,4) |
| `test_parse_truncated_file_too_small` | 3-byte file | Raises ValueError("too small") |
| `test_parse_oversized_header` | Claims 1MB header but file is 10 bytes | Raises ValueError("exceeds remaining file size") |
| `test_parse_truncated_json` | Claims 100 bytes but only 7 present | Raises ValueError("truncated") |
| `test_parse_unsupported_dtype` | dtype="F128" (nonexistent) | Raises ValueError("Unsupported safetensors dtype") |
| `test_parse_nonexistent_file` | Non-existent path | Raises FileNotFoundError/OSError |
| `test_parse_exact_offset_boundaries` | 100-byte U8 data | data_start=0, nbytes=100 |

### `test_read_group.py` — Read-Group Construction & File Planning (32 tests, pure Python)

**Objective**: Verify `align_up` / `align_down` alignment functions, `build_read_groups` merge logic, and `build_file_plan` slot assignment logic.

**Method**: Constructs tensor meta dicts (`{name: (dtype, shape, data_start, nbytes)}`) as input, asserts returned `ReadGroup` / `FilePlan` field values. End-to-end tests use `make_safetensors` to create real files.

| Group | Tests | Coverage |
|-------|-------|----------|
| `TestAlignUp` | 7 | Boundary values: 0, exact multiple, off-by-one, 1 byte, just-below, large value (1M), alignment=1 |
| `TestAlignDown` | 8 | Same as above + idempotency verification |
| `TestBuildReadGroups` | 8 | Empty input, single tensor, adjacent merge, large-gap no-merge, custom threshold, unsorted ordering, alignment properties, partial merge, dict copy isolation |
| `TestBuildFilePlan` | 5 | Empty plan, single group slot=0, multi-group cumulative slots, slot 4K alignment, path/header_size preservation |
| `TestEndToEndPipeline` | 2 | parse → build_read_groups → build_file_plan full pipeline, including selective subset loading |

### `test_phxloader.py` — PhxLoader Hardware Tests (15 tests, device + phxfs required)

**Objective**: Verify `PhxLoader` construction, regmem / deregmem, sync / async DMA loading, DMA timer, and close semantics.

**Method**: Uses `phx_loader` (session-scoped instance) and `gpu_buffer` fixture. DMA tests use `make_aligned_file` to create aligned files, write `torch.arange` data, DMA-read to device memory, and verify with `torch.equal`.

| Test | Data | Verifies |
|------|------|----------|
| `test_loader_init` | — | Instantiation succeeds |
| `test_regmem_returns_valid_addr` | 8 MiB device buffer | `regmem` returns a non-zero address |
| `test_deregmem_unregistered_noop` | Unregistered device buffer | `deregmem` does not raise |
| `test_regmem_deregmem_multiple` | 2× 2 MiB device buffers | Both return non-zero addresses |
| `test_load_tensors_into_buffer` | `torch.arange(4096, uint8)` written to file | DMA read data matches |
| `test_load_tensors_into_buffer_empty_batch` | Empty batch | No-op, no exception |
| `test_load_unregistered_ptr_raises` | Unregistered device pointer | Raises exception |
| `test_load_async_and_wait` | `torch.arange(4096, uint8)` | Async DMA + `wait_dma` data matches |
| `test_wait_dma_without_async_noop` | — | `wait_dma` does not raise |
| `test_async_overlaps_previous` | 2 files (arange 0-4096 and 4096-8192) | Two consecutive async + wait |
| `test_dma_timer_reset_and_read` | — | reset → 0, returns float |
| `test_dma_timer_after_dma` | `torch.arange(4096, uint8)` | reset → 0, DMA >0, reset → 0 |
| `test_multi_entry_batch` | arange(0-4096) + arange(4096-8192) | Single batch with 2 entries, verified independently |
| `test_close_idempotent` | — | `close()` twice does not raise |
| `test_invalid_device_raises` | `device_id=99999` | Raises exception + device context reset (**must run last in module**) |
