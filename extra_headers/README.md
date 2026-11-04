# External kernel headers

Phoenix vendors NVIDIA's `nv-p2p.h` here so building the NVIDIA kernel module
does not require downloading the entire driver source tree or CUDA Toolkit.
Each header is copied unchanged from the official `NVIDIA/open-gpu-kernel-modules`
repository. Its original MIT license and copyright notice remain in the file;
each version directory records the source URL and SHA256 checksum.

These headers declare kernel P2P interfaces. The implementation and exported
symbols come from the loaded NVIDIA kernel driver. Headers are selected by
**driver version**, independently of the CUDA Toolkit version.

## Selection

`cmake ..` does not search for these headers. When `make modules` configures its
component, it uses:

1. An explicit `NVIDIA` directory containing `nv-p2p.h`, if configured.
2. A header for the exact loaded driver version under `PHXFS_EXTRA_HEADERS`
   (default: this directory), or installed driver sources under `/usr/src`.
3. A header directly in `PHXFS_EXTRA_HEADERS`, if supplied by the user.
4. The bundled sample for the same driver branch.

The last fallback is a convenience, not a guarantee that every release in a
branch has the same ABI. Phoenix checks `NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE`
and requires 64 KiB pages before using the returned page table. An unsupported
ABI or page size causes registration to return `-EOPNOTSUPP`.

For an unbundled branch, add its official header under
`nvidia-<driver-version>/nv-p2p.h`, or configure
`cmake -DNVIDIA=/path/to/driver/headers ..`. Keep the source version, checksum,
and upstream license with any added header.

## Sampled changes

Comparison of the six bundled upstream tags, performed on November 4, 2026:

| Upstream tags | Page-table version | Changes relevant to Phoenix |
| --- | --- | --- |
| `535.129.03`, `550.54.15` | `0x00010002` | Headers are byte-for-byte identical. |
| `570.124.06`, `575.57.08` | `0x00010002` | Add `NVIDIA_P2P_FLAGS_DEFAULT` and `NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING` for persistent APIs; other changes from the 535 sample are comments. |
| `580.95.05`, `590.44.01` | `0x00020000` | Append `uint32_t flags` to `nvidia_p2p_page_table`, add `NVIDIA_P2P_PAGE_TABLE_FLAGS_CPU_CACHEABLE`, and retain the persistent API flags. |

The ordinary `nvidia_p2p_get_pages`, `nvidia_p2p_put_pages`, and
`nvidia_p2p_free_page_table` signatures and `nvidia_p2p_page` layout used by
Phoenix are unchanged in these samples. The page-table **major version**
change still requires matching headers and the runtime version check. The new
cacheability flag may require additional mapping support on affected hardware;
matching the version alone does not establish hardware compatibility.

This comparison covers these exact tags, not every release or future driver.
The local verification built the module against `535.129.03` for a host running
that driver; no new module was loaded as part of this work.
