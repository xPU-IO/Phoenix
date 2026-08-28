"""Build script for phxcache pybind11 extension."""

import os
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Try to find a library using pkg-config or search paths
def find_library(name, search_paths=None):
    try:
        result = subprocess.run(
            ["pkg-config", "--libs", name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if search_paths is None:
        search_paths = [
            "/usr/lib", "/usr/lib64", "/usr/local/lib", "/usr/local/lib64",
        ]

    for path in search_paths:
        for ext in [".so", ".a"]:
            if os.path.exists(os.path.join(path, f"lib{name}{ext}")):
                return f"-L{path} -l{name}"
    return None


# libphoenix is installed system-wide at /usr/local/lib and /usr/local/include.
# Do NOT use the workspace-local phoenix/build/ copy — it may be stale and
# cause ABI mismatch (see 2026-08-07 investigation: phxfs_read signature
# changed from 5 params to 6 params between versions).
LIBPHOENIX_INCLUDE = Path("/usr/local/include")
LIBPHOENIX_LIB = Path("/usr/local/lib")

phxfs_vendor = os.environ.get("PHXFS_VENDOR", "NVIDIA")

if phxfs_vendor == "NVIDIA":
    sdk_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    _link_libraries = ["phoenix", "cuda", "cudart"]
elif phxfs_vendor == "METAX":
    sdk_home = os.environ.get("MACA_HOME", "/opt/maca")
    _link_libraries = ["phoenix", "mcruntime"]

if find_library("uring"):
    _link_libraries.append("uring")

ext_modules = [
    Pybind11Extension(
        "phxcache._phxcache",
        sources=[
            "src/phx_cache.cpp",
            "src/bindings.cpp",
        ],
        include_dirs=[
            str(LIBPHOENIX_INCLUDE),
            str(Path(__file__).resolve().parent / "src"),
            os.path.join(sdk_home, "include"),
        ],
        library_dirs=[
            str(LIBPHOENIX_LIB),
            os.path.join(sdk_home, "lib64"),
            os.path.join(sdk_home, "lib"),
        ],
        libraries=_link_libraries,
        extra_compile_args=["-std=c++17", "-O2", "-fPIC"],
        extra_link_args=["-std=c++17"],
    ),
]

setup(
    name="phxcache",
    version="0.1.0",
    description="Phoenix KV cache adapter for LMCache (phxfs DMA)",
    packages=["phxcache"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[],
    python_requires=">=3.10",
)
