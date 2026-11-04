#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Convenience script to install/reinstall phxloader in conda environment.
# Prerequisite: conda activate /mnt/nvme4/dataset/conda_envs/phoenix_lmcache
# Usage: bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Require conda environment ──
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: No conda environment active."
    echo "  Run: conda activate /mnt/nvme4/dataset/conda_envs/phoenix_lmcache"
    exit 1
fi

# ── Environment setup ──
# CUDA_HOME: use conda's CUDA toolkit to match torch
export CUDA_HOME="${CONDA_PREFIX}"

# LIBRARY_PATH: compile-time linker search path
# - /usr/local/lib: libphoenix.so (system-wide, always up-to-date)
# - /usr/lib64: liburing.so (conda compiler doesn't search system paths)
export LIBRARY_PATH="/usr/local/lib:/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

# LD_LIBRARY_PATH: runtime shared library search path
# - $CONDA_PREFIX/lib first: conda's libstdc++ (GLIBCXX_3.4.29+) for phxloader.so
# - /usr/local/lib: libphoenix.so
# Note: /usr/lib64 (liburing, libcuda) is in the default linker search path
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CPATH: system headers (liburing.h etc.) — use CPATH instead of
# CPLUS_INCLUDE_PATH to avoid breaking GCC's #include_next <stdlib.h>
export CPATH="/usr/include${CPATH:+:$CPATH}"

if [ -f /opt/rh/gcc-toolset-13/enable ]; then
    source /opt/rh/gcc-toolset-13/enable
elif [ -f /opt/rh/gcc-toolset-12/enable ]; then
    source /opt/rh/gcc-toolset-12/enable
fi

# ── Install ──
echo ">>> Installing phxloader to ${CONDA_PREFIX} (no-deps, no-build-isolation)..."
python -m pip install . \
    --no-deps \
    --no-build-isolation \
    --force-reinstall

# ── Verify (-P flag prevents current dir from shadowing installed package) ──
echo ">>> Verifying import..."
python -P -c 'from phxloader import PhxLoader; print("phxloader import OK")'

echo ">>> Verifying torch unchanged..."
python -P -c 'import torch; print(f"torch=={torch.__version__}, cuda={torch.cuda.is_available()}")'

echo ">>> Done."
