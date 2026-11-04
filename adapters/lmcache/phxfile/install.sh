#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build and install libphxfile.so — the frozen-ABI shim between LMCache and
# libphoenix (see README.md). Installs to /usr/local; the build itself is
# unprivileged, only the install step needs sudo.
#
# Usage: bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Prerequisite: /usr/local/include/phoenix.h must declare the phxfs
# stream API. A stale header (deployed .so newer than the installed
# header) would compile the shim against implicit declarations — UB that
# happens to link on x86-64 but must never ship; the Makefile also turns
# that into a hard error. Self-heal by syncing from the workspace header.
SRC_HEADER="$SCRIPT_DIR/../../../libphoenix/include/phoenix.h"
INST_HEADER=/usr/local/include/phoenix.h

if ! grep -q "phxfs_read_stream" "$INST_HEADER" 2>/dev/null; then
    if [ ! -f "$SRC_HEADER" ] || ! grep -q "phxfs_read_stream" "$SRC_HEADER"; then
        echo "ERROR: $INST_HEADER lacks the phxfs stream API and the workspace" >&2
        echo "       header ($SRC_HEADER) does not have it either —" >&2
        echo "       libphoenix source tree is too old for this shim." >&2
        exit 1
    fi
    echo ">>> $INST_HEADER is stale (no phxfs stream API); syncing from workspace..."
    sudo cp "$SRC_HEADER" "$INST_HEADER"
    sudo cp "$SCRIPT_DIR/../../../libphoenix/include/common.h" \
        /usr/local/include/common.h
fi

echo ">>> Building libphxfile.so..."
make -C "$SCRIPT_DIR" clean all

echo ">>> Installing to /usr/local (sudo)..."
sudo make -C "$SCRIPT_DIR" install PREFIX=/usr/local
sudo ldconfig

echo ">>> Verifying exported phxFile* symbols..."
count=$(nm -D /usr/local/lib/libphxfile.so | grep -c " T phxFile")
if [ "$count" -ne 10 ]; then
    echo "ERROR: expected 10 phxFile* symbols, found ${count}" >&2
    exit 1
fi
echo "OK: ${count} phxFile* symbols exported"

echo ">>> Done. LMCache's _phx_async now binds libphxfile (ctypes: 'phxfile')."
