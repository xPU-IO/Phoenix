#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# run_tests.sh — Run all phxloader unit tests with one command
#
# Usage:
#   bash run_tests.sh              # Run all tests (hardware tests auto-skip if no device)
#   bash run_tests.sh --python     # Run pure-Python tests only (no device needed)
#   bash run_tests.sh --hardware   # Run hardware-dependent tests only
#   bash run_tests.sh --verbose    # Verbose output
#   bash run_tests.sh --help       # Show help
#
# Prerequisites:
#   1. Run in a Python environment where phxloader is installed (no env switching)
#   2. phxloader extension compiled and installed (bash install.sh)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Colored output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="all"
VERBOSE=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)  MODE="python";  shift ;;
        --hardware) MODE="hardware"; shift ;;
        --verbose|-v) VERBOSE="-v --tb=short"; shift ;;
        --help|-h)  SHOW_HELP=true; shift ;;
        *) error "Unknown argument: $1"; SHOW_HELP=true; shift ;;
    esac
done

if $SHOW_HELP; then
    cat <<'EOF'
run_tests.sh — phxloader unit test runner

Usage:
  bash run_tests.sh [options]

Options:
  --python     Run pure-Python tests only (test_import, test_safetensors_parser, test_read_group)
               No device / phxfs needed, just phxloader extension
  --hardware   Run hardware-dependent tests only (test_phxloader)
               Requires device + phxfs; auto-skips if unavailable
  --verbose    Verbose output (equivalent to -v --tb=short)
  --help       Show this help

Default (no args): Run all tests

Test tiers:
  Tier 1  Pure Python logic   — test_safetensors_parser.py, test_read_group.py
  Tier 2  Import & API surface — test_import.py
  Tier 3  Hardware-dependent   — test_phxloader.py (@skip_no_hardware auto-skips)

Requirements:
  - Run in a Python environment with phxloader installed (activate your conda env)
  - phxloader extension compiled and installed
  - Hardware tests require: device + /dev/phxfs_dev
EOF
    exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "Python: $(which python)"
info "pytest: $(python -m pytest --version 2>&1 || echo 'NOT FOUND')"

# Check phxloader is importable
if ! python -P -c "import phxloader" 2>/dev/null; then
    error "phxloader extension not installed! Run: bash $SCRIPT_DIR/../install.sh"
    exit 1
fi
ok "phxloader extension importable"

# Check device availability
if python -P -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE_COUNT=$(python -P -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    ok "Device available: ${DEVICE_COUNT} device(s)"
    HAS_DEVICE=true
else
    warn "Device not available — hardware tests will be skipped"
    HAS_DEVICE=false
fi

# ---------------------------------------------------------------------------
# Build pytest command
# ---------------------------------------------------------------------------
# python -P: do not add cwd to sys.path (prevents source dir shadowing installed package)
# --import-mode=importlib: avoids conftest.py conflicts across directories
PYTEST_COMMON="python -P -m pytest --import-mode=importlib"

case "$MODE" in
    python)
        info "Running pure-Python tests (no hardware required)"
        PYTEST_CMD="$PYTEST_COMMON $VERBOSE \
            $TESTS_DIR/test_import.py \
            $TESTS_DIR/test_safetensors_parser.py \
            $TESTS_DIR/test_read_group.py"
        ;;
    hardware)
        if [ "$HAS_DEVICE" = "false" ]; then
            warn "Device not available, hardware tests will be skipped"
        fi
        info "Running hardware-dependent tests"
        PYTEST_CMD="$PYTEST_COMMON $VERBOSE $TESTS_DIR/test_phxloader.py"
        ;;
    all)
        info "Running all tests"
        PYTEST_CMD="$PYTEST_COMMON $VERBOSE $TESTS_DIR"
        ;;
esac

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo
info "Executing: $PYTEST_CMD"
echo "============================================================================="
echo

# Temporarily disable set -e (pytest exit code 1 = test failure, not script error)
set +e
eval $PYTEST_CMD
EXIT_CODE=$?
set -e

echo "============================================================================="
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [ $EXIT_CODE -eq 0 ]; then
    ok "All tests passed!"
elif [ $EXIT_CODE -eq 5 ]; then
    warn "No tests collected"
else
    error "Some tests failed (exit code: $EXIT_CODE)"
    echo
    echo "Tips:"
    echo "  - Hardware tests (test_phxloader.py) auto-skip without device"
    echo "  - To run pure-Python tests only: bash run_tests.sh --python"
fi

exit $EXIT_CODE
