cmake_minimum_required (VERSION 3.18)
project (phoenix-test)

set (phoenix_root "${CMAKE_CURRENT_SOURCE_DIR}/..")
set (phoenix_include "${phoenix_root}/libphoenix/include")

# Test file path (override with -DTEST_FILE=/path/to/file)
set (TEST_FILE "/mnt/nvme0/phxfs_test.bin" CACHE STRING "Path for test I/O file")

find_package (Threads REQUIRED)

# Vendor dispatch: CUDA and CUDA-like runtimes (NVIDIA CUDA, MetaX via the
# MACA cu-bridge) build from cuda/*.cu; AMD builds from amd/*.hip
# (hipify-perl output — regenerate with: hipify-perl cuda/test_X.cu > amd/test_X.hip).
if (PHXFS_VENDOR STREQUAL "AMD")
    add_subdirectory (amd)
else ()
    add_subdirectory (cuda)
endif ()

# CTest target. `batch_cpu` runs the module-free CPU-buffer correctness (it
# skips the GPU sections gracefully); point it at a writable scratch path so
# it works in CI.
enable_testing ()
add_test (NAME batch_cpu COMMAND test_batch 1)
set_tests_properties (batch_cpu PROPERTIES
    ENVIRONMENT "PHXFS_TEST_FILE=${CMAKE_BINARY_DIR}/phxfs_ctest")
