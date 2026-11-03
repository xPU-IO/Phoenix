cmake_minimum_required (VERSION 3.18)
project (phoenix-test)

set (phoenix_root "${CMAKE_CURRENT_SOURCE_DIR}/..")
set (phoenix_include "${phoenix_root}/libphoenix/include")

# Test file path (override with -DTEST_FILE=/path/to/file)
set (TEST_FILE "/mnt/nvme0/phxfs_test.bin" CACHE STRING "Path for test I/O file")

find_package (Threads REQUIRED)

# CPU-only synchronous batch API test. It forces the sync engine internally,
# so it needs neither a GPU nor the phxfs kernel module.
add_executable (test_batch_sync test_batch_sync.cpp)
add_dependencies (test_batch_sync libphoenix)
target_include_directories (test_batch_sync PRIVATE "${phoenix_include}")
target_link_libraries (test_batch_sync libphoenix Threads::Threads)

# CPU-only asynchronous batch API test (submit / wait / destroy lifecycle
# and worker-pool queueing). Same property: forced sync engine, no GPU and
# no kernel module needed.
add_executable (test_batch_async test_batch_async.cpp)
add_dependencies (test_batch_async libphoenix)
target_include_directories (test_batch_async PRIVATE "${phoenix_include}")
target_link_libraries (test_batch_async libphoenix Threads::Threads)

# Helper for the test matrix (test/kernel_module_matrix/run_matrix.sh): holds a WC mapping
# over a GPU BAR via sysfs resource*_wc to exercise phoenixfs's PAT-conflict
# handling at load time. Plain C, no GPU runtime and no libphoenix needed.
add_executable (bar_holder kernel_module_matrix/bar_holder.c)

# Vendor dispatch: CUDA and CUDA-like runtimes (NVIDIA CUDA, MetaX via the
# MACA cu-bridge) build from cuda/*.cu; AMD builds from amd/*.hip
# (hipify-perl output — regenerate with: hipify-perl cuda/test_X.cu > amd/test_X.hip).
# Iluvatar has no GPU language target: its tests are plain C++.
if (PHXFS_VENDOR STREQUAL "AMD")
    add_subdirectory (amd)
elseif (PHXFS_VENDOR STREQUAL "ILUVATAR")
    add_subdirectory (iluvatar)
else ()
    add_subdirectory (cuda)
endif ()

enable_testing ()
add_test (NAME batch_sync_cpu COMMAND test_batch_sync)
add_test (NAME batch_async_cpu COMMAND test_batch_async)
