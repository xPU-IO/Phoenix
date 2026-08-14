cmake_minimum_required (VERSION 3.18)
project (phoenix-test)

if (PHXFS_VENDOR STREQUAL "NVIDIA")
    find_package (CUDA 12.4 REQUIRED)
endif ()

if (PHXFS_MAP_MODE STREQUAL "staging")
	set (PHXFS_MAP_MODE_STAGING ON)
endif ()

set (phoenix_root "${PROJECT_SOURCE_DIR}/..")
set (phoenix_include "${phoenix_root}/libphoenix/include")

# Test file path (override with -DTEST_FILE=/path/to/file)
set (TEST_FILE "/mnt/nvme0/phxfs_test.bin" CACHE STRING "Path for test I/O file")

# --- test_regmem: memory registration lifecycle ---
add_executable (test_regmem test_regmem.cu)
add_dependencies (test_regmem libphoenix)
target_include_directories (test_regmem PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_regmem libphoenix CUDA::cudart)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_regmem libphoenix)
endif ()
target_compile_definitions (test_regmem PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_io: read/write correctness + performance ---
add_executable (test_io test_io.cu)
add_dependencies (test_io libphoenix)
target_include_directories (test_io PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_io libphoenix CUDA::cudart)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_io libphoenix)
endif ()
target_compile_definitions (test_io PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_batch: batch I/O (io_uring) correctness + performance ---
find_package (Threads REQUIRED)
add_executable (test_batch test_batch.cu)
add_dependencies (test_batch libphoenix)
target_include_directories (test_batch PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_batch libphoenix CUDA::cudart Threads::Threads)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_batch libphoenix Threads::Threads)
endif ()
target_compile_definitions (test_batch PRIVATE TEST_FILE="${TEST_FILE}")

# CTest target. `batch_cpu` runs the module-free CPU-buffer correctness (it
# skips the GPU sections gracefully); point it at a writable scratch path so
# it works in CI.
enable_testing ()
add_test (NAME batch_cpu COMMAND test_batch 1)
set_tests_properties (batch_cpu PROPERTIES
    ENVIRONMENT "PHXFS_TEST_FILE=${CMAKE_BINARY_DIR}/phxfs_ctest")
