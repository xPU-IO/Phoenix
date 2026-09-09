# CUDA / CUDA-like tests (NVIDIA CUDA, MetaX via MACA cu-bridge).
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    find_package (CUDA 12.4 REQUIRED)
endif ()

set (TEST_GPU_SUFFIX ".cu")

# --- test_regmem: memory registration lifecycle ---
add_executable (test_regmem test_regmem${TEST_GPU_SUFFIX})
add_dependencies (test_regmem libphoenix)
target_include_directories (test_regmem PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_regmem libphoenix CUDA::cudart)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_regmem libphoenix)
endif ()
target_compile_definitions (test_regmem PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_io: read/write correctness + performance ---
add_executable (test_io test_io${TEST_GPU_SUFFIX})
add_dependencies (test_io libphoenix)
target_include_directories (test_io PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_io libphoenix CUDA::cudart)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_io libphoenix)
endif ()
target_compile_definitions (test_io PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_batch: batch I/O (io_uring) correctness + performance ---
add_executable (test_batch test_batch${TEST_GPU_SUFFIX})
add_dependencies (test_batch libphoenix)
target_include_directories (test_batch PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_batch libphoenix CUDA::cudart Threads::Threads)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_batch libphoenix Threads::Threads)
endif ()
target_compile_definitions (test_batch PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_stream: stream-ordered async I/O correctness (host-func model) ---
add_executable (test_stream test_stream${TEST_GPU_SUFFIX})
add_dependencies (test_stream libphoenix)
target_include_directories (test_stream PRIVATE "${phoenix_include}")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries (test_stream libphoenix CUDA::cudart Threads::Threads)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    target_link_libraries (test_stream libphoenix Threads::Threads)
endif ()
target_compile_definitions (test_stream PRIVATE TEST_FILE="${TEST_FILE}")
