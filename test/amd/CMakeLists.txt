# AMD tests: hipify-perl-converted *.hip sources (cuda/test_X.cu is the
# source of truth; regenerate with: hipify-perl cuda/test_X.cu > test_X.hip).
# The CMake HIP language drives hipcc, which links the HIP runtime
# (amdhip64) automatically.
enable_language (HIP)
if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set (CMAKE_HIP_ARCHITECTURES "gfx942")
endif ()

# --- test_regmem: memory registration lifecycle ---
add_executable (test_regmem test_regmem.hip)
add_dependencies (test_regmem libphoenix)
target_include_directories (test_regmem PRIVATE "${phoenix_include}")
target_link_libraries (test_regmem libphoenix Threads::Threads)
target_compile_definitions (test_regmem PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_io: read/write correctness + performance ---
add_executable (test_io test_io.hip)
add_dependencies (test_io libphoenix)
target_include_directories (test_io PRIVATE "${phoenix_include}")
target_link_libraries (test_io libphoenix Threads::Threads)
target_compile_definitions (test_io PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_batch: batch I/O (io_uring) correctness + performance ---
add_executable (test_batch test_batch.hip)
add_dependencies (test_batch libphoenix)
target_include_directories (test_batch PRIVATE "${phoenix_include}")
target_link_libraries (test_batch libphoenix Threads::Threads)
target_compile_definitions (test_batch PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_stream: stream-ordered async I/O correctness (host-func model) ---
add_executable (test_stream test_stream.hip)
add_dependencies (test_stream libphoenix)
target_include_directories (test_stream PRIVATE "${phoenix_include}")
target_link_libraries (test_stream libphoenix Threads::Threads)
target_compile_definitions (test_stream PRIVATE TEST_FILE="${TEST_FILE}")
