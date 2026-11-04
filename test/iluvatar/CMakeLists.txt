# SPDX-License-Identifier: Apache-2.0
# Needs phoenixfs in FULL mode and an O_DIRECT file.
add_executable (test_iluvatar_roundtrip test_full_roundtrip.cc)
add_dependencies (test_iluvatar_roundtrip libphoenix)
target_include_directories (test_iluvatar_roundtrip PRIVATE
    "${phoenix_include}" "${COREX_PATH}/include")
target_link_libraries (test_iluvatar_roundtrip libphoenix Threads::Threads)

add_test (NAME iluvatar_roundtrip COMMAND test_iluvatar_roundtrip ${TEST_FILE})

# Full correctness suite: registration, direct I/O, sync/async batch,
# cancel/destroy, stream-ordered I/O, host path, 256MiB multi-chunk.
add_executable (test_iluvatar_suite suite.cc)
add_dependencies (test_iluvatar_suite libphoenix)
target_include_directories (test_iluvatar_suite PRIVATE
    "${phoenix_include}" "${COREX_PATH}/include")
target_link_libraries (test_iluvatar_suite libphoenix Threads::Threads)

add_test (NAME iluvatar_suite COMMAND test_iluvatar_suite ${TEST_FILE})
set_tests_properties (iluvatar_suite PROPERTIES TIMEOUT 300)
