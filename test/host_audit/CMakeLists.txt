cmake_minimum_required(VERSION 3.18)
project(phoenix_host_audit LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(default_root "../.." ABSOLUTE)
set(PHX_ROOT "${default_root}" CACHE PATH "Production source tree under test")
option(AUDIT_ASAN "AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(AUDIT_LEGACY_PROBE "Supply the old NVIDIA-only probe symbols" ON)
find_package(Threads REQUIRED)
add_library(audit_core STATIC
    ${PHX_ROOT}/libphoenix/phx_device.cpp
    ${PHX_ROOT}/libphoenix/phx_mem.cpp
    ${PHX_ROOT}/libphoenix/phx_io.cpp
    ${PHX_ROOT}/libphoenix/phx_stream.cpp
    ${PHX_ROOT}/libphoenix/phx_staging.cpp
    ${PHX_ROOT}/libphoenix/phx_host_staging.cpp
    ${PHX_ROOT}/libphoenix/io_engine/io_pool.cpp
    ${PHX_ROOT}/libphoenix/io_engine/io_engine_sync.cpp
    mock.cpp)
target_include_directories(audit_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PHX_ROOT}/libphoenix ${PHX_ROOT}/libphoenix/include)
target_link_libraries(audit_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(AUDIT_LEGACY_PROBE)
    target_compile_definitions(audit_core PRIVATE AUDIT_LEGACY_PROBE)
endif()
target_compile_options(audit_core PUBLIC -g -Wall -Wextra)
target_link_options(audit_core PUBLIC
    -Wl,--wrap=open -Wl,--wrap=mmap -Wl,--wrap=munmap -Wl,--wrap=ioctl
    -Wl,--wrap=close -Wl,--wrap=pread -Wl,--wrap=pwrite -Wl,--wrap=pthread_create)
if(AUDIT_ASAN)
    target_compile_options(audit_core PUBLIC -O1 -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -fno-pie)
    target_link_options(audit_core PUBLIC -fsanitize=address,undefined -no-pie)
endif()
enable_testing()
file(GLOB cases CONFIGURE_DEPENDS "case_*.cpp")
foreach(src IN LISTS cases)
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE audit_core)
    if(NOT name MATCHES "^case_r0[47]$" AND NOT name STREQUAL "case_r05_sync_lifetime")
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES TIMEOUT 30 ENVIRONMENT "PHXFS_IO_ENGINE=sync")
    endif()
endforeach()
if(TARGET case_r03)
    target_link_options(case_r03 PRIVATE -Wl,--wrap=__pread_chk)
endif()
if(TARGET case_close)
    target_link_options(case_close PRIVATE -Wl,--wrap=pthread_cond_wait)
endif()
if(TARGET case_r05)
    target_link_options(case_r05 PRIVATE -Wl,--wrap=pthread_cond_wait)
endif()

foreach(name case_open case_staging_stream)
    if(TARGET ${name})
        target_link_options(${name} PRIVATE -Wl,--wrap=pthread_cond_wait)
    endif()
endforeach()

if(TARGET case_r05_sync_lifetime)
    target_link_options(case_r05_sync_lifetime PRIVATE -Wl,--wrap=pthread_cond_wait)
    foreach(mode host full)
        foreach(action dereg close)
            foreach(direction read write)
                set(t "case_r05_sync_${mode}_${action}_${direction}")
                add_test(NAME ${t} COMMAND case_r05_sync_lifetime ${mode} ${action} ${direction})
                set_tests_properties(${t} PROPERTIES TIMEOUT 30 ENVIRONMENT "PHXFS_IO_ENGINE=sync")
            endforeach()
        endforeach()
    endforeach()
endif()
if(TARGET case_r05_sync_failure)
    target_link_options(case_r05_sync_failure PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc)
endif()

# Run the unchanged upstream API/cancellation tests with the CPU audit connector.
# The async API test uses a temporary file on tmpfs; all audit cases use memfd.
add_executable(upstream_batch_async ${PHX_ROOT}/test/test_batch_async.cpp)
target_link_libraries(upstream_batch_async PRIVATE audit_core)
add_test(NAME upstream_batch_async COMMAND upstream_batch_async)
set_tests_properties(upstream_batch_async PROPERTIES TIMEOUT 45
    ENVIRONMENT "PHXFS_IO_ENGINE=sync;TMPDIR=/dev/shm")
add_executable(upstream_host_dereg ${PHX_ROOT}/test/host_dereg/test_host_dereg.cpp)
target_link_libraries(upstream_host_dereg PRIVATE audit_core)
target_link_options(upstream_host_dereg PRIVATE -Wl,--wrap=pthread_cond_wait)
add_test(NAME upstream_host_dereg COMMAND upstream_host_dereg sync)
set_tests_properties(upstream_host_dereg PROPERTIES TIMEOUT 45 ENVIRONMENT "PHXFS_IO_ENGINE=sync")

# Independently runnable policy and ordinal checks, also used against upstream.
if(TARGET case_r04)
    foreach(scenario open-ENOENT open-ENODEV open-ENXIO open-EOPNOTSUPP open-ENOSYS open-EACCES open-EPERM open-ENOMEM open-EIO full-ENODEV full-ENXIO full-EOPNOTSUPP full-ENOSYS full-EACCES full-EPERM full-ENOMEM full-EIO full-EINVAL stage-ENODEV stage-ENXIO stage-EOPNOTSUPP stage-ENOSYS stage-EACCES stage-EPERM stage-ENOMEM stage-EIO all-windows second-window invalid-page alloc-ENOMEM discovery-missing discovery-EACCES discovery-EIO direct-open direct-stage)
        add_test(NAME case_r04_${scenario} COMMAND case_r04 ${scenario})
        set_tests_properties(case_r04_${scenario} PROPERTIES TIMEOUT 30 ENVIRONMENT "PHXFS_IO_ENGINE=sync")
    endforeach()
endif()
if(TARGET case_r07)
    foreach(mode host auto)
        foreach(scenario invalid-find invalid-open zero-find zero-open valid)
            add_test(NAME case_r07_${mode}_${scenario} COMMAND case_r07 ${mode} ${scenario})
            set_tests_properties(case_r07_${mode}_${scenario} PROPERTIES TIMEOUT 30 ENVIRONMENT "PHXFS_IO_ENGINE=sync")
        endforeach()
    endforeach()
endif()
