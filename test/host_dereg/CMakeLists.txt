cmake_minimum_required(VERSION 3.18)
project(phoenix_host_dereg LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
option(PHX_TEST_NVIDIA "Use the real NVIDIA connector (requires a reserved GPU)" OFF)
option(PHX_TEST_ASAN "Enable AddressSanitizer" ON)
option(PHX_TEST_URING "Build the io_uring engine" OFF)
get_filename_component(PHX_ROOT "../.." ABSOLUTE)
find_package(Threads REQUIRED)
set(core
    ${PHX_ROOT}/libphoenix/phx_device.cpp
    ${PHX_ROOT}/libphoenix/phx_mem.cpp
    ${PHX_ROOT}/libphoenix/phx_io.cpp
    ${PHX_ROOT}/libphoenix/phx_staging.cpp
    ${PHX_ROOT}/libphoenix/phx_host_staging.cpp
    ${PHX_ROOT}/libphoenix/io_engine/io_pool.cpp
    ${PHX_ROOT}/libphoenix/io_engine/io_engine_sync.cpp)
if(PHX_TEST_NVIDIA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND core ${PHX_ROOT}/libphoenix/connectors/nvidia_connector.cpp)
else()
    list(APPEND core mock_connector.cpp)
endif()
add_executable(test_host_dereg test_host_dereg.cpp ${core})
target_include_directories(test_host_dereg PRIVATE
    ${PHX_ROOT}/libphoenix ${PHX_ROOT}/libphoenix/include)
target_link_libraries(test_host_dereg PRIVATE Threads::Threads)
# Observe the actual deregistration wait without sleeps or production hooks.
target_link_options(test_host_dereg PRIVATE -Wl,--wrap=pthread_cond_wait)
if(PHX_TEST_ASAN)
    target_compile_options(test_host_dereg PRIVATE -g -O1 -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(test_host_dereg PRIVATE -fsanitize=address)
endif()
if(PHX_TEST_NVIDIA)
    target_link_libraries(test_host_dereg PRIVATE CUDA::cudart)
endif()
if(PHX_TEST_URING)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing)
    target_sources(test_host_dereg PRIVATE ${PHX_ROOT}/libphoenix/io_engine/io_engine_uring.cpp)
    target_compile_definitions(test_host_dereg PRIVATE PHXFS_HAVE_LIBURING)
    target_link_libraries(test_host_dereg PRIVATE PkgConfig::URING)
endif()
enable_testing()
add_test(NAME host_dereg_sync COMMAND test_host_dereg sync)
set_tests_properties(host_dereg_sync PROPERTIES TIMEOUT 45)
if(PHX_TEST_URING)
    add_test(NAME host_dereg_uring COMMAND test_host_dereg io_uring)
    set_tests_properties(host_dereg_uring PROPERTIES TIMEOUT 45)
endif()
