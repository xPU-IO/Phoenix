# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
# Public targets configure their own toolchains when make invokes them.
project(Phoenix LANGUAGES NONE)

set(PHXFS_VENDOR "NVIDIA" CACHE STRING "Target GPU vendor")
set_property(CACHE PHXFS_VENDOR PROPERTY STRINGS NVIDIA METAX AMD HUAWEI ILUVATAR)
if(NOT PHXFS_VENDOR MATCHES "^(NVIDIA|METAX|AMD|HUAWEI|ILUVATAR)$")
    message(FATAL_ERROR "Unsupported PHXFS_VENDOR: ${PHXFS_VENDOR}")
endif()
set(PHXFS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PHXFS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(PHXFS_EXTRA_HEADERS "${PHXFS_ROOT}/extra_headers" CACHE PATH "External kernel headers")
set(PHXFS_MAP_MODE "staging" CACHE STRING "Default BAR mapping mode: staging or full")
set_property(CACHE PHXFS_MAP_MODE PROPERTY STRINGS staging full)
set(NVIDIA "" CACHE PATH "Optional NVIDIA driver header directory")
set(KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Kernel build tree")
option(BUILD_SHARED_LIBS "Build a shared user library" ON)
option(no_liburing "Disable the io_uring engine" OFF)
option(phx_nvtx "Emit NVTX ranges from libphoenix" ON)
include(ProcessorCount)
ProcessorCount(cpu_count)
if(NOT cpu_count OR cpu_count GREATER 4)
    set(cpu_count 4)
endif()
set(PHXFS_BUILD_JOBS "${cpu_count}" CACHE STRING "Parallel jobs in each component build")
set(PHXFS_GENERATOR "${CMAKE_GENERATOR}")
find_program(PHXFS_MAKE NAMES gmake make REQUIRED)
if(BUILD_SHARED_LIBS)
    set(PHXFS_LIBRARY_FILE "${PHXFS_OUTPUT_DIR}/libphoenix${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    set(PHXFS_LIBRARY_FILE "${PHXFS_OUTPUT_DIR}/libphoenix${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

# Preserve explicit toolchain/SDK settings and old cache entries without
# enabling any language or inspecting optional SDKs in this top-level project.
set(PHXFS_FORWARDED_VARIABLES PHXFS_ROOT PHXFS_OUTPUT_DIR PHXFS_VENDOR
    PHXFS_EXTRA_HEADERS PHXFS_MAP_MODE NVIDIA KERNEL BUILD_SHARED_LIBS no_liburing
    phx_nvtx PHXFS_BUILD_JOBS PHXFS_GENERATOR PHXFS_MAKE PHXFS_LIBRARY_FILE
    CMAKE_INSTALL_PREFIX)
get_cmake_property(cache_variables CACHE_VARIABLES)
foreach(name IN LISTS cache_variables)
    if(name MATCHES "^CMAKE_(C|CXX|CUDA|HIP)_(COMPILER|HOST_COMPILER|ARCHITECTURES|FLAGS(_[A-Z]+)?)$"
            OR name MATCHES "^(CMAKE_BUILD_TYPE|CMAKE_TOOLCHAIN_FILE|CMAKE_PREFIX_PATH|CUDAToolkit_ROOT|HIP_PATH|COREX_PATH|TEST_FILE|no_cc)$")
        list(APPEND PHXFS_FORWARDED_VARIABLES "${name}")
    endif()
endforeach()
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
    list(APPEND PHXFS_FORWARDED_VARIABLES CUDAToolkit_ROOT)
endif()
set(PHXFS_SETTINGS "${PHXFS_OUTPUT_DIR}/.components/settings.cmake")
file(MAKE_DIRECTORY "${PHXFS_OUTPUT_DIR}/.components")
set(settings "set(PHXFS_FORWARDED_VARIABLES [==[${PHXFS_FORWARDED_VARIABLES}]==])\n")
foreach(name IN LISTS PHXFS_FORWARDED_VARIABLES)
    string(APPEND settings "set(${name} [==[${${name}}]==])\n")
endforeach()
file(WRITE "${PHXFS_SETTINGS}" "${settings}")

function(phxfs_target name component target)
    add_custom_target(${name}
        COMMAND "${CMAKE_COMMAND}" "-DPHXFS_SETTINGS=${PHXFS_SETTINGS}"
            "-DPHXFS_COMPONENT=${component}" "-DPHXFS_TARGET=${target}"
            -P "${PHXFS_ROOT}/cmake/BuildComponent.cmake"
        USES_TERMINAL VERBATIM)
endfunction()
phxfs_target(libphoenix library libphoenix)
add_custom_target(library DEPENDS libphoenix)
add_custom_target(default_library ALL DEPENDS libphoenix)
phxfs_target(modules module all)
phxfs_target(clean_modules module clean)
phxfs_target(install_modules module install)
add_dependencies(install_modules modules)
phxfs_target(with_module module install)
add_dependencies(with_module modules)
foreach(target insmod insmod-full insmod-staging rmmod)
    add_custom_target(${target}
        COMMAND "${CMAKE_COMMAND}" "-DPHXFS_SETTINGS=${PHXFS_SETTINGS}"
            -DPHXFS_COMPONENT=module "-DPHXFS_TARGET=${target}"
            "-DMAP_MODE=$(MAP_MODE)" "-DDEBUG_MODE=$(DEBUG_MODE)"
            -P "${PHXFS_ROOT}/cmake/BuildComponent.cmake"
        USES_TERMINAL VERBATIM)
endforeach()
foreach(target test_batch_sync test_batch_async)
    phxfs_target(${target} cpu ${target})
    add_dependencies(${target} libphoenix)
endforeach()
phxfs_target(cpu_tests cpu all)
add_dependencies(cpu_tests libphoenix)
phxfs_target(gpu_tests gpu all)
add_dependencies(gpu_tests libphoenix)
if(PHXFS_VENDOR STREQUAL "ILUVATAR")
    set(gpu_targets test_iluvatar_roundtrip test_iluvatar_suite)
else()
    set(gpu_targets test_regmem test_io test_batch test_stream)
    if(NOT PHXFS_VENDOR STREQUAL "AMD")
        list(APPEND gpu_targets test_ioctl)
    endif()
endif()
foreach(target IN LISTS gpu_targets)
    phxfs_target(${target} gpu ${target})
    add_dependencies(${target} libphoenix)
endforeach()
phxfs_target(bar_holder tools bar_holder)
add_custom_target(tests DEPENDS cpu_tests gpu_tests)
# Native clean must also remove objects built by the component projects.
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    "${PHXFS_OUTPUT_DIR}/.components/library"
    "${PHXFS_OUTPUT_DIR}/.components/cpu"
    "${PHXFS_OUTPUT_DIR}/.components/gpu"
    "${PHXFS_OUTPUT_DIR}/.components/tools"
    "${PHXFS_OUTPUT_DIR}/libphoenix${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "${PHXFS_OUTPUT_DIR}/libphoenix${CMAKE_STATIC_LIBRARY_SUFFIX}")
foreach(target test_batch_sync test_batch_async bar_holder ${gpu_targets})
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        "${PHXFS_OUTPUT_DIR}/bin/${target}")
endforeach()
enable_testing()
add_test(NAME batch_sync_cpu COMMAND "${PHXFS_OUTPUT_DIR}/bin/test_batch_sync")
add_test(NAME batch_async_cpu COMMAND "${PHXFS_OUTPUT_DIR}/bin/test_batch_async")

# make install builds the default library, then installs its headers/library.
install(CODE "set(PHXFS_SETTINGS \"${PHXFS_SETTINGS}\")\nset(PHXFS_COMPONENT library)\nset(PHXFS_TARGET install)\ninclude(\"${PHXFS_ROOT}/cmake/BuildComponent.cmake\")")
message(STATUS "Targets: modules, libphoenix, cpu_tests, gpu_tests, tests, install_modules")
message(STATUS "Dependencies are checked when the selected target is built; make builds the user library")
