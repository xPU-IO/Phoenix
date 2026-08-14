cmake_minimum_required (VERSION 3.18)

# Vendor selection must be set before project() so we can conditionally
# enable the CUDA language.
set (PHXFS_VENDOR "NVIDIA" CACHE STRING "Target GPU vendor: NVIDIA, METAX, AMD, HUAWEI")
set_property (CACHE PHXFS_VENDOR PROPERTY STRINGS NVIDIA METAX AMD HUAWEI)

if (PHXFS_VENDOR STREQUAL "NVIDIA")
    project (libphoenix LANGUAGES CUDA C CXX)
    set (CMAKE_CUDA_ARCHITECTURES 90)
    set (CMAKE_CUDA_STANDARD 20)
elseif (PHXFS_VENDOR STREQUAL "METAX")
    if(DEFINED ENV{MACA_PATH})
        set(WCUDA_PATH $ENV{MACA_PATH}/tools/cu-bridge)
    else()
        message(FATAL_ERROR "Please set env parameter MACA_PATH")
    endif()
    set(CMAKE_MODULE_PATH "${WCUDA_PATH}/cmake_module/maca")
    set(CMAKE_C_COMPILER "${WCUDA_PATH}/bin/cucc")
    set(CMAKE_CXX_COMPILER "${WCUDA_PATH}/bin/cucc")
    set(CMAKE_CUDA_COMPILER "${WCUDA_PATH}/bin/cucc")
    set(CMAKE_LINKER "${WCUDA_PATH}/bin/cucc")
    set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    set(CMAKE_CUDA_LINK_EXECUTABLE "<CMAKE_LINKER> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

    project (libphoenix LANGUAGES CUDA C CXX)
    set (CMAKE_CUDA_STANDARD 20)
else ()
    project (libphoenix LANGUAGES C CXX)
endif ()

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Threads REQUIRED)
find_library (DL_LIB dl REQUIRED)

# CUDA is only required for NVIDIA vendor
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    find_package (CUDA 12.4 REQUIRED)
    find_package (CUDAToolkit REQUIRED)
    find_library (CUFILE_LIBRARIES cufile ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif ()

find_package(PkgConfig REQUIRED)

set (no_liburing false CACHE BOOL "Do not use liburing (io_uring async I/O)")
set (phx_nvtx true CACHE BOOL "Emit NVTX ranges from libphoenix (for nsys)")

if (NOT no_liburing)
    pkg_check_modules(LIBURING liburing)
    if (LIBURING_FOUND)
        message(STATUS "liburing found: ${LIBURING_INCLUDE_DIRS}")
    else ()
        message(STATUS "liburing NOT found - io_uring async disabled")
    endif ()
else ()
    message(STATUS "liburing disabled by no_liburing=ON")
endif ()

if (LIBURING_FOUND)
    add_compile_definitions(PHXFS_HAVE_LIBURING)
endif ()

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

set (nvidia_archs "90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")
# Default BAR mapping mode baked into phoenixfs.ko: "staging" (default) or
# "full". Still overridable at load time via the phxfs_map_mode module param.
set (PHXFS_MAP_MODE "staging" CACHE STRING "default BAR mapping mode: full or staging")
set_property (CACHE PHXFS_MAP_MODE PROPERTY STRINGS full staging)

if (no_smartio)
    set (no_smartio_samples true)
    set (no_smartio_benchmarks true)
endif ()

# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()

# C_FLAGS
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
# CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)

# GCC 8 requires linking stdc++fs for std::filesystem support
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    set(FS_LIBRARY stdc++fs)
else()
    set(FS_LIBRARY "")
endif()


if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()

include_directories ("${PROJECT_SOURCE_DIR}/libphoenix/include")

if (PHXFS_VENDOR STREQUAL "NVIDIA")
    include_directories ("${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/")
endif ()

# Set vendor-specific backend object file and compile define
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    set (VENDOR_BACKEND "nvidia-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_NVIDIA")
elseif (PHXFS_VENDOR STREQUAL "METAX")
    set (VENDOR_BACKEND "metax-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_METAX")
elseif (PHXFS_VENDOR STREQUAL "AMD")
    set (VENDOR_BACKEND "amd-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_AMD")
elseif (PHXFS_VENDOR STREQUAL "HUAWEI")
    set (VENDOR_BACKEND "huawei-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_HUAWEI")
else ()
    message (FATAL_ERROR "Unsupported PHXFS_VENDOR: ${PHXFS_VENDOR}. Use NVIDIA, METAX, AMD, or HUAWEI.")
endif ()

# Try to find Nvidia driver headers (only needed for NVIDIA backend)
set (driver_include "")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    # Directly glob for nv-p2p.h in common locations
    file (GLOB nv_p2p_files
        "/usr/src/nvidia-*/nvidia-peermem/nv-p2p.h"
        "/usr/src/nvidia-*/nv-p2p.h"
        "${NVIDIA}/nv-p2p.h"
        "${NVIDIA}/nvidia-peermem/nv-p2p.h")

    if (nv_p2p_files)
        list (REVERSE nv_p2p_files)
        list (GET nv_p2p_files 0 first_match)
        get_filename_component (driver_include "${first_match}" DIRECTORY)
    endif ()

    if (driver_include)
        message(STATUS "Found NVIDIA driver header: ${driver_include}/nv-p2p.h")
    else ()
        message(WARNING
            "nv-p2p.h not found! Searched:\n"
            "  - /usr/src/nvidia-*/nvidia-peermem/nv-p2p.h\n"
            "  - /usr/src/nvidia-*/nv-p2p.h\n"
            "  - NVIDIA cache var: '${NVIDIA}'\n"
            "Pass -DNVIDIA=/path/to/nvidia/driver/src to cmake.")
    endif ()
endif ()

# # Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
set (module_output "${CMAKE_BINARY_DIR}/module")
# for Makefile.in, do not delete
if (driver_include)
	set (module_ccflags "-I${driver_include} -D${VENDOR_DEFINE}")
else ()
	set (module_ccflags "-D${VENDOR_DEFINE}")
endif ()
# Bake the default BAR mapping mode into phoenixfs.ko.
if (PHXFS_MAP_MODE STREQUAL "full")
	set (PHXFS_MAP_MODE_DEFAULT 0)
else ()
	set (PHXFS_MAP_MODE_DEFAULT 1)
endif ()
set (module_ccflags "${module_ccflags} -DPHXFS_MAP_MODE_DEFAULT=${PHXFS_MAP_MODE_DEFAULT}")
configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
message (STATUS "Configuring module build in ${module_output}")

add_custom_target(
    modules
    ALL
    COMMAND make
    WORKING_DIRECTORY ${module_output}
    COMMENT "Building kernel module"
)

add_custom_target(
    clean_modules
    COMMAND make clean
    WORKING_DIRECTORY ${module_output}
    COMMENT "Cleaning kernel module"
)

add_custom_target(
    insmod
    # Optional override: `make insmod MAP_MODE=0|1` forces the BAR map mode for
    # this load, overriding the mode baked in at cmake time (PHXFS_MAP_MODE).
    COMMAND sudo make insmod MAP_MODE=$(MAP_MODE)
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module"
)
# Named variants so the mode is TAB-completable at the top level too.
add_custom_target(
    insmod-staging
    COMMAND sudo make insmod-staging
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module (STAGING mode)"
)
add_custom_target(
    insmod-full
    COMMAND sudo make insmod-full
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module (FULL mode)"
)

add_custom_target(
    rmmod
    COMMAND sudo make rmmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Removing kernel module"
)


# Library source files
set(libphoenix_root "${PROJECT_SOURCE_DIR}/libphoenix")
file(GLOB libphoenix_source "${libphoenix_root}/*.cpp" "${libphoenix_root}/io_engine/*.cpp")
file(GLOB libphoenix_include "${libphoenix_root}/include/*.h")

# io_engine_uring.cpp requires liburing; drop it when unavailable so the
# sync fallback engine is still built.
if (NOT LIBURING_FOUND)
    list(REMOVE_ITEM libphoenix_source "${libphoenix_root}/io_engine/io_engine_uring.cpp")
endif ()

# Select vendor-specific connector source file
if (PHXFS_VENDOR STREQUAL "NVIDIA" OR PHXFS_VENDOR STREQUAL "METAX")
    set(devconn_source "${libphoenix_root}/connectors/nvidia_connector.cpp")
elseif (PHXFS_VENDOR STREQUAL "AMD")
    set(devconn_source "${libphoenix_root}/connectors/amd_connector.cc")
elseif (PHXFS_VENDOR STREQUAL "HUAWEI")
    set(devconn_source "${libphoenix_root}/connectors/huawei_connector.cc")
endif ()

add_library(libphoenix ${libphoenix_source} ${devconn_source} ${libphoenix_include})
set_target_properties (libphoenix PROPERTIES OUTPUT_NAME "phoenix")
target_include_directories(libphoenix PRIVATE "${libphoenix_root}")

# The library itself depends on cudart (nvidia_connector) and pthreads (io
# pool); declare them as link deps so libphoenix.so carries the right
# DT_NEEDED and can be dlopen()'d standalone (P1-2), not left to consumers.
target_link_libraries(libphoenix Threads::Threads)
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    target_link_libraries(libphoenix CUDA::cudart)
    # NVTX ranges (phx.batch.*, phx.staging.*) for nsys. NVTX v3 is
    # header-only, so this adds no link dependency; disable with
    # cmake -Dphx_nvtx=false.
    if (phx_nvtx)
        target_compile_definitions(libphoenix PRIVATE PHX_NVTX=1)
    endif ()
endif ()

# Link liburing for the io_uring batch engine when available.
if (LIBURING_FOUND)
    target_include_directories(libphoenix PRIVATE ${LIBURING_INCLUDE_DIRS})
    target_link_libraries(libphoenix ${LIBURING_LIBRARIES})
    target_link_directories(libphoenix PRIVATE ${LIBURING_LIBRARY_DIRS})
endif ()
# Set install locations
install (FILES ${libphoenix_include} DESTINATION "include")
install (TARGETS libphoenix DESTINATION "lib")
# Also install the phoenixfs kernel module into the running kernel's module
# tree (/lib/modules/$(uname -r)/extra) + depmod. A failure here (e.g. missing
# root or a mismatched kernel tree) only warns — it must not abort the library
# install above. Run `sudo make install` for a system-path install.
install (CODE "execute_process (COMMAND make -C ${module_output} install RESULT_VARIABLE _phx_mod_install_res)
	if (NOT _phx_mod_install_res EQUAL 0)
		message (WARNING \"phoenixfs.ko was not installed (see output above). libphoenix was still installed.\")
	endif ()")


# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} Threads::Threads)
endmacro ()

# Output directories (shared by libphoenix and the test targets)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Tests require CUDA (they use .cu files)
enable_testing ()
if (PHXFS_VENDOR STREQUAL "NVIDIA" OR PHXFS_VENDOR STREQUAL "METAX")
    add_subdirectory ("${PROJECT_SOURCE_DIR}/test/")
endif ()

