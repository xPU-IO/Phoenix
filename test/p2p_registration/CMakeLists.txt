cmake_minimum_required (VERSION 3.18)
project (p2p-registration-test LANGUAGES CXX)

find_package (CUDAToolkit REQUIRED)
add_executable (test_p2p_registration test_p2p_registration.cpp)
target_compile_features (test_p2p_registration PRIVATE cxx_std_11)
target_link_libraries (test_p2p_registration PRIVATE CUDA::cudart)

execute_process (COMMAND uname -r OUTPUT_VARIABLE KERNEL_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set (KERNEL_BUILD_DIR "/lib/modules/${KERNEL_VERSION}/build" CACHE PATH "Kernel build tree")
set (PHOENIX_MODULE_SYMVERS "${CMAKE_CURRENT_SOURCE_DIR}/../../build/module/Module.symvers"
    CACHE FILEPATH "Module.symvers from the Phoenix module build")
find_program (KERNEL_MAKE NAMES make REQUIRED)
set (CONSUMER_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/consumer-module")
file (MAKE_DIRECTORY "${CONSUMER_BUILD_DIR}")
configure_file (test_consumer.c "${CONSUMER_BUILD_DIR}/test_consumer.c" COPYONLY)
configure_file (test_ioctl.h "${CONSUMER_BUILD_DIR}/test_ioctl.h" COPYONLY)
file (WRITE "${CONSUMER_BUILD_DIR}/Makefile"
    "obj-m += phxfs_p2p_test.o\nphxfs_p2p_test-y := test_consumer.o\nccflags-y += -I${CMAKE_CURRENT_SOURCE_DIR}/../../module\n")
add_custom_target (p2p_registration_module
    COMMAND "${KERNEL_MAKE}" -C "${KERNEL_BUILD_DIR}" "M=${CONSUMER_BUILD_DIR}"
        "KBUILD_EXTRA_SYMBOLS=${PHOENIX_MODULE_SYMVERS}" modules
    VERBATIM)
