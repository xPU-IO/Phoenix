# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(phoenix_tools LANGUAGES C)
add_executable(bar_holder "${PHXFS_ROOT}/test/kernel_module_matrix/bar_holder.c")
set_target_properties(bar_holder PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PHXFS_OUTPUT_DIR}/bin")
