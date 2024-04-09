# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.10)

project(MoE_test)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message (STATUS "No CMAKE_BUILD_TYPE selected, defaulting to ${CMAKE_BUILD_TYPE}")
elseif()
  message(STATUS "CMAKE_BUILD_TYPE has beed set to ${CMAKE_BUILD_TYPE}")
endif()


set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_VERSION)
if (TORCH_VERSION VERSION_LESS "1.5.0")
    message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
endif()
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
message(STATUS "TORCH_DIR: ${TORCH_DIR}")

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES})
  
  # Get NVIDIA GPU arch
  execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
if torch.cuda.is_available():
  device = torch.device(\"cuda:0\")
  capability = torch.cuda.get_device_capability(device)
  major_arch, minor_arch = capability
  gpu_arch = major_arch * 10 + minor_arch
  print(gpu_arch)"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE GPU_ARCH)
  message (STATUS "Autodetected NVIDIA GPU arch: ${GPU_ARCH}")

  if (GPU_ARCH EQUAL 70)
    add_definitions("-DARCH_70")
  elseif (GPU_ARCH LESS 80)
    add_definitions("-DARCH_75")
  elseif (GPU_ARCH LESS_EQUAL 90)
    add_definitions("-DARCH_80")
  else ()
    message (SEND_ERROR "GPU arch not supported!")
  endif()

  string(SUBSTRING ${GPU_ARCH} 0 1 GPU_MAJOR_ARCH)
  string(SUBSTRING ${GPU_ARCH} 1 1 GPU_MINOR_ARCH)
  set(GPU_ARCH_TORCH "${GPU_MAJOR_ARCH}.${GPU_MINOR_ARCH}")

  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
  set(TORCH_CUDA_ARCH_LIST ${GPU_ARCH_TORCH})
  find_package(Torch REQUIRED)

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O0;-lineinfo;-lcudart;--expt-extended-lambda;--expt-relaxed-constexpr;--generate-code=arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH};--use_fast_math;")
  message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
else (CUDA_FOUND)
  message (SEND_ERROR "Unable to locate CUDA")
endif()

set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/../third_party/cutlass/include)
set(CUTLASS_UTIL_DIR ${PROJECT_SOURCE_DIR}/../third_party/cutlass/tools/util/include)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUTLASS_HEADER_DIR}
  ${CUTLASS_UTIL_DIR}
)
include_directories(${COMMON_HEADER_DIRS})
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")


if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag")
endif()

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message("CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()

option(BUILD_CUTLASS_MOE "Builds CUTLASS kernels supporting MoE GEMM" ON)
if(BUILD_CUTLASS_MOE)
  message(STATUS "Add DBUILD_CUTLASS_MOE, requires CUTLASS. Increases compilation time")
  add_definitions("-DBUILD_CUTLASS_MOE")
endif()

set(moe_files pybind.cu sinkhorn.cu)
file(GLOB moe_files ${moe_files} permute/*.cu)
file(GLOB moe_files ${moe_files} groupedgemm/*.cu)

set(LIB_NAME "moe_unit_ops")
add_library(${LIB_NAME} SHARED ${moe_files})
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)
target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")

