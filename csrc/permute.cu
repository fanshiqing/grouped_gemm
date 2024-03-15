/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute.h"

#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/array.h"

using torch::Tensor;

namespace grouped_gemm {

template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

template <typename T, int kElementsPerAccess>
__global__ void moe_permute_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *row_id_map,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = row_id_map[dest_row];

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x * kElementsPerAccess; tid < num_cols; tid += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<float4, sizeof(float4), cutlass::arch::CacheOperation::LastUse>(
            *(float4 *)(dest_row_ptr + tid), (source_row_ptr + tid), true);
    }
}

template <typename T, int kElementsPerAccess>
__global__ void moe_recover_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *row_id_map,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int source_row = blockIdx.x;

    if (source_row >= num_rows)
        return;

    int dest_row = row_id_map[source_row];

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x * kElementsPerAccess; tid < num_cols; tid += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<float4, sizeof(float4), cutlass::arch::CacheOperation::LastUse>(
            *(float4 *)(dest_row_ptr + tid), (source_row_ptr + tid), true);
    }
}


template <typename T, bool FWD, int kElementsPerAccess>
void moe_permute_kernel_launcher(
    const T *original_input,
    T *permuted_output,
    const int *row_id_map,
    const int num_rows,
    const int num_cols,
    cudaStream_t stream)
{
    if (num_cols & 0x7 != 0)
        throw std::runtime_error("num_cols of input activations must be multiples of 8.");

    const int blocks = num_rows;
    const int threads = std::min(num_cols / kElementsPerAccess, 1024);

    if (FWD)
    {
        moe_permute_kernel<T, kElementsPerAccess><<<blocks, threads, 0, stream>>>(original_input,
                                                                                  permuted_output,
                                                                                  row_id_map,
                                                                                  num_rows,
                                                                                  num_cols);
    }
    else
    {
        moe_recover_kernel<T, kElementsPerAccess><<<blocks, threads, 0, stream>>>(original_input,
                                                                                  permuted_output,
                                                                                  row_id_map,
                                                                                  num_rows,
                                                                                  num_cols);
    }
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<Tensor>> moe_permute_op(
    Tensor original_input,
    Tensor expert_for_rows,
    std::vector<Tensor> workspace,
    int64_t max_token_num)
{
    // initialize the workspace on the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        Tensor row_id = torch::range(0, max_token_num - 1, 1, options);
        Tensor sorted_expert_for_rows = torch::empty(max_token_num, options);

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(row_id);
        workspace.push_back(sorted_expert_for_rows);
        workspace.push_back(temp_storage);
    }

    const int num_rows = original_input.size(0);
    const int num_cols = original_input.size(1);

    // activations type
    const at::ScalarType _st = original_input.scalar_type();

    // Output buffer alloc
    Tensor permuted_output =
        torch::empty({num_rows, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map =
        torch::empty(num_rows, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *expert_for_rows_ptr = get_ptr<int>(expert_for_rows);
    int *row_id_ptr = get_ptr<int>(workspace[0]);
    int *sorted_expert_for_rows_ptr = get_ptr<int>(workspace[1]);
    int *row_id_map_ptr = get_ptr<int>(row_id_map);

    // Run sorting operation
    void *d_temp_storage = get_ptr<void>(workspace[2]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    expert_for_rows_ptr, sorted_expert_for_rows_ptr,
                                    row_id_ptr, row_id_map_ptr, num_rows);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType, true, 4>(
            original_input_ptr,
            permuted_output_ptr,
            row_id_map_ptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType, true, 8>(
            original_input_ptr,
            permuted_output_ptr,
            row_id_map_ptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType, true, 8>(
            original_input_ptr,
            permuted_output_ptr,
            row_id_map_ptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    /// Removed to align with pytorch
    // cudaStreamSynchronize(stream);

    return std::make_tuple(permuted_output, row_id_map, workspace);
}

torch::Tensor moe_recover_op(
    Tensor permuted_input,
    Tensor row_id_map)
{
    const int num_rows = permuted_input.size(0);
    const int num_cols = permuted_input.size(1);

    // activations type
    const at::ScalarType _st = permuted_input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_rows, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_kernel_launcher<dType, false, 4>(
            permuted_input_ptr,
            unpermuted_output_ptr,
            row_id_map_ptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_kernel_launcher<dType, false, 8>(
            permuted_input_ptr,
            unpermuted_output_ptr,
            row_id_map_ptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_kernel_launcher<dType, false, 8>(
            permuted_input_ptr,
            unpermuted_output_ptr,
            row_id_map_ptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    /// Removed to align with pytorch
    // cudaStreamSynchronize(stream);

    return unpermuted_output;
}


}  // namespace grouped_gemm
