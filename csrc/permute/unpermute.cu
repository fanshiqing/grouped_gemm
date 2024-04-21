/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute_kernels.h"

using torch::Tensor;

Tensor moe_recover_topK_op(
    Tensor  input,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

std::tuple<Tensor, Tensor> moe_recover_topK_bwd_op(
    Tensor  input_bwd,
    Tensor  input_fwd,
    Tensor  row_id_map,
    Tensor  prob)
{
    const int num_tokens = prob.size(0);
    const int num_topK = prob.size(1);
    const int num_cols = input_bwd.size(1);

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = get_ptr<float>(prob);

    // activations type
    const at::ScalarType _st = input_bwd.scalar_type();

    // Output buffer alloc
    Tensor act_grad =
        torch::empty({input_fwd.size(0), num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor prob_grad =
        torch::empty({num_tokens, num_topK}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    float *prob_grad_ptr = get_ptr<float>(prob_grad);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(act_grad, prob_grad);
}