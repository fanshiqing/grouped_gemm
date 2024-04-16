/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cutlass_extensions/groupedgemm_varM_template.h"
#include "cublas_wrapper.h"

using torch::Tensor;

bool USE_CUBLAS = true;
bool cublas_init = false;
cublasHandle_t cublas_handle[NUM_STREAM];
cudaStream_t cublas_stream[NUM_STREAM];
cudaEvent_t cublas_event[NUM_STREAM];

void use_cublas_for_groupedgemm(bool enable)
{
    USE_CUBLAS = enable;
}

// act type, weight type
template <typename T, typename WeightType>
Tensor run_group_gemm_helper(Tensor              input_activations,
                             std::vector<Tensor> fc1_expert_weights_list,
                             Tensor              tokens_per_expert,
                             bool                transB)
{
    const int gemm_m = input_activations.size(0);
    int gemm_n;
    if (transB) gemm_n = fc1_expert_weights_list[0].size(0);
    else gemm_n = fc1_expert_weights_list[0].size(1);
    const int gemm_k = input_activations.size(1);
    const int num_experts = tokens_per_expert.size(0);

    if (gemm_k & 0x7 != 0)
    {
        throw std::runtime_error("gemm_k of grouped gemm with variable M must be a multiple of 8.");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int *tokens_per_expert_ptr = get_ptr<int>(tokens_per_expert);

    T *input_act_ptr = get_ptr<T>(input_activations);
    WeightType *fc1_expert_weights_ptr_list[num_experts];
    for (size_t i = 0; i < num_experts; i++)
    {
        fc1_expert_weights_ptr_list[i] = get_ptr<WeightType>(fc1_expert_weights_list[i]);
    }

    const at::ScalarType _st = input_activations.scalar_type();
    auto fc1_output =
        torch::empty({gemm_m, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T *fc1_output_ptr = get_ptr<T>(fc1_output);

    // int sm_ = getSMVersion();

    // if ((sm_ != 90) && (USE_CUBLAS == false))
    if (USE_CUBLAS == false)
    {
        MoeGemmRunner<T, WeightType> moe_gemm_runner_;

        moe_gemm_runner_.moe_gemm(input_act_ptr,
                                  fc1_expert_weights_ptr_list,
                                  fc1_output_ptr,
                                  tokens_per_expert_ptr, // gemm_m
                                  gemm_n,                // gemm_n
                                  gemm_k,                // gemm_k
                                  gemm_m,                // num_tokens
                                  num_experts,
                                  transB,
                                  stream);
    }
    else
    {
        cublas_group_gemm_helper<T>(
            input_act_ptr,
            fc1_expert_weights_ptr_list,
            fc1_output_ptr,
            tokens_per_expert_ptr, // gemm_m
            gemm_n,                // gemm_n
            gemm_k,                // gemm_k
            num_experts,
            transB,
            stream);
    }

    return fc1_output;
}

Tensor moe_group_gemm_op(Tensor              input_activations,
                         std::vector<Tensor> fc1_expert_weights_list,
                         Tensor              tokens_per_expert,
                         bool                transB)
{
    Tensor output_tensor;

    // activations type
    const at::ScalarType _st = input_activations.scalar_type();
    switch (_st) {
        case at::ScalarType::Float: {
            output_tensor = run_group_gemm_helper<float, float>(
                input_activations,
                fc1_expert_weights_list,
                tokens_per_expert,
                transB);
            break;
        }
        case at::ScalarType::Half: {
            output_tensor = run_group_gemm_helper<half, half>(
                input_activations,
                fc1_expert_weights_list,
                tokens_per_expert,
                transB);
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            output_tensor = run_group_gemm_helper<__nv_bfloat16, __nv_bfloat16>(
                input_activations,
                fc1_expert_weights_list,
                tokens_per_expert,
                transB);
            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong activation tensor type.");
    }
    return output_tensor;
}