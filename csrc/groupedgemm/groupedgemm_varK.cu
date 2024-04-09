/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cutlass_extensions/groupedgemm_varK_template.h"
#include "cublas_wrapper.h"

using torch::Tensor;

template <typename T,
          typename WeightType,
          typename AccumGradType>
void group_gemm_varK_algo_dispatcher(T*              A,
                                     WeightType*     B,
                                     T*              C,
                                     AccumGradType** weight_grad_list,
                                     int64_t         gemm_m,
                                     int64_t         gemm_n,
                                     int*            gemm_k_per_expert,
                                     int             num_tokens,
                                     int             num_experts,
                                     bool            transC,
                                     cudaStream_t    stream)
{
    int sm_ = getSMVersion();

    if ((sm_ != 90) && (USE_CUBLAS == false))
    {
        MoeGemmRunner<T, WeightType> moe_gemm_runner_;

        moe_gemm_runner_.moe_gemm_backward(
            A,
            B,
            C,
            weight_grad_list,
            gemm_m,
            gemm_n,
            gemm_k_per_expert,
            num_tokens,
            num_experts,
            transC,
            stream);
    }
    else
    {
        if (!cublas_init)
            cublas_handle_init();

        cublas_group_gemm_helper<T, AccumGradType>(
            A,
            B,
            C,
            weight_grad_list,
            gemm_m,
            gemm_n,
            gemm_k_per_expert,
            num_experts,
            transC,
            stream);
    }
}

// act type, weight type
template <typename T, typename WeightType>
Tensor run_group_gemm_backward_helper(Tensor input_activations,
                                      Tensor fc1_expert_weights,
                                      Tensor tokens_per_expert,
                                      bool   transC,
                                      std::vector<Tensor> weight_grad_list)
{
    // Matrix A: X      shape(m, k)
    // Matrix B: dL/dY  shape(m, n)
    // Output C: dL/dW  shape(k, n)

    const int gemm_m = input_activations.size(1);
    const int gemm_n = fc1_expert_weights.size(1);
    const int gemm_k = input_activations.size(0);
    const int num_experts = tokens_per_expert.size(0);

    if ((gemm_m & 0x7 != 0) || (gemm_n & 0x7 != 0))
    {
        throw std::runtime_error("gemm_m and gemm_n of grouped gemm with variable K must be multiples of 8.");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int *tokens_per_expert_ptr = get_ptr<int>(tokens_per_expert);

    T *input_act_ptr = get_ptr<T>(input_activations);
    WeightType *fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);

    const at::ScalarType _st = input_activations.scalar_type();
    Tensor fc1_output;

    if (weight_grad_list.empty())
    {
        if (transC)
        {
            fc1_output = torch::empty({num_experts, gemm_n, gemm_m}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        }
        else
        {
            fc1_output = torch::empty({num_experts, gemm_m, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        }

        T *fc1_output_ptr = get_ptr<T>(fc1_output);
        group_gemm_varK_algo_dispatcher<T, WeightType, T>(
            input_act_ptr,
            fc1_expert_weights_ptr,
            fc1_output_ptr,
            nullptr,
            gemm_m,                // gemm_m
            gemm_n,                // gemm_n
            tokens_per_expert_ptr, // gemm_k
            gemm_k,                // num_tokens
            num_experts,
            transC,
            stream);
    }
    else
    {
        const at::ScalarType _st = weight_grad_list[0].scalar_type();
        switch (_st) {
            case at::ScalarType::Float: {
                using dType = float;

                dType *weight_grad_ptr_list[num_experts];
                for (size_t i = 0; i < num_experts; i++)
                {
                    weight_grad_ptr_list[i] = get_ptr<dType>(weight_grad_list[i]);
                }

                group_gemm_varK_algo_dispatcher<T, WeightType, dType>(
                    input_act_ptr,
                    fc1_expert_weights_ptr,
                    nullptr,
                    weight_grad_ptr_list,
                    gemm_m,                // gemm_m
                    gemm_n,                // gemm_n
                    tokens_per_expert_ptr, // gemm_k
                    gemm_k,                // num_tokens
                    num_experts,
                    transC,
                    stream);

                break;
            }
            case at::ScalarType::Half: {
                using dType = half;

                dType *weight_grad_ptr_list[num_experts];
                for (size_t i = 0; i < num_experts; i++)
                {
                    weight_grad_ptr_list[i] = get_ptr<dType>(weight_grad_list[i]);
                }

                group_gemm_varK_algo_dispatcher<T, WeightType, dType>(
                    input_act_ptr,
                    fc1_expert_weights_ptr,
                    nullptr,
                    weight_grad_ptr_list,
                    gemm_m,                // gemm_m
                    gemm_n,                // gemm_n
                    tokens_per_expert_ptr, // gemm_k
                    gemm_k,                // num_tokens
                    num_experts,
                    transC,
                    stream);

                break;
            }
#ifdef ENABLE_BF16
            case at::ScalarType::BFloat16: {
                using dType = __nv_bfloat16;

                dType *weight_grad_ptr_list[num_experts];
                for (size_t i = 0; i < num_experts; i++)
                {
                    weight_grad_ptr_list[i] = get_ptr<dType>(weight_grad_list[i]);
                }

                group_gemm_varK_algo_dispatcher<T, WeightType, dType>(
                    input_act_ptr,
                    fc1_expert_weights_ptr,
                    nullptr,
                    weight_grad_ptr_list,
                    gemm_m,                // gemm_m
                    gemm_n,                // gemm_n
                    tokens_per_expert_ptr, // gemm_k
                    gemm_k,                // num_tokens
                    num_experts,
                    transC,
                    stream);

                break;
            }
#endif
            default:
                throw std::runtime_error("Wrong main_grad tensor data type.");
        }
    }

    return fc1_output;
}

Tensor moe_group_gemm_backward_op(Tensor input_activations,
                                  Tensor fc1_expert_weights,
                                  Tensor tokens_per_expert,
                                  bool   transC,
                                  std::vector<Tensor> weight_grad_list)
{
    Tensor output_tensor;

    // activations type
    const at::ScalarType _st = input_activations.scalar_type();
    switch (_st) {
        case at::ScalarType::Float: {
            output_tensor = run_group_gemm_backward_helper<float, float>(
                input_activations,
                fc1_expert_weights,
                tokens_per_expert,
                transC,
                weight_grad_list);

            break;
        }
        case at::ScalarType::Half: {
            output_tensor = run_group_gemm_backward_helper<half, half>(
                input_activations,
                fc1_expert_weights,
                tokens_per_expert,
                transC,
                weight_grad_list);

            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            output_tensor = run_group_gemm_backward_helper<__nv_bfloat16, __nv_bfloat16>(
                input_activations,
                fc1_expert_weights,
                tokens_per_expert,
                transC,
                weight_grad_list);

            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong activation tensor type.");
    }
    return output_tensor;
}