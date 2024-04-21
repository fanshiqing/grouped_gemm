/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint> // uint8_t type on AMD cpus
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cub/cub.cuh>

#include "common.h"

using torch::Tensor;

/***************************************************************************************************
 * sinkhorn
 **************************************************************************************************/

Tensor sinkhorn(Tensor cost, const double tol=0.0001);

/***************************************************************************************************
 * permute
 **************************************************************************************************/

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num);

Tensor moe_recover_topK_op(
    Tensor  input,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK);

std::tuple<Tensor, Tensor> moe_recover_topK_bwd_op(
    Tensor input_bwd,
    Tensor input_fwd,
    Tensor row_id_map,
    Tensor prob);

/***************************************************************************************************
 * grouped gemm
 **************************************************************************************************/

void use_cublas_for_groupedgemm(bool enable);

Tensor moe_group_gemm_op(
    Tensor              input_activations,
    std::vector<Tensor> fc1_expert_weights_list,
    Tensor              tokens_per_expert,
    bool                transB);

Tensor moe_group_gemm_backward_op(
    Tensor              input_activations,
    Tensor              fc1_expert_weights,
    Tensor              tokens_per_expert,
    bool                transC,
    std::vector<Tensor> weight_grad_list);