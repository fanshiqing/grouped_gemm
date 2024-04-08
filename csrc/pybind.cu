/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

TORCH_LIBRARY(moe_unit_ops, m) {
  m.def("sinkhorn", sinkhorn);
  m.def("moe_permute_topK_op", moe_permute_topK_op);
  m.def("moe_recover_topK_op", moe_recover_topK_op);
  m.def("moe_recover_topK_bwd_op", moe_recover_topK_bwd_op);
  m.def("use_cublas_for_groupedgemm", use_cublas_for_groupedgemm);
  m.def("moe_group_gemm_op", moe_group_gemm_op);
  m.def("moe_group_gemm_backward_op", moe_group_gemm_backward_op);
}