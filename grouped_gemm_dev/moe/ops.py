# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import os
from sys import stderr

so_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/build'
torch.classes.load_library(so_dir + '/libmoe_unit_ops.so')

# TODO by Jiang Shao, add parameter `out` which can be optionally given to be used as output buffers.

################################################################################################
##
## PermuteMoE
##
################################################################################################

class PermuteMoE(torch.autograd.Function):
  
  workspace_fw=None
  dtype=None
  max_token_num=0

  @staticmethod
  def forward(ctx, 
              unpermuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              max_token_num: int):

    # Device check
    if unpermuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"unpermuted_inputs\" of permute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
      print("[Warning] The input \"expert_for_rows\" of permute op is on the device: CPU!", file=stderr)
      expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if unpermuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"[Error] permute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {unpermuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
      print("[Warning] The data type of the input \"expert_for_rows\" of permute op is Int64! "
            "The recommended type is int32.", file=stderr)
      expert_for_rows = expert_for_rows.to(torch.int32)

    # Contiguous check
    if not unpermuted_inputs.is_contiguous():
      print("[Warning] The input \"unpermuted_inputs\" of permute op is discontiguous!", file=stderr)
      unpermuted_inputs = unpermuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
      print("[Warning] The input \"expert_for_rows\" of permute op is discontiguous!", file=stderr)
      expert_for_rows = expert_for_rows.contiguous()

    input_max_token_num = max(max_token_num, unpermuted_inputs.size(0))
    if PermuteMoE.max_token_num < input_max_token_num:
      # print("Permute op workspace reset!")
      PermuteMoE.max_token_num = input_max_token_num
      PermuteMoE.workspace_fw = []

    if PermuteMoE.dtype != unpermuted_inputs.dtype:
      # print("Permute op workspace reset!")
      PermuteMoE.dtype = unpermuted_inputs.dtype
      PermuteMoE.workspace_fw = []


    permuted_inputs, source_row_to_dest_row, PermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs,
      expert_for_rows,
      PermuteMoE.workspace_fw,
      PermuteMoE.max_token_num)

    ctx.source_row_to_dest_row = source_row_to_dest_row

    return permuted_inputs, source_row_to_dest_row

  @staticmethod
  def backward(ctx, permuted_inputs_grad, _):
    if not permuted_inputs_grad.is_contiguous():
      permuted_inputs_grad = permuted_inputs_grad.contiguous()
    source_row_to_dest_row = ctx.source_row_to_dest_row

    original_output = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs_grad,
      source_row_to_dest_row)

    return original_output, None, None

################################################################################################
##
## UnpermuteMoE
##
################################################################################################

class UnpermuteMoE(torch.autograd.Function):

  workspace_fw=None
  dtype=None
  max_token_num=0
  
  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              source_row_to_dest_row: torch.Tensor,
              max_token_num: int):

    # Device check
    if permuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"permuted_inputs\" of unpermute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
      print("[Warning] The input \"expert_for_rows\" of unpermute op is on the device: CPU!", file=stderr)
      expert_for_rows = expert_for_rows.cuda()
    if source_row_to_dest_row.is_cpu:
      print("[Warning] The input \"source_row_to_dest_row\" of unpermute op is on the device: CPU!", file=stderr)
      source_row_to_dest_row = source_row_to_dest_row.cuda()

    # Shape check
    if permuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"[Error] unpermute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {permuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
      print("[Warning] The data type of the input \"expert_for_rows\" of unpermute op is Int64! "
            "The recommended type is int32.", file=stderr)
      expert_for_rows = expert_for_rows.to(torch.int32)
    if source_row_to_dest_row.dtype != torch.int32:
      print("[Warning] The data type of the input \"source_row_to_dest_row\" of unpermute op is Int64! "
            "The recommended type is int32.", file=stderr)
      source_row_to_dest_row = source_row_to_dest_row.to(torch.int32)

    # Contiguous check
    if not permuted_inputs.is_contiguous():
      print("[Warning] The input \"permuted_inputs\" of unpermute op is discontiguous!", file=stderr)
      permuted_inputs = permuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
      print("[Warning] The input \"expert_for_rows\" of unpermute op is discontiguous!", file=stderr)
      expert_for_rows = expert_for_rows.contiguous()
    if not source_row_to_dest_row.is_contiguous():
      print("[Warning] The input \"source_row_to_dest_row\" of unpermute op is discontiguous!", file=stderr)
      source_row_to_dest_row = source_row_to_dest_row.contiguous()

    if UnpermuteMoE.max_token_num < max_token_num:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = max_token_num
      UnpermuteMoE.workspace_fw = []

    if UnpermuteMoE.max_token_num < permuted_inputs.size(0) or UnpermuteMoE.dtype != permuted_inputs.dtype:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = permuted_inputs.size(0)
      UnpermuteMoE.dtype = permuted_inputs.dtype
      UnpermuteMoE.workspace_fw = []

    ctx.expert_for_rows = expert_for_rows

    original_output = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs,
      source_row_to_dest_row)
    
    return original_output
  
  @staticmethod
  def backward(ctx, unpermuted_inputs_grad):
    if not unpermuted_inputs_grad.is_contiguous():
      unpermuted_inputs_grad = unpermuted_inputs_grad.contiguous()
    expert_for_rows = ctx.expert_for_rows

    permuted_inputs, _, UnpermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs_grad,
      expert_for_rows,
      UnpermuteMoE.workspace_fw,
      UnpermuteMoE.max_token_num)

    return permuted_inputs, None, None, None

################################################################################################
##
## GroupedGemmMoE
##
################################################################################################

class GroupedGemmMoE(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              weights: torch.Tensor,
              tokens_per_expert: torch.Tensor,
              transB: bool):

    # Device check
    if permuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"permuted_inputs\" of groupedgemm op is on the device: CPU!")
    if weights.is_cpu:
      raise RuntimeError("[Error] The input \"weights\" of groupedgemm op is on the device: CPU!")
    if tokens_per_expert.is_cpu:
      print("[Warning] The input \"tokens_per_expert\" of groupedgemm op is on the device: CPU!", file=stderr)
      tokens_per_expert = tokens_per_expert.cuda()

    # Shape check
    if not transB:
      if permuted_inputs.size(1) != weights.size(1):
        raise RuntimeError(f"[Error] groupedgemm op input \"weights\" shape mismatch! "
                           f"Expect {permuted_inputs.size(1)}, but got {weights.size(1)}.")
    else:
      if permuted_inputs.size(1) != weights.size(2):
        raise RuntimeError(f"[Error] groupedgemm op input \"weights\" shape mismatch! "
                           f"Expect {permuted_inputs.size(1)}, but got {weights.size(2)}.")

    # Data type check
    if permuted_inputs.dtype != weights.dtype:
      raise RuntimeError(f"[Error] groupedgemm op input data type mismatch! "
                         f"\"permuted_inputs\": {permuted_inputs.dtype}, \"weights\": {weights.dtype}.")
    if tokens_per_expert.dtype != torch.int32:
      print("[Warning] The data type of the input \"tokens_per_expert\" of groupedgemm op is Int64! "
            "The recommended type is int32.", file=stderr)
      tokens_per_expert = tokens_per_expert.to(torch.int32)

    # Contiguous check
    if not permuted_inputs.is_contiguous():
      print("[Warning] The input \"permuted_inputs\" of groupedgemm op is discontiguous!", file=stderr)
      permuted_inputs = permuted_inputs.contiguous()
    if not weights.is_contiguous():
      print("[Warning] The input \"weights\" of groupedgemm op is discontiguous!", file=stderr)
      weights = weights.contiguous()

    output = torch.ops.moe_unit_ops.moe_group_gemm_op(
      permuted_inputs,
      weights,
      tokens_per_expert,
      transB)
    
    ctx.save_for_backward(permuted_inputs, tokens_per_expert, weights)
    ctx.transB = transB

    return output


  @staticmethod
  def backward(ctx, permuted_inputs_grad):
      
    permuted_inputs, tokens_per_expert, weights = ctx.saved_tensors
    transB = ctx.transB

    if not permuted_inputs_grad.is_contiguous():
      permuted_inputs_grad = permuted_inputs_grad.contiguous()

    activation_grad = None
    if ctx.needs_input_grad[0]:
      activation_grad = torch.ops.moe_unit_ops.moe_group_gemm_op(
        permuted_inputs_grad,
        weights,
        tokens_per_expert,
        not transB)

    weight_grad = None
    if ctx.needs_input_grad[1]:
      weight_grad = torch.ops.moe_unit_ops.moe_group_gemm_backward_op(
        permuted_inputs,
        permuted_inputs_grad,
        tokens_per_expert,
        transB)

    return activation_grad, weight_grad, None, None

################################################################################################
##
## Ops Wrapper
##
################################################################################################

def permute(unpermuted_inputs, expert_for_rows, max_token_num=0):
  return PermuteMoE.apply(unpermuted_inputs, expert_for_rows, max_token_num)

def unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num=0):
  return UnpermuteMoE.apply(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num)

def gmm(permuted_inputs, weights, tokens_per_expert, trans_b=False):
  return GroupedGemmMoE.apply(permuted_inputs, weights, tokens_per_expert, trans_b)

def sinkhorn_kernel(cost, tol=0.0001):
    return torch.ops.moe_unit_ops.sinkhorn(cost, tol)
