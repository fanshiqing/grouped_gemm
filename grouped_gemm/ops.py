from grouped_gemm import backend
import torch
import warnings

from sys import stderr
import torch.cuda.nvtx as nvtx


# For debug's convenience
ENABLE_NVTX=False

class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)

def sinkhorn_kernel(cost, tol=0.0001):
    return backend.sinkhorn(cost, tol)

################################################################################################
##
## PermuteMoE topK
##
################################################################################################

class PermuteMoE_topK(torch.autograd.Function):

  workspace_fw=None
  dtype=None
  max_expanded_token_num=0

  @staticmethod
  def forward(ctx, 
              input_act: torch.Tensor,
              indices: torch.Tensor,
              num_out_tokens: int,
              max_token_num: int):
    '''
    indices: for topK=1, indices in a 1-d tensor of shape [num_tokens],
             otherwise, it's a 2-d tensor of shape [num_tokens, topK]
    '''
    if ENABLE_NVTX:
      nvtx.range_push("permute_topK forward")
    # Empty input check
    if not input_act.numel():
      if ENABLE_NVTX:
        nvtx.range_pop()
      return input_act, None

    # For top1 case, view the indices as 2D tensor to unify the shape for topk>=2 cases.
    if indices.dim() == 1:
      indices = indices.view(-1, 1)

    # Device check
    if input_act.is_cpu:
      raise RuntimeError("[Error] The input `input_act` of permute_topK op is on the device: CPU!")
    if indices.is_cpu:
      warnings.warn("The input `indices` of permute_topK op is on the device: CPU!")
      expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if input_act.size(0) != indices.size(0):
      raise RuntimeError(f"[Error] permute_topK op input `indices` shape mismatch! "
                         f"Expect {input_act.size(0)}, but got {indices.size(0)}.")

    # Data type check
    if indices.dtype != torch.int32:
      warnings.warn(f"The data type of the input `indices` of permute_topK op is {indices.dtype}! "
            "The recommended type is torch.int32.")
      indices = indices.to(torch.int32)

    # Contiguous check
    if not input_act.is_contiguous():
      warnings.warn("The input `input_act` of permute_topK op is discontiguous!")
      input_act = input_act.contiguous()
    if not indices.is_contiguous():
      warnings.warn("The input `indices` of permute_topK op is discontiguous!")
      indices = indices.contiguous()

    num_topK = indices.size(1)

    input_max_expanded_token_num = max(max_token_num, input_act.size(0)) * num_topK
    if PermuteMoE_topK.max_expanded_token_num < input_max_expanded_token_num:
      PermuteMoE_topK.max_expanded_token_num = input_max_expanded_token_num
      PermuteMoE_topK.workspace_fw = []

    if PermuteMoE_topK.dtype != input_act.dtype:
      PermuteMoE_topK.dtype = input_act.dtype
      PermuteMoE_topK.workspace_fw = []

    permuted_act, row_id_map, PermuteMoE_topK.workspace_fw = backend.permute(
      input_act,
      indices,
      num_out_tokens,
      PermuteMoE_topK.workspace_fw,
      PermuteMoE_topK.max_expanded_token_num)

    ctx.row_id_map = row_id_map
    ctx.num_tokens = indices.size(0)
    ctx.num_topK = num_topK
    if ENABLE_NVTX:
      nvtx.range_pop()
    return permuted_act, row_id_map


  @staticmethod
  def backward(ctx, permuted_act_grad, _):
    if ENABLE_NVTX:
      nvtx.range_push("permute_topK backward")
    # Empty input check
    if not permuted_act_grad.numel():
      if ENABLE_NVTX:
        nvtx.range_pop()
      return permuted_act_grad, None, None, None

    if not permuted_act_grad.is_contiguous():
      permuted_act_grad = permuted_act_grad.contiguous()

    row_id_map = ctx.row_id_map
    num_tokens = ctx.num_tokens
    num_topK = ctx.num_topK

    unpermuted_act_grad = backend.unpermute(
      permuted_act_grad,
      row_id_map,
      torch.tensor([]),
      num_tokens,
      num_topK)
    if ENABLE_NVTX:
      nvtx.range_pop()
    return unpermuted_act_grad, None, None, None

################################################################################################
##
## UnpermuteMoE topK
##
################################################################################################

class UnpermuteMoE_topK(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              input_act: torch.Tensor,
              row_id_map: torch.Tensor,
              probs: torch.Tensor = None):
    if ENABLE_NVTX:
      nvtx.range_push("unpermute_topK forward")
    # Empty input check
    if not input_act.numel():
      ctx.probs = probs
      if ENABLE_NVTX:
        nvtx.range_pop()
      return input_act

    # Device check
    if input_act.is_cpu:
      raise RuntimeError("[Error] The input `input_act` of unpermute_topK op is on the device: CPU!")
    if row_id_map.is_cpu:
      warnings.warn("The input `row_id_map` of unpermute_topK op is on the device: CPU!")
      row_id_map = row_id_map.cuda()
    if probs is not None and probs.is_cpu:
      warnings.warn("The input `probs` of unpermute_topK op is on the device: CPU!")
      probs = probs.cuda()

    # Shape check
    if probs is not None and row_id_map.size(0) != probs.size(0) * probs.size(1):
      raise RuntimeError(f"[Error] unpermute_topK op input `probs` shape mismatch! "
                         f"Expect {row_id_map.size(0)}, but got {probs.size(0) * probs.size(1)}.")

    # Data type check
    if row_id_map.dtype != torch.int32:
      warnings.warn(f"The data type of the input `row_id_map` of unpermute_topK op is {row_id_map.dtype}! "
            "The recommended type is torch.int32.")
      row_id_map = row_id_map.to(torch.int32)
    if probs is not None and probs.dtype != torch.float32:
      warnings.warn(f"The data type of the input `probs` of unpermute_topK op is {probs.dtype}! "
            "The recommended type is torch.float32.")
      probs = probs.to(torch.float32)

    # Contiguous check
    if not input_act.is_contiguous():
      warnings.warn("The input `input_act` of unpermute_topK op is discontiguous!")
      input_act = input_act.contiguous()
    if not row_id_map.is_contiguous():
      warnings.warn("The input `row_id_map` of unpermute_topK op is discontiguous!")
      row_id_map = row_id_map.contiguous()
    if probs is not None and not probs.is_contiguous():
      warnings.warn("The input `probs` of unpermute_topK op is discontiguous!")
      probs = probs.contiguous()

    num_tokens = probs.size(0) if probs is not None else input_act.size(0)
    num_topK = probs.size(1) if probs is not None else 1

    unpermuted_output = backend.unpermute(
      input_act,
      row_id_map,
      probs if probs is not None else torch.tensor([]),
      num_tokens,
      num_topK)

    ctx.save_for_backward(input_act, row_id_map, probs)
    if ENABLE_NVTX:
      nvtx.range_pop()
    return unpermuted_output

  @staticmethod
  def backward(ctx, unpermuted_act_grad):
    if ENABLE_NVTX:
      nvtx.range_push("unpermute_topK backward")
    # Empty input check
    if not unpermuted_act_grad.numel():
      if ENABLE_NVTX:
        nvtx.range_pop()
      return unpermuted_act_grad, None, ctx.probs

    if not unpermuted_act_grad.is_contiguous():
      unpermuted_act_grad = unpermuted_act_grad.contiguous()

    input_act, row_id_map, probs = ctx.saved_tensors

    act_grad = None
    if ctx.needs_input_grad[0]:
      act_grad, prob_grad = backend.unpermute_bwd(
        unpermuted_act_grad,
        input_act,
        row_id_map,
        probs)

    if not ctx.needs_input_grad[2]:
      prob_grad = None
    if ENABLE_NVTX:
      nvtx.range_pop()
    return act_grad, None, prob_grad

def permute(input_act, indices, num_out_tokens=None, max_token_num=0):
  num_out_tokens = 0 if num_out_tokens is None else num_out_tokens
  return PermuteMoE_topK.apply(input_act, indices, num_out_tokens, max_token_num)

def unpermute(input_act, row_id_map, probs=None):
  return UnpermuteMoE_topK.apply(input_act, row_id_map, probs)
