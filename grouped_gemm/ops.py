from grouped_gemm import backend
import torch
import warnings

from sys import stderr


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
      warnings.warn("Got {} dtype for \"expert_for_rows\", will be casted into torch.int32".format(expert_for_rows.dtype))
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

    permuted_inputs, row_id_map, PermuteMoE.workspace_fw = backend.permute(
      unpermuted_inputs,
      expert_for_rows,
      PermuteMoE.workspace_fw,
      PermuteMoE.max_token_num)

    ctx.row_id_map = row_id_map

    return permuted_inputs, row_id_map

  @staticmethod
  def backward(ctx, permuted_inputs_grad, _):
    if not permuted_inputs_grad.is_contiguous():
      permuted_inputs_grad = permuted_inputs_grad.contiguous()
    row_id_map = ctx.row_id_map

    original_output = backend.unpermute(
      permuted_inputs_grad,
      row_id_map)

    return original_output, None, None

class UnpermuteMoE(torch.autograd.Function):

  workspace_fw=None
  dtype=None
  max_token_num=0
  
  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              row_id_map: torch.Tensor,
              max_token_num: int):

    # Device check
    if permuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"permuted_inputs\" of unpermute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
      print("[Warning] The input \"expert_for_rows\" of unpermute op is on the device: CPU!", file=stderr)
      expert_for_rows = expert_for_rows.cuda()
    if row_id_map.is_cpu:
      print("[Warning] The input \"row_id_map\" of unpermute op is on the device: CPU!", file=stderr)
      row_id_map = row_id_map.cuda()

    # Shape check
    if permuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"[Error] unpermute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {permuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
      warnings.warn("Got {} dtype for \"expert_for_rows\", will be casted into torch.int32".format(expert_for_rows.dtype))
      expert_for_rows = expert_for_rows.to(torch.int32)
    if row_id_map.dtype != torch.int32:
      print("[Warning] The data type of the input \"row_id_map\" of unpermute op is int64! "
            "The recommended type is int32.", file=stderr)
      row_id_map = row_id_map.to(torch.int32)

    # Contiguous check
    if not permuted_inputs.is_contiguous():
      print("[Warning] The input \"permuted_inputs\" of unpermute op is discontiguous!", file=stderr)
      permuted_inputs = permuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
      print("[Warning] The input \"expert_for_rows\" of unpermute op is discontiguous!", file=stderr)
      expert_for_rows = expert_for_rows.contiguous()
    if not row_id_map.is_contiguous():
      print("[Warning] The input \"row_id_map\" of unpermute op is discontiguous!", file=stderr)
      row_id_map = row_id_map.contiguous()

    input_max_token_num = max(max_token_num, permuted_inputs.size(0))
    if UnpermuteMoE.max_token_num < input_max_token_num:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = input_max_token_num
      UnpermuteMoE.workspace_fw = []

    if UnpermuteMoE.dtype != permuted_inputs.dtype:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.dtype = permuted_inputs.dtype
      UnpermuteMoE.workspace_fw = []

    ctx.expert_for_rows = expert_for_rows

    original_output = backend.unpermute(
      permuted_inputs,
      row_id_map)
    
    return original_output
  
  @staticmethod
  def backward(ctx, unpermuted_inputs_grad):
    if not unpermuted_inputs_grad.is_contiguous():
      unpermuted_inputs_grad = unpermuted_inputs_grad.contiguous()
    expert_for_rows = ctx.expert_for_rows

    permuted_inputs, _, UnpermuteMoE.workspace_fw = backend.permute(
      unpermuted_inputs_grad,
      expert_for_rows,
      UnpermuteMoE.workspace_fw,
      UnpermuteMoE.max_token_num)

    return permuted_inputs, None, None, None

def permute(unpermuted_inputs, expert_for_rows, max_token_num=0):
  return PermuteMoE.apply(unpermuted_inputs, expert_for_rows, max_token_num)

def unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num=0):
  return UnpermuteMoE.apply(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num)