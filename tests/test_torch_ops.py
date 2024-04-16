# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import unittest
import torch.cuda.nvtx as nvtx
import triton

try:
  from grouped_gemm.ops import permute, unpermute, groupedgemm, set_grouped_gemm_algo
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  import sys
  sys.path.append("..")
  from ops import permute, unpermute, groupedgemm, set_grouped_gemm_algo


class TestMoeOps(unittest.TestCase):

  def setUp(self) -> None:
    torch.manual_seed(734876213)

################################################################################################
##
## Test Helpers
##
################################################################################################

  def permute_ops_helper(self,
                         num_rows,
                         max_token_num,
                         num_cols,
                         num_experts,
                         dtype,
                         atol,
                         execution_times,
                         PRINT):

    # Prepare inputs
    _1_expert_for_rows = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda().unsqueeze(-1)
    _2_expert_for_rows = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda().unsqueeze(-1)
    _probs = torch.ones_like(_1_expert_for_rows, dtype=torch.float32)

    # unpermuted_inputs = torch.rand(size=(num_rows, num_cols), dtype=torch.float32).type(dtype).cuda()
    # unpermuted_inputs = torch.randint(size=(num_rows, num_cols), low=0, high=400, dtype=torch.int32).type(dtype).cuda()
    unpermuted_inputs = torch.empty(size=(num_rows, num_cols), dtype=torch.float32)
    for i in range(num_rows):
        unpermuted_inputs[i] = i % 300
    unpermuted_inputs = unpermuted_inputs.type(dtype).cuda()
    unpermuted_inputs.requires_grad_(True)
    original_inputs = unpermuted_inputs.detach()

    # Build network
    for _ in range(execution_times):
      # Forward
      nvtx.range_push("permute op forward")
      _1_permuted_inputs, _1_row_id_map = permute(unpermuted_inputs, _1_expert_for_rows, max_token_num)
      nvtx.range_pop()

      nvtx.range_push("unpermute op forward")
      _1_unpermute_outputs = unpermute(_1_permuted_inputs, _1_row_id_map, _probs)
      nvtx.range_pop()

      nvtx.range_push("permute op forward")
      _2_permuted_inputs, _2_row_id_map = permute(_1_unpermute_outputs, _2_expert_for_rows, max_token_num)
      nvtx.range_pop()

      nvtx.range_push("unpermute op forward")
      _2_unpermute_outputs = unpermute(_2_permuted_inputs, _2_row_id_map, _probs)
      nvtx.range_pop()

      # Reset grad to avoid accumulation
      unpermuted_inputs.grad = None
      _1_permuted_inputs.grad = None
      _1_unpermute_outputs.grad = None
      _2_permuted_inputs.grad = None

      # Backward
      nvtx.range_push("permute & unpermute op backward")
      _2_unpermute_outputs.backward(_2_unpermute_outputs.detach())
      nvtx.range_pop()

    if PRINT:
      print("unpermuted_inputs: {}".format(unpermuted_inputs))
      print("_1_expert_for_rows: {}".format(_1_expert_for_rows))
      print("_1_permuted_inputs: {}".format(_1_permuted_inputs))
      print("_1_row_id_map: {}".format(_1_row_id_map))
      print("_1_unpermute_outputs: {}".format(_1_unpermute_outputs))
      print("_2_expert_for_rows: {}".format(_2_expert_for_rows))
      print("_2_permuted_inputs: {}".format(_2_permuted_inputs))
      print("_2_row_id_map: {}".format(_2_row_id_map))
      print("_2_unpermute_outputs: {}".format(_2_unpermute_outputs))
      print("original_inputs: {}".format(original_inputs))
      print("backward: {}".format(unpermuted_inputs.grad))

    if not unpermuted_inputs.numel():
      print("permute & unpermute empty input activation test passed.")
      return

    # Result check
    original_inputs = original_inputs.float().cpu().numpy().flatten()
    original_output = _2_unpermute_outputs.float().cpu().detach().numpy().flatten()
    max_abs_error = abs(original_inputs - original_output).max()
    print(f"permute & unpermute forward max error: \t\t\t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_permute failed!"

    original_output = unpermuted_inputs.grad.float().cpu().numpy().flatten()
    max_abs_error = abs(original_inputs - original_output).max()
    print(f"permute & unpermute backward max error: \t\t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_permute failed!"

  def groupedgemm_ops_helper(self,
                             num_rows,
                             hidden_size,
                             inter_size,
                             num_experts,
                             transB,
                             dtype,
                             atol,
                             gradient_accumulation_fusion=False,
                             execution_times=1,
                             PRINT=False,
                             BENCHMARK=False):
    # Prepare inputs
    rand_mean = 0
    rand_std = 0.02

    expert_for_rows = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda()
    tokens_per_expert = torch.bincount(expert_for_rows, minlength=num_experts)
    tokens_per_expert = tokens_per_expert.to(torch.int32).cpu()

    permuted_inputs = torch.empty([num_rows, hidden_size], dtype=dtype, device="cuda").normal_(rand_mean, rand_std)
    permuted_inputs.requires_grad_(True)

    weights_list = []
    for i in range(num_experts):
      if not transB:
        weights = torch.empty([hidden_size, inter_size], dtype=dtype, device="cuda").normal_(rand_mean, rand_std)    
      else:
        weights = torch.empty([inter_size, hidden_size], dtype=dtype, device="cuda").normal_(rand_mean, rand_std)    
      weights_list.append(weights.detach())
      weights_list[i].requires_grad_(True)
      weights_list[i].main_grad = torch.zeros_like(weights_list[i], dtype=torch.float32)

    # Build network
    for _ in range(execution_times):
      # Forward
      nvtx.range_push("grouped gemm op forward")
      gemm_output = groupedgemm(permuted_inputs,
                                tokens_per_expert,
                                *weights_list,
                                transB=transB,
                                gradient_accumulation_fusion=gradient_accumulation_fusion)
      nvtx.range_pop()

      # Reset grad to avoid accumulation
      permuted_inputs.grad = torch.zeros_like(permuted_inputs)
      for weights in weights_list:
        weights.grad = torch.zeros_like(weights)

      # Backward
      nvtx.range_push("grouped gemm op backward")
      gemm_output.backward(gemm_output.detach(), retain_graph=True)
      nvtx.range_pop()

    weights_grad = []
    for weights in weights_list:
        weights_grad.append(weights.grad.unsqueeze(0))
    weights_grad = torch.cat(weights_grad, dim=0)

    # Ref calculation
    gemm_output_ref_list = []
    weight_grad_ref_list = []
    activation_grad_ref_list = []

    rows_idx_for_expert = torch.cumsum(tokens_per_expert, dim=0)
    rows_idx_for_expert = torch.cat((torch.tensor([0]), rows_idx_for_expert[:-1]))

    for expert_id in range(num_experts):
      row_start_id = rows_idx_for_expert[expert_id]
      row_end_id = row_start_id + tokens_per_expert[expert_id]

      activations_expert = permuted_inputs[row_start_id:row_end_id].detach()
      weights_expert = weights_list[expert_id].detach()
      if transB:
        weights_expert = weights_expert.T
      activations_expert.requires_grad_(True)
      weights_expert.requires_grad_(True)
      
      gemm_output_ref = torch.matmul(activations_expert, weights_expert)
      gemm_output_ref.backward(gemm_output_ref.detach())

      gemm_output_ref_list.append(gemm_output_ref)
      weights_expert_grad = weights_expert.grad
      if gradient_accumulation_fusion:
        weights_expert_grad = weights_expert_grad * execution_times

      if transB:
        weights_expert_grad = weights_expert_grad.T
      weight_grad_ref_list.append(weights_expert_grad.unsqueeze(0))
      activation_grad_ref_list.append(activations_expert.grad)

    gemm_output_ref = torch.cat(gemm_output_ref_list, dim=0)
    weight_grad_ref = torch.cat(weight_grad_ref_list, dim=0)
    activation_grad_ref = torch.cat(activation_grad_ref_list, dim=0)

    if PRINT:
      print(expert_for_rows)
      # Forward
      print("    gemm output: ", gemm_output)
      print("ref gemm output: ", gemm_output_ref)
      # Backward
      print("    act grad: ", permuted_inputs.grad)
      print("ref act grad: ", activation_grad_ref)
      print("    weight grad: ", weights_grad)
      print("ref weight grad: ", weight_grad_ref)

    # Result check
    if not permuted_inputs.numel():
      print("grouped gemm empty input activation test passed.")
      return

    result = gemm_output.float().cpu().detach().numpy().flatten()
    result_ref = gemm_output_ref.float().cpu().detach().numpy().flatten()
    max_abs_error = abs(result - result_ref).max()
    print(f"grouped gemm forward max error: \t\t\t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_groupedgemm failed!"

    result = permuted_inputs.grad.float().cpu().detach().numpy()
    result_ref = activation_grad_ref.float().cpu().detach().numpy()
    max_abs_error = abs(result - result_ref).max()
    print(f"grouped gemm backward activation.grad max error: \t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_groupedgemm failed!"

    result = weights_grad.float().cpu().detach().numpy().flatten()
    result_ref = weight_grad_ref.float().cpu().detach().numpy().flatten()
    max_abs_error = abs(result - result_ref).max()
    print(f"grouped gemm backward weight.grad max error: \t\t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_groupedgemm failed!"

    # Benchmark
    if BENCHMARK:
      print(f"-------- Benchmark --------")
      t = triton.testing.do_bench(lambda: groupedgemm(permuted_inputs,   \
                                                      tokens_per_expert, \
                                                      *weights_list,     \
                                                      transB=transB,     \
                                                      gradient_accumulation_fusion=gradient_accumulation_fusion))
      print(f"grouped gemm fwd: {t:.3f} ms")

      bwd_input = torch.rand_like(gemm_output)
      t = triton.testing.do_bench(lambda: gemm_output.backward(bwd_input, retain_graph=True))
      print(f"grouped gemm bwd: {t:.3f} ms")

################################################################################################
##
## Test Cases
##
################################################################################################

  def test_moe_permute(self):
    num_rows =        4096 * 2
    max_token_num =   num_rows + 10
    num_cols =        2048
    num_experts =     8
    atol =            1e-5
    execution_times = 10
    PRINT =           False

    print()
    dtype = torch.float32
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    dtype = torch.float16
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    dtype = torch.bfloat16
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    dtype = torch.float8_e5m2
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    dtype = torch.float8_e4m3fn
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    num_rows = 0
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)

  def test_moe_groupedgemm(self):
    # Note that the test directly uses the forward result as the input for the backward process, 
    # so the max error of the backward result is the accumulation of errors from both the forward 
    # and backward processes.

    num_rows =                     4096 * 2
    hidden_size =                  2048
    inter_size =                   hidden_size * 4
    num_experts =                  8
    atol =                         1e-2
    gradient_accumulation_fusion = True
    execution_times =              5
    PRINT =                        False
    BENCHMARK =                    False

    set_grouped_gemm_algo(True)

    print()
    transB = False
    dtype = torch.float32
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol,
                                gradient_accumulation_fusion, execution_times, PRINT, BENCHMARK)
    dtype = torch.float16
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol,
                                gradient_accumulation_fusion, execution_times, PRINT, BENCHMARK)
    dtype = torch.bfloat16
    transB = True
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol,
                                gradient_accumulation_fusion, execution_times, PRINT, BENCHMARK)
    gradient_accumulation_fusion = False
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol,
                                gradient_accumulation_fusion, execution_times, PRINT, BENCHMARK)
    num_rows = 0
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol,
                                gradient_accumulation_fusion, execution_times, PRINT, BENCHMARK)
    transB = False
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol,
                                gradient_accumulation_fusion, execution_times, PRINT, BENCHMARK)

def test_ops():
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromTestCase(TestMoeOps)
  runner = unittest.TextTestRunner()
  runner.run(suite)


if __name__ == '__main__':
  test_ops()
