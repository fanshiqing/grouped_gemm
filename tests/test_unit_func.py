# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import unittest
import sys
import time
import torch.cuda.nvtx as nvtx

try:
  import grouped_gemm
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  torch.classes.load_library("../csrc/build/libmoe_unit_ops.so")


def random_cuda_tensor(shape, dtype, mean=0, std=1):
    # https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html
    # torch.Tensor.normal_
    # Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
    return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

def basic_moe_fc(activations, expert_for_row, weights, biases):
  if weights.dtype != torch.bfloat16 and weights.dtype != torch.float16 and weights.dtype != torch.float32:
      raise ValueError("Invalid data type for weights")

  res = torch.zeros(size=[activations.shape[0], weights.shape[-1]], dtype=activations.dtype, device='cuda')
  for row in range(activations.shape[0]):
      row_expert = expert_for_row[row]

      torch.matmul(activations[row], weights[row_expert], out=res[row : row + 1, :])
      res[row] += biases[row_expert]

  return res

def basic_moe_fc_backward(activations, expert_for_row, weights):
  if weights.dtype != torch.bfloat16 and weights.dtype != torch.float16 and weights.dtype != torch.float32:
      raise ValueError("Invalid data type for weights")

  rows_per_expert = torch.bincount(expert_for_row)
  # print("rows_per_expert = ", rows_per_expert)
  expert_num = rows_per_expert.shape[0]

  rows_idx_for_expert = torch.cumsum(rows_per_expert, dim=0)
  rows_idx_for_expert = torch.cat((torch.tensor([0]).cuda(), rows_idx_for_expert[:-1]))
  # print("rows_idx_for_expert = ", rows_idx_for_expert)

  res = torch.zeros(size=[expert_num, activations.shape[1], weights.shape[1]], dtype=activations.dtype, device='cuda')
  # print("activations.shape = ", activations.shape)
  # print("weights.shape = ", weights.shape)
  # print("res.shape = ", res.shape)

  for expert_id in range(expert_num):
      row_start_id = rows_idx_for_expert[expert_id]
      row_end_id = row_start_id + rows_per_expert[expert_id]

      activations_expert = activations[row_start_id:row_end_id, :].T
      weights_expert = weights[row_start_id:row_end_id, :]

      # print("expert: {}, rows_num: {}, rows from {} to {}".format(expert_id, rows_per_expert[expert_id], row_start_id, row_end_id))
      # print("activations[{}].shape = {}, weights[{}].shape = {}".format(expert_id, activations_expert.shape, expert_id, weights_expert.shape))
      # print("activations_expert = ", activations_expert)
      # print("weights_expert = ", weights_expert)

      torch.matmul(activations_expert, weights_expert, out=res[expert_id])

  return res

class TestMoe(unittest.TestCase):

  def setUp(self) -> None:
    torch.manual_seed(734876213)
    self.moe_group_gemm_op = torch.ops.moe_unit_ops.moe_group_gemm_op
    self.moe_group_gemm_backward_op = torch.ops.moe_unit_ops.moe_group_gemm_backward_op
    self.moe_permute_op = torch.ops.moe_unit_ops.moe_permute_topK_op
    self.moe_recover_op = torch.ops.moe_unit_ops.moe_recover_topK_op

  def run_ref_moe_shao(self, input_dict, backward: bool = False):
    if backward:
      # backward          
      moe_fc_1_result = basic_moe_fc_backward(input_dict["permuted_inputs"], input_dict["expert_for_rows"], 
                                    input_dict["fc1_expert_weights_for_ft"])
    else:
      # forward
      moe_fc_1_result = basic_moe_fc(input_dict["input_activations"], input_dict["expert_for_rows"], 
                                    input_dict["fc1_expert_weights_for_ft"], input_dict["fc1_expert_biases"])
    
    return moe_fc_1_result

  def moe_permute_helper(self,
                         num_rows,
                         num_cols,
                         num_experts,
                         dtype,
                         warmup_times,
                         execution_times,
                         atol,
                         PRINT):    
    if PRINT:
      print("\n----------------------------------------- test_moe_permute -----------------------------------------")

    expert_for_rows = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda().unsqueeze(-1)
    probs = torch.ones_like(expert_for_rows, dtype=torch.float32)

    if PRINT:
      print("expert_for_rows: {}".format(expert_for_rows))

    # unpermuted_inputs = torch.rand(size=(num_rows, num_cols), dtype=torch.float32).type(dtype).cuda()
    # unpermuted_inputs = torch.randint(size=(num_rows, num_cols), low=0, high=400, dtype=torch.int32).type(dtype).cuda()
    unpermuted_inputs = torch.empty(size=(num_rows, num_cols), dtype=torch.float32)
    for i in range(num_rows):
        unpermuted_inputs[i] = i % 300
    unpermuted_inputs = unpermuted_inputs.type(dtype).cuda()
    if PRINT:
      print("unpermuted_inputs: {}".format(unpermuted_inputs))

    original_inputs = unpermuted_inputs.detach()

    for _ in range(warmup_times):
      permuted_inputs, row_id_map, _ = self.moe_permute_op(unpermuted_inputs, expert_for_rows, [], num_rows)

    nvtx.range_push("permute test")
    nvtx.range_push("permute op")
    start_time = time.perf_counter()
    for _ in range(execution_times):
      permuted_inputs, row_id_map, _ = self.moe_permute_op(unpermuted_inputs, expert_for_rows, [], num_rows)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / execution_times * 1000
    nvtx.range_pop()

    if PRINT:
      print("permuted_inputs: {}".format(permuted_inputs))
      print("row_id_map: {}".format(row_id_map))
      print("elapsed_time: {} ms".format(elapsed_time))

    nvtx.range_push("unpermute op")
    start_time = time.perf_counter()
    for _ in range(execution_times):
      original_output = self.moe_recover_op(permuted_inputs, row_id_map, probs, num_rows, 1)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / execution_times * 1000
    nvtx.range_pop()
    nvtx.range_pop()

    if PRINT:
      print("original_inputs: {}".format(original_inputs))
      print("original_output: {}".format(original_output))
      print("elapsed_time: {} ms".format(elapsed_time))

    # Result check
    original_inputs = original_inputs.float().cpu().numpy().flatten()
    original_output = original_output.float().cpu().numpy().flatten()
    max_abs_error = abs(original_inputs - original_output).max()
    print("Permute max error between ref1 & ref2: ", max_abs_error)
    assert (max_abs_error < atol), "test_moe_permute failed!"


  def grouped_gemm_helper(self,
                          num_rows,
                          hidden_size,
                          inter_size,
                          num_experts,
                          dtype,
                          atol,
                          RAND_INPUT_ACT,
                          RAND_INPUT_WEIGHT,
                          PRINT):

    rand_mean = 0
    rand_std = 0.02

    if PRINT:
      print("\n----------------------------------------- test_grouped_gemm -----------------------------------------")

    inputs = dict()
    inputs["expert_for_rows"] = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda()
    if PRINT:
      print("expert_for_rows = {}".format(inputs["expert_for_rows"]))

    if RAND_INPUT_ACT:
      inputs["input_activations"] = random_cuda_tensor([num_rows, hidden_size], dtype, mean=rand_mean, std=rand_std)
    else:
      inputs["input_activations"] = torch.empty(size=(num_rows, hidden_size), dtype=dtype, device="cuda")
      for i in range(num_rows):
          inputs["input_activations"][i] = i
    inputs["input_activations_ref"] = inputs["input_activations"].clone()
    
    if PRINT:
      print("unpermuted_inputs: {}".format(inputs["input_activations"]))

    weights = dict()
    if RAND_INPUT_WEIGHT:
      weights["fc1_expert_weights_for_ft"] = random_cuda_tensor([num_experts, hidden_size, inter_size], dtype, mean=rand_mean, std=rand_std)    
    else:
      weights["fc1_expert_weights_for_ft"] = torch.empty((num_experts, hidden_size, inter_size), dtype=dtype, device="cuda")
      for i in range(num_experts):
          weights["fc1_expert_weights_for_ft"][i] = i

    # weights["fc1_expert_biases"] = random_cuda_tensor([num_experts, inter_size], dtype, mean=rand_mean, std=rand_std)
    weights["fc1_expert_biases"] = torch.zeros([num_experts, inter_size], dtype=dtype, device="cuda")
    # weights["fc1_expert_biases"] = torch.ones([num_experts, inter_size], dtype=dtype, device="cuda")

    nvtx.range_push("grouped gemm fwd test")
    nvtx.range_push("permute op")
    permuted_inputs, row_id_map, _ = self.moe_permute_op(inputs["input_activations"], inputs["expert_for_rows"].unsqueeze(-1), [], num_rows)
    nvtx.range_pop()

    if PRINT:
      print("permuted_inputs: {}".format(permuted_inputs))
      print("weights: {}".format(weights["fc1_expert_weights_for_ft"]))
      print("row_id_map: {}".format(row_id_map))

    input_dict = inputs
    input_dict.update(weights)
    if PRINT:
      print(input_dict.keys())

    ref_gemm1_output = self.run_ref_moe_shao(input_dict)

    rows_per_expert = torch.bincount(inputs["expert_for_rows"], minlength=num_experts)
    # Cast from int64 to int32 to meet the kernel's requirement.
    rows_per_expert = rows_per_expert.to(torch.int32).cpu()

    weights_list = []
    for i in range(num_experts):
      weights_list.append(input_dict["fc1_expert_weights_for_ft"][i])

    nvtx.range_push("forward cutlass grouped gemm")
    gemm1_output = self.moe_group_gemm_op(
        permuted_inputs,
        weights_list,
        rows_per_expert,
        False
    )
    nvtx.range_pop()

    probs = torch.ones_like(inputs["expert_for_rows"].unsqueeze(-1), dtype=torch.float32)
    nvtx.range_push("unpermute op")
    original_output = self.moe_recover_op(gemm1_output, row_id_map, probs, num_rows, 1)
    nvtx.range_pop()
    nvtx.range_pop()

    # Result check
    moe_gemm_1_result_np = ref_gemm1_output.cpu().float().numpy()
    gemm1_output_np = original_output.cpu().float().numpy()

    if PRINT:
      fp = sys.stdout
      print_num = 10
      for i in range(gemm1_output_np.shape[0]):
        print(i, ": ", end='')
        for j in range(print_num):
          print(moe_gemm_1_result_np[i,j], " ", end='')
        print()

        print(i, ": ", end='')
        for j in range(print_num):
          print(gemm1_output_np[i,j], " ", end='')
        print()
      fp.flush()

    moe_gemm_1_result_np = moe_gemm_1_result_np.flatten()
    gemm1_output_np = gemm1_output_np.flatten()
    max_abs_error = abs(moe_gemm_1_result_np - gemm1_output_np).max()
    print("Grouped GEMM forward max error between ref1 & ref2: ", max_abs_error)
    assert (max_abs_error < atol), "test_grouped_gemm failed!"

  def grouped_gemm_backward_helper(self,
                                   num_rows,
                                   hidden_size,
                                   inter_size,
                                   num_experts,
                                   dtype,
                                   atol,
                                   RAND_INPUT_ACT,
                                   RAND_INPUT_WEIGHT,
                                   PRINT):

    rand_mean = 0
    rand_std = 0.02

    if PRINT:
      print("\n----------------------------------------- test_grouped_gemm_backward -----------------------------------------")

    inputs = dict()
    inputs["expert_for_rows"] = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda()
    if PRINT:
      print("expert_for_rows = {}".format(inputs["expert_for_rows"]))

    if RAND_INPUT_ACT:
      inputs["input_activations"] = random_cuda_tensor([num_rows, hidden_size], dtype, mean=rand_mean, std=rand_std)
    else:
      inputs["input_activations"] = torch.empty(size=(num_rows, hidden_size), dtype=dtype, device="cuda")
      for i in range(num_rows):
          inputs["input_activations"][i] = i
    inputs["input_activations_ref"] = inputs["input_activations"].clone()
    
    if PRINT:
      print("unpermuted_inputs: {}".format(inputs["input_activations"]))


    weights = dict()
    if RAND_INPUT_WEIGHT:
      weights["fc1_expert_weights_for_ft"] = random_cuda_tensor([num_rows, inter_size], dtype, mean=rand_mean, std=rand_std)
    else:
      weights["fc1_expert_weights_for_ft"] = torch.empty(size=(num_rows, inter_size), dtype=dtype, device="cuda")
      for i in range(num_rows):
        for j in range(inter_size):
          weights["fc1_expert_weights_for_ft"][i, j] = i

    if PRINT:
      print("fc1_expert_weights_for_ft = {}".format(weights["fc1_expert_weights_for_ft"]))

    nvtx.range_push("grouped gemm bwd test")
    nvtx.range_push("permute op")
    # Permutation on activations based on expert id
    inputs["permuted_inputs"], row_id_map, _ = self.moe_permute_op(inputs["input_activations"], inputs["expert_for_rows"].unsqueeze(-1), [], num_rows)
    nvtx.range_pop()

    if PRINT:
      print("permuted_inputs: {}".format(inputs["permuted_inputs"]))
      print("weights: {}".format(weights["fc1_expert_weights_for_ft"]))
      print("row_id_map: {}".format(row_id_map))

    input_dict = inputs
    input_dict.update(weights)
    if PRINT:
      print(input_dict.keys())

    ref_gemm1_output = self.run_ref_moe_shao(input_dict, backward=True)

    rows_per_expert = torch.bincount(inputs["expert_for_rows"], minlength=num_experts)
    # Cast from int64 to int32 to meet the kernel's requirement.
    rows_per_expert = rows_per_expert.to(torch.int32).cpu()

    nvtx.range_push("backward cutlass grouped gemm")
    # Grouped GEMM for the case of fixed M, N and variable K
    gemm1_output = self.moe_group_gemm_backward_op(
        inputs["permuted_inputs"],
        input_dict["fc1_expert_weights_for_ft"],
        rows_per_expert,
        False,
        [])
    nvtx.range_pop()
    nvtx.range_pop()

    # Result check
    moe_gemm_1_result_np = ref_gemm1_output.cpu().float().numpy()
    gemm1_output_np = gemm1_output.cpu().float().numpy()

    if PRINT:
      fp = sys.stdout
      print_num = 5
      for expert_id in range(num_experts):
        print("expert: ", expert_id, "-----------------------------------------------")
        for i in range(moe_gemm_1_result_np.shape[1]):
          print(i, ": ", end='')
          for j in range(print_num):
            print(moe_gemm_1_result_np[expert_id, i, j], " ", end='')
          print()

          print(i, ": ", end='')
          for j in range(print_num):
            print(gemm1_output_np[expert_id, i, j], " ", end='')
          print()
      fp.flush()

    moe_gemm_1_result_np = moe_gemm_1_result_np.flatten()
    gemm1_output_np = gemm1_output_np.flatten()
    max_abs_error = abs(moe_gemm_1_result_np - gemm1_output_np).max()
    print("Grouped GEMM backward max error between ref1 & ref2: ", max_abs_error)
    assert (max_abs_error < atol), "test_grouped_gemm_backward failed!"

################################################################################################
##
## Test Cases
##
################################################################################################

  def test_moe_permute(self):
    PRINT = False

    num_rows = 4096 * 2
    num_cols = 2048
    num_experts = 8

    warmup_times = 0
    execution_times = 1
    atol = 1e-4

    print()
    dtype = torch.float32
    self.moe_permute_helper(num_rows, num_cols, num_experts, dtype, warmup_times, execution_times, atol, PRINT)
    dtype = torch.float16
    self.moe_permute_helper(num_rows, num_cols, num_experts, dtype, warmup_times, execution_times, atol, PRINT)
    dtype = torch.bfloat16
    self.moe_permute_helper(num_rows, num_cols, num_experts, dtype, warmup_times, execution_times, atol, PRINT)
    dtype = torch.float8_e5m2
    self.moe_permute_helper(num_rows, num_cols, num_experts, dtype, warmup_times, execution_times, atol, PRINT)
    dtype = torch.float8_e4m3fn 
    self.moe_permute_helper(num_rows, num_cols, num_experts, dtype, warmup_times, execution_times, atol, PRINT)

  def test_grouped_gemm(self):
    RAND_INPUT_ACT = True
    RAND_INPUT_WEIGHT = True
    PRINT = False

    num_rows = 4096 * 2
    hidden_size = 2048
    inter_size = hidden_size * 4
    num_experts = 8
    
    atol = 1e-3

    print()
    dtype = torch.float32
    self.grouped_gemm_helper(num_rows, hidden_size, inter_size, num_experts, dtype, atol, RAND_INPUT_ACT, RAND_INPUT_WEIGHT, PRINT)
    dtype = torch.float16
    self.grouped_gemm_helper(num_rows, hidden_size, inter_size, num_experts, dtype, atol, RAND_INPUT_ACT, RAND_INPUT_WEIGHT, PRINT)
    dtype = torch.bfloat16
    self.grouped_gemm_helper(num_rows, hidden_size, inter_size, num_experts, dtype, atol, RAND_INPUT_ACT, RAND_INPUT_WEIGHT, PRINT)

  def test_grouped_gemm_backward(self):
    RAND_INPUT_ACT = True
    RAND_INPUT_WEIGHT = True
    PRINT = False

    num_rows = 4096 * 2
    hidden_size = 2048
    inter_size = hidden_size * 4
    num_experts = 8
    
    atol = 1e-3

    print()
    dtype = torch.float32
    self.grouped_gemm_backward_helper(num_rows, hidden_size, inter_size, num_experts, dtype, atol, RAND_INPUT_ACT, RAND_INPUT_WEIGHT, PRINT)
    dtype = torch.float16
    self.grouped_gemm_backward_helper(num_rows, hidden_size, inter_size, num_experts, dtype, atol, RAND_INPUT_ACT, RAND_INPUT_WEIGHT, PRINT)
    dtype = torch.bfloat16
    self.grouped_gemm_backward_helper(num_rows, hidden_size, inter_size, num_experts, dtype, atol, RAND_INPUT_ACT, RAND_INPUT_WEIGHT, PRINT)


def test_func():
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromTestCase(TestMoe)
  runner = unittest.TextTestRunner()
  runner.run(suite)


if __name__ == '__main__':
  test_func()