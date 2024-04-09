# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import torch.cuda.nvtx as nvtx

try:
  from grouped_gemm.ops import permute as permute_topK, unpermute as unpermute_topK
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  import sys
  sys.path.append("..")
  from ops import permute as permute_topK, unpermute as unpermute_topK

def permute(tokens, indices, expand_factor: int = 1, is_fp8=False):
    """Permute the tokens based on the indices.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token2expert indices tensor.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    expand_factor = indices.size(1)

    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, merge_factor: int = 1):
    """Unpermute the sorted tokens based on the indices.
    
    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The sorted indices tensor.
        probs (torch.Tensor, optional): The probabilities tensor. Defaults to None.
        merge_factor (int, optional): The merge factor. Defaults to 1.

    Returns:
        torch.Tensor: The unpermuted tensor.
    """
    merge_factor = probs.size(1)

    if merge_factor > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // merge_factor
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // merge_factor
        assert (
            probs.size(1) == merge_factor
        ), f"probs size {probs.size()} merge_factor {merge_factor}"

    # unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens = permuted_tokens.index_copy(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, merge_factor, permuted_tokens.size(-1))

    if probs is not None:
        dtype = unpermuted_tokens.dtype
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.to(dtype)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens

def permute_topK_test(
    dtype,
    num_token,
    num_expert,
    hidden_size,
    num_topK,
    PRINT,
    BENCHMARK):
    
    print(f"{dtype} token:{num_token} hidden_size:{hidden_size} expert:{num_expert} topK:{num_topK}")

    is_fp8 = dtype in [torch.float8_e5m2, torch.float8_e4m3fn]

    permute_input = torch.rand((num_token, hidden_size), dtype=torch.float32).cuda()
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     permute_input[i][j] = i * 100 + j
    permute_input = permute_input.to(dtype)
    if is_fp8:
        permute_input = permute_input.half()

    permute_input.requires_grad_(True)
    
    if num_token > 0:
        indices = torch.stack([torch.randperm(num_expert)[:num_topK] for _ in range(num_token)])
    else:
        indices = torch.empty((num_token, num_topK))
    indices = indices.to(torch.int32).cuda()

    # probs = torch.tensor([[0.1, 0.9],
    #                       [0.2, 0.8],
    #                       [0.3, 0.7]])
    # 0.5
    # probs = torch.ones_like(indices) / 2
    # rand
    probs = torch.rand(num_token, num_topK).cuda()
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs.requires_grad_(True)

    if PRINT:
        print(permute_input)
        print(indices)
        print(probs)

    ###################################################################################################################################
    #
    # PyTorch
    #
    ###################################################################################################################################
    nvtx.range_push("PyTorch permute forward")
    permute_output, sorted_indices = permute(permute_input, indices, num_topK, is_fp8)
    nvtx.range_pop()

    permute_bwd_input = torch.rand_like(permute_output)
    # for i in range(num_token * num_topK):
    #   for j in range(hidden_size):
    #     permute_bwd_input[i][j] = i * 100 + j

    nvtx.range_push("PyTorch permute backward")
    permute_output.backward(permute_bwd_input, retain_graph=True)
    nvtx.range_pop()

    unpermute_input = permute_output.detach()
    unpermute_input.requires_grad_(True)

    unpermute_output = unpermute(
        unpermute_input, sorted_indices, probs=probs, merge_factor=num_topK)

    if PRINT:
        print("--------------unpermute fwd permute_input--------------")
        print(unpermute_input)
        print("--------------unpermute fwd output--------------")
        print(unpermute_output)

    unpermute_bwd_input = torch.rand_like(unpermute_output)
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     unpermute_bwd_input[i][j] = i * 2000 + j * 20

    if PRINT:
        print("--------------unpermute bwd permute_input--------------")
        print(unpermute_bwd_input)

    unpermute_output.backward(unpermute_bwd_input, retain_graph=True)
    if PRINT:
        print("--------------unpermute bwd output act grad--------------")
        print(permute_output.grad)
        print("--------------unpermute bwd output probs grad--------------")
        print(probs.grad)

    ###################################################################################################################################
    #
    # Mine
    #
    ###################################################################################################################################
    new_permute_input = permute_input.detach().to(dtype)
    new_permute_bwd_input = permute_bwd_input.detach().to(dtype)
    new_unpermute_bwd_input = unpermute_bwd_input.detach().to(dtype)
    new_permute_input.requires_grad_(True)

    new_permute_output, row_id_map = permute_topK(new_permute_input, indices)

    assert torch.allclose(permute_output.float(), new_permute_output.float())

    if PRINT:
        print("--------------row_id_map--------------")
        print(row_id_map)
        print("--------------new_permute_input--------------")
        print(new_permute_input)
        print("--------------new_permute_output--------------")
        print(new_permute_output)

    new_permute_output.backward(new_permute_bwd_input, retain_graph=True)

    if torch.allclose(permute_input.grad.float(), new_permute_input.grad.float()) == False:
        original_inputs = new_permute_input.grad.float().cpu().numpy().flatten()
        original_output = permute_input.grad.float().cpu().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"permute_topK bwd max error (mine vs pytorch): \t\t\t{max_abs_error:.3e} ({dtype})")

        if PRINT:
            print(permute_input.grad)
            print(new_permute_input.grad)

    new_probs = probs.detach()
    new_probs.requires_grad_(True)
    new_unpermute_input = new_permute_output.detach()
    new_unpermute_input.requires_grad_(True)

    new_unpermute_output = unpermute_topK(new_unpermute_input, row_id_map, new_probs)

    if torch.allclose(unpermute_output.float(), new_unpermute_output.float()) == False:
        original_inputs = unpermute_output.float().cpu().detach().numpy().flatten()
        original_output = new_unpermute_output.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK fwd max error (mine vs pytorch): \t\t{max_abs_error:.3e} ({dtype})")

        if PRINT:
            print(unpermute_output)
            print(new_unpermute_output)

    new_unpermute_output.backward(new_unpermute_bwd_input, retain_graph=True)

    if torch.allclose(unpermute_input.grad.float(), new_unpermute_input.grad.float()) == False:
        original_inputs = unpermute_input.grad.float().cpu().detach().numpy().flatten()
        original_output = new_unpermute_input.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK bwd act_grad max error (mine vs pytorch): \t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(new_unpermute_input.grad)
            print(unpermute_input.grad)

    if num_topK > 1 and torch.allclose(new_probs.grad, probs.grad) == False:
        original_inputs = new_probs.grad.float().cpu().detach().numpy().flatten()
        original_output = probs.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK bwd prob_grad max error (mine vs pytorch): \t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(new_probs.grad)
            print(probs.grad)

    if not permute_input.numel():
      print("Empty permute_input activation test passed.")
      return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    def backward_wrapper(act, backward_input, forward_input=[], retain_graph=True, accumulate_grad=False):
        # Set forward_input.grad to None to avoid grad accumulation.
        if accumulate_grad == False:
            for i in forward_input:
                i.grad = None
        return act.backward(backward_input, retain_graph=retain_graph)

    if BENCHMARK:
        print(f"----permute topK----")
        t = perf_test_cuda_kernel(lambda: permute(permute_input, indices, 2))
        print(f"pytorch fwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(lambda: permute_topK(new_permute_input, indices))
        print(f"new     fwd: {t:.3f} ms")

        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(permute_output, permute_bwd_input, forward_input=[permute_input], retain_graph=True, accumulate_grad=False))
        print(f"pytorch bwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(new_permute_output, new_permute_bwd_input, forward_input=[new_permute_input], retain_graph=True, accumulate_grad=False))
        print(f"new     bwd: {t:.3f} ms")

        print(f"----unpermute topK----")
        t = perf_test_cuda_kernel(
            lambda: unpermute(unpermute_input, sorted_indices, probs=probs, merge_factor=num_topK))
        print(f"pytorch fwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: unpermute_topK(new_unpermute_input, row_id_map, new_probs))
        print(f"new     fwd: {t:.3f} ms")

        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(unpermute_output, unpermute_bwd_input, forward_input=[unpermute_input, probs], retain_graph=True, accumulate_grad=False))
        print(f"pytorch bwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(new_unpermute_output, new_unpermute_bwd_input, forward_input=[new_unpermute_input, new_probs], retain_graph=True, accumulate_grad=False))
        print(f"new     bwd: {t:.3f} ms")


def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        # create CUDA event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(50):
            cuda_kernel_fn()

        start_event.record()
        for _ in range(100):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"Elapsed Time: {elapsed_time_ms / 100} ms")
        return elapsed_time_ms / 100
    else:
        print("CUDA is not available.")

def test_permute_topK():

    torch.manual_seed(1)

    num_token = 4096 * 2
    num_expert = 8
    hidden_size = 4096
    num_topK = 1

    Benchmark = False
    print("GPU:", torch.cuda.get_device_name(0))

    dtype = torch.float32
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.float16
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.float8_e5m2
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.float8_e4m3fn
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, 4, hidden_size, 1, False, Benchmark)
    permute_topK_test(dtype, num_token, 5, hidden_size, 2, False, Benchmark)
    permute_topK_test(dtype, num_token, 6, hidden_size, 3, False, Benchmark)
    permute_topK_test(dtype, num_token, 7, hidden_size, 4, False, Benchmark)
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, False, Benchmark)
    num_token = 0
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, False, Benchmark)

if __name__ == "__main__":
    test_permute_topK()