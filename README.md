<div align="center">

Grouped GEMM for MoE
===========================
<h4>A PyTorch Toolbox for Grouped GEMM in MoE Model Training</h4>

[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

<div align="left">

- [Steps for Using](#steps-for-using)
  - [pip install](#pip-install)
  - [Build from Source](#build-from-source)
- [Support Matrix](#support-matrix)
- [Ops Usage](#ops-usage)
  - [permute](#permute)
  - [unpermute](#unpermute)
  - [groupedgemm](#groupedgemm)
- [Unit Tests](#unit-tests)

---

# Steps for Using

## pip install
```bash
pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@jiangs/groupedgemm_shiqing
```

## Build from Source
```bash
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..

# unit function test
python test_unit_func.py
# pytorch ops test
python test_torch_ops.py
# topK permute & unpermute ops test
python test_permuteTopK.py
```

# Support Matrix

## permute & unpermute

| GPU Arch   | FP32  | FP16  | BF16  | FP8   |
| :--------- | :---: | :---: | :---: | :---: |
| SM 70      |   Y   |   Y   |   .   |   Y   |
| SM 75      |   Y   |   Y   |   .   |   Y   |
| SM 80      |   Y   |   Y   |   Y   |   Y   |
| SM 86      |   Y   |   Y   |   Y   |   Y   |
| SM 89      |   Y   |   Y   |   Y   |   Y   |
| SM 90      |   Y   |   Y   |   Y   |   Y   |

## groupedgemm

| GPU Arch   | FP32  | FP16  | BF16  |
| :--------- | :---: | :---: | :---: |
| SM 70      |   Y   |   Y   |   .   |
| SM 75      |   Y   |   Y   |   .   |
| SM 80      |   Y   |   Y   |   Y   |
| SM 86      |   Y   |   Y   |   Y   |
| SM 89      |   Y   |   Y   |   Y   |
| SM 90      |   Y   |   Y   |   Y   |

# Ops Usage

## permute

> ```py
> grouped_gemm.ops.permute(
>   input_act: torch.Tensor,
>   indices: torch.Tensor,
>   num_out_tokens: int = 0,
>   max_token_num: int = 0) -> tuple
> ```

The output tuple of `(torch.Tensor, torch.Tensor)` that contains two tensors `permuted_act` and `row_id_map`.

* `permuted_act` is the permutation of the original tensor `input_act` with its first dimension permuted according to `indices`.
* `row_id_map` is the mapping table for the row indices of the input activations before and after `grouped_gemm.ops.permute`, which is used for the following `unpermute` op.

### Parameters

* **input_act** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The input activations with each row (token) corresponds to topK experts.

* **indices** (torch.Tensor)  
    &emsp;shape = [tokens_num, topK_num]  
    &emsp;The topK expert indices for each row (token) of activations. The `int32` type is recommended.

* **num_out_tokens** (int)  
    &emsp;The number of output tokens (rows) used for token drop feature.

* **max_token_num** (int)  
    &emsp;The maximum number of tokens (rows) used for workspace pre-allocation.

<p align="center"><img src=figures/figure_permute.png></p>

## unpermute

> ```py
> grouped_gemm.ops.unpermute(
>   input_act: torch.Tensor,
>   row_id_map: torch.Tensor,
>   probs) -> torch.Tensor
> ```

The mirror operator of `grouped_gemm.ops.permute`.

### Parameters

* **input_act** (torch.Tensor)  
    &emsp;shape = [tokens_num * topK_num, hidden_size]  
    &emsp;The permuted activations produced by `grouped_gemm.ops.permute`.

* **row_id_map** (torch.Tensor)  
    &emsp;shape = [tokens_num * topK_num]  
    &emsp;The mapping table for the row indices of the activations before and after `grouped_gemm.ops.permute`. The second output tensor of `grouped_gemm.ops.permute`.

* **probs** (torch.Tensor)  
    &emsp;shape = [tokens_num, topK_num]  
    &emsp;Sum weights for same-origin tokens from different experts.

<p align="center"><img src=figures/figure_unpermute.png></p>

### Example

```py
import torch
from grouped_gemm import permute, unpermute

indices = torch.tensor([[1, 2], [0, 1], [0, 2], [1, 2]], dtype=torch.int32, device='cuda')
input_act = torch.tensor([[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3]], dtype=torch.float32, device='cuda')
probs = torch.ones_like(indices, dtype=torch.float32)
permuted_inputs, row_id_map = permute(input_act, indices)
unpermute_outputs = unpermute(permuted_inputs, row_id_map, probs)

print(row_id_map)
print(input_act)
print(permuted_inputs)
print(unpermute_outputs)

# Output
# tensor([2, 0, 1, 4, 5, 3, 6, 7], device='cuda:0', dtype=torch.int32)
# tensor([[0., 0., 0., 0.],
#         [1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]], device='cuda:0')
# tensor([[1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [0., 0., 0., 0.],
#         [1., 1., 1., 1.],
#         [3., 3., 3., 3.],
#         [0., 0., 0., 0.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]], device='cuda:0')
# tensor([[0., 0., 0., 0.],
#         [2., 2., 2., 2.],
#         [4., 4., 4., 4.],
#         [6., 6., 6., 6.]], device='cuda:0')
```

## groupedgemm
> ```py
> grouped_gemm.ops.groupedgemm(
>   permuted_inputs: torch.Tensor,
>   tokens_per_expert: torch.Tensor,
>   weights_list: list,
>   transB: bool = False,
>   gradient_accumulation_fusion: bool = False) -> torch.Tensor
> ```

Grouped matrix product of two tensors activations and weights for each expert.

### Parameters

* **permuted_inputs** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The permuted input activations with each row sorted according to expert id via `grouped_gemm.ops.permute`.

* **tokens_per_expert** (torch.Tensor)  
    &emsp;shape = [num_experts]  
    &emsp;The number of tokens for each expert. The `int32` type is recommended.

* **weights_list** (list)  
    &emsp;experts_num x [hidden_size, inter_size] for `transB = False`  
    &emsp;experts_num x [inter_size, hidden_size] for `transB = True`  
    &emsp;A list of weight tensors for each expert.

* **transB** (bool)  
    &emsp;Whether to transpose weight tensor.

* **gradient_accumulation_fusion** (bool)  
    &emsp;Whether to do gradient accumulation. If this is set, we are assuming that each input weight tensor in `weights_list` has a `main_grad` field.

<p align="center"><img src=figures/figure_groupedgemm.png></p>

# Unit Tests

```py
from grouped_gemm import tests

tests.test_func()
tests.test_ops()
tests.test_permute_topK()
tests.test_sinkhorn()
```