<div align="center">

Grouped GEMM for MoE
===========================
<h4>A PyTorch Toolbox for Grouped GEMM in MoE Model Training</h4>

[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

<div align="left">

- [Grouped GEMM for MoE](#grouped-gemm-for-moe)
- [Steps for Using](#steps-for-using)
  - [pip install](#pip-install)
  - [Build from Source](#build-from-source)
- [Support Matrix](#support-matrix)
  - [permute \& unpermute](#permute--unpermute)
- [Ops Usage](#ops-usage)
  - [permute](#permute)
    - [Parameters](#parameters)
  - [unpermute](#unpermute)
    - [Parameters](#parameters-1)
    - [Example](#example)

---

# Steps for Using

## pip install
```bash
pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
```

## Build from Source
```bash
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..

# GroupedGEMM ops test
python grouped_gemm/ops_test.py

# topK permute & unpermute ops test
python grouped_gemm/permute_test.py

# sinkhorn kernel test
python grouped_gemm/sinkhorn_test.py
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

# Ops Usage

## permute

> ```py
> grouped_gemm.ops.permute(
>   input_act: torch.Tensor,
>   indices: torch.Tensor,
>   num_out_tokens: int = 0,
>   max_token_num=0: int) -> tuple
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
from grouped_gemm.ops import permute, unpermute

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

