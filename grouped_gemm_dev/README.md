<div align="center">

Grouped GEMM for MoE
===========================
<h4>A PyTorch Toolbox for Grouped GEMM in MoE Model Training</h4>

[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

<div align="left">

- [Steps for Using](#steps-for-using)
- [Sketch](#sketch)
- [Support Matrix](#support-matrix)
- [Ops Usage](#ops-usage)
  - [permute](#permute)
  - [unpermute](#unpermute)
  - [groupedgemm](#groupedgemm)

---

# Steps for Using

```bash
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j8
cd ..

# unit function test
python test_unit_func.py
# pytorch ops test
python test_torch_ops.py
```
# Sketch

<p align="center"><img src=figures/figure_permute.png></p>
<p align="center"><img src=figures/figure_unpermute.png></p>
<p align="center"><img src=figures/figure_groupedgemm.png></p>

# Support Matrix

| GPU Arch   | FP32  | FP16  | BF16  |
| :--------- | :---: | :---: | :---: |
| SM 70      |   Y   |   Y   |   .   |
| SM 75      |   Y   |   Y   |   .   |
| SM 80      |   Y   |   Y   |   Y   |
| SM 86      |   Y   |   Y   |   Y   |
| SM 89      |   Y   |   Y   |   Y   |

# Ops Usage

## permute

> ```py
> moe.ops.permute(
>   unpermuted_inputs: torch.Tensor,
>   expert_for_rows: torch.Tensor,
>   max_token_num=0: int) -> tuple
> ```

The output tuple of `(torch.Tensor, torch.Tensor)` that contains two tensors `permuted_inputs` and `row_id_map`.

* `permuted_inputs` is a view of the original tensor `unpermuted_inputs` with its first dimension permuted according to `expert_for_rows`.
* `row_id_map` is the mapping table for the row indices of the input activations before and after `moe.ops.permute`. 
    &emsp;For example, given an `expert_for_rows` of `[2, 0, 1, 1, 2, 1, 3, 2, 1, 0]`, then it will be permuted to `[0, 0, 1, 1, 1, 1, 2, 2, 2, 3]`.
    &emsp;The original row indices `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` will be changed to `[1, 9, 2, 3, 5, 8, 0, 4, 7, 6]`, which is also the value of `row_id_map`, so that `row_id_map[source_row_id] = dest_row_id`.

### Parameters

* **unpermuted_inputs** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The input activations with each row corresponds to a single expert.

* **expert_for_rows** (torch.Tensor)  
    &emsp;shape = [tokens_num]  
    &emsp;The expert index for each row of activations. The `int32` type is recommended.

* **max_token_num** (int)  
    &emsp;The maximum number of tokens (rows) used for workspace allocation.


## unpermute

> ```py
> moe.ops.unpermute(
>   permuted_inputs: torch.Tensor,
>   row_id_map: torch.Tensor) -> torch.Tensor
> ```

The mirror operator of `moe.ops.permute`.

### Parameters

* **permuted_inputs** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The permuted activations output by `moe.ops.permute`.

* **row_id_map** (torch.Tensor)  
    &emsp;shape = [tokens_num]  
    &emsp;The mapping table for the row indices of the original unpermuted activations before and after `moe.ops.permute`. The second output tensor of `moe.ops.permute`.

### Example

```py
from grouped_gemm import permute

expert_for_rows = torch.tensor([2, 0, 1, 0], dtype=torch.int32, device='cuda')
unpermuted_inputs = torch.tensor([[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3]], dtype=torch.float32, device='cuda')
permuted_inputs, row_id_map = permute(unpermuted_inputs, expert_for_rows)
unpermute_outputs = unpermute(permuted_inputs, row_id_map)

print(row_id_map)
print(unpermuted_inputs)
print(permuted_inputs)
print(unpermute_outputs)

# Output
# tensor([1, 3, 2, 0], device='cuda:0', dtype=torch.int32)
# tensor([[0., 0., 0., 0.],
#         [1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]], device='cuda:0')
# tensor([[1., 1., 1., 1.],
#         [3., 3., 3., 3.],
#         [2., 2., 2., 2.],
#         [0., 0., 0., 0.]], device='cuda:0')
# tensor([[0., 0., 0., 0.],
#         [1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]], device='cuda:0')
```

## groupedgemm
> ```py
> moe.ops.groupedgemm(
>   permuted_inputs: torch.Tensor,
>   weights: torch.Tensor,
>   tokens_per_expert: torch.Tensor,
>   transB=False: bool) -> torch.Tensor
> ```

Matrix product of two tensors `permuted_inputs` and `weights` for each expert.

### Parameters

* **permuted_inputs** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The permuted input activations with each row sorted according to expert id via `moe.ops.permute`.

* **weights** (torch.Tensor)  
    &emsp;shape = [experts_num, hidden_size, inter_size] for `transB = False`  
    &emsp;shape = [experts_num, inter_size, hidden_size] for `transB = True`  
    &emsp;Weight matrices for each expert.

* **tokens_per_expert** (torch.Tensor)  
    &emsp;shape = [num_experts]  
    &emsp;The number of tokens for each expert. The `int32` type is recommended.

* **transB** (bool)  
    &emsp;Whether to transpose `Weights`.
