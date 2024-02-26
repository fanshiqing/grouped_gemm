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

<p align="center"><img src=figures/figure1.png></p>

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

The output tuple of `(torch.Tensor, torch.Tensor)` that contains two tensors `permuted_inputs` and `source_row_to_dest_row`.

* `permuted_inputs` is a view of the original tensor `unpermuted_inputs` with its first dimension permuted according to `expert_for_rows`.
* `source_row_to_dest_row` is the mapping table for the row indices of the input activations before and after `moe.ops.permute`. For example, given an `expert_for_rows` of `[3, 1, 0, 4, 2, 0, 2, 1, 3, 1]`, the corresponding `source_row_to_dest_row` will be `[7, 2, 0, 9, 5, 1, 6, 3, 8, 4]`.

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
>   expert_for_rows: torch.Tensor,
>   source_row_to_dest_row: torch.Tensor,
>   max_token_num=0: int) -> torch.Tensor
> ```

The mirror operator of `moe.ops.permute`.

### Parameters

* **permuted_inputs** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The permuted activations output by `moe.ops.permute`.

* **expert_for_rows** (torch.Tensor)  
    &emsp;shape = [tokens_num]  
    &emsp;The expert index for each row of original unpermuted activations. The `int32` type is recommended.

* **source_row_to_dest_row** (torch.Tensor)  
    &emsp;shape = [tokens_num]  
    &emsp;The mapping table for the row indices of the original unpermuted activations before and after `moe.ops.permute`. The second output tensor of `moe.ops.permute`.

* **max_token_num** (int)  
    &emsp;The maximum number of tokens (rows) used for workspace allocation.

### Example

```py
from moe.ops import permute

expert_for_rows = torch.tensor([2, 0, 1, 0], dtype=torch.int32, device='cuda')
unpermuted_inputs = torch.tensor([[0,0], [1,1], [2,2], [3,3]], dtype=torch.float16, device='cuda')
permuted_inputs, source_row_to_dest_row = permute(unpermuted_inputs, expert_for_rows)
unpermute_outputs = unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row)

print(source_row_to_dest_row)
print(unpermuted_inputs)
print(permuted_inputs)
print(unpermute_outputs)

# Output
# tensor([3, 0, 2, 1], device='cuda:0', dtype=torch.int32)
# tensor([[0., 0.],
#         [1., 1.],
#         [2., 2.],
#         [3., 3.]], device='cuda:0', dtype=torch.float16)
# tensor([[1., 1.],
#         [3., 3.],
#         [2., 2.],
#         [0., 0.]], device='cuda:0', dtype=torch.float16)
# tensor([[0., 0.],
#         [1., 1.],
#         [2., 2.],
#         [3., 3.]], device='cuda:0', dtype=torch.float16)
```

## groupedgemm
> ```py
> moe.ops.gmm(
>   permuted_inputs: torch.Tensor,
>   weights: torch.Tensor,
>   tokens_per_expert: torch.Tensor,
>   trans_b=False: bool) -> torch.Tensor
> ```

Matrix product of two tensors `permuted_inputs` and `weights` for each expert.

### Parameters

* **permuted_inputs** (torch.Tensor)  
    &emsp;shape = [tokens_num, hidden_size]  
    &emsp;The permuted input activations with each row sorted according to expert id via `moe.ops.permute`.

* **weights** (torch.Tensor)  
    &emsp;shape = [experts_num, hidden_size, inter_size] for `trans_b = False`
    &emsp;shape = [experts_num, inter_size, hidden_size] for `trans_b = True`
    &emsp;Weight matrices for each expert.

* **tokens_per_expert** (torch.Tensor)  
    &emsp;shape = [num_experts]  
    &emsp;The number of tokens for each expert. The `int32` type is recommended.

* **trans_b** (bool)
    &emsp;Whether to transpose `Weights`.
