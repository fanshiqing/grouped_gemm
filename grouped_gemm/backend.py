# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import grouped_gemm_backend as backend


def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

def gmm(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm(a, b, c, batch_sizes, trans_a, trans_b)
    return c

def sinkhorn(cost, tol=0.0001):
    return backend.sinkhorn(cost, tol)

def permute(unpermuted_inputs, expert_for_rows, worksapce, max_token_num):
    return backend.permute(unpermuted_inputs, expert_for_rows, worksapce, max_token_num)

def unpermute(permuted_inputs, row_id_map):
    return backend.unpermute(permuted_inputs, row_id_map)
