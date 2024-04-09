# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest

from absl.testing import parameterized

try:
  from grouped_gemm.ops import sinkhorn
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  import sys
  sys.path.append("..")
  from ops import sinkhorn

import torch

# Notes: the source of this func implementation:
#     https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/core/transformer/switch_mlp.py#L17-L31
def baseline_sinkhorn(cost, tol=0.0001):
    "Sinkhorn based MoE routing function"
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    iter_count = 0
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
        iter_count = iter_count + 1
    return d1 * cost * d0.unsqueeze(1)

_TEST_PROBLEMS = (
    # (128,  2, 0.1),
    # (256,  2, 0.1),
    # (1024, 4, 0.01),
    # (2048, 8, 0.0001),
    (4096, 8, 0.0001),
    (8192, 8, 0.0001),
    (8192, 16, 0.0001),
)

@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    def test_sinkhorn_kernel(self, m, k, tol=0.0001):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.manual_seed(0)
        cost = torch.rand(m, k, device='cuda', dtype=torch.float32)

        start.record()
        expected_out = baseline_sinkhorn(cost, tol)
        end.record()
        torch.cuda.synchronize()
        baseline_time = start.elapsed_time(end)

        start.record()
        out = sinkhorn(cost, tol)
        end.record()
        torch.cuda.synchronize()
        kernel_time = start.elapsed_time(end)
        print("===================================")
        print("Problem size: [%d]x[%d], kernel speedup: %fX" % (m, k, baseline_time/kernel_time))
        print("===================================")
        self.assertTrue(torch.allclose(out, expected_out))

def test_sinkhorn():
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromTestCase(OpsTest)
  runner = unittest.TextTestRunner()
  runner.run(suite)

if __name__ == '__main__':
    test_sinkhorn()
