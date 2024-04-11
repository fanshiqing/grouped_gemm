
import torch
import triton
# pip install git+https://github.com/tgale96/grouped_gemm@main
from grouped_gemm.ops import gmm

import sys
sys.path.append("..")
from ops import permute, unpermute, groupedgemm, set_grouped_gemm_algo



dtype = torch.bfloat16
token_num = 4096 * 2
expert_num = 8
n = 14336
k = 4096
batch_sizes = (torch.ones([expert_num]) * (token_num // expert_num)).to(torch.int64)
BENCHMARK = True
warmups = 100

a = torch.empty((token_num, k), dtype=dtype).cuda()
for i in range(a.size(0)):
    a[i] = i
b = torch.ones((expert_num, k, n), dtype=dtype).cuda()


################################################################################################
##
## nv
##
################################################################################################

nv_a = a.detach()
nv_b = b.detach()
nv_a.requires_grad_(True)
nv_b.requires_grad_(True)

weights_list = []
for i in range(batch_sizes.size(0)):
    weights_list.append(nv_b[i].detach())
    weights_list[i].requires_grad_(True)

batch_sizes_nv = batch_sizes.cuda().to(torch.int32)

# grouped gemm
nv_c = groupedgemm(nv_a,
                   batch_sizes_nv,
                   *weights_list,
                   transB=False,
                   gradient_accumulation_fusion=False)
# print(nv_c)

# Benchmark
if BENCHMARK:
    # warmups
    for _ in range(warmups):
        groupedgemm(nv_a,   \
                    batch_sizes_nv, \
                    *weights_list,     \
                    transB=False,     \
                    gradient_accumulation_fusion=False)
    
    print(f"-------- Benchmark --------")
    t = triton.testing.do_bench(lambda: groupedgemm(nv_a,   \
                                                    batch_sizes_nv, \
                                                    *weights_list,     \
                                                    transB=False,     \
                                                    gradient_accumulation_fusion=False))
    print(f"nv gemm fwd: {t:.3f} ms")

    bwd_input = torch.rand_like(nv_c)
    t = triton.testing.do_bench(lambda: nv_c.backward(bwd_input, retain_graph=True))
    print(f"nv gemm bwd: {t:.3f} ms")


################################################################################################
##
## megablocks
##
################################################################################################
mega_a = a.detach()
mega_b = b.detach()
mega_a.requires_grad_(True)
mega_b.requires_grad_(True)

# grouped gemm
mega_c = gmm(mega_a, mega_b, batch_sizes, trans_b=False)
# print(mega_c)

# Benchmark
if BENCHMARK:
    # warmups
    for _ in range(warmups):
        gmm(mega_a, mega_b, batch_sizes, trans_b=False)

    print(f"-------- Benchmark --------")
    t = triton.testing.do_bench(
        lambda: gmm(mega_a, mega_b, batch_sizes, trans_b=False))
    print(f"megablocks fwd: {t:.3f} ms")

    bwd_input = torch.rand_like(mega_c)
    t = triton.testing.do_bench(lambda: mega_c.backward(bwd_input, retain_graph=True))
    print(f"megablocks bwd: {t:.3f} ms")


################################################################################################
##
## pytorch
##
################################################################################################
def pytorch_gmm(a_list, b_list, batch_sizes):
    c_list = []
    start_id = 0
    for i in range(batch_sizes.size(0)):
        e = batch_sizes[i]
        c_list.append(torch.matmul(a_list[i], b_list[i]))
        start_id += e
    return c_list

def pytorch_gmm_bwd(c_list, bwd_input_list):
    for i in range(len(c_list)):
        c_list[i].backward(bwd_input_list[i], retain_graph=True)

    
a_list = []
b_list = []
start_id = 0
for i in range(batch_sizes.size(0)):
    e = batch_sizes[i]
    pytorch_a = (a.detach())[start_id:start_id + e].clone()
    pytorch_b = (b.detach())[i].clone()
    pytorch_a.requires_grad_(True)
    pytorch_b.requires_grad_(True)
    a_list.append(pytorch_a)
    b_list.append(pytorch_b)
    start_id += e

# grouped gemm
c_list = pytorch_gmm(a_list, b_list, batch_sizes)
# print(c_list)

bwd_input_list = []
for c in c_list:
    bwd_input_list.append(torch.rand_like(c))

# Benchmark
if BENCHMARK:
    # warmups
    for _ in range(warmups):
        pytorch_gmm(a_list, b_list, batch_sizes)

    print(f"-------- Benchmark --------")
    t = triton.testing.do_bench(
        lambda: pytorch_gmm(a_list, b_list, batch_sizes))
    print(f"pytorch fwd: {t:.3f} ms")

    t = triton.testing.do_bench(lambda: pytorch_gmm_bwd(c_list, bwd_input_list))
    print(f"pytorch bwd: {t:.3f} ms")