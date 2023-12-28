from grouped_gemm import backend
import torch


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)

def sinkhorn_kernel(cost, tol=0.0001):
    return backend.sinkhorn(cost, tol)

