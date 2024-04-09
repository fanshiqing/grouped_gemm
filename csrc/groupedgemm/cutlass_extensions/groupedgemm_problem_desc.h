/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <cub/cub.cuh>
#include "cutlass/gemm_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Grouped GEMM Problem Descriptor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#define NUM_EXPERTS 1024

template <typename ElementA,
          typename ElementB,
          typename ElementC>
class GroupedGemmProblemDesc
{
private:
    /* data */
public:
    GroupedGemmProblemDesc(int num_experts);
    // ~GroupedGemmProblemDesc();

    cutlass::gemm::GemmCoord *problem_sizes;
    ElementA **device_ptr_A;
    ElementB **device_ptr_B;
    ElementC **device_ptr_C;
    int64_t *device_lda;
    int64_t *device_ldb;
    int64_t *device_ldc;
};

template <typename ElementA,
          typename ElementB,
          typename ElementC>
GroupedGemmProblemDesc<ElementA,
                       ElementB,
                       ElementC>::GroupedGemmProblemDesc(int num_experts)
{
    void *device_ptr;
    // num_experts * (12 + 3 x 8 + 3 x 8)
    size_t MemInBytes = num_experts * (sizeof(cutlass::gemm::GemmCoord) +
                                       sizeof(ElementA *) +
                                       sizeof(ElementB *) +
                                       sizeof(ElementC *) +
                                       3 * sizeof(int64_t));

    cudaMalloc((void **)&device_ptr, MemInBytes);

    device_ptr_A = reinterpret_cast<ElementA **>(device_ptr);
    device_ptr_B = reinterpret_cast<ElementB **>(device_ptr_A + num_experts);
    device_ptr_C = reinterpret_cast<ElementC **>(device_ptr_B + num_experts);
    device_lda = reinterpret_cast<int64_t *>(device_ptr_C + num_experts);
    device_ldb = device_lda + num_experts;
    device_ldc = device_ldb + num_experts;
    problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord *>(device_ldc + num_experts);
}

// For Variable K
template <typename ElementA,
          typename ElementB,
          typename ElementC,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void setGroupedGemmProblemDesc(
    GroupedGemmProblemDesc<ElementA, ElementB, ElementC> problem_desc,
    int64_t gemm_m,
    int64_t gemm_n,
    int *gemm_k_per_expert,
    ElementA *ptr_A,
    ElementB *ptr_B,
    ElementC *ptr_C,
    bool set_C)
{
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<int64_t, NUM_EXPERTS> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_id = threadIdx.x;

    int64_t gemm_k = gemm_k_per_expert[expert_id];
    problem_desc.problem_sizes[expert_id] = cutlass::gemm::GemmCoord(gemm_m, gemm_n, gemm_k);
    problem_desc.device_lda[expert_id] = LayoutA::packed({gemm_m, gemm_k}).stride(0);
    problem_desc.device_ldb[expert_id] = LayoutB::packed({gemm_k, gemm_n}).stride(0);
    problem_desc.device_ldc[expert_id] = LayoutC::packed({gemm_m, gemm_n}).stride(0);

    int64_t stride_A = gemm_m * gemm_k;
    int64_t stride_B = gemm_k * gemm_n;
    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(stride_A, stride_A);
    BlockScan(temp_storage).ExclusiveSum(stride_B, stride_B);
    problem_desc.device_ptr_A[expert_id] = ptr_A + stride_A;
    problem_desc.device_ptr_B[expert_id] = ptr_B + stride_B;

    if (set_C)
    {
        int64_t stride_C = gemm_m * gemm_n;
        BlockScan(temp_storage).ExclusiveSum(stride_C, stride_C);
        problem_desc.device_ptr_C[expert_id] = ptr_C + stride_C;
    }
}

// For Variable M
template <typename ElementA,
          typename ElementB,
          typename ElementC,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void setGroupedGemmProblemDesc(
    GroupedGemmProblemDesc<ElementA, ElementB, ElementC> problem_desc,
    int *gemm_m_per_expert,
    int64_t gemm_n,
    int64_t gemm_k,
    ElementA *ptr_A,
    ElementC *ptr_C)
{
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<int64_t, NUM_EXPERTS> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_id = threadIdx.x;

    int64_t gemm_m = gemm_m_per_expert[expert_id];
    problem_desc.problem_sizes[expert_id] = cutlass::gemm::GemmCoord(gemm_m, gemm_n, gemm_k);
    problem_desc.device_lda[expert_id] = LayoutA::packed({gemm_m, gemm_k}).stride(0);
    problem_desc.device_ldb[expert_id] = LayoutB::packed({gemm_k, gemm_n}).stride(0);
    problem_desc.device_ldc[expert_id] = LayoutC::packed({gemm_m, gemm_n}).stride(0);

    int64_t stride_A = gemm_m * gemm_k;
    int64_t stride_C = gemm_m * gemm_n;

    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(stride_A, stride_A);
    BlockScan(temp_storage).ExclusiveSum(stride_C, stride_C);

    problem_desc.device_ptr_A[expert_id] = ptr_A + stride_A;
    problem_desc.device_ptr_C[expert_id] = ptr_C + stride_C;
}

// For Variable K
template <typename ElementA,
          typename ElementB,
          typename ElementC,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void setGroupedGemmProblemDescFromDevice(
    GroupedGemmProblemDesc<ElementA, ElementB, ElementC> &problem_desc,
    int num_experts,
    int64_t gemm_m,
    int64_t gemm_n,
    int *gemm_k_per_expert,
    ElementA *ptr_A,
    ElementB *ptr_B,
    ElementC *ptr_C,
    bool set_C,
    cudaStream_t stream)
{
    if (num_experts > NUM_EXPERTS)
    {
        throw std::runtime_error(
            "[Grouped GEMM Runner] The number of experts cannot exceed NUM_EXPERTS!");
    }
    setGroupedGemmProblemDesc<ElementA, ElementB, ElementC,
                              LayoutA, LayoutB, LayoutC><<<1, num_experts, 0, stream>>>(
        problem_desc,
        gemm_m, gemm_n, gemm_k_per_expert,
        ptr_A, ptr_B, ptr_C, set_C);
}

// For Variable M
template <typename ElementA,
          typename ElementB,
          typename ElementC,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void setGroupedGemmProblemDescFromDevice(
    GroupedGemmProblemDesc<ElementA, ElementB, ElementC> &problem_desc,
    int num_experts,
    int *gemm_m_per_expert,
    int64_t gemm_n,
    int64_t gemm_k,
    ElementA *ptr_A,
    ElementC *ptr_C,
    cudaStream_t stream)
{
    if (num_experts > NUM_EXPERTS)
    {
        throw std::runtime_error(
            "[Grouped GEMM Runner] The number of experts cannot exceed NUM_EXPERTS!");
    }
    setGroupedGemmProblemDesc<ElementA,
                              ElementB,
                              ElementC,
                              LayoutA,
                              LayoutB,
                              LayoutC><<<1, num_experts, 0, stream>>>(
                                    problem_desc,
                                    gemm_m_per_expert, gemm_n, gemm_k,
                                    ptr_A, ptr_C);
}
