/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/util/device_memory.h"

#include "groupedgemm_traits.h"
#include "groupedgemm_problem_desc.h"

#include "extensions.h"

enum class CutlassTileConfig {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    // CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape128x32x64,

    // CUTLASS Grouped GEMM config
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x32_WarpShape64x64x32,
    CtaShape128x64x32_WarpShape64x32x32,
    CtaShape64x128x32_WarpShape32x64x32
};

struct CutlassGemmConfig {
    CutlassTileConfig tile_config    = CutlassTileConfig::ChooseWithHeuristic;
    int               stages         = -1;
};

template<typename T,         /* Data Type of Input and Output Activations */
         typename WeightType /* Weight Data Type */>
class MoeGemmRunner {
public:
    MoeGemmRunner()
    {
        int device{-1};
        cudaGetDevice(&device);
        sm_ = getSMVersion();
        cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device);
    }

    void moe_gemm(T*           A,
                  WeightType** B_list,
                  T*           C,
                  int*         gemm_m_per_expert,
                  int64_t      gemm_n,
                  int64_t      gemm_k,
                  int          num_tokens,
                  int          num_experts,
                  bool         transB,
                  cudaStream_t stream);

    template<typename AccumGradType /* Data Type of Accumulated Gradient */>
    void moe_gemm_backward(T*              A,
                           WeightType*     B,
                           T*              C,
                           AccumGradType** weight_grad_list,
                           int64_t         gemm_m,
                           int64_t         gemm_n,
                           int*            gemm_k_per_expert,
                           int             num_tokens,
                           int             num_experts,
                           bool            transC,
                           cudaStream_t    stream);

private:
    template<bool TransB /* Whether to transpose weights */>
    void dispatch_to_arch(T*                A,
                          WeightType**      B_list,
                          T*                C,
                          int*              gemm_m_per_expert,
                          int64_t           gemm_n,
                          int64_t           gemm_k,
                          int               num_experts,
                          CutlassGemmConfig gemm_config,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr);

    template<bool TransB /* Whether to transpose weights */>
    void run_gemm(T*           A,
                  WeightType** B_list,
                  T*           C,
                  int*         gemm_m_per_expert,
                  int64_t      gemm_n,
                  int64_t      gemm_k,
                  int          num_tokens,
                  int          num_experts,
                  cudaStream_t stream);

    template<typename AccumGradType /* Data Type of Accumulated Gradient */,
             bool TransC /* Whether to transpose outputs */>
    void dispatch_to_arch_backward(T*                A,
                                   WeightType*       B,
                                   T*                C,
                                   AccumGradType**   weight_grad_list,
                                   int64_t           gemm_m,
                                   int64_t           gemm_n,
                                   int*              gemm_k_per_expert,
                                   int               num_experts,
                                   CutlassGemmConfig gemm_config,
                                   cudaStream_t      stream,
                                   int*              occupancy = nullptr);

    template<typename AccumGradType /* Data Type of Accumulated Gradient */,
             bool TransC /* Whether to transpose outputs */>
    void run_gemm_backward(T*              A,
                           WeightType*     B,
                           T*              C,
                           AccumGradType** weight_grad_list,
                           int64_t         gemm_m,
                           int64_t         gemm_n,
                           int*            gemm_k_per_expert,
                           int             num_tokens,
                           int             num_experts,
                           cudaStream_t    stream);

private:
    int sm_;
    int multi_processor_count_;
};
