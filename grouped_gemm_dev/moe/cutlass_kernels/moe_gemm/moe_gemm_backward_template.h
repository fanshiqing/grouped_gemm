/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/util/device_memory.h"

#include "default_fpA_intB_traits.h"
#include "compute_occupancy.h"

#include "moe/cutlass_kernels/cutlass_heuristic.h"
#include "moe/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "moe/cutlass_kernels/moe_gemm/grouped_gemm_problem_desc.h"

#include <nvtx3/nvToolsExt.h>

namespace groupedgemmformoe {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Kernel Launcher
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T,
         typename WeightType,
         bool     TransC,
         typename arch,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
void generic_moe_gemm_backward_kernelLauncher(
    T*           A,
    WeightType*  B,
    T*           C,
    int64_t      gemm_m,
    int64_t      gemm_n,
    int*         gemm_k_per_expert,
    int          num_experts,
    cudaStream_t stream,
    int*         kernel_occupancy = nullptr)
{
    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t,
                                                ElementType_>::type;
#else
    using ElementType       = ElementType_;
#endif

    using CutlassWeightType_ = typename cutlass::platform::
        conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t, WeightType>::type;
#ifdef ENABLE_BF16
    using CutlassWeightType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t,
                                                CutlassWeightType_>::type;
#else
    using CutlassWeightType = CutlassWeightType_;
#endif

    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator  = float;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                                    MixedGemmArchTraits::ElementsPerAccessC,
                                                                    ElementAccumulator,
                                                                    ElementAccumulator,
                                                                    cutlass::epilogue::thread::ScaleType::Default>;
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = typename cutlass::platform::conditional<TransC, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    using ElementA = ElementType;
    using ElementB = CutlassWeightType;
    using ElementC = ElementType;
    
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      typename MixedGemmArchTraits::OperatorClass,
      arch,
      ThreadblockShape,
      WarpShape,
      typename MixedGemmArchTraits::InstructionShape,
      EpilogueOp,
      // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
      // This parameter is passed in at present to match the APIs of other kernels. The parameter
      // is unused within the kernel.
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Stages>::GemmKernel;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    if (kernel_occupancy != nullptr) {
        *kernel_occupancy = compute_occupancy_for_kernel<GemmKernel>();
        return;
    }

    ElementA *ptr_A = reinterpret_cast<ElementA *>(A);
    ElementB *ptr_B = reinterpret_cast<ElementB *>(B);
    ElementC *ptr_C = reinterpret_cast<ElementC *>(C);

    GroupedGemmProblemDesc<ElementA, ElementB, ElementC> problem_desc(num_experts);
    setGroupedGemmProblemDescFromDevice<ElementA,
                                        ElementB,
                                        ElementC,
                                        LayoutA,
                                        LayoutB,
                                        LayoutC>(problem_desc, num_experts,
                                                 gemm_m, gemm_n, gemm_k_per_expert,
                                                 ptr_A, ptr_B, ptr_C, stream);

    std::vector<cutlass::gemm::GemmCoord> host_problem_sizes(num_experts);
    cudaMemcpyAsync(host_problem_sizes.data(), problem_desc.problem_sizes, 
                    num_experts * sizeof(cutlass::gemm::GemmCoord),
                    cudaMemcpyDeviceToHost,
                    stream);

    int threadblock_count = GemmGrouped::sufficient(host_problem_sizes.data(), num_experts);
    if (!threadblock_count)
    {
        throw std::runtime_error(
            "[FT Error][MoE Runner] GPU lacks resources to run GroupedGEMM kernel.");
    }

    typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f), ElementAccumulator(0.f));

    typename GemmGrouped::Arguments args(
        problem_desc.problem_sizes,
        num_experts,
        threadblock_count,
        epilogue_op,
        problem_desc.device_ptr_A,
        problem_desc.device_ptr_B,
        problem_desc.device_ptr_C,
        problem_desc.device_ptr_C,
        problem_desc.device_lda,
        problem_desc.device_ldb,
        problem_desc.device_ldc,
        problem_desc.device_ldc,
        host_problem_sizes.data());

    GemmGrouped gemm;
    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg =
            "MoE backward kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg = "Failed to initialize cutlass grouped gemm. Error: "
                              + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass grouped gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Switch Stages
// SM >= 80
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T,
          typename WeightType,
          bool     TransC,
          typename arch,
          typename ThreadblockShape,
          typename WarpShape,
          typename std::enable_if<
              std::is_same<arch, cutlass::arch::Sm80>::value>::type* = nullptr>
void dispatch_gemm_config(
    T*                A,
    WeightType*       B,
    T*                C,
    int64_t           gemm_m,
    int64_t           gemm_n,
    int*              gemm_k_per_expert,
    int               num_experts,
    CutlassGemmConfig gemm_config,
    cudaStream_t      stream,
    int*              occupancy = nullptr)
{
    switch (gemm_config.stages) {
        case 2:
            generic_moe_gemm_backward_kernelLauncher<T, WeightType, TransC, arch, ThreadblockShape, WarpShape, 2>(
                A,
                B,
                C,
                gemm_m,
                gemm_n,
                gemm_k_per_expert,
                num_experts,
                stream,
                occupancy);
            break;
        case 4:
            generic_moe_gemm_backward_kernelLauncher<T, WeightType, TransC, arch, ThreadblockShape, WarpShape, 4>(
                A,
                B,
                C,
                gemm_m,
                gemm_n,
                gemm_k_per_expert,
                num_experts,
                stream,
                occupancy);
            break;
        case 6:
            generic_moe_gemm_backward_kernelLauncher<T, WeightType, TransC, arch, ThreadblockShape, WarpShape, 6>(
                A,
                B,
                C,
                gemm_m,
                gemm_n,
                gemm_k_per_expert,
                num_experts,
                stream,
                occupancy);
            break;
        default:
            std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
            throw std::runtime_error("[FT Error][MoE][dispatch_gemm_config] " + err_msg);
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Switch Stages
// SM < 80
// T =! bf16
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T,
          typename WeightType,
          bool     TransC,
          typename arch,
          typename ThreadblockShape,
          typename WarpShape,
          typename std::enable_if<
              !std::is_same<arch, cutlass::arch::Sm80>::value &&
              !std::is_same<T, __nv_bfloat16>::value>::type* = nullptr>
void dispatch_gemm_config(
    T*                A,
    WeightType*       B,
    T*                C,
    int64_t           gemm_m,
    int64_t           gemm_n,
    int*              gemm_k_per_expert,
    int               num_experts,
    CutlassGemmConfig gemm_config,
    cudaStream_t      stream,
    int*              occupancy = nullptr)
{
    switch (gemm_config.stages) {
        case 2:
            generic_moe_gemm_backward_kernelLauncher<T, WeightType, TransC, arch, ThreadblockShape, WarpShape, 2>(
                A,
                B,
                C,
                gemm_m,
                gemm_n,
                gemm_k_per_expert,
                num_experts,
                stream,
                occupancy);
            break;
        default:
            std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
            throw std::runtime_error("[FT Error][MoE][dispatch_gemm_config] " + err_msg);
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Switch Stages
// SM < 80
// T == bf16
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T,
          typename WeightType,
          bool     TransC,
          typename arch,
          typename ThreadblockShape,
          typename WarpShape,
          typename std::enable_if<
              !std::is_same<arch, cutlass::arch::Sm80>::value &&
              std::is_same<T, __nv_bfloat16>::value>::type * = nullptr>
void dispatch_gemm_config(
    T*                A,
    WeightType*       B,
    T*                C,
    int64_t           gemm_m,
    int64_t           gemm_n,
    int*              gemm_k_per_expert,
    int               num_experts,
    CutlassGemmConfig gemm_config,
    cudaStream_t      stream,
    int*              occupancy = nullptr)
{
    std::string err_msg = "GPU with arch < 80 does not support bfloat16 types!";
    throw std::runtime_error("[FT Error][MoE][dispatch_gemm_config] " + err_msg);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Switch GEMM Config
// T != float
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T,
         typename WeightType,
         bool     TransC,
         typename arch,
         typename std::enable_if<
            !std::is_same<T, float>::value &&
            std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(T*                A,
                                  WeightType*       B,
                                  T*                C,
                                  int64_t           gemm_m,
                                  int64_t           gemm_n,
                                  int*              gemm_k_per_expert,
                                  int               num_experts,
                                  CutlassGemmConfig gemm_config,
                                  cudaStream_t      stream,
                                  int*              occupancy = nullptr)
{
    switch (gemm_config.tile_config) {
        case CutlassTileConfig::CtaShape128x128x32_WarpShape64x64x32:
            dispatch_gemm_config<T, WeightType, TransC, arch,
                                 cutlass::gemm::GemmShape<128, 128, 32>,
                                 cutlass::gemm::GemmShape<64, 64, 32>>(A,
                                                                       B,
                                                                       C,
                                                                       gemm_m,
                                                                       gemm_n,
                                                                       gemm_k_per_expert,
                                                                       num_experts,
                                                                       gemm_config,
                                                                       stream,
                                                                       occupancy);
            break;
        default:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass] Config is invalid for same type MoE tensorop GEMM.");
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Switch GEMM Config
// T == float
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T,
         typename WeightType,
         bool     TransC,
         typename arch,
         typename std::enable_if<
            std::is_same<T, float>::value &&
            std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(T*                A,
                                  WeightType*       B,
                                  T*                C,
                                  int64_t           gemm_m,
                                  int64_t           gemm_n,
                                  int*              gemm_k_per_expert,
                                  int               num_experts,
                                  CutlassGemmConfig gemm_config,
                                  cudaStream_t      stream,
                                  int*              occupancy = nullptr)
{
    switch (gemm_config.tile_config) {
        case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
            dispatch_gemm_config<T, WeightType, TransC, arch,
                                 cutlass::gemm::GemmShape<128, 128, 8>,
                                 cutlass::gemm::GemmShape<64, 64, 8>>(A,
                                                                      B,
                                                                      C,
                                                                      gemm_m,
                                                                      gemm_n,
                                                                      gemm_k_per_expert,
                                                                      num_experts,
                                                                      gemm_config,
                                                                      stream,
                                                                      occupancy);

            break;
        default:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass] Config is invalid for same type MoE tensorop GEMM.");
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Switch Arch
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T,
          typename WeightType>
template <bool     TransC>
void MoeGemmRunner<T, WeightType>::dispatch_to_arch_backward(
    T*                A,
    WeightType*       B,
    T*                C,
    int64_t           gemm_m,
    int64_t           gemm_n,
    int*              gemm_k_per_expert,
    int               num_experts,
    CutlassGemmConfig gemm_config,
    cudaStream_t      stream,
    int*              occupancy)
{
    if (sm_ >= 70 && sm_ < 75) {
#ifdef ARCH_70
        dispatch_moe_gemm_to_cutlass<T, WeightType, TransC, cutlass::arch::Sm70>(
            A,
            B,
            C,
            gemm_m,
            gemm_n,
            gemm_k_per_expert,
            num_experts,
            gemm_config,
            stream,
            occupancy);
#endif // ARCH_70
    }
    else if (sm_ >= 75 && sm_ < 80) {
#ifdef ARCH_75
        dispatch_moe_gemm_to_cutlass<T, WeightType, TransC, cutlass::arch::Sm75>(
            A,
            B,
            C,
            gemm_m,
            gemm_n,
            gemm_k_per_expert,
            num_experts,
            gemm_config,
            stream,
            occupancy);
#endif // ARCH_75
    }
    else if (sm_ >= 80 && sm_ <= 90) {
#ifdef ARCH_80
        dispatch_moe_gemm_to_cutlass<T, WeightType, TransC, cutlass::arch::Sm80>(
            A,
            B,
            C,
            gemm_m,
            gemm_n,
            gemm_k_per_expert,
            num_experts,
            gemm_config,
            stream,
            occupancy);
#endif // ARCH_80
    }
    else {
        throw std::runtime_error("[FT Error][MoE][GEMM Dispatch] Arch unsupported for MoE GEMM");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// CUTLASS Heuristic
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T,
          typename WeightType>
template <bool     TransC>
void MoeGemmRunner<T, WeightType>::run_gemm_backward(
    T*           A,
    WeightType*  B,
    T*           C,
    int64_t      gemm_m,
    int64_t      gemm_n,
    int*         gemm_k_per_expert,
    int          num_tokens,
    int          num_experts,
    cudaStream_t stream)
{
    CutlassGemmConfig chosen_config = CutlassGemmConfig();

    if (std::is_same<T, float>::value)
        chosen_config.tile_config = CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8;
    else
        chosen_config.tile_config = CutlassTileConfig::CtaShape128x128x32_WarpShape64x64x32;

    if (sm_ > 80)
    {
        if (sm_ == 86 || sm_ == 89)
            chosen_config.stages = 4;
        else
            chosen_config.stages = 6;
    }
    else
        chosen_config.stages = 2;

    dispatch_to_arch_backward<TransC>(A,
                                      B,
                                      C,
                                      gemm_m,
                                      gemm_n,
                                      gemm_k_per_expert,
                                      num_experts,
                                      chosen_config,
                                      stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// MoE Grouped GEMM for Backward
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm_backward(T*           A,
                                                     WeightType*  B,
                                                     T*           C,
                                                     int64_t      gemm_m,
                                                     int64_t      gemm_n,
                                                     int*         gemm_k_per_expert,
                                                     int          num_tokens,
                                                     int          num_experts,
                                                     bool         transC,
                                                     cudaStream_t stream)
{
    if (transC)
    {
        run_gemm_backward<true>(
            A, B, C, gemm_m, gemm_n, gemm_k_per_expert, num_tokens, num_experts, stream);
    }
    else
    {
        run_gemm_backward<false>(
            A, B, C, gemm_m, gemm_n, gemm_k_per_expert, num_tokens, num_experts, stream);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace groupedgemmformoe

