/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/gemm/gemm.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename TypeA, typename TypeB, typename TypeC,
         typename arch, typename Enable = void>
struct MixedGemmArchTraits {
};

template<typename TypeC, typename arch>
struct MixedGemmArchTraits<float, float, TypeC, arch> {
    static constexpr int ElementsPerAccessA = 1;
    static constexpr int ElementsPerAccessB = 1;
    static constexpr int ElementsPerAccessC = 1;
    using OperatorClass                     = cutlass::arch::OpClassSimt;
    using InstructionShape                  = cutlass::gemm::GemmShape<1, 1, 1>;
};

// ========================= Volta Traits ===========================
// Volta will always dequantize after the global memory load.
// This will instantiate any HMMA tensorcore kernels for Volta.
// Note that volta does not have native bfloat support so weights and activations will be casted to fp16
// and compute will happen in fp16 then will be converted for bf16 output.
template<typename TypeA, typename TypeB, typename TypeC>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    TypeC,
    cutlass::arch::Sm70,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {

public:
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = 128 / cutlass::sizeof_bits<TypeB>::value;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeC>::value;
    using OperatorClass                     = cutlass::arch::OpClassTensorOp;
    using InstructionShape                  = cutlass::gemm::GemmShape<8, 8, 4>;
};

// ======================= Turing Traits ==============================
// Note that turing does not have native bfloat support so weights and activations will be casted to fp16
// and compute will happen in fp16 then will be converted for bf16 output.
template<typename TypeA, typename TypeB, typename TypeC>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    TypeC,
    cutlass::arch::Sm75,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {

public:
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = 128 / cutlass::sizeof_bits<TypeB>::value;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeC>::value;
    using OperatorClass                     = cutlass::arch::OpClassTensorOp;
    using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 8>;
};

// ======================= Ampere Traits ==============================
template<typename TypeA, typename TypeB, typename TypeC>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    TypeC,
    cutlass::arch::Sm80,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {

public:
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = 128 / cutlass::sizeof_bits<TypeB>::value;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeC>::value;
    using OperatorClass                     = cutlass::arch::OpClassTensorOp;
    using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 16>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass