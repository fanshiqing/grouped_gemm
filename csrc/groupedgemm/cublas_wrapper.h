/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cublas_v2.h"

#define NUM_STREAM 4

extern bool USE_CUBLAS;
extern bool cublas_init;
extern cublasHandle_t cublas_handle[NUM_STREAM];
extern cudaStream_t cublas_stream[NUM_STREAM];
extern cudaEvent_t cublas_event[NUM_STREAM];

inline void cublas_handle_init()
{
    cublas_init = true;

    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamCreate(&cublas_stream[i]);
        cublasCreate(&cublas_handle[i]);
        cublasSetStream(cublas_handle[i], cublas_stream[i]);
        cudaEventCreate(&cublas_event[i]);
    }
}

inline void cublas_sync_streams(cudaStream_t stream)
{
    for (int s = 0; s < NUM_STREAM; s++)
    {
        cudaEventRecord(cublas_event[s], cublas_stream[s]);
    }

    for (int s = 0; s < NUM_STREAM; s++)
    {
        cudaStreamWaitEvent(stream, cublas_event[s]);
    }
}

template <typename T>
void cublas_group_gemm_helper(
    T *A,
    T **B_list,
    T *C,
    int *gemm_m_per_expert,
    int64_t gemm_n,
    int64_t gemm_k,
    const int num_experts,
    bool transB,
    cudaStream_t stream)
{
    // variable M grouped gemm

    if (!cublas_init)
        cublas_handle_init();

    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

    if (std::is_same<T, float>::value)
    {
        Atype = CUDA_R_32F;
        Btype = CUDA_R_32F;
        Ctype = CUDA_R_32F;
    }
    else if (std::is_same<T, half>::value)
    {
        Atype = CUDA_R_16F;
        Btype = CUDA_R_16F;
        Ctype = CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value)
    {
        Atype = CUDA_R_16BF;
        Btype = CUDA_R_16BF;
        Ctype = CUDA_R_16BF;
    }
#endif

    cublasOperation_t trans_A = CUBLAS_OP_N;
    cublasOperation_t trans_B = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasGemmAlgo_t cublas_algo = CUBLAS_GEMM_DEFAULT;

    float alpha = 1.0f;
    float beta = 0.0f;

    int ldb = transB ? gemm_k : gemm_n;

    for (int e = 0; e < num_experts; e++)
    {
        int i = e % NUM_STREAM;

        cublasGemmEx(
            cublas_handle[i],
            trans_B,
            trans_A,
            gemm_n,
            gemm_m_per_expert[e],
            gemm_k,
            &alpha,
            B_list[e],
            Btype,
            ldb,
            A,
            Atype,
            gemm_k,
            &beta,
            C,
            Ctype,
            gemm_n,
            computeType,
            cublas_algo);

        A = A + gemm_m_per_expert[e] * gemm_k;
        C = C + gemm_m_per_expert[e] * gemm_n;
    }

    cublas_sync_streams(stream);
}

template <typename T,
          typename AccumGradType>
void cublas_group_gemm_helper(
    T *A,
    T *B,
    T *C,
    AccumGradType **weight_grad_list,
    int64_t gemm_m,
    int64_t gemm_n,
    int *gemm_k_per_expert,
    const int num_experts,
    bool transC,
    cudaStream_t stream)
{
    // variable K grouped gemm

    if (!cublas_init)
        cublas_handle_init();

    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

    float beta;

    if (std::is_same<T, float>::value)
    {
        Atype = CUDA_R_32F;
        Btype = CUDA_R_32F;
    }
    else if (std::is_same<T, half>::value)
    {
        Atype = CUDA_R_16F;
        Btype = CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value)
    {
        Atype = CUDA_R_16BF;
        Btype = CUDA_R_16BF;
    }
#endif

    if (C != nullptr)
    {
        beta = 0.0f;
        Ctype = Atype;
    }
    else
    {
        beta = 1.0f;

        if (std::is_same<AccumGradType, float>::value)
        {
            Ctype = CUDA_R_32F;
        }
        else if (std::is_same<AccumGradType, half>::value)
        {
            Ctype = CUDA_R_16F;
        }
#ifdef ENABLE_BF16
        else if (std::is_same<AccumGradType, __nv_bfloat16>::value)
        {
            Ctype = CUDA_R_16BF;
        }
#endif
    }

    cublasGemmAlgo_t cublas_algo = CUBLAS_GEMM_DEFAULT;
    float alpha = 1.0f;

    if (!transC)
    {
        for (int e = 0; e < num_experts; e++)
        {
            cublasOperation_t trans_A = CUBLAS_OP_T;
            cublasOperation_t trans_B = CUBLAS_OP_N;

            int i = e % NUM_STREAM;

            void *C_ptr;
            if (C != nullptr)
            {
                C_ptr = reinterpret_cast<void *>(C);
                C = C + gemm_m * gemm_n;
            }
            else
            {
                C_ptr = reinterpret_cast<void *>(weight_grad_list[e]);
            }

            cublasGemmEx(
                cublas_handle[i],
                trans_B,
                trans_A,
                gemm_n,
                gemm_m,
                gemm_k_per_expert[e],
                &alpha,
                B,
                Btype,
                gemm_n,
                A,
                Atype,
                gemm_m,
                &beta,
                C_ptr,
                Ctype,
                gemm_n,
                computeType,
                cublas_algo);

            A = A + gemm_m * gemm_k_per_expert[e];
            B = B + gemm_n * gemm_k_per_expert[e];
        }
    }
    else
    {
        for (int e = 0; e < num_experts; e++)
        {
            cublasOperation_t trans_A = CUBLAS_OP_N;
            cublasOperation_t trans_B = CUBLAS_OP_T;

            int i = e % NUM_STREAM;

            void *C_ptr;
            if (C != nullptr)
            {
                C_ptr = reinterpret_cast<void *>(C);
                C = C + gemm_m * gemm_n;
            }
            else
            {
                C_ptr = reinterpret_cast<void *>(weight_grad_list[e]);
            }

            cublasGemmEx(
                cublas_handle[i],
                trans_A,
                trans_B,
                gemm_m,
                gemm_n,
                gemm_k_per_expert[e],
                &alpha,
                A,
                Atype,
                gemm_m,
                B,
                Btype,
                gemm_n,
                &beta,
                C_ptr,
                Ctype,
                gemm_m,
                computeType,
                cublas_algo);

            A = A + gemm_m * gemm_k_per_expert[e];
            B = B + gemm_n * gemm_k_per_expert[e];
        }
    }

    cublas_sync_streams(stream);
}