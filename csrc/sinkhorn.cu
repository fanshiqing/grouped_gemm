/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

__global__ void sinkhorn_kernel(float *cost, const int rows, const int cols, float tol) {
    assert(rows >= cols && cols < blockDim.x);

    extern __shared__ float shared_memory[];
    float *shared_d0 = shared_memory;                   // For d0
    float *shared_d1 = (float*)&shared_d0[rows];        // For d1
    float *shared_d1_old = (float*)&shared_d1[cols];    // For d1_old
    float *abs_diff_sum = (float*)&shared_d1_old[cols]; // For sum of absolute differences

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Exponentiate cost matrix
    if (idx < rows * cols) {
        for (int flat_idx = idx; flat_idx < rows * cols; flat_idx += blockDim.x)
            cost[flat_idx] = expf(cost[flat_idx]);
    }

    if (idx >= rows) return;

    // Initilization for d0, d1, d1_old and abs_diff_sum vector.
    for (int row_idx = idx; row_idx < rows; row_idx += blockDim.x) {
        shared_d0[row_idx] = 1.0f;
    }

    if (idx < cols) {
        shared_d1[idx] = 1.0f;
        shared_d1_old[idx] = 1.0f;
        abs_diff_sum[idx] = 0.0f;
    }
    __syncthreads();

    tol = tol * cols;     // error mean check --> error sum check
    const float eps = 1e-8;
    float local_error = 0.0f;
    do {
        local_error = 0.0f;

        // Update d0.
        for (int row_idx = idx; row_idx < rows; row_idx += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                sum += shared_d1[j] * cost[row_idx * cols + j];
            }
            // Using __fdividef for fast division
            shared_d0[row_idx] = __fdividef(1.0f, (sum + eps) * rows);
        }
        __syncthreads();

        // Update d1 and calculate absolute differences.
        if (idx < cols) {
            float sum = 0.0f;
            for (int i = 0; i < rows; ++i) {
                sum += shared_d0[i] * cost[i * cols + idx];
            }
            float new_d1 = __fdividef(1.0, (sum + eps) * cols);
            abs_diff_sum[idx] = fabsf(new_d1 - shared_d1_old[idx]);
            shared_d1[idx] = new_d1;
            // Update shared_d1_old for the next iteration
            shared_d1_old[idx] = new_d1;
        }
        __syncthreads();

        // Compute the sum absolute difference error.
        for (int i = 0; i < cols; ++i) {
            local_error += abs_diff_sum[i];
        }

    } while (local_error > tol);

    // Final multiplication.
    for (int row_idx = idx; row_idx < rows; row_idx += blockDim.x) {
        for (int j = 0; j < cols; ++j) {
            cost[row_idx * cols + j] *= shared_d1[j] * shared_d0[row_idx];
        }
    }
}

void sinkhorn_launch(float *cost, int rows, int cols, float tol) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = 1;
    // Allocate enough shared memory for d0, d1, d1_old and abs_diff_sum
    size_t sharedMemSize = (rows + cols * 2 + cols) * sizeof(float);
    sinkhorn_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(cost, rows, cols, tol);
    // cudaDeviceSynchronize();
}

// Wrapper function
torch::Tensor sinkhorn(torch::Tensor cost, const double tol) {
    sinkhorn_launch(cost.data_ptr<float>(), cost.size(0), cost.size(1), tol);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return cost;
}