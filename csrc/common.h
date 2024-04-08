/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

inline int getSMVersion()
{
    int device{-1};
    cudaGetDevice(&device);
    int sm_major = 0;
    int sm_minor = 0;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    return sm_major * 10 + sm_minor;
}