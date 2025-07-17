import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Supported NVIDIA GPU architectures.
NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}

# TORCH_CUDA_ARCH_LIST can have one or more architectures,
# e.g. "9.0" or "7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX". Here,
# the "9.0+PTX" option asks the
# compiler to additionally include PTX code that can be runtime-compiled
# and executed on the 8.6 or newer architectures. While the PTX code will
# not give the best performance on the newer architectures, it provides
# forward compatibility.
env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
if env_arch_list:
    # Let PyTorch builder to choose device to target for.
    device_capability = ""
else:
    device_capability = torch.cuda.get_device_capability()
    device_capability = f"{device_capability[0]}{device_capability[1]}"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16",  # Enable BF16 for cuda_version >= 11
    # "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
]

if device_capability:
    nvcc_flags.extend(
        [
            f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}",
            f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability}",
        ]
    )

if "CUBLAS_VERSION" in os.environ:
    nvcc_flags.append(f"-DCUBLAS_VERSION={os.environ['CUBLAS_VERSION']}")

ext_modules = [
    CUDAExtension(
        "grouped_gemm_backend",
        [
            "csrc/ops.cu",
            "csrc/grouped_gemm.cu",
            "csrc/sinkhorn.cu",
            "csrc/permute.cu",
        ],
        include_dirs=[f"{cwd}/third_party/cutlass/include/", f"{cwd}/csrc"],
        extra_compile_args={
            "cxx": ["-fopenmp", "-fPIC", "-Wno-strict-aliasing"],
            "nvcc": nvcc_flags,
        },
    )
]


setup(
    name="nv_grouped_gemm",
    version="1.1.4",
    author="Trevor Gale, Jiang Shao, Shiqing Fan",
    author_email="tgale@stanford.edu, jiangs@nvidia.com, shiqingf@nvidia.com",
    description="GEMM Grouped",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/fanshiqing/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["absl-py", "numpy", "torch"],
    include_package_data=True,
)
