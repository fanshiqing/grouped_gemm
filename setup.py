import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from packaging.version import parse
import sys
import platform
import re
import ast
import urllib
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# Supported NVIDIA GPU architectures.
NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
FORCE_BUILD = os.getenv("GROUPED_GEMM_FORCE_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("GROUPED_GEMM_FORCE_CXX11_ABI", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("GROUPED_GEMM_SKIP_CUDA_BUILD", "FALSE") == "TRUE"

# HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
# torch._C._GLIBCXX_USE_CXX11_ABI
# https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
if FORCE_CXX11_ABI:
    torch._C._GLIBCXX_USE_CXX11_ABI = True

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16",  # Enable BF16 for cuda_version >= 11
    # "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
]

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


BASE_WHEEL_URL = (
    "https://github.com/fanshiqing/grouped_gemm/releases/download/{tag_name}/{wheel_name}"
)
PACKAGE_NAME = "nv_grouped_gemm"


def get_package_version():
    with open(Path(this_dir) / "grouped_gemm" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("GROUPED_GEMM_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_wheel_url():
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    grouped_gemm_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.cuda)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.3
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = (
        parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
    )
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}"

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{grouped_gemm_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"

    wheel_url = BASE_WHEEL_URL.format(
        tag_name=f"v{grouped_gemm_version}", wheel_name=wheel_filename
    )

    return wheel_url, wheel_filename


ext_modules = []

if not SKIP_CUDA_BUILD:
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

    if device_capability:
        nvcc_flags.extend(
            [
                f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}",
                f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability}",
            ]
        )

    ext_modules.append(
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
    )


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all grouped gemm installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


setup(
    name="nv_grouped_gemm",
    version=get_package_version(),
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
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": CachedWheelsCommand},
    install_requires=["absl-py", "numpy", "torch"],
    include_package_data=True,
)
