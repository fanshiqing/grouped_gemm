import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import subprocess
import re
import shutil
import os

def found_cmake() -> bool:
    """"Check if valid CMake is available

    CMake 3.18 or newer is required.

    """

    # Check if CMake is available
    try:
        _cmake_bin = cmake_bin()
    except FileNotFoundError:
        return False

    # Query CMake for version info
    output = subprocess.run(
        [_cmake_bin, "--version"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"version\s*([\d.]+)", output.stdout)
    version = match.group(1).split('.')
    version = tuple(int(v) for v in version)
    return version >= (3, 18)

def cmake_bin() -> Path:
    """Get CMake executable

    Throws FileNotFoundError if not found.

    """

    # Search in CMake Python package
    _cmake_bin = None
    try:
        import cmake
    except ImportError:
        pass
    else:
        cmake_dir = Path(cmake.__file__).resolve().parent
        _cmake_bin = cmake_dir / "data" / "bin" / "cmake"
        if not _cmake_bin.is_file():
            _cmake_bin = None

    # Search in path
    if _cmake_bin is None:
        _cmake_bin = shutil.which("cmake")
        if _cmake_bin is not None:
            _cmake_bin = Path(_cmake_bin).resolve()

    # Return executable if found
    if _cmake_bin is None:
        raise FileNotFoundError("Could not find CMake executable")
    return _cmake_bin


setup_requires = []

if not found_cmake():
    setup_requires.append("cmake>=3.18")

build_dir = os.path.dirname(os.path.abspath(__file__)) + '/csrc/build'
cmake_dir = os.path.dirname(os.path.abspath(__file__)) + '/csrc'

if not os.path.exists(build_dir):
    os.makedirs(build_dir)
    print(f"Directory '{build_dir}' created.")
else:
    print(f"Directory '{build_dir}' already exists.")

_cmake_bin = str(cmake_bin())
subprocess.run([_cmake_bin, "-S", cmake_dir, "-B", build_dir])
subprocess.run([_cmake_bin, "--build", build_dir, "--parallel"])


setup(
    name="grouped_gemm",
    version="0.5",
    author="Jiang Shao, Shiqing Fan",
    author_email="jiangs@nvidia.com, shiqingf@nvidia.com",
    description="GroupedGEMM CUDA Extension of Pytorch for MoE",
    url="https://github.com/fanshiqing/grouped_gemm@releases/nv_grouped_gemm_v0.5",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    package_dir={'grouped_gemm': './'},
    packages=['grouped_gemm', 'grouped_gemm.tests'],
    package_data={
    'grouped_gemm': ['csrc/build/libmoe_unit_ops.so'],
    },
    cmdclass={"build_ext": BuildExtension},
    install_requires=["absl-py", "numpy", "torch"],
    license_files=("LICENSE",),
)