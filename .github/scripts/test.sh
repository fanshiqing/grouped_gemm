#!/bin/bash

set -exou pipefail

pip install dist/*.whl
python -c "import grouped_gemm; print(grouped_gemm.__version__)"