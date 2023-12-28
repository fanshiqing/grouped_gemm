#include <torch/extension.h>

namespace grouped_gemm {

torch::Tensor sinkhorn(torch::Tensor cost, const float tol=0.0001);

}  // namespace grouped_gemm
