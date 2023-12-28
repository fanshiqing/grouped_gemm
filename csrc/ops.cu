#include "grouped_gemm.h"
#include "sinkhorn.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  m.def("sinkhorn", &sinkhorn, "Sinkhorn kernel");
}

}  // namespace grouped_gemm
