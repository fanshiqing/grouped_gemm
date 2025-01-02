#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

namespace grouped_gemm {

#define NUM_STREAM 4

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

#define CUBLAS_CALL(code)					  \
  do {								  \
    cublasStatus_t status = code;				  \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS Error"); \
  } while (0)

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
  GROUPED_GEMM_STRINGIFY_HELPER(x)

// TODO(tgale): Update this for SM90 when it's supported by CUTLASS.
using GroupedGemmKernelNN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
  // Non-transposed A operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // Non-transposed B operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // C operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 128, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
  // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
  // This parameter is passed in at present to match the APIs of other kernels. The parameter
  // is unused within the kernel.
  ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  // TODO(tgale): Experiment with GroupScheduleMode.
  // TODO(tgale): Tune this for SM90.
  4>::GemmKernel;
using GemmGroupedNN = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernelNN>;

std::vector<cutlass::gemm::GemmCoord> MakeProblemSizes(torch::Tensor b, torch::Tensor batch_sizes) {
  const size_t num_experts = batch_sizes.size(0);
  const size_t k = b.size(1), n = b.size(2);
  std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    problem_sizes[i] = cutlass::gemm::GemmCoord(batch_sizes.data_ptr<int64_t>()[i], n, k);
  }
  return problem_sizes;
}

template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device) {
  size_t bytes = x.size() * sizeof(T);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
  torch::Tensor out = torch::empty(bytes, options);

  CUDA_CALL(cudaMemcpyAsync(out.data_ptr(),
			    x.data(), bytes,
			    cudaMemcpyHostToDevice,
			    c10::cuda::getCurrentCUDAStream()));
  return out;
}

template <typename Gemm>
typename Gemm::Arguments MakeArguments(torch::Tensor a,
				       torch::Tensor b,
				       torch::Tensor c,
				       torch::Tensor batch_sizes) {
  auto problem_sizes_host = MakeProblemSizes(b, batch_sizes);

  // Calculate the number of threadblocks to use and validate the result.
  int64_t num_experts = problem_sizes_host.size();

  // NOTE: This is borrowed from FasterTransformer.
  int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), num_experts);
  if (!threadblock_count) {
    TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
  }

  // Create the host arrays of leading dimension data and pointer data.
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  std::vector<int64_t> lda_host(num_experts), offsets_a(num_experts);
  std::vector<int64_t> ldb_host(num_experts), offsets_b(num_experts);
  std::vector<int64_t> ldc_host(num_experts), offsets_c(num_experts);
  int64_t elements_a = 0, elements_b = 0, elements_c = 0;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  std::vector<ElementA *> ptr_a_host(num_experts);
  std::vector<ElementB *> ptr_b_host(num_experts);
  std::vector<ElementC *> ptr_c_host(num_experts);

  for (int i = 0; i < num_experts; ++i) {
    auto problem = problem_sizes_host[i];
    lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    offsets_a[i] = elements_a;
    offsets_b[i] = elements_b;
    offsets_c[i] = elements_c;

    ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
    ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
    ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

    elements_a += problem.m() * problem.k();
    elements_b += problem.k() * problem.n();
    elements_c += problem.m() * problem.n();
  }

  // Copy the problem sizes, pointers and leading dimension data to the device.
  torch::Tensor lda = CopyToDevice(lda_host, a.device());
  torch::Tensor ldb = CopyToDevice(ldb_host, a.device());
  torch::Tensor ldc = CopyToDevice(ldc_host, a.device());
  torch::Tensor ptr_a = CopyToDevice(ptr_a_host, a.device());
  torch::Tensor ptr_b = CopyToDevice(ptr_b_host, a.device());
  torch::Tensor ptr_c = CopyToDevice(ptr_c_host, a.device());
  torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, a.device());

  typename Gemm::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f, /*beta=*/0.0f);
  typename Gemm::Arguments arguments((cutlass::gemm::GemmCoord*)problem_sizes.data_ptr(),
  				     (int)num_experts,
  				     (int)threadblock_count,
  				     epilogue_op,
  				     (ElementA**)ptr_a.data_ptr(),
  				     (ElementB**)ptr_b.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     /*lda=*/(int64_t*)lda.data_ptr(),
  				     /*ldb=*/(int64_t*)ldb.data_ptr(),
  				     /*ldc=*/(int64_t*)ldc.data_ptr(),
  				     /*ldd=*/(int64_t*)ldc.data_ptr(),
  				     (cutlass::gemm::GemmCoord*)problem_sizes_host.data());
  return arguments;
}

torch::Tensor CutlassGroupedGemm(torch::Tensor a,
				 torch::Tensor b,
				 torch::Tensor c,
				 torch::Tensor batch_sizes) {
  using Gemm = GemmGroupedNN;
  Gemm gemm;

  auto arguments = MakeArguments<Gemm>(a, b, c, batch_sizes);
  int64_t workspace_size = gemm.get_workspace_size(arguments);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  torch::Tensor workspace = torch::empty(workspace_size, options);

  // Initialize the kernel.
  if(gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel in the current stream.
  if(gemm.run(c10::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
  }
  return c;
}

cublasHandle_t cublas_handle[NUM_STREAM];
cudaStream_t cublas_stream[NUM_STREAM];
cudaEvent_t cublas_event[NUM_STREAM];
bool cublas_init = false;

void cublas_handle_init()
{
    cublas_init = true;

    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamCreateWithFlags(&cublas_stream[i], cudaStreamNonBlocking);
        cublasCreate(&cublas_handle[i]);
        cublasSetStream(cublas_handle[i], cublas_stream[i]);
        cudaEventCreate(&cublas_event[i]);
    }
}

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12500

#define MAX_GROUPSIZE 1024

cublasOperation_t trans_array_T[MAX_GROUPSIZE];
cublasOperation_t trans_array_N[MAX_GROUPSIZE];
int m_array[MAX_GROUPSIZE];
int n_array[MAX_GROUPSIZE];
int k_array[MAX_GROUPSIZE];
float alpha_array[MAX_GROUPSIZE];
float beta_array[MAX_GROUPSIZE];

void * Aarray[MAX_GROUPSIZE];
int lda_array[MAX_GROUPSIZE];
void * Barray[MAX_GROUPSIZE];
int ldb_array[MAX_GROUPSIZE];
void * Carray[MAX_GROUPSIZE];
int ldc_array[MAX_GROUPSIZE];

// on device
void **d_Aarray = nullptr;
void **d_Barray = nullptr;
void **d_Carray = nullptr;

int group_size[MAX_GROUPSIZE];

bool cublas_grouped_gemm_init = false;

void cublas_grouped_gemm_global_var_init()
{
    cublas_grouped_gemm_init = true;

    for (int i = 0; i < MAX_GROUPSIZE; i++)
    {
        alpha_array[i] = 1.0;
        beta_array[i] = 0.0;
        group_size[i] = 1;
        trans_array_T[i] = CUBLAS_OP_T;
        trans_array_N[i] = CUBLAS_OP_N;
    }

    CUDA_CALL(cudaMallocAsync(
        &d_Aarray,
        MAX_GROUPSIZE * sizeof(void *),
        c10::cuda::getCurrentCUDAStream()));
    CUDA_CALL(cudaMallocAsync(
        &d_Barray,
        MAX_GROUPSIZE * sizeof(void *),
        c10::cuda::getCurrentCUDAStream()));
    CUDA_CALL(cudaMallocAsync(
        &d_Carray,
        MAX_GROUPSIZE * sizeof(void *),
        c10::cuda::getCurrentCUDAStream()));
}

void CublasGemmGroupedBatched(torch::Tensor a,
		       torch::Tensor b,
		       torch::Tensor c,
		       torch::Tensor batch_sizes,
           bool trans_a, bool trans_b)
{
  if (!cublas_grouped_gemm_init)
    cublas_grouped_gemm_global_var_init();

  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();

  int a_rows, a_cols, b_rows, b_cols, c_rows, c_cols;

  int group_count = 0;
  for (int i = 0; i < batch_sizes.size(0); i++)
  {
    int bs = batch_sizes.data_ptr<int64_t>()[i];
    if (trans_a) {
      a_rows = bs;
      a_cols = a.size(1);

      // b.dims() == 2 here
      b_rows = bs;
      b_cols = b.size(1);

      c_rows = a_cols;
      c_cols = b_cols;
    } else {
      a_rows = bs;
      a_cols = a.size(1);

      // b.dims() == 3 here
      b_rows = b.size(1);
      b_cols = b.size(2);

      c_rows = a_rows;
      c_cols = trans_b ? b_rows : b_cols;
    }

    if (bs != 0) {
      int m = trans_b ? b_rows : b_cols;
      int k = trans_b ? b_cols : b_rows;
      int n = trans_a ? a_cols : a_rows;
      m_array[group_count] = m;
      n_array[group_count] = n;
      k_array[group_count] = k;

      lda_array[group_count] = trans_a ? n : k;
      ldb_array[group_count] = trans_b ? k : m;
      ldc_array[group_count] = c_cols;
      
      Aarray[group_count] = a_ptr;
      Barray[group_count] = b_ptr;
      Carray[group_count] = c_ptr;

      group_count++;
    }

    a_ptr += a_rows * a_cols;
    b_ptr += b_rows * b_cols;
    c_ptr += c_rows * c_cols;
  }

  CUDA_CALL(cudaMemcpyAsync(d_Aarray, Aarray,
                              sizeof(void *) * group_count,
                              cudaMemcpyHostToDevice,
                              c10::cuda::getCurrentCUDAStream()));
  CUDA_CALL(cudaMemcpyAsync(d_Barray, Barray,
                              sizeof(void *) * group_count,
                              cudaMemcpyHostToDevice,
                              c10::cuda::getCurrentCUDAStream()));
  CUDA_CALL(cudaMemcpyAsync(d_Carray, Carray,
                              sizeof(void *) * group_count,
                              cudaMemcpyHostToDevice,
                              c10::cuda::getCurrentCUDAStream()));

  CUBLAS_CALL(cublasGemmGroupedBatchedEx(
      at::cuda::getCurrentCUDABlasHandle(),
      trans_b ? trans_array_T : trans_array_N,
      trans_a ? trans_array_T : trans_array_N,
      m_array,
      n_array,
      k_array,
      alpha_array,
      d_Barray,
      CUDA_R_16BF,
      ldb_array,
      d_Aarray,
      CUDA_R_16BF,
      lda_array,
      beta_array,
      d_Carray,
      CUDA_R_16BF,
      ldc_array,
      group_count,
      group_size,
      CUBLAS_COMPUTE_32F));
}

#endif

inline void cublas_current_wait_streams(cudaStream_t stream)
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

inline void cublas_streams_wait_current(cudaStream_t stream)
{
    cudaEventRecord(cublas_event[0], stream);

    for (int s = 0; s < NUM_STREAM; s++)
    {
        cudaStreamWaitEvent(cublas_stream[s], cublas_event[0]);
    }
}

void CublasGemm(cublasHandle_t cublas_handle,
    c10::BFloat16 *a, int64_t a_rows, int64_t a_cols, bool trans_a,
		c10::BFloat16 *b, int64_t b_rows, int64_t b_cols, bool trans_b,
		c10::BFloat16 *c, int64_t c_rows, int64_t c_cols) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  int lda = trans_a ? n : k;
  int ldb = trans_b ? k : m;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;


  float alpha = 1.0, beta = 0.0;
  CUBLAS_CALL(cublasGemmEx(cublas_handle,
			   transpose_b, transpose_a,
			   m, n, k, &alpha,
			   b, CUDA_R_16BF, ldb,
			   a, CUDA_R_16BF, lda,
			   &beta,
			   c, CUDA_R_16BF, c_cols, CUDA_R_32F,
			   CUBLAS_GEMM_DEFAULT));
}

void CublasGroupedGemm(torch::Tensor a,
		       torch::Tensor b,
		       torch::Tensor c,
		       torch::Tensor batch_sizes,
		       bool trans_b) {

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12500
  CublasGemmGroupedBatched(a, b, c, batch_sizes, false, trans_b);
  return;
#endif

  if (!cublas_init)
    cublas_handle_init();

  int64_t bs = batch_sizes.size(0), k = a.size(1);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();

  cublas_streams_wait_current(c10::cuda::getCurrentCUDAStream());

  for (int i = 0; i < bs; ++i) {

    int64_t m = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(cublas_handle[i % NUM_STREAM], a_ptr, m, k, /*trans_a=*/false,
	       b_ptr, b_rows, b_cols, trans_b,
	       c_ptr, m, n);
    a_ptr += m * k;
    b_ptr += b_rows * b_cols;
    c_ptr += m * n;
  }

  cublas_current_wait_streams(c10::cuda::getCurrentCUDAStream());
}

void CublasGroupedGemmVariableK(torch::Tensor a,
				torch::Tensor b,
				torch::Tensor c,
				torch::Tensor batch_sizes) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12500
  CublasGemmGroupedBatched(a, b, c, batch_sizes, true, false);
  return;
#endif

  if (!cublas_init)
    cublas_handle_init();

  int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();

  cublas_streams_wait_current(c10::cuda::getCurrentCUDAStream());

  for (int i = 0; i < bs; ++i) {
    int64_t k = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(cublas_handle[i % NUM_STREAM], a_ptr, k, m, /*trans_a=*/true,
	       b_ptr, k, n, /*trans_b=*/false,
	       c_ptr, m, n);
    a_ptr += k * m;
    b_ptr += k * n;
    c_ptr += m * n;
  }

  cublas_current_wait_streams(c10::cuda::getCurrentCUDAStream());
}

void GroupedGemmVariableK(torch::Tensor a,
			  torch::Tensor b,
			  torch::Tensor c,
			  torch::Tensor batch_sizes) {
  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 2);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the dimensions.
  int64_t tokens = a.size(0), num_experts = batch_sizes.size(0);
  int64_t m = a.size(1), n = b.size(1);

  // Validate that we have the same contraction dimension.
  TORCH_CHECK(tokens == b.size(0));

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 3);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == num_experts);
  TORCH_CHECK(c.size(1) == m);
  TORCH_CHECK(c.size(2) == n);

  // Run the computation.
  CublasGroupedGemmVariableK(a, b, c, batch_sizes);
}

// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

  // Defer to the variable 'k' helper for the rest of the op.
  if (trans_a) {
    GroupedGemmVariableK(a, b, c, batch_sizes);
    return;
  }

  // We expected a CUDA tensor with three dimensions and shape
  // (num_experts, hidden_in, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 3);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the contraction dimensions match.
  int64_t tokens = a.size(0), num_experts = b.size(0);
  int64_t hidden_in = trans_b ? b.size(2) : b.size(1);
  int64_t hidden_out = trans_b ? b.size(1) : b.size(2);
  TORCH_CHECK(hidden_in == a.size(1));

  // Validate that we have one size per expert.
  TORCH_CHECK(batch_sizes.size(0) == num_experts);

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 2);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == tokens);
  TORCH_CHECK(c.size(1) == hidden_out);

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());

  // NOTE: Use cuBLAS for SM90 until CUTLASS supports SM90-optimized grouped-gemm.
#if !defined(GROUPED_GEMM_DEVICE_CAPABILITY) || GROUPED_GEMM_DEVICE_CAPABILITY != 80
  CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
  return;
#else
  // TODO(tgale): Support transposition with CUTLASS grouped GEMM.
  if (trans_b) {
    CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
    return;
  }
  CutlassGroupedGemm(a, b, c, batch_sizes);
  return;
#endif
}

}  // namespace grouped_gemm
