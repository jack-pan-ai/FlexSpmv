#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
static std::tuple<torch::Tensor, torch::Tensor> merged_spmv_launch_typed(
    torch::Tensor scatter,
    torch::Tensor area,
    torch::Tensor h,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(scatter.is_cuda(), "scatter must be a CUDA tensor");
  TORCH_CHECK(scatter.is_contiguous(), "scatter must be contiguous");
  TORCH_CHECK(area.is_cuda(), "area must be a CUDA tensor");
  TORCH_CHECK(area.is_contiguous(), "area must be contiguous");
  TORCH_CHECK(h.is_cuda(), "h must be a CUDA tensor");
  TORCH_CHECK(h.is_contiguous(), "h must be contiguous");
  TORCH_CHECK(row_end_offsets.is_cuda(), "row_end_offsets must be a CUDA tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(scatter.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "scatter dtype mismatch");
  TORCH_CHECK(area.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "area dtype mismatch");
  TORCH_CHECK(h.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "h dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = num_cols;
  TORCH_CHECK(ne > 0, "selector index must be non-empty");
  // TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");


  auto options_val = torch::TensorOptions().dtype(scatter.scalar_type()).device(scatter.device());
  torch::Tensor out_0_truediv_2 = torch::zeros({ne}, options_val);
  torch::Tensor out_1_add_10 = torch::zeros({ne}, options_val);


  FlexParams<ValueT, OffsetT> params;
  params.scatter_ptr =         reinterpret_cast<ValueT*>(scatter.data_ptr());
  params.area_ptr =         reinterpret_cast<ValueT*>(area.data_ptr());
  params.h_ptr =         reinterpret_cast<ValueT*>(h.data_ptr());


  params.output_y_truediv_2_ptr =                 reinterpret_cast<ValueT*>(out_0_truediv_2.data_ptr());
  params.output_y_add_10_ptr =                 reinterpret_cast<ValueT*>(out_1_add_10.data_ptr());

  params.d_row_end_offsets = reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr());
  params.num_rows = static_cast<int>(num_rows);
  params.num_cols = static_cast<int>(num_cols);
  params.num_nonzeros = static_cast<int>(ne);

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch (size query) failed: ", cudaGetErrorString(err));

  if (temp_storage_bytes > 0) {
    cudaError_t alloc_err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    TORCH_CHECK(alloc_err == cudaSuccess, "cudaMalloc temp storage failed: ", cudaGetErrorString(alloc_err));
  }

  err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  if (d_temp_storage) { cudaFree(d_temp_storage); }
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch failed: ", cudaGetErrorString(err));

  return std::make_tuple(out_0_truediv_2, out_1_add_10);
}

static std::tuple<torch::Tensor, torch::Tensor> merged_spmv_launch_bind(
    torch::Tensor scatter,
    torch::Tensor area,
    torch::Tensor h,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(scatter.device().is_cuda(), "CUDA device required");
  switch (scatter.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(scatter, area, h, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(scatter, area, h, row_end_offsets, num_rows, num_cols);
        
    default:
      std::cerr << "Unsupported dtype for ValueT. dtype code: " << static_cast<int>(scatter.scalar_type()) << std::endl;
      // TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel"
      , py::arg("scatter")
      , py::arg("area")
      , py::arg("h")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


