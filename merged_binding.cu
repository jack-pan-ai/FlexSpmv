
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
static std::tuple<torch::Tensor, torch::Tensor> merged_spmv_launch_typed(
    torch::Tensor bsx,
    torch::Tensor bsy,
    torch::Tensor gather_b_1_idx,
    torch::Tensor add_10,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols) {
  TORCH_CHECK(bsx.is_cuda(), "bsx must be a CUDA tensor");
  TORCH_CHECK(bsy.is_cuda(), "bsy must be a CUDA tensor");
  TORCH_CHECK(add_10.is_cuda(), "add_10 must be a CUDA tensor");
  TORCH_CHECK(gather_b_1_idx.is_cuda(), "gather_b_1_idx must be a CUDA tensor");
  TORCH_CHECK(row_end_offsets.is_cuda(), "row_end_offsets must be a CUDA tensor");

  TORCH_CHECK(bsx.is_contiguous(), "bsx must be contiguous");
  TORCH_CHECK(bsy.is_contiguous(), "bsy must be contiguous");
  TORCH_CHECK(add_10.is_contiguous(), "add_10 must be contiguous");
  TORCH_CHECK(gather_b_1_idx.is_contiguous(), "gather_b_1_idx must be contiguous");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(bsx.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "bsx dtype mismatch");
  TORCH_CHECK(bsy.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "bsy dtype mismatch");
  TORCH_CHECK(add_10.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "add_10 dtype mismatch");
  TORCH_CHECK(gather_b_1_idx.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "gather_b_1_idx dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");

  const int64_t ne = gather_b_1_idx.numel();
  TORCH_CHECK(ne > 0, "gather_b_1_idx must be non-empty");
  TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");

  // Infer reducer dims from inputs
  TORCH_CHECK(bsx.numel() % ne == 0, "bsx.numel() must be a multiple of ne");
  TORCH_CHECK(bsy.numel() % ne == 0, "bsy.numel() must be a multiple of ne");
  const int64_t ne1_dim = bsx.numel() / ne;
  const int64_t ne2_dim = bsy.numel() / ne;

  // Allocate outputs
  auto options_val = torch::TensorOptions().dtype(bsx.scalar_type()).device(bsx.device());
  torch::Tensor reducer_1 = torch::zeros({num_rows * ne1_dim}, options_val);
  torch::Tensor reducer_2 = torch::zeros({num_rows * ne2_dim}, options_val);

  // Build params
  FlexParams<ValueT, OffsetT> params;
  params.bsx_ptr = reinterpret_cast<ValueT*>(bsx.data_ptr());
  params.bsy_ptr = reinterpret_cast<ValueT*>(bsy.data_ptr());
  params.gather_b_1_ptr = reinterpret_cast<OffsetT*>(gather_b_1_idx.data_ptr());
  params.add_10_ptr = reinterpret_cast<ValueT*>(add_10.data_ptr());
  params.output_y_scatter_b_2_ptr = reinterpret_cast<ValueT*>(reducer_1.data_ptr());
  params.output_y_scatter_b_3_ptr = reinterpret_cast<ValueT*>(reducer_2.data_ptr());
  params.d_row_end_offsets = reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr());
  params.num_rows = static_cast<int>(num_rows);
  params.num_cols = static_cast<int>(num_cols);
  params.num_nonzeros = static_cast<int>(ne);

  // Query temp storage size and run
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
  if (d_temp_storage) {
    cudaFree(d_temp_storage);
  }
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch failed: ", cudaGetErrorString(err));

  return {reducer_1, reducer_2};
}

static std::tuple<torch::Tensor, torch::Tensor> merged_spmv_launch_bind(
    torch::Tensor bsx,
    torch::Tensor bsy,
    torch::Tensor gather_b_1_idx,
    torch::Tensor add_10,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols) {
  TORCH_CHECK(bsx.device().is_cuda(), "CUDA device required");
  switch (bsx.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(bsx, bsy, gather_b_1_idx, add_10, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(bsx, bsy, gather_b_1_idx, add_10, row_end_offsets, num_rows, num_cols);
    default:
      TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel",
      py::arg("bsx"),
      py::arg("bsy"),
      py::arg("gather_b_1_idx"),
      py::arg("add_10"),
      py::arg("row_end_offsets"),
      py::arg("num_rows"),
      py::arg("num_cols"));
}
