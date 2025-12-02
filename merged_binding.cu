#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
static std::tuple<torch::Tensor> merged_spmv_launch_typed(
    torch::Tensor spm_1,
    torch::Tensor y_add_1,
    torch::Tensor spm_2,
    torch::Tensor y_add_2,
    torch::Tensor vector_x,
    torch::Tensor selector_1_idx,
    torch::Tensor selector_2_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(spm_1.is_cuda(), "spm_1 must be a CUDA tensor");
  TORCH_CHECK(spm_1.is_contiguous(), "spm_1 must be contiguous");
  TORCH_CHECK(y_add_1.is_cuda(), "y_add_1 must be a CUDA tensor");
  TORCH_CHECK(y_add_1.is_contiguous(), "y_add_1 must be contiguous");
  TORCH_CHECK(spm_2.is_cuda(), "spm_2 must be a CUDA tensor");
  TORCH_CHECK(spm_2.is_contiguous(), "spm_2 must be contiguous");
  TORCH_CHECK(y_add_2.is_cuda(), "y_add_2 must be a CUDA tensor");
  TORCH_CHECK(y_add_2.is_contiguous(), "y_add_2 must be contiguous");
  TORCH_CHECK(vector_x.is_cuda(), "vector_x must be a CUDA tensor");
  TORCH_CHECK(vector_x.is_contiguous(), "vector_x must be contiguous");
  TORCH_CHECK(selector_1_idx.is_cuda(), "selector_1_idx must be a CUDA tensor");
  TORCH_CHECK(selector_1_idx.is_contiguous(), "selector_1_idx must be contiguous");
  TORCH_CHECK(selector_2_idx.is_cuda(), "selector_2_idx must be a CUDA tensor");
  TORCH_CHECK(selector_2_idx.is_contiguous(), "selector_2_idx must be contiguous");
  TORCH_CHECK(row_end_offsets.is_cuda(), "row_end_offsets must be a CUDA tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(spm_1.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "spm_1 dtype mismatch");
  TORCH_CHECK(y_add_1.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "y_add_1 dtype mismatch");
  TORCH_CHECK(spm_2.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "spm_2 dtype mismatch");
  TORCH_CHECK(y_add_2.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "y_add_2 dtype mismatch");
  TORCH_CHECK(vector_x.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "vector_x dtype mismatch");
  TORCH_CHECK(selector_1_idx.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "selector_1_idx dtype mismatch");
  TORCH_CHECK(selector_2_idx.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "selector_2_idx dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = selector_1_idx.numel();
  TORCH_CHECK(ne > 0, "selector index must be non-empty");
  // TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");


  auto options_val = torch::TensorOptions().dtype(spm_1.scalar_type()).device(spm_1.device());
  torch::Tensor out_0_reducer_1 = torch::zeros({num_rows, 2}, options_val);


  FlexParams<ValueT, OffsetT> params;
  params.spm_1_ptr =         reinterpret_cast<ValueT*>(spm_1.data_ptr());
  params.y_add_1_ptr =         reinterpret_cast<ValueT*>(y_add_1.data_ptr());
  params.spm_2_ptr =         reinterpret_cast<ValueT*>(spm_2.data_ptr());
  params.y_add_2_ptr =         reinterpret_cast<ValueT*>(y_add_2.data_ptr());
  params.vector_x_ptr =         reinterpret_cast<ValueT*>(vector_x.data_ptr());

  params.selector_1_ptr =         reinterpret_cast<OffsetT*>((selector_1_idx).data_ptr());
  params.selector_2_ptr =         reinterpret_cast<OffsetT*>((selector_2_idx).data_ptr());

  params.output_y_reducer_1_ptr =                 reinterpret_cast<ValueT*>(out_0_reducer_1.data_ptr());

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

  return std::make_tuple(out_0_reducer_1);
}

static std::tuple<torch::Tensor> merged_spmv_launch_bind(
    torch::Tensor spm_1,
    torch::Tensor y_add_1,
    torch::Tensor spm_2,
    torch::Tensor y_add_2,
    torch::Tensor vector_x,
    torch::Tensor selector_1_idx,
    torch::Tensor selector_2_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(spm_1.device().is_cuda(), "CUDA device required");
  switch (spm_1.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(spm_1, y_add_1, spm_2, y_add_2, vector_x, selector_1_idx, selector_2_idx, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(spm_1, y_add_1, spm_2, y_add_2, vector_x, selector_1_idx, selector_2_idx, row_end_offsets, num_rows, num_cols);
        
    default:
      std::cerr << "Unsupported dtype for ValueT. dtype code: " << static_cast<int>(spm_1.scalar_type()) << std::endl;
      // TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel"
      , py::arg("spm_1")
      , py::arg("y_add_1")
      , py::arg("spm_2")
      , py::arg("y_add_2")
      , py::arg("vector_x")
      , py::arg("selector_1_idx")
      , py::arg("selector_2_idx")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


