#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_typed(
    torch::Tensor dst_p_0,
    torch::Tensor dst_p_1,
    torch::Tensor dst_p_2,
    torch::Tensor cells,
    torch::Tensor selector_dst_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(dst_p_0.is_cuda(), "dst_p_0 must be a CUDA tensor");
  TORCH_CHECK(dst_p_0.is_contiguous(), "dst_p_0 must be contiguous");
  TORCH_CHECK(dst_p_1.is_cuda(), "dst_p_1 must be a CUDA tensor");
  TORCH_CHECK(dst_p_1.is_contiguous(), "dst_p_1 must be contiguous");
  TORCH_CHECK(dst_p_2.is_cuda(), "dst_p_2 must be a CUDA tensor");
  TORCH_CHECK(dst_p_2.is_contiguous(), "dst_p_2 must be contiguous");
  TORCH_CHECK(cells.is_cuda(), "cells must be a CUDA tensor");
  TORCH_CHECK(cells.is_contiguous(), "cells must be contiguous");
  TORCH_CHECK(selector_dst_idx.is_cuda(), "selector_dst_idx must be a CUDA tensor");
  TORCH_CHECK(selector_dst_idx.is_contiguous(), "selector_dst_idx must be contiguous");
  TORCH_CHECK(row_end_offsets.is_cuda(), "row_end_offsets must be a CUDA tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(dst_p_0.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "dst_p_0 dtype mismatch");
  TORCH_CHECK(dst_p_1.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "dst_p_1 dtype mismatch");
  TORCH_CHECK(dst_p_2.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "dst_p_2 dtype mismatch");
  TORCH_CHECK(cells.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "cells dtype mismatch");
  TORCH_CHECK(selector_dst_idx.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "selector_dst_idx dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = selector_dst_idx.numel();
  TORCH_CHECK(ne > 0, "selector index must be non-empty");
  // TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");


  auto options_val = torch::TensorOptions().dtype(dst_p_0.scalar_type()).device(dst_p_0.device());


  FlexParams<ValueT, OffsetT> params;
  params.dst_p_0_ptr =         reinterpret_cast<ValueT*>(dst_p_0.data_ptr());
  params.dst_p_1_ptr =         reinterpret_cast<ValueT*>(dst_p_1.data_ptr());
  params.dst_p_2_ptr =         reinterpret_cast<ValueT*>(dst_p_2.data_ptr());
  params.cells_ptr =         reinterpret_cast<ValueT*>(cells.data_ptr());

  params.selector_dst_ptr =         reinterpret_cast<OffsetT*>((selector_dst_idx).data_ptr());


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

  
}

void merged_spmv_launch_bind(
    torch::Tensor dst_p_0,
    torch::Tensor dst_p_1,
    torch::Tensor dst_p_2,
    torch::Tensor cells,
    torch::Tensor selector_dst_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(dst_p_0.device().is_cuda(), "CUDA device required");
  switch (dst_p_0.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(dst_p_0, dst_p_1, dst_p_2, cells, selector_dst_idx, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(dst_p_0, dst_p_1, dst_p_2, cells, selector_dst_idx, row_end_offsets, num_rows, num_cols);
        case torch::kLong:
      return merged_spmv_launch_typed<long, int>(dst_p_0, dst_p_1, dst_p_2, cells, selector_dst_idx, row_end_offsets, num_rows, num_cols);
    
    default:
      std::cerr << "Unsupported dtype for ValueT. dtype code: " << static_cast<int>(dst_p_0.scalar_type()) << std::endl;
      // TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel"
      , py::arg("dst_p_0")
      , py::arg("dst_p_1")
      , py::arg("dst_p_2")
      , py::arg("cells")
      , py::arg("selector_dst_idx")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


