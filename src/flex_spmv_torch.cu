#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include "../include/merged_spmv.cuh"

// Forward declaration of the CUDA implementation
template <typename ValueT, typename OffsetT>
torch::Tensor launch_flex_spmv_cuda(torch::Tensor spm_k, torch::Tensor spm_l,
                      torch::Tensor row_offsets, 
                      torch::Tensor col_indices_i, torch::Tensor col_indices_j, 
                      torch::Tensor vector_x, torch::Tensor vector_y,
                      int num_rows, int num_cols, int num_nonzeros);

// Python-facing function that will be called from Python
torch::Tensor flex_spmv(torch::Tensor spm_k, torch::Tensor spm_l,
                        torch::Tensor row_offsets, 
                        torch::Tensor col_indices_i, torch::Tensor col_indices_j, 
                        torch::Tensor vector_x, torch::Tensor vector_y) {

  // Check input tensors are contiguous and on CUDA
  TORCH_CHECK(spm_k.is_cuda(), "spm_k must be a CUDA tensor");
  TORCH_CHECK(spm_l.is_cuda(), "spm_l must be a CUDA tensor");
  TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");
  TORCH_CHECK(col_indices_i.is_cuda(), "col_indices_i must be a CUDA tensor");
  TORCH_CHECK(col_indices_j.is_cuda(), "col_indices_j must be a CUDA tensor");
  TORCH_CHECK(vector_x.is_cuda(), "vector_x must be a CUDA tensor");
  TORCH_CHECK(vector_y.is_cuda(), "vector_y must be a CUDA tensor");

  TORCH_CHECK(spm_k.is_contiguous(), "spm_k must be contiguous");
  TORCH_CHECK(spm_l.is_contiguous(), "spm_l must be contiguous");
  TORCH_CHECK(row_offsets.is_contiguous(), "row_offsets must be contiguous");
  TORCH_CHECK(col_indices_i.is_contiguous(),
              "col_indices_i must be contiguous");
  TORCH_CHECK(col_indices_j.is_contiguous(),
              "col_indices_j must be contiguous");
  TORCH_CHECK(vector_x.is_contiguous(), "vector_x must be contiguous");
  TORCH_CHECK(vector_y.is_contiguous(), "vector_y must be contiguous");

  // Get dimensions
  int num_rows = row_offsets.size(0) - 1;
  int num_cols = vector_x.size(0);
  int num_nonzeros = spm_k.size(0);

  // Check tensor types
  if (spm_k.scalar_type() == torch::ScalarType::Float) {
    return launch_flex_spmv_cuda<float, int>(spm_k, spm_l, row_offsets,
                                             col_indices_i, col_indices_j,
                                             vector_x, vector_y,
                                             num_rows, num_cols,
                                             num_nonzeros);
  } else if (spm_k.scalar_type() == torch::ScalarType::Double) {
    return launch_flex_spmv_cuda<double, int>(spm_k, spm_l, row_offsets,
                                              col_indices_i, col_indices_j,
                                              vector_x, vector_y,
                                              num_rows, num_cols, 
                                              num_nonzeros);
  } else {
    TORCH_CHECK(false,
                "Unsupported data type. Only float and double are supported.");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flex_spmv", &flex_spmv,
        "FlexSpmv sparse matrix-vector multiplication");
}