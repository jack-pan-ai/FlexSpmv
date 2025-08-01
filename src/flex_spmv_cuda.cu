#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <c10/cuda/CUDAStream.h>

#include "../include/merged_spmv.cuh"

// Caching allocator for device memory
cub::CachingDeviceAllocator g_allocator(true);

template <typename ValueT, typename OffsetT>
torch::Tensor launch_flex_spmv_cuda(
    torch::Tensor spm_k,
    torch::Tensor spm_l,
    torch::Tensor row_offsets,
    torch::Tensor col_indices_i,
    torch::Tensor col_indices_j,
    torch::Tensor vector_x,
    torch::Tensor vector_y,
    int num_rows,
    int num_cols,
    int num_nonzeros) {
    
    // Setup FlexParams struct with PyTorch tensor data
    FlexParams<ValueT, OffsetT> params;
    params.spm_k_ptr = spm_k.data_ptr<ValueT>();
    // params.d_spm_nnz = spm_l.data_ptr<ValueT>(); // use it later
    params.row_end_offsets_ptr = row_offsets.data_ptr<OffsetT>();
    params.selector_i_ptr = col_indices_i.data_ptr<OffsetT>();
    params.selector_j_ptr = col_indices_j.data_ptr<OffsetT>();
    params.vector_x_ptr = vector_x.data_ptr<ValueT>();
    params.d_vector_y = vector_y.data_ptr<ValueT>();
    params.num_rows = num_rows;
    params.num_cols = num_cols;
    params.num_nonzeros = num_nonzeros;
    
    // Get current CUDA stream from PyTorch
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    
    // Get amount of temporary storage needed
    cudaError_t error = merged::merged_spmv_launch<ValueT, OffsetT>(
        params, d_temp_storage, temp_storage_bytes, false, stream);
    
    if (error != cudaSuccess) {
        throw std::runtime_error("Error in merged_spmv_launch: " + 
                                std::string(cudaGetErrorString(error)));
    }
    
    // Allocate temporary storage
    error = g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    if (error != cudaSuccess) {
        throw std::runtime_error("Error allocating temporary storage: " + 
                                std::string(cudaGetErrorString(error)));
    }
    
    // Launch the SpMV kernel
    error = merged::merged_spmv_launch<ValueT, OffsetT>(
        params, d_temp_storage, temp_storage_bytes, false, stream);
    
    if (error != cudaSuccess) {
        g_allocator.DeviceFree(d_temp_storage);
        throw std::runtime_error("Error in merged_spmv_launch: " + 
                                std::string(cudaGetErrorString(error)));
    }
    
    // Free temporary storage
    g_allocator.DeviceFree(d_temp_storage);
    
    return vector_y;
}

// Explicit instantiation for float and double types
template torch::Tensor launch_flex_spmv_cuda<float, int>(
    torch::Tensor spm_k,
    torch::Tensor spm_l,
    torch::Tensor row_offsets,
    torch::Tensor col_indices_i,
    torch::Tensor col_indices_j,
    torch::Tensor vector_x,
    torch::Tensor vector_y,
    int num_rows,
    int num_cols,
    int num_nonzeros);

template torch::Tensor launch_flex_spmv_cuda<double, int>(
    torch::Tensor spm_k,
    torch::Tensor spm_l,
    torch::Tensor row_offsets,
    torch::Tensor col_indices_i,
    torch::Tensor col_indices_j,
    torch::Tensor vector_x,
    torch::Tensor vector_y,
    int num_rows,
    int num_cols,
    int num_nonzeros); 