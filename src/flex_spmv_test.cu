/**
 * @file flex_spmv_test.cu
 * @brief Test for Flexible Spmv implementation
 */

#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "../include/device_flex_spmv.cuh"

using namespace cub;

// CachingDeviceAllocator for managing device memory
CachingDeviceAllocator g_allocator(true);

/**
 * @brief Generate a random sparse matrix in CSR format and a corresponding dense matrix
 */
template <typename ValueT, typename OffsetT>
void generateTestData(
    ValueT* &h_dense_matrix,        // Host dense matrix
    OffsetT* &h_column_indices_A,   // Host column indices for dense matrix
    OffsetT* &h_row_offsets,        // Host row offsets for sparse matrix
    OffsetT* &h_column_indices,     // Host column indices for sparse matrix
    ValueT* &h_vector_x,            // Host input vector
    ValueT* &h_vector_y_reference,  // Host reference output vector
    ValueT* &h_vector_y,            // Host output vector
    ValueT* &d_dense_matrix,        // Device dense matrix
    OffsetT* &d_column_indices_A,   // Device column indices for dense matrix
    OffsetT* &d_row_offsets,        // Device row offsets for sparse matrix
    OffsetT* &d_column_indices,     // Device column indices for sparse matrix
    ValueT* &d_vector_x,            // Device input vector
    ValueT* &d_vector_y,            // Device output vector
    int num_rows,                   // Number of rows
    int num_cols,                   // Number of columns
    int dense_matrix_width,         // Width of dense matrix
    int nnz_per_row)                // Nonzeros per row
{
    // Calculate total number of nonzeros
    int num_nonzeros = num_rows * nnz_per_row;

    // Allocate host memory
    h_dense_matrix = new ValueT[num_rows * dense_matrix_width];
    h_column_indices_A = new OffsetT[num_nonzeros];
    h_row_offsets = new OffsetT[num_rows + 1];
    h_column_indices = new OffsetT[num_nonzeros];
    h_vector_x = new ValueT[num_cols];
    h_vector_y_reference = new ValueT[num_rows];
    h_vector_y = new ValueT[num_rows];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<ValueT> val_dist(0.1, 1.0);
    std::uniform_int_distribution<OffsetT> col_dist(0, num_cols - 1);

    // Generate dense matrix with random values
    for (int i = 0; i < num_rows * dense_matrix_width; ++i) {
        h_dense_matrix[i] = val_dist(gen);
    }

    // Generate input vector with random values
    for (int i = 0; i < num_cols; ++i) {
        h_vector_x[i] = val_dist(gen);
    }

    // Initialize output vectors to zero
    for (int i = 0; i < num_rows; ++i) {
        h_vector_y_reference[i] = 0.0;
        h_vector_y[i] = 0.0;
    }

    // Generate sparse matrix structure
    h_row_offsets[0] = 0;
    for (int i = 0; i < num_rows; ++i) {
        h_row_offsets[i + 1] = h_row_offsets[i] + nnz_per_row;
        
        // Generate random columns for this row
        std::set<OffsetT> used_cols;
        for (int j = 0; j < nnz_per_row; ++j) {
            OffsetT col;
            do {
                col = col_dist(gen);
            } while (used_cols.count(col) > 0);
            
            used_cols.insert(col);
            OffsetT idx             = h_row_offsets[i] + j;
            h_column_indices[idx]   = col;
            h_column_indices_A[idx] = col;

            // Compute the reference result
            ValueT val = h_dense_matrix[i * dense_matrix_width + col];
            h_vector_y_reference[i] += val * h_vector_x[col];      
        }
    }

    // Allocate device memory
    g_allocator.DeviceAllocate((void**)&d_dense_matrix, sizeof(ValueT) * num_rows * dense_matrix_width);
    g_allocator.DeviceAllocate((void**)&d_column_indices_A, sizeof(OffsetT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_row_offsets, sizeof(OffsetT) * (num_rows + 1));
    g_allocator.DeviceAllocate((void**)&d_column_indices, sizeof(OffsetT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_vector_x, sizeof(ValueT) * num_cols);
    g_allocator.DeviceAllocate((void**)&d_vector_y, sizeof(ValueT) * num_rows);

    // Copy data to device
    cudaMemcpy(d_dense_matrix, h_dense_matrix, sizeof(ValueT) * num_rows * dense_matrix_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices_A, h_column_indices_A, sizeof(OffsetT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(OffsetT) * (num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, sizeof(OffsetT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_x, h_vector_x, sizeof(ValueT) * num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_y, h_vector_y, sizeof(ValueT) * num_rows, cudaMemcpyHostToDevice);
}

/**
 * @brief Clean up memory
 */
template <typename ValueT, typename OffsetT>
void cleanupTestData(
    ValueT* h_dense_matrix,
    OffsetT* h_column_indices_A,
    OffsetT* h_row_offsets,
    OffsetT* h_column_indices,
    ValueT* h_vector_x,
    ValueT* h_vector_y_reference,
    ValueT* h_vector_y,
    ValueT* d_dense_matrix,
    OffsetT* d_column_indices_A,
    OffsetT* d_row_offsets,
    OffsetT* d_column_indices,
    ValueT* d_vector_x,
    ValueT* d_vector_y)
{
    // Free host memory
    delete[] h_dense_matrix;
    delete[] h_column_indices_A;
    delete[] h_row_offsets;
    delete[] h_column_indices;
    delete[] h_vector_x;
    delete[] h_vector_y_reference;
    delete[] h_vector_y;

    // Free device memory
    g_allocator.DeviceFree(d_dense_matrix);
    g_allocator.DeviceFree(d_column_indices_A);
    g_allocator.DeviceFree(d_row_offsets);
    g_allocator.DeviceFree(d_column_indices);
    g_allocator.DeviceFree(d_vector_x);
    g_allocator.DeviceFree(d_vector_y);
}

/**
 * @brief Check result against reference
 */
template <typename ValueT>
bool checkResult(
    ValueT* h_result,
    ValueT* h_reference,
    int num_elements,
    ValueT tolerance = 1e-5)
{
    for (int i = 0; i < num_elements; ++i) {
        ValueT diff = std::abs(h_result[i] - h_reference[i]);
        // std::cout << "Error at index " << i << ": " 
        //           << h_result[i] << " != " << h_reference[i] 
        //           << " (diff = " << diff << ")" << std::endl;
        if (diff > tolerance) {
            std::cout << "Error at index " << i << ": " 
                      << h_result[i] << " != " << h_reference[i] 
                      << " (diff = " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Main test function
 */
int main() {
    // Test parameters
    const int num_rows = 1000;
    const int num_cols = dense_matrix_width = 100;
    const int nnz_per_row = 10;
    const int num_nonzeros = num_rows * nnz_per_row;
    
    std::cout << "Running Flexible Spmv test with:" << std::endl;
    std::cout << "  Rows: " << num_rows << std::endl;
    std::cout << "  Columns: " << num_cols << std::endl;
    std::cout << "  Dense matrix width: " << dense_matrix_width << std::endl;
    std::cout << "  Nonzeros per row: " << nnz_per_row << std::endl;
    std::cout << "  Total nonzeros: " << num_nonzeros << std::endl;
    
    // Declare host and device arrays
    double* h_dense_matrix = nullptr;
    int* h_column_indices_A = nullptr;
    int* h_row_offsets = nullptr;
    int* h_column_indices = nullptr;
    double* h_vector_x = nullptr;
    double* h_vector_y_reference = nullptr;
    double* h_vector_y = nullptr;
    
    double* d_dense_matrix = nullptr;
    int* d_column_indices_A = nullptr;
    int* d_row_offsets = nullptr;
    int* d_column_indices = nullptr;
    double* d_vector_x = nullptr;
    double* d_vector_y = nullptr;
    
    // Generate test data
    generateTestData(
        h_dense_matrix, h_column_indices_A, h_row_offsets, h_column_indices,
        h_vector_x, h_vector_y_reference, h_vector_y,
        d_dense_matrix, d_column_indices_A, d_row_offsets, d_column_indices,
        d_vector_x, d_vector_y,
        num_rows, num_cols, dense_matrix_width, nnz_per_row);
    
    // Allocate temporary storage for DeviceFlexSpmv
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;
    
    // Get required temporary storage size
    cudaError_t error = DeviceFlexSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        d_dense_matrix, d_column_indices_A, dense_matrix_width,
        d_row_offsets, d_column_indices,
        d_vector_x, d_vector_y,
        num_rows, num_cols, num_nonzeros);
    
    if (error != cudaSuccess) {
        std::cerr << "Error in DeviceFlexSpmv::CsrMV sizing: " << cudaGetErrorString(error) << std::endl;
        cleanupTestData(
            h_dense_matrix, h_column_indices_A, h_row_offsets, h_column_indices,
            h_vector_x, h_vector_y_reference, h_vector_y,
            d_dense_matrix, d_column_indices_A, d_row_offsets, d_column_indices,
            d_vector_x, d_vector_y);
        return 1;
    }
    
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    
    // Execute the Flexible Spmv
    error = DeviceFlexSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        d_dense_matrix, d_column_indices_A, dense_matrix_width,
        d_row_offsets, d_column_indices,
        d_vector_x, d_vector_y,
        num_rows, num_cols, num_nonzeros);
    
    if (error != cudaSuccess) {
        std::cerr << "Error in DeviceFlexSpmv::CsrMV execution: " << cudaGetErrorString(error) << std::endl;
        g_allocator.DeviceFree(d_temp_storage);
        cleanupTestData(
            h_dense_matrix, h_column_indices_A, h_row_offsets, h_column_indices,
            h_vector_x, h_vector_y_reference, h_vector_y,
            d_dense_matrix, d_column_indices_A, d_row_offsets, d_column_indices,
            d_vector_x, d_vector_y);
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_vector_y, d_vector_y, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
    
    // Check result
    bool result_correct = checkResult(h_vector_y, h_vector_y_reference, num_rows);
    
    if (result_correct) {
        std::cout << "Flexible Spmv test PASSED!" << std::endl;
    } else {
        std::cout << "Flexible Spmv test FAILED!" << std::endl;
    }
    
    // Clean up
    g_allocator.DeviceFree(d_temp_storage);
    cleanupTestData(
        h_dense_matrix, h_column_indices_A, h_row_offsets, h_column_indices,
        h_vector_x, h_vector_y_reference, h_vector_y,
        d_dense_matrix, d_column_indices_A, d_row_offsets, d_column_indices,
        d_vector_x, d_vector_y);
    
    return result_correct ? 0 : 1;
} 
