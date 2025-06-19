/**
 * @file flex_spmv_test.cu
 * @brief Test for Flexible Spmv implementation
 */

#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "../include/device_flex_spmv.cuh"
#include "../utils.h"

using namespace cub;

// CachingDeviceAllocator for managing device memory
CachingDeviceAllocator g_allocator(true);

/**
 * @brief Generate a random sparse matrix in CSR format and a corresponding dense matrix
 */
template <typename ValueT, typename OffsetT>
void generateTestData(
    ValueT* &h_spm_A,               // Host sparse matrix A
    ValueT* &h_spm_B,               // Host sparse matrix B
    OffsetT* &h_column_indices_A,   // Host column indices for sparse matrix A
    OffsetT* &h_column_indices_1,   // Host column indices for sparse matrix 1
    OffsetT* &h_column_indices_2,   // Host column indices for sparse matrix 2
    OffsetT* &h_row_offsets,        // Host row offsets for sparse matrix
    ValueT* &h_vector_x,            // Host input vector
    ValueT* &h_vector_y_reference,  // Host reference output vector
    ValueT* &h_vector_y,            // Host output vector
    ValueT* &d_spm_A,               // Device sparse matrix A
    ValueT* &d_spm_B,               // Device sparse matrix B
    OffsetT* &d_column_indices_A,   // Device column indices for sparse matrix A
    OffsetT* &d_column_indices_1,   // Device column indices for sparse matrix 1
    OffsetT* &d_column_indices_2,   // Device column indices for sparse matrix 2
    OffsetT* &d_row_offsets,        // Device row offsets for sparse matrix
    ValueT* &d_vector_x,            // Device input vector
    ValueT* &d_vector_y,            // Device output vector
    int num_rows,                   // Number of rows
    int num_cols,                   // Number of columns
    int nnz_per_row,
    int dimension)                // Dimension of the spring mass input
{
    // Calculate total number of nonzeros
    int num_nonzeros = num_rows * nnz_per_row;

    // pre-allocate memory for the intermediate result
    ValueT* r_ij = new ValueT[dimension];
    ValueT* r_i = new ValueT[dimension];
    ValueT* r_j = new ValueT[dimension];
    ValueT* e_ij = new ValueT[dimension];
    ValueT* f_ij = new ValueT[dimension];

    // Allocate host memory
    h_spm_A = new ValueT[num_nonzeros];
    h_spm_B = new ValueT[num_nonzeros];
    h_column_indices_A = new OffsetT[num_nonzeros];
    h_column_indices_1 = new OffsetT[num_nonzeros];
    h_column_indices_2 = new OffsetT[num_nonzeros];
    h_row_offsets = new OffsetT[num_rows + 1];
    h_vector_x = new ValueT[num_cols * dimension];
    h_vector_y_reference = new ValueT[num_rows * dimension];
    h_vector_y = new ValueT[num_rows * dimension];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<ValueT> val_dist(0.1, 1.0);
    std::uniform_int_distribution<OffsetT> col_dist(0, num_cols - 1);
    std::uniform_int_distribution<OffsetT> col_dist_A(0, num_nonzeros - 1);

    // Generate dense matrix with random values
    for (int i = 0; i < num_nonzeros; ++i) {
        h_spm_A[i] = val_dist(gen);
        h_spm_B[i] = val_dist(gen);
    }

    // Generate input vector with random values
    for (int i = 0; i < num_cols * dimension; ++i) {
        h_vector_x[i] = val_dist(gen);
    }

    // Initialize output vectors to zero
    for (int i = 0; i < num_rows * dimension; ++i) {
        h_vector_y_reference[i] = 0.0;
        h_vector_y[i] = 0.0;
    }

    // Generate sparse matrix structure
    h_row_offsets[0] = 0;
    for (int i = 0; i < num_rows; ++i) {
        h_row_offsets[i + 1] = h_row_offsets[i] + nnz_per_row;
    }

    for (int i = 0; i < num_rows; ++i) {        
        // Generate random columns for this row
        std::set<OffsetT> used_cols;
        std::set<OffsetT> used_cols_A;
        for (int j = 0; j < nnz_per_row; ++j) {
            OffsetT col, col_A;
            do {
                col = col_dist(gen);
            } while (used_cols.count(col) > 0);
            do {
                col_A = col_dist_A(gen);
            } while (used_cols_A.count(col_A) > 0);
            
            used_cols.insert(col);
            used_cols_A.insert(col_A);

            // Store the column indices
            OffsetT idx             = h_row_offsets[i] + j;
            OffsetT idx_A           = h_row_offsets[i] + j;
            h_column_indices_1[idx]   = i;
            h_column_indices_2[idx]   = col;
            h_column_indices_A[idx_A] = col_A;

            // Compute the reference result
            for (int k = 0; k < dimension; ++k) {
                r_i[k] = h_vector_x[i * dimension + k];
                r_j[k] = h_vector_x[col * dimension + k];
            }

            for (int k = 0; k < dimension; ++k) {
                r_ij[k] = r_i[k] - r_j[k];
            }
            ValueT r_ij_norm = 0.0;
            for (int k = 0; k < dimension; ++k) {
                r_ij_norm += r_ij[k] * r_ij[k];
            }
            r_ij_norm = std::sqrt(r_ij_norm);
            
            // e_ij = r_ij / r_ij_norm
            for (int k = 0; k < dimension; ++k) {
                e_ij[k] = r_ij[k] / r_ij_norm;
            }

            // f_ij = - B[idx_A] * (r_ij_norm - A[idx_A]) * e_ij
            for (int k = 0; k < dimension; ++k) {
                f_ij[k] = - h_spm_B[idx_A] * (r_ij_norm - h_spm_A[idx_A]) * e_ij[k];
            }

            // reduce f_ij to the reference output vector along the j
            for (int k = 0; k < dimension; ++k) {
                h_vector_y_reference[i * dimension + k] += f_ij[k];
            }
        }
    }

    // Allocate device memory
    g_allocator.DeviceAllocate((void**)&d_spm_A, sizeof(ValueT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_spm_B, sizeof(ValueT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_column_indices_A, sizeof(OffsetT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_column_indices_1, sizeof(OffsetT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_column_indices_2, sizeof(OffsetT) * num_nonzeros);
    g_allocator.DeviceAllocate((void**)&d_row_offsets, sizeof(OffsetT) * (num_rows + 1));
    g_allocator.DeviceAllocate((void**)&d_vector_x, sizeof(ValueT) * num_cols * dimension);
    g_allocator.DeviceAllocate((void**)&d_vector_y, sizeof(ValueT) * num_rows * dimension);

    // Copy data to device
    cudaMemcpy(d_spm_A, h_spm_A, sizeof(ValueT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spm_B, h_spm_B, sizeof(ValueT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices_A, h_column_indices_A, sizeof(OffsetT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices_1, h_column_indices_1, sizeof(OffsetT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices_2, h_column_indices_2, sizeof(OffsetT) * num_nonzeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(OffsetT) * (num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_x, h_vector_x, sizeof(ValueT) * num_cols * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_y, h_vector_y, sizeof(ValueT) * num_rows * dimension, cudaMemcpyHostToDevice);
}

/**
 * @brief Clean up memory
 */
template <typename ValueT, typename OffsetT>
void cleanupTestData(
    ValueT* h_spm_A,
    ValueT* h_spm_B,
    OffsetT* h_column_indices_A,
    OffsetT* h_row_offsets,
    OffsetT* h_column_indices_1,
    OffsetT* h_column_indices_2,
    ValueT* h_vector_x,
    ValueT* h_vector_y_reference,
    ValueT* h_vector_y,
    ValueT* d_spm_A,
    ValueT* d_spm_B,
    OffsetT* d_column_indices_A,
    OffsetT* d_row_offsets,
    OffsetT* d_column_indices_1,
    OffsetT* d_column_indices_2,
    ValueT* d_vector_x,
    ValueT* d_vector_y)
{
    // Free host memory
    delete[] h_spm_A;
    delete[] h_spm_B;
    delete[] h_column_indices_A;
    delete[] h_row_offsets;
    delete[] h_column_indices_1;
    delete[] h_column_indices_2;
    delete[] h_vector_x;
    delete[] h_vector_y_reference;
    delete[] h_vector_y;

    // Free device memory
    g_allocator.DeviceFree(d_spm_A);
    g_allocator.DeviceFree(d_spm_B);
    g_allocator.DeviceFree(d_column_indices_A);
    g_allocator.DeviceFree(d_row_offsets);
    g_allocator.DeviceFree(d_column_indices_1);
    g_allocator.DeviceFree(d_column_indices_2);
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
    int num_rows,
    int dimension,
    ValueT tolerance = 1e-5)
{
    for (int i = 0; i < num_rows * dimension; ++i) {
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
 * Compares the equivalence of two arrays
 */
template <typename ValueT>
int checkResult_2(ValueT* computed, ValueT* reference, int num_rows, int dimension, bool verbose = true)
{
    ValueT meps = std::numeric_limits<ValueT>::epsilon();
    ValueT fmeps = std::numeric_limits<ValueT>::epsilon();
 
    for (int i = 0; i < num_rows * dimension; i++)
    {
        float   a           = computed[i];
        float   b           = reference[i];
        int     int_diff    = std::abs(*(int*)&a - *(int*)&b);
        float   sqrt_diff   = sqrt(float(int_diff));

        if (sqrt_diff > num_rows)      
        {
            if (verbose) std::cout << "INCORRECT (sqrt_diff: " << sqrt_diff << "): [" << i << "]: "
                 << computed[i] << " != "
                 << reference[i]; 
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
    const int num_cols = 100;
    const int nnz_per_row = 10;
    const int dimension = 2;
    const int num_nonzeros = num_rows * nnz_per_row;
    
    std::cout << "Running Flexible Spmv test with:" << std::endl;
    std::cout << "  Rows: " << num_rows << std::endl;
    std::cout << "  Columns: " << num_cols << std::endl;
    std::cout << "  Nonzeros per row: " << nnz_per_row << std::endl;
    std::cout << "  Total nonzeros: " << num_nonzeros << std::endl;
    
    // Spring Mass Matrix flexible Spmv
    double* h_spm_A = nullptr;  // K
    double* h_spm_B = nullptr;  // L
    int* h_column_indices_A = nullptr;
    int* h_column_indices_1 = nullptr;
    int* h_column_indices_2 = nullptr;

    // Spring Mass others
    int* h_row_offsets = nullptr;
    double* h_vector_x = nullptr;
    double* h_vector_y_reference = nullptr;
    double* h_vector_y = nullptr;
    
    double* d_spm_A = nullptr;
    double* d_spm_B = nullptr;
    int* d_column_indices_A = nullptr;
    int* d_column_indices_1 = nullptr;
    int* d_column_indices_2 = nullptr;

    int* d_row_offsets = nullptr;
    double* d_vector_x = nullptr;
    double* d_vector_y = nullptr;
    
    // Generate test data
    generateTestData(
        h_spm_A, h_spm_B, h_column_indices_A, h_row_offsets, h_column_indices_1, h_column_indices_2,
        h_vector_x, h_vector_y_reference, h_vector_y,
        d_spm_A, d_spm_B, d_column_indices_A, d_row_offsets, d_column_indices_1, d_column_indices_2,
        d_vector_x, d_vector_y,
        num_rows, num_cols, nnz_per_row, dimension);
    
    // Allocate temporary storage for DeviceFlexSpmv
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;
    
    // Get required temporary storage size
    cudaError_t error = DeviceFlexSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        d_spm_A, d_spm_B, d_column_indices_A, d_column_indices_1, d_column_indices_2,
        d_row_offsets,
        d_vector_x, d_vector_y,
        num_rows, num_cols, num_nonzeros);
    
    if (error != cudaSuccess) {
        std::cerr << "Error in DeviceFlexSpmv::CsrMV sizing: " << cudaGetErrorString(error) << std::endl;
        cleanupTestData(
            h_spm_A, h_spm_B, h_column_indices_A, h_row_offsets, h_column_indices_1, h_column_indices_2,
            h_vector_x, h_vector_y_reference, h_vector_y,
            d_spm_A, d_spm_B, d_column_indices_A, d_row_offsets, d_column_indices_1, d_column_indices_2,
            d_vector_x, d_vector_y);
        return 1;
    }
    
    // Allocate temporary storage
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    
    // Execute the Flexible Spmv
    error = DeviceFlexSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        d_spm_A, d_spm_B, d_column_indices_A, d_column_indices_1, d_column_indices_2,
        d_row_offsets,
        d_vector_x, d_vector_y,
        num_rows, num_cols, num_nonzeros);
    
    if (error != cudaSuccess) {
        std::cerr << "Error in DeviceFlexSpmv::CsrMV execution: " << cudaGetErrorString(error) << std::endl;
        g_allocator.DeviceFree(d_temp_storage);
        cleanupTestData(
            h_spm_A, h_spm_B, h_column_indices_A, h_row_offsets, h_column_indices_1, h_column_indices_2,
            h_vector_x, h_vector_y_reference, h_vector_y,
            d_spm_A, d_spm_B, d_column_indices_A, d_row_offsets, d_column_indices_1, d_column_indices_2,
            d_vector_x, d_vector_y);
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_vector_y, d_vector_y, sizeof(double) * num_rows * dimension, cudaMemcpyDeviceToHost);
    
    // Check result
    bool result_correct = checkResult_2(h_vector_y, h_vector_y_reference, num_rows, dimension);
    
    if (result_correct) {
        std::cout << "\n Flexible Spmv test PASSED!" << std::endl;
    } else {
        std::cout << "\n Flexible Spmv test FAILED!" << std::endl;
    }
    
    // Clean up
    g_allocator.DeviceFree(d_temp_storage);
    cleanupTestData(
        h_spm_A, h_spm_B, h_column_indices_A, h_row_offsets, h_column_indices_1, h_column_indices_2,
        h_vector_x, h_vector_y_reference, h_vector_y,
        d_spm_A, d_spm_B, d_column_indices_A, d_row_offsets, d_column_indices_1, d_column_indices_2,
        d_vector_x, d_vector_y);
    
    return result_correct ? 0 : 1;
} 
