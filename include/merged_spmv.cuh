// merged_spmv.cuh
#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define INIT_KERNEL_THREADS 128 // INFO: this is from cub config
#define DIM_OUTPUT 1             // Dimension of the output vector
#define DIM_INPUT_VECTOR_X 1     // Dimension of the input vector x
#define DIM_INPUT_MATRIX_A 1     // Dimension of the input matrix A

template <typename OffsetT, typename ValueT, int Dim>
struct Tensor
{
    OffsetT key;
    ValueT values[Dim];

    // Constructor
    __host__ __device__ Tensor()
    {
        for (int i = 0; i < Dim; ++i)
            values[i] = 0.0f;
    }

    // Constructor with key and values
    __host__ __device__ Tensor(const OffsetT &k, const ValueT *v) : key(k)
    {
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Element-wise operations: +, -, *, /, *scalar, +scalar
    __host__ __device__ Tensor operator+(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] + other.values[i];
        return result;
    }

    __host__ __device__ Tensor operator-(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] - other.values[i];
        return result;
    }

    __host__ __device__ Tensor operator*(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] * other.values[i];
        return result;
    }

    __host__ __device__ Tensor operator/(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        for (int i = 0; i < Dim; ++i)
            result.values[i] = (other.values[i] != 0.0) ? (values[i] / other.values[i]) : 0.0;
        return result;
    }

    __host__ __device__ Tensor operator*(ValueT scalar) const
    {
        Tensor result;
        result.key = key;
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] * scalar;
        return result;
    }

    __host__ __device__ Tensor operator+(ValueT scalar) const
    {
        Tensor result;
        result.key = key;
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] + scalar;
        return result;
    }

    // Indexing operator for value access
    __host__ __device__ ValueT &operator[](int idx)
    {
        return values[idx];
    }

    __host__ __device__ const ValueT &operator[](int idx) const
    {
        return values[idx];
    }

    // Host-only print function
    __host__ void print() const
    {
        printf("Key: %d, Values: [", (int)key);
        for (int i = 0; i < Dim; ++i)
        {
            printf("%.2f", values[i]);
            if (i < Dim - 1)
                printf(", ");
        }
        printf("]\n");
    }
};

// FlexSpmvParams [code transformation]
template <
    typename ValueT,  ///< Matrix and vector value type
    typename OffsetT> ///< Signed integer type for sequence offsets
struct FlexSpmvParams
{
    ValueT *d_spm_nnz;          ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT *d_row_end_offsets; ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    OffsetT *d_column_indices;  ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT *d_vector_x;         ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT *d_vector_y;         ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int num_rows;               ///< Number of rows of matrix <b>A</b>.
    int num_cols;               ///< Number of columns of matrix <b>A</b>.
    int num_nonzeros;           ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT alpha;               ///< Alpha multiplicand
    ValueT beta;                ///< Beta addend-multiplicand
};

/**
 * Launch kernel configuration. <cub>
 */
struct LaunchKernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    __host__ __forceinline__ void Init()
    {
        block_threads = PolicyT::BLOCK_THREADS;
        items_per_thread = PolicyT::ITEMS_PER_THREAD;
        tile_items = block_threads * items_per_thread;
    }
};

/// Host/device‐side launcher (calls the kernel)
template <
    typename ValueT,
    typename OffsetT>
__host__ __device__ __forceinline__ static cudaError_t merged_spmv_launch(
    FlexSpmvParams<ValueT, OffsetT> spmv_params,
    LaunchKernelConfig spmv_config,
    LaunchKernelConfig segment_fixup_config,
    bool debug_synchronous = false,
    cudaStream_t stream = 0);
