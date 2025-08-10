#pragma once

#define INIT_KERNEL_THREADS 128 // INFO: this is from cub config
#define DIM_OUTPUT_VECTOR_Y 2   // [code generation] Dimension of the output vector
#define DIM_INPUT_VECTOR_X 2    // [code generation] Dimension of the input vector x
#define DIM_INPUT_MATRIX_A 1    // Dimension of the input matrix A

template <typename OffsetT, typename ValueT, int Dim>
struct Tensor
{
    typedef ValueT Value;
    OffsetT key;
    ValueT values[Dim];

    // Constructor
    __host__ __device__ Tensor()
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = 0.0f;
    }

    // Constructor with key and values
    __host__ __device__ Tensor(const OffsetT &k, const ValueT *v) : key(k)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Element-wise operations: +, -, *, /, *scalar, +scalar
    __host__ __device__ Tensor operator+(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] + other.values[i];
        return result;
    }

    __host__ __device__ Tensor operator-(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] - other.values[i];
        return result;
    }

    __host__ __device__ Tensor operator*(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] * other.values[i];
        return result;
    }

    __host__ __device__ Tensor operator/(const Tensor &other) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            result.values[i] = (other.values[i] != 0.0) ? (values[i] / other.values[i]) : 0.0;
        return result;
    }

    __host__ __device__ Tensor operator/(ValueT scalar) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] / scalar;
        return result;
    }

    __host__ __device__ Tensor operator*(ValueT scalar) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            result.values[i] = values[i] * scalar;
        return result;
    }

    __host__ __device__ Tensor operator+(ValueT scalar) const
    {
        Tensor result;
        result.key = key;
        #pragma unroll
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
    
    // L2 norm (Euclidean norm)
    __host__ __device__ ValueT l2Norm() const
    {
        ValueT sum = 0.0;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            sum += values[i] * values[i];
        return sqrt(sum);
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

// FlexParams
template <
    typename ValueT,  ///< Matrix and vector value type
    typename OffsetT> ///< Signed integer type for sequence offsets
struct FlexParams
{
    // [code generation]
    OffsetT *d_row_end_offsets;  ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
      ValueT *vector_x_ptr; 
  OffsetT *selector_i_ptr; 
  OffsetT *selector_j_ptr; 
  ValueT *spm_l_ptr; 
  ValueT *spm_k_ptr; 
  ValueT *output_y_sum_1_ptr; 
  ValueT *output_y_sum_2_ptr; 

    int num_rows;                ///< Number of rows of matrix <b>A</b>.
    int num_cols;                ///< Number of columns of matrix <b>A</b>.
    int num_nonzeros;            ///< Number of nonzero elements of matrix <b>A</b>.
};

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

// Non-member operator to enable scalar * Tensor
template <typename OffsetT, typename ValueT, int Dim>
__host__ __device__ Tensor<OffsetT, ValueT, Dim> operator*(ValueT scalar, const Tensor<OffsetT, ValueT, Dim>& tensor)
{
    return tensor * scalar;  // Reuse the existing Tensor * scalar operator
}

// Non-member operator to enable scalar / Tensor
template <typename OffsetT, typename ValueT, int Dim>
__host__ __device__ Tensor<OffsetT, ValueT, Dim> operator/(ValueT scalar, const Tensor<OffsetT, ValueT, Dim>& tensor)
{
    return tensor / scalar;
}