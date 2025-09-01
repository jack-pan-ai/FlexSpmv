#pragma once
// The data structures shared by GPU and CPU
// here we define the tensor struct, and overloading functions for the tensor

// Macro definitions for cross-platform compatibility
#ifdef __CUDACC__
#define TENSOR_INLINE __host__ __device__ __forceinline__
#define TENSOR_HOST_ONLY __host__
#define TENSOR_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TENSOR_INLINE inline
#define TENSOR_HOST_ONLY
#define TENSOR_PRAGMA_UNROLL
#endif

// Values-only Tensor
template <typename ValueT, int Dim>
struct Tensor
{
    typedef ValueT Value;
    ValueT values[Dim];

    // Constructor
    TENSOR_INLINE Tensor()
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = 0.0f;
    }

    // Constructor with values
    TENSOR_INLINE Tensor(const ValueT *v)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Constructor with iterator
    template<typename IteratorT>
    TENSOR_INLINE Tensor(IteratorT iter)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = static_cast<ValueT>(iter[i]);
    }

    // Compound assignment operators (more efficient, no temporaries)
    TENSOR_INLINE Tensor& operator+=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] += other.values[i];
        return *this;
    }

    TENSOR_INLINE Tensor& operator-=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] -= other.values[i];
        return *this;
    }

    TENSOR_INLINE Tensor& operator*=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] *= other.values[i];
        return *this;
    }

    TENSOR_INLINE Tensor& operator/=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = (other.values[i] != 0.0) ? (values[i] / other.values[i]) : 0.0;
        return *this;
    }

    TENSOR_INLINE Tensor& operator+=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] += scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator-=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] -= scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator*=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] *= scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator/=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] /= scalar;
        return *this;
    }

    // Binary operators - optimized to avoid temporary variables
    TENSOR_INLINE Tensor operator+(const Tensor &other) const
    {
        Tensor result = *this;
        result += other;
        return result;
    }

    TENSOR_INLINE Tensor operator-(const Tensor &other) const
    {
        Tensor result = *this;
        result -= other;
        return result;
    }

    TENSOR_INLINE Tensor operator*(const Tensor &other) const
    {
        Tensor result = *this;
        result *= other;
        return result;
    }

    TENSOR_INLINE Tensor operator/(const Tensor &other) const
    {
        Tensor result = *this;
        result /= other;
        return result;
    }

    TENSOR_INLINE Tensor operator+(ValueT scalar) const
    {
        Tensor result = *this;
        result += scalar;
        return result;
    }

    TENSOR_INLINE Tensor operator-(ValueT scalar) const
    {
        Tensor result = *this;
        result -= scalar;
        return result;
    }

    TENSOR_INLINE Tensor operator*(ValueT scalar) const
    {
        Tensor result = *this;
        result *= scalar;
        return result;
    }

    TENSOR_INLINE Tensor operator/(ValueT scalar) const
    {
        Tensor result = *this;
        result /= scalar;
        return result;
    }

    // Indexing operator for value access
    TENSOR_INLINE ValueT &operator[](int idx)
    {
        return values[idx];
    }

    TENSOR_INLINE const ValueT &operator[](int idx) const
    {
        return values[idx];
    }

    // L2 norm (Euclidean norm)
    TENSOR_INLINE ValueT l2Norm() const
    {
        ValueT sum = 0.0;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            sum += values[i] * values[i];
        return sqrt(sum);
    }

    // Add this to the Tensor struct after the constructors
    TENSOR_INLINE Tensor& operator=(const Tensor& other)
    {
        if (this != &other) {  // Self-assignment check
            TENSOR_PRAGMA_UNROLL
            for (int i = 0; i < Dim; ++i)
                values[i] = other.values[i];
        }
        return *this;
    }

    TENSOR_INLINE void set(const ValueT* array)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = array[i];
    }

    TENSOR_INLINE void set(const ValueT val)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
    }

    // Also add copy constructor
    TENSOR_INLINE Tensor(const Tensor& other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = other.values[i];
    }

    // Assignment from a scalar (e.g., 0), e.g. Tensor a; a = 0.0;
    TENSOR_INLINE Tensor& operator=(const ValueT val)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
        return *this;
    }

    // Host-only print function
    TENSOR_HOST_ONLY void print() const
    {
        printf("Values: [");
        for (int i = 0; i < Dim; ++i)
        {
            printf("%.2f", values[i]);
            if (i < Dim - 1)
                printf(", ");
        }
        printf("]\n");
    }
};

// Non-member operator to enable scalar * Tensor (values-only)
template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator*(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    return tensor * scalar;  // Reuse the existing Tensor * scalar operator
}

// Non-member operator to enable scalar / Tensor (values-only)
template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator/(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i)
        result.values[i] = (tensor.values[i] != 0.0) ? (scalar / tensor.values[i]) : 0.0;
    return result;
}

// Macro to define tensor binary operators with broadcasting
#define DEFINE_TENSOR_BINARY_OP(op_symbol) \
template <typename ValueT, int N, int M> \
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator op_symbol(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b) \
{ \
    constexpr int OUT_DIM = (N > M ? N : M); \
    Tensor<ValueT, OUT_DIM> result; \
    TENSOR_PRAGMA_UNROLL \
    for (int i = 0; i < OUT_DIM; ++i) \
        result[i] = a[i % N] op_symbol b[i % M]; \
    return result; \
}

// Define all binary operators
DEFINE_TENSOR_BINARY_OP(+)
DEFINE_TENSOR_BINARY_OP(-)
DEFINE_TENSOR_BINARY_OP(*)
DEFINE_TENSOR_BINARY_OP(/)