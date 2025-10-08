#pragma once
// =============================================================================
// Tensor Data Structure - Cross-Platform Implementation
// =============================================================================
// High-performance tensor class supporting:
// - Element-wise operations with broadcasting
// - Scalar operations with type safety  
// - CUDA and host compatibility
// - Template-based dimension support
// =============================================================================

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>
#else
#include <cmath>
#include <type_traits>
#endif

// =============================================================================
// Cross-Platform Compatibility Macros
// =============================================================================
#ifdef __CUDACC__
#define TENSOR_INLINE __host__ __device__ __forceinline__
#define TENSOR_HOST_ONLY __host__
#define TENSOR_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TENSOR_INLINE inline
#define TENSOR_HOST_ONLY
#define TENSOR_PRAGMA_UNROLL
#endif

// =============================================================================
// Helper Functions
// =============================================================================
template<typename T>
TENSOR_INLINE T tensor_pow(T base, T exp) {
#ifdef __CUDA_ARCH__
    return pow(base, exp);
#else
    return std::pow(base, exp);
#endif
}

// =============================================================================
// Main Tensor Class
// =============================================================================
template <typename ValueT, int Dim>
struct Tensor
{
    // Type aliases
    typedef ValueT Value;
    
    // Data storage
    ValueT values[Dim];

    // =============================================================================
    // Constructors and Assignment
    // =============================================================================
    
    TENSOR_INLINE Tensor()
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = 0.0f;
    }

    TENSOR_INLINE Tensor(const ValueT *v)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    template<typename IteratorT>
    TENSOR_INLINE Tensor(IteratorT iter)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = static_cast<ValueT>(iter[i]);
    }

    TENSOR_INLINE Tensor(const Tensor& other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = other.values[i];
    }

    TENSOR_INLINE Tensor& operator=(const Tensor& other)
    {
        if (this != &other) {
            TENSOR_PRAGMA_UNROLL
            for (int i = 0; i < Dim; ++i)
                values[i] = other.values[i];
        }
        return *this;
    }

    TENSOR_INLINE Tensor& operator=(const ValueT val)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
        return *this;
    }

    // =============================================================================
    // Element Access
    // =============================================================================
    
    TENSOR_INLINE ValueT &operator[](int idx) { return values[idx]; }
    TENSOR_INLINE const ValueT &operator[](int idx) const { return values[idx]; }

    // =============================================================================
    // Compound Assignment Operators (Tensor-Tensor)
    // =============================================================================
    
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

    TENSOR_INLINE Tensor& operator^=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = tensor_pow(values[i], other.values[i]);
        return *this;
    }

    // =============================================================================
    // Compound Assignment Operators (Tensor-Scalar)
    // =============================================================================
    
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

    TENSOR_INLINE Tensor& operator^=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = tensor_pow(values[i], scalar);
        return *this;
    }

    // =============================================================================
    // Binary Operators - Auto-generated from compound assignments
    // =============================================================================
    
    // Tensor-Tensor operations
    TENSOR_INLINE Tensor operator+(const Tensor &other) const { Tensor result = *this; result += other; return result; }
    TENSOR_INLINE Tensor operator-(const Tensor &other) const { Tensor result = *this; result -= other; return result; }
    TENSOR_INLINE Tensor operator*(const Tensor &other) const { Tensor result = *this; result *= other; return result; }
    TENSOR_INLINE Tensor operator/(const Tensor &other) const { Tensor result = *this; result /= other; return result; }
    TENSOR_INLINE Tensor operator^(const Tensor &other) const { Tensor result = *this; result ^= other; return result; }

    // Tensor-Scalar operations  
    TENSOR_INLINE Tensor operator+(ValueT scalar) const { Tensor result = *this; result += scalar; return result; }
    TENSOR_INLINE Tensor operator-(ValueT scalar) const { Tensor result = *this; result -= scalar; return result; }
    TENSOR_INLINE Tensor operator*(ValueT scalar) const { Tensor result = *this; result *= scalar; return result; }
    TENSOR_INLINE Tensor operator/(ValueT scalar) const { Tensor result = *this; result /= scalar; return result; }
    TENSOR_INLINE Tensor operator^(ValueT scalar) const { Tensor result = *this; result ^= scalar; return result; }

    // =============================================================================
    // Utility Methods
    // =============================================================================
    
    TENSOR_INLINE Tensor pow(ValueT exponent) const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_pow(values[i], exponent);
        return result;
    }

    TENSOR_INLINE Tensor pow(const Tensor &exponents) const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_pow(values[i], exponents.values[i]);
        return result;
    }

    TENSOR_INLINE ValueT l2Norm() const
    {
        ValueT sum = 0.0;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            sum += values[i] * values[i];
        return sqrt(sum);
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

// =============================================================================
// Scalar-Tensor Binary Operators
// =============================================================================

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator*(ValueT scalar, const Tensor<ValueT, Dim>& tensor) { return tensor * scalar; }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator+(ValueT scalar, const Tensor<ValueT, Dim>& tensor) { return tensor + scalar; }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator-(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = scalar - tensor.values[i];
    return result;
}

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator/(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = (tensor.values[i] != 0.0) ? (scalar / tensor.values[i]) : 0.0;
    return result;
}

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator^(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = tensor_pow(scalar, tensor.values[i]);
    return result;
}

// =============================================================================
// Mixed-Type Operators (double with non-double tensors)
// =============================================================================

#define DEFINE_MIXED_DOUBLE_OP(op) \
template <typename ValueT, int Dim> \
TENSOR_INLINE typename std::enable_if<!std::is_same<ValueT, double>::value, Tensor<ValueT, Dim>>::type \
operator op(double scalar, const Tensor<ValueT, Dim>& tensor) { return static_cast<ValueT>(scalar) op tensor; } \
\
template <typename ValueT, int Dim> \
TENSOR_INLINE typename std::enable_if<!std::is_same<ValueT, double>::value, Tensor<ValueT, Dim>>::type \
operator op(const Tensor<ValueT, Dim>& tensor, double scalar) { return tensor op static_cast<ValueT>(scalar); }

DEFINE_MIXED_DOUBLE_OP(+)
DEFINE_MIXED_DOUBLE_OP(-)
DEFINE_MIXED_DOUBLE_OP(*)
DEFINE_MIXED_DOUBLE_OP(/)
DEFINE_MIXED_DOUBLE_OP(^)

#undef DEFINE_MIXED_DOUBLE_OP

// =============================================================================
// Utility Functions
// =============================================================================

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> pow(const Tensor<ValueT, Dim>& base, ValueT exponent) { return base.pow(exponent); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> pow(const Tensor<ValueT, Dim>& base, const Tensor<ValueT, Dim>& exponents) { return base.pow(exponents); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> pow(ValueT base, const Tensor<ValueT, Dim>& exponents)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = tensor_pow(base, exponents.values[i]);
    return result;
}

// =============================================================================
// Broadcasting Operations (Different Dimensions)
// =============================================================================

#define DEFINE_BROADCASTING_OP(op) \
template <typename ValueT, int N, int M> \
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator op(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b) \
{ \
    constexpr int OUT_DIM = (N > M ? N : M); \
    Tensor<ValueT, OUT_DIM> result; \
    TENSOR_PRAGMA_UNROLL \
    for (int i = 0; i < OUT_DIM; ++i) result[i] = a[i % N] op b[i % M]; \
    return result; \
}

DEFINE_BROADCASTING_OP(+)
DEFINE_BROADCASTING_OP(-)
DEFINE_BROADCASTING_OP(*)
DEFINE_BROADCASTING_OP(/)

// Power operator requires special handling
template <typename ValueT, int N, int M>
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator^(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b)
{
    constexpr int OUT_DIM = (N > M ? N : M);
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) result[i] = tensor_pow(a[i % N], b[i % M]);
    return result;
}

#undef DEFINE_BROADCASTING_OP
