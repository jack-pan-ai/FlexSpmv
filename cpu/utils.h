#pragma once

#include <cmath>
#include <cstdio>

// Values-only Tensor
template <typename ValueT, int Dim> struct Tensor {
  typedef ValueT Value;
  ValueT values[Dim];

  // Constructor
  inline Tensor() {

    for (int i = 0; i < Dim; ++i)
      values[i] = 0.0f;
  }

  // Constructor with values
  inline Tensor(const ValueT *v) {

    for (int i = 0; i < Dim; ++i)
      values[i] = v[i];
  }

  // Constructor with iterator
  template <typename IteratorT> inline Tensor(IteratorT iter) {

    for (int i = 0; i < Dim; ++i)
      values[i] = static_cast<ValueT>(iter[i]);
  }

  // Compound assignment operators (more efficient, no temporaries)
  inline Tensor &operator+=(const Tensor &other) {

    for (int i = 0; i < Dim; ++i)
      values[i] += other.values[i];
    return *this;
  }

  inline Tensor &operator-=(const Tensor &other) {

    for (int i = 0; i < Dim; ++i)
      values[i] -= other.values[i];
    return *this;
  }

  inline Tensor &operator*=(const Tensor &other) {

    for (int i = 0; i < Dim; ++i)
      values[i] *= other.values[i];
    return *this;
  }

  inline Tensor &operator/=(const Tensor &other) {

    for (int i = 0; i < Dim; ++i)
      values[i] =
          (other.values[i] != 0.0) ? (values[i] / other.values[i]) : 0.0;
    return *this;
  }

  inline Tensor &operator+=(ValueT scalar) {

    for (int i = 0; i < Dim; ++i)
      values[i] += scalar;
    return *this;
  }

  inline Tensor &operator-=(ValueT scalar) {

    for (int i = 0; i < Dim; ++i)
      values[i] -= scalar;
    return *this;
  }

  inline Tensor &operator*=(ValueT scalar) {

    for (int i = 0; i < Dim; ++i)
      values[i] *= scalar;
    return *this;
  }

  inline Tensor &operator/=(ValueT scalar) {

    for (int i = 0; i < Dim; ++i)
      values[i] /= scalar;
    return *this;
  }

  // Binary operators - optimized to avoid temporary variables
  inline Tensor operator+(const Tensor &other) const {
    Tensor result = *this;
    result += other;
    return result;
  }

  inline Tensor operator-(const Tensor &other) const {
    Tensor result = *this;
    result -= other;
    return result;
  }

  inline Tensor operator*(const Tensor &other) const {
    Tensor result = *this;
    result *= other;
    return result;
  }

  inline Tensor operator/(const Tensor &other) const {
    Tensor result = *this;
    result /= other;
    return result;
  }

  inline Tensor operator+(ValueT scalar) const {
    Tensor result = *this;
    result += scalar;
    return result;
  }

  inline Tensor operator-(ValueT scalar) const {
    Tensor result = *this;
    result -= scalar;
    return result;
  }

  inline Tensor operator*(ValueT scalar) const {
    Tensor result = *this;
    result *= scalar;
    return result;
  }

  inline Tensor operator/(ValueT scalar) const {
    Tensor result = *this;
    result /= scalar;
    return result;
  }

  // Indexing operator for value access
  inline ValueT &operator[](int idx) { return values[idx]; }

  inline const ValueT &operator[](int idx) const { return values[idx]; }

  // L2 norm (Euclidean norm)
  inline ValueT l2Norm() const {
    ValueT sum = 0.0;

    for (int i = 0; i < Dim; ++i)
      sum += values[i] * values[i];
    return sqrt(sum);
  }

  // Add this to the Tensor struct after the constructors
  inline Tensor &operator=(const Tensor &other) {
    if (this != &other) { // Self-assignment check

      for (int i = 0; i < Dim; ++i)
        values[i] = other.values[i];
    }
    return *this;
  }

  inline void set(const ValueT *array) {

    for (int i = 0; i < Dim; ++i)
      values[i] = array[i];
  }

  inline void set(const ValueT val) {

    for (int i = 0; i < Dim; ++i)
      values[i] = val;
  }

  // Also add copy constructor
  inline Tensor(const Tensor &other) {

    for (int i = 0; i < Dim; ++i)
      values[i] = other.values[i];
  }

  // Assignment from a scalar (e.g., 0), e.g. Tensor a; a = 0.0;
  inline Tensor &operator=(const ValueT val) {

    for (int i = 0; i < Dim; ++i)
      values[i] = val;
    return *this;
  }

  // Print function
  void print() const {
    printf("Values: [");
    for (int i = 0; i < Dim; ++i) {
      printf("%.2f", values[i]);
      if (i < Dim - 1)
        printf(", ");
    }
    printf("]\n");
  }
};

// Non-member operator to enable scalar * Tensor (values-only)
template <typename ValueT, int Dim>
inline Tensor<ValueT, Dim> operator*(ValueT scalar,
                                     const Tensor<ValueT, Dim> &tensor) {
  return tensor * scalar; // Reuse the existing Tensor * scalar operator
}

// Non-member operator to enable scalar / Tensor (values-only)
template <typename ValueT, int Dim>
inline Tensor<ValueT, Dim> operator/(ValueT scalar,
                                     const Tensor<ValueT, Dim> &tensor) {
  Tensor<ValueT, Dim> result;

  for (int i = 0; i < Dim; ++i)
    result.values[i] =
        (tensor.values[i] != 0.0) ? (scalar / tensor.values[i]) : 0.0;
  return result;
}

// Macro to define tensor binary operators with broadcasting
#define DEFINE_TENSOR_BINARY_OP(op_symbol)                                     \
  template <typename ValueT, int N, int M>                                     \
  inline Tensor<ValueT, (N > M ? N : M)> operator op_symbol(                   \
      const Tensor<ValueT, N> &a, const Tensor<ValueT, M> &b) {                \
    constexpr int OUT_DIM = (N > M ? N : M);                                   \
    Tensor<ValueT, OUT_DIM> result;                                            \
    for (int i = 0; i < OUT_DIM; ++i) result[i] =                              \
        a[i % N] op_symbol b[i % M];                                           \
    return result;                                                             \
  }
         
// Define all binary operators
DEFINE_TENSOR_BINARY_OP(+)
DEFINE_TENSOR_BINARY_OP(-)
DEFINE_TENSOR_BINARY_OP(*)
DEFINE_TENSOR_BINARY_OP(/)