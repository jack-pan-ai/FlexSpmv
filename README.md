# Dense Matrix-based SpMV Implementation

This project extends CUB's merge-based SpMV implementation to work with a dense matrix and column indices instead of pre-computed sparse matrix values. The implementation allows for on-the-fly generation of sparse matrix values from a dense matrix.

## Overview

The traditional SpMV (Sparse Matrix-Vector Multiplication) operation computes:

```
y = A * x
```

where:
- `A` is a sparse matrix in CSR format
- `x` is a dense vector
- `y` is the resulting dense vector

Our implementation modifies this to compute:

```
y = D[colIdx] * x
```

where:
- `D` is a dense matrix 
- `colIdx` are indices that map sparse matrix elements to rows in the dense matrix
- The sparse matrix structure (row offsets and column indices) is preserved

This allows for dynamic generation of sparse matrix values without storing them explicitly.

## Directory Structure

```
merge_dense_spmv/
├── include/
│   ├── device_dense_spmv.cuh       - Public API for the Flexible Spmv implementation
│   ├── dispatch_dense_spmv.cuh     - Dispatch logic for kernel launching
│   └── agent_dense_spmv.cuh        - The core kernel implementation
├── src/
│   └── dense_spmv_test.cu          - Test program demonstrating usage
├── Makefile                        - Build configuration
└── README.md                       - This file
```

## Implementation Details

1. **FlexSpmvParams**: Extends CUB's SpmvParams to include a dense matrix and column indices.

2. **AgentFlexSpmv**: Extends AgentSpmv to compute sparse matrix values on-the-fly from a dense matrix.

3. **DeviceFlexSpmv**: Provides the public API for using the Flexible Spmv implementation.

The key difference in our implementation is in the `ConsumeTile` method of `AgentFlexSpmv`, where instead of directly using pre-computed sparse matrix values, we:

1. Use the column indices of the sparse matrix as usual
2. Use additional column indices to determine which row of the dense matrix to use
3. Compute each sparse matrix value by fetching from the dense matrix
4. Continue with the standard merge-based SpMV algorithm

## Usage

Here's a simple example of how to use the implementation:

```cpp
#include <device_dense_spmv.cuh>

// Allocate arrays and initialize data
// ...

// Allocate temporary storage
size_t temp_storage_bytes = 0;
void* d_temp_storage = nullptr;

// Get required storage size
DeviceFlexSpmv::CsrMV(
    d_temp_storage, temp_storage_bytes,
    d_dense_matrix, d_column_indices_A, dense_matrix_width,
    d_row_offsets, d_column_indices,
    d_vector_x, d_vector_y,
    num_rows, num_cols, num_nonzeros);

// Allocate storage
cudaMalloc(&d_temp_storage, temp_storage_bytes);

// Execute SpMV
DeviceFlexSpmv::CsrMV(
    d_temp_storage, temp_storage_bytes,
    d_dense_matrix, d_column_indices_A, dense_matrix_width,
    d_row_offsets, d_column_indices,
    d_vector_x, d_vector_y,
    num_rows, num_cols, num_nonzeros);
```

## Building and Running the Test

To build the test program:

```bash
cd merge_dense_spmv
make
```

To run the test:

```bash
./dense_spmv_test
```

The test will generate a random sparse matrix structure and dense matrix, compute the SpMV result, and verify it against a CPU reference implementation.

## Dependencies

- CUDA Toolkit (tested with CUDA 11.0+)
- CUB library (part of CUDA Toolkit since CUDA 11.0)

## Customization

You can modify the implementation for different use cases:

- Change the way values are computed from the dense matrix
- Extend the parameters to support more complex computations
- Optimize for specific matrix patterns or architectures

## Performance Considerations

- The on-the-fly computation may be slower than using pre-computed values for simple cases
- This approach is beneficial when:
  - The dense matrix is frequently updated
  - The sparse pattern remains constant
  - Memory usage needs to be reduced
  - Computation can be fused with other operations 