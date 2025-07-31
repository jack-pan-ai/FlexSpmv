# FlexSpmv PyTorch Extension

This repository contains a PyTorch C++/CUDA extension for the FlexSpmv sparse matrix-vector multiplication algorithm, specifically designed for spring-mass systems.

## Features

- High-performance sparse matrix-vector multiplication on GPU
- Support for both float and double precision
- Seamless integration with PyTorch tensors
- Optimized for spring-mass system simulations

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- CUDA Toolkit 10.2+
- C++14 compatible compiler

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/FlexSpmv.git
cd FlexSpmv

# Install the package
pip install -e .
```

### Option 2: JIT compilation

You can also use the extension with JIT compilation:

```python
import torch.utils.cpp_extension

flex_spmv = torch.utils.cpp_extension.load(
    name="flex_spmv",
    sources=["src/flex_spmv_torch.cpp", "src/flex_spmv_cuda.cu"],
    extra_include_paths=["include"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda"
    ],
    verbose=True
)
```

## Usage

Here's a simple example of how to use the FlexSpmv extension:

```python
import torch
import flex_spmv

# Create input tensors on GPU
values = torch.rand(1000, dtype=torch.float32, device='cuda')
row_offsets = torch.zeros(501, dtype=torch.int32, device='cuda')
col_indices_i = torch.zeros(1000, dtype=torch.int32, device='cuda')
col_indices_j = torch.zeros(1000, dtype=torch.int32, device='cuda')
x = torch.rand(1000, dtype=torch.float32, device='cuda')

# Initialize CSR format (example)
for i in range(1, 501):
    row_offsets[i] = i * 2

# Run the SpMV operation
y = flex_spmv.flex_spmv(values, row_offsets, col_indices_i, col_indices_j, x)
```

See `flex_spmv_example.py` for a more complete example.

## API Reference

### `flex_spmv.flex_spmv(values, row_offsets, col_indices_i, col_indices_j, x)`

Performs sparse matrix-vector multiplication for spring-mass systems.

**Parameters:**

- `values` (torch.Tensor): Tensor containing the values for the spring-mass system (k_ij and l_ij)
- `row_offsets` (torch.Tensor): CSR row offsets tensor (int32)
- `col_indices_i` (torch.Tensor): Column indices for the first point in each spring (int32)
- `col_indices_j` (torch.Tensor): Column indices for the second point in each spring (int32)
- `x` (torch.Tensor): Input vector containing point positions

**Returns:**

- `y` (torch.Tensor): Output vector containing the forces on each point

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
