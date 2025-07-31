import torch
import numpy as np
import time
import random

# Import our extension
import flex_spmv

def create_spring_mass_example(nnz = 1000, num_rows=100, num_cols=100, x_dim=2):
    """
    Create a simple spring-mass system example
    """
    items_per_row = nnz // num_rows

    # Create random values for the spring-mass system
    spm_k = torch.rand(nnz, dtype=torch.float32, device='cuda')  # k_ij values
    spm_l = torch.rand(nnz, dtype=torch.float32, device='cuda')  # l_ij values
    
    # Create row offsets (CSR format)
    row_offsets = torch.zeros(num_rows + 1, dtype=torch.int32, device='cuda')
    for i in range(1, num_rows + 1):
        row_offsets[i] = i * items_per_row

    # Create column indices for i and j
    col_indices_i = torch.zeros(nnz, dtype=torch.int32, device='cuda')
    col_indices_j = torch.zeros(nnz, dtype=torch.int32, device='cuda')
    
    for i in range(nnz):
        col_indices_i[i] = random.randint(0, num_cols - 1)
        col_indices_j[i] = random.randint(0, num_cols - 1)
    
    # Create input vector x (positions)
    vector_x = torch.rand(num_cols * x_dim, dtype=torch.float32, device='cuda')  # 2D positions
    vector_y = torch.zeros(num_rows * x_dim, dtype=torch.float32, device='cuda')  # 2D positions
    return spm_k, spm_l, row_offsets, col_indices_i, col_indices_j, vector_x, vector_y

def main():
    # Create example data
    print("Creating example data...")
    spm_k, spm_l, row_offsets, col_indices_i, col_indices_j, vector_x, vector_y = create_spring_mass_example(10000, 1000, 1000, 2)
    
    # Print shapes
    print(f"spm_k shape: {spm_k.shape}")
    print(f"spm_l shape: {spm_l.shape}")
    print(f"row_offsets shape: {row_offsets.shape}")
    print(f"col_indices_i shape: {col_indices_i.shape}")
    print(f"col_indices_j shape: {col_indices_j.shape}")
    print(f"vector_x shape: {vector_x.shape}")
    print(f"vector_y shape: {vector_y.shape}")

    # Run the SpMV operation
    print("\nRunning FlexSpmv...")
    start_time = time.time()
    # flex_spmv.flex_spmv(spm_k, spm_l, row_offsets, col_indices_i, col_indices_j, vector_x, vector_y)
    from src.flex_spmv_jit import flex_spmv_op
    flex_spmv_op(spm_k, spm_l, row_offsets, col_indices_i, col_indices_j, vector_x, vector_y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"FlexSpmv completed in {(end_time - start_time) * 1000:.2f} ms")
    print(f"Output shape: {vector_y.shape}")
    print(f"First few elements of output: {vector_y[:10]}")

if __name__ == "__main__":
    main() 