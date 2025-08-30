import torch
import os
from torch.utils.cpp_extension import load

# JIT compile the extension
flex_spmv = load(
    name="flex_spmv",
    sources=[
        os.path.join(os.path.dirname(__file__), "flex_spmv_torch.cu"),
        os.path.join(os.path.dirname(__file__), "flex_spmv_cuda.cu")
    ],
    extra_include_paths=[os.path.join(os.path.dirname(__file__), "..", "include")],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-arch=sm_70"
    ],
    verbose=True
)

def flex_spmv_op(spm_k, spm_l, row_offsets, col_indices_i, col_indices_j, vector_x, output_y_reducer_i, output_y_reducer_j):
    """
    Wrapper function for the FlexSpmv operation
    
    Args:
        spm_k (torch.Tensor): Tensor containing the spring constants
        spm_l (torch.Tensor): Tensor containing the rest lengths
        row_offsets (torch.Tensor): CSR row offsets tensor
        col_indices_i (torch.Tensor): Column indices for the first point in each spring
        col_indices_j (torch.Tensor): Column indices for the second point in each spring
        vector_x (torch.Tensor): Input vector containing point positions
        output_y_reducer_i (torch.Tensor): Output vector for storing forces
        output_y_reducer_j (torch.Tensor): Output vector for storing forces
        
    Returns:
        torch.Tensor: Output vector containing the forces on each point
    """
    return flex_spmv.flex_spmv(spm_k, spm_l, row_offsets, col_indices_i, col_indices_j, vector_x, output_y_reducer_i, output_y_reducer_j)