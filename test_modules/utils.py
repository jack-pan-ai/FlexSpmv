import os
import numpy as np
from scipy import sparse


def save_to_coo_format(
        spm_k,
        spm_l,
        row_end_offset,
        col_indices_i,
        col_indices_j,
        vector_x,
        output_dir="saved_data"):
    """
    Save the sparse matrix and vectors in COO Matrix Market format.

    Args:
        spm_k: Spring constant values (N, 1)
        spm_l: Spring length values (N, 1)
        row_end_offset: Row end offsets (M + 1)
        col_indices_i: Column indices for i (N)
        col_indices_j: Column indices for j (N)
        vector_x: Position vectors (N, D)
        output_dir: Directory to save the files
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert to CPU and numpy arrays
    spm_k_np = spm_k.cpu().numpy()
    spm_l_np = spm_l.cpu().numpy()
    row_end_offset_np = row_end_offset.cpu().numpy()
    col_indices_i_np = col_indices_i.cpu().numpy()
    col_indices_j_np = col_indices_j.cpu().numpy()
    vector_x_np = vector_x.cpu().numpy()

    # Generate row indices from row_end_offset
    num_rows = len(row_end_offset_np) - 1
    num_cols = vector_x_np.shape[0]
    num_nnz = len(col_indices_i_np)
    row_indices = np.zeros(num_nnz, dtype=np.int32)

    for i in range(num_rows):
        start = row_end_offset_np[i]
        end = row_end_offset_np[i + 1]
        row_indices[start:end] = i

    # Create COO matrices
    # 1. Matrix for spring constants (spm_k)
    coo_k = sparse.coo_matrix(
        (spm_k_np.flatten(), (row_indices, col_indices_i_np)), shape=(
            num_rows, num_cols))
    # print(coo_k)
    # print(col_indices_i_np)
    # Sort the COO matrix coo_k by column index
    # Get the permutation that sorts the column indices
    # sort_perm = np.argsort(coo_k.col, kind='stable')
    # coo_k = sparse.coo_matrix(
    #     (coo_k.data[sort_perm], (coo_k.row[sort_perm], coo_k.col[sort_perm])),
    #     shape=coo_k.shape
    # )

    # 2. Matrix for spring lengths (spm_l)
    coo_l = sparse.coo_matrix(
        (spm_l_np.flatten(), (row_indices, col_indices_i_np)), shape=(
            num_rows, num_cols))

    # Sort the COO matrix coo_l by column index
    # sort_perm = np.argsort(coo_l.col, kind='stable')
    # coo_l = sparse.coo_matrix(
    #     (coo_l.data[sort_perm], (coo_l.row[sort_perm], coo_l.col[sort_perm])),
    #     shape=coo_l.shape
    # )

    # Custom function to write Matrix Market files with decimal format
    def write_mtx_file(filename, coo_matrix, comment=""):
        with open(filename, 'w') as f:
            # Write header
            f.write("%%MatrixMarket matrix coordinate real general\n")
            if comment:
                f.write(f"%{comment}\n")

            # Write dimensions and number of non-zeros
            f.write(
                f"{coo_matrix.shape[0]} {coo_matrix.shape[1]} {coo_matrix.nnz}\n")

            # Write data in coordinate format (1-indexed for Matrix Market)
            for i, j, v in zip(
                    coo_matrix.row, coo_matrix.col, coo_matrix.data):
                # Use 1-indexed format for Matrix Market and format value as
                # decimal
                f.write(f"{i + 1} {j + 1} {v:.6f}\n")

    # Save the matrices in Matrix Market format with decimal values
    write_mtx_file(os.path.join(output_dir, "spm_k_matrix.mtx"),
                   coo_k, comment="Spring constant matrix (k)")
    write_mtx_file(os.path.join(output_dir, "spm_l_matrix.mtx"),
                   coo_l, comment="Spring length matrix (l)")

    # Save vectors in a simple format that can be read in C++
    # For vector_x, save as a text file with shape information in the header
    with open(os.path.join(output_dir, "vector_x.txt"), 'w') as f:
        # Header: rows cols
        f.write(f"{vector_x_np.shape[0]} {vector_x_np.shape[1]}\n")
        for i in range(vector_x_np.shape[0]):
            for j in range(vector_x_np.shape[1]):
                f.write(f"{vector_x_np[i, j]:.6f} ")
            f.write("\n")
    # Also save row indices, col_indices_i, and col_indices_j in a format
    # readable by C++
    with open(os.path.join(output_dir, "indices.txt"), 'w') as f:
        # Header: num_rows num_cols num_nnz
        f.write(f"{num_rows} {num_cols} {num_nnz}\n")
        f.write("# Row indices\n")
        for i in range(num_nnz):
            f.write(f"{row_indices[i]} ")
        f.write("\n# Column indices i\n")
        for i in range(num_nnz):
            f.write(f"{col_indices_i_np[i]} ")
        f.write("\n# Column indices j\n")
        for i in range(num_nnz):
            f.write(f"{col_indices_j_np[i]} ")
        f.write("\n# Row end offsets\n")
        for i in range(num_rows + 1):
            f.write(f"{row_end_offset_np[i]} ")

    print(f"Data saved successfully to {output_dir}/")
    print(f"Files saved:")
    print(f"  - {output_dir}/spm_k_matrix.mtx (Matrix Market format)")
    print(f"  - {output_dir}/spm_l_matrix.mtx (Matrix Market format)")
    print(f"  - {output_dir}/vector_x.txt (Text format)")
    print(f"  - {output_dir}/indices.txt (Text format with row/column indices)")
