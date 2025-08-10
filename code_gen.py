import torch
import torch.fx
import os
import numpy as np
import string

import easier as esr
from easier.core.jit import EasierTracer
import scipy.sparse
from scipy.io import mmwrite

from test_modules.models import AggregatorTest, SpringMassSystem
from test_modules.models import AggregatorTest_calculation, SpringMassSystem_calculation

# Part 2: Trace the model with torch.fx
def trace_model(model):
    tracer = EasierTracer()
    traced_model = tracer.trace(model)
    print("FX Graph:")
    info_nodes = {
        "vector_x": {"shape": (2,)},
        "vector_x1": {"shape": (2,)},
        "sub": {"shape": (2,)},
        "norm": {"shape": (1,)},
        "truediv": {"shape": (2,)},
        "spm_l": {"shape": (1,)},
        "spm_k": {"shape": (1,)},
        "spm_l_1": {"shape": (1,)},
        "spm_k_1": {"shape": (1,)},
        "sub_1": {"shape": (1,)},
        "mul_1": {"shape": (2,)},
        "mul": {"shape": (1,)},
        "mul_2": {"shape": (1,)},
        "mul_3": {"shape": (1,)},
        "mul_4": {"shape": (2,)},
        "selector_i": {"shape": (2,)},
        "selector_j": {"shape": (2,)},
        "reducer_i": {"shape": (2,)},
        "reducer_j": {"shape": (2,)},
        "sum_1": {"shape": (2,)},
        "sum_2": {"shape": (2,)}
    }
    traced_model.print_tabular()    
    return traced_model, info_nodes

# Part 3: Generate CUDA code from the graph
def generate_cuda_code_from_graph(traced_model, info_nodes):
    # Create directory for generated code
    os.makedirs("cuda_code", exist_ok=True)
    
    # Analyze the graph and extract operations
    inputs = []
    input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    output_var = None

#  -------------------------------------------------------------
#  Analyze the graph and extract operations, 1) input and output, 2) selector, 3) map
#  -------------------------------------------------------------

    #  -------------------------------------------------------------
    #  obtain the input and output
    #  -------------------------------------------------------------
    for i in range(len(traced_model.nodes)):
        node = list(traced_model.nodes)[i - 1]
        if node.op == 'get_attr':
            if node.target not in input_keys:
                inputs.append(
                    {
                        "name": node.name,
                        "target": node.target,
                        "dtype": "scalar_t"
                    }
                )
                input_keys.add(node.target)
        elif node.op == 'call_module':
            if 'selector' in str(node.target):
                target_name = 'selector'
                inputs.append(
                    {
                        "name": node.name + "_idx",
                        "target": node.target,
                        "dtype": "int"
                    }
                )
            elif 'reducer' in str(node.target):
                target_name = 'reducer'
                inputs.append(
                    {
                        "name": "output_y_" + node.name + "_ptr",
                        "target": node.target,
                        "dtype": "ValueT"
                    }
                )
        elif node.op == 'call_function':
            # this is used for the aggregator
            if 'sum' in str(node.target):
                inputs.append(
                    {
                        "name": "output_y_" + node.name + "_ptr",
                        "target": node.name,
                        "dtype": "ValueT"
                    }
                )
        elif node.op == 'output':
            output_var = node.args[0]
    
    #  -------------------------------------------------------------
    #  obtain the selector in intermediate register
    #  -------------------------------------------------------------
    for i in range(len(traced_model.nodes)):
        # current node and forward node 
        node_current = list(traced_model.nodes)[i]
        if i < len(traced_model.nodes) - 1:
            node_forward = list(traced_model.nodes)[i + 1]
        else:
            node_forward = None
        if node_current.op == 'get_attr':
            if node_forward.op == 'call_module' and 'selector' in str(node_forward.target):
                selector_register.append(
                    {
                        "name": node_current.name,
                        "target": node_current.target,
                        "dtype": "TensorT",
                        "selector": 1, # represents the selector is used in the forward pass
                        "selector_name": node_forward.target
                    }
                )
            else:
                selector_register.append(
                    {
                        "name": node_current.name,
                        "target": node_current.target,
                        "dtype": "ValueT",
                        "selector": 0, # represents the selector is not used in the forward pass
                        "selector_name": None
                    }
                )

    # # debug print
    # for inter in selector_register:
    #     print(f"Selector register: {inter}")

    #  -------------------------------------------------------------
    #  obtain the map operations
    #  -------------------------------------------------------------
    for i in range(len(traced_model.nodes)):
        node = list(traced_model.nodes)[i]
        if node.op == 'call_module':
            if 'selector' in str(node.target):
                continue
            elif 'reducer' in str(node.target):
                target_name = 'reducer'
                map_operations.append({
                    'output': node.name,
                    'op': target_name,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                    'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                    'shape': info_nodes[node.name]['shape']
                })
            else:
                map_operations.append({
                    'output': node.name,
                    'op': node.target,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                    'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                    'shape': info_nodes[node.name]['shape']
                })
        elif node.op == 'call_function' or node.op == 'call_method':
            # this is used for the aggregator
            target_name = str(node.target).split('.')[-1]
            if 'add' in str(node.target):
                target_name = 'add'
            elif 'mul' in str(node.target):
                target_name = 'mul'
            elif 'sub' in str(node.target):
                target_name = 'sub'
            elif 'norm' in str(node.target):
                target_name = 'norm'
            elif 'truediv' in str(node.target):
                target_name = 'truediv'
            elif 'norm' in str(node.target):
                target_name = 'norm'
            elif 'sum' in str(node.target):
                target_name = 'sum'
            else:
                # error
                raise ValueError(f"Operation {node.target} not supported")
            map_operations.append({
                'output': node.name,
                'op': target_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                'shape': info_nodes[node.name]['shape']
            })
        elif node.op == 'output':
            output_var = node.args[0]
    
    #  -------------------------------------------------------------
    #  obtain the reducer and aggregator operations
    #  -------------------------------------------------------------
    for i in range(len(traced_model.nodes)):
        node = list(traced_model.nodes)[i]
        if 'reducer' in str(node.target):
            op_name = 'reducer'
            reducer_operations.append({
                'output': node.name,
                'op': op_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None]
            })
        elif 'sum' in str(node.target):
            op_name = 'sum'
            aggregator_operations.append({
                'output': node.name,
                'op': op_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None]
            })

    # # debug print
    # for inp in inputs:
    #     print(f"Input: {inp}")
    # for inter in selector_register:
    #     print(f"Selector register: {inter}")
    # for op in map_operations:
    #     print(f"Operation: {op}")
    # print(f"Output: {output_var}")

#  -------------------------------------------------------------
#  Generate the CUDA code
#  -------------------------------------------------------------

    #  -------------------------------------------------------------
    #  Generate the input code
    #  -------------------------------------------------------------
    input_declarations_code = []
    input_init_code = []
    input_declarations_utils_code = []
    for inp in inputs:
        if inp['dtype'] == 'int':
            # column indices
            input_declarations_utils_code.append(f"  OffsetT *{inp['target'] + '_ptr'}; \n")
            input_declarations_code.append(f"  ColumnIndicesIteratorT {inp['target'] + '_ptr'}; \n")
            input_init_code.append(f"    {inp['target'] + '_ptr'}(spmv_params.{inp['target'] + '_ptr'}), \n")
        else:
            # vector value and spm nnz or aggregator
            if 'reducer' in str(inp['target']) or 'sum' in str(inp['target']):
                input_declarations_utils_code.append(f"  ValueT *{"output_y_" + inp['target'] + "_ptr"}; \n")
            else:
                input_declarations_utils_code.append(f"  ValueT *{inp['target'] + '_ptr'}; \n")
                input_declarations_code.append(f"  VectorValueIteratorT {inp['target'] + '_ptr'}; \n")
                input_init_code.append(f"    {inp['target'] + '_ptr'}(spmv_params.{inp['target'] + '_ptr'}), \n")
    
    # # debug print
    # for inp in input_declarations_utils_code:
    #     print(f"Input declarations utils: {inp}")
    # for inp in input_declarations_code:
    #     print(f"Input declarations: {inp}")
    # for inp in input_init_code:
    #     print(f"Input init code: {inp}")
    
    #  -------------------------------------------------------------
    #  Generate the selector register code
    #  -------------------------------------------------------------
    selector_code = []
    for inter in selector_register:
        if inter['selector'] == 1:
            # load the selector register
            selector_code.append(f"    ColumnIndicesIteratorT {inter['name'] + '_ptr_current'} = {inter['selector_name'] + '_ptr'} + tile_start_coord.y + nonzero_idx; \n")
            selector_code.append(f"    TensorT {inter['selector_name']}; \n")
            selector_code.append(f"    #pragma unroll\n    for (int i = 0; i < DIM_INPUT_VECTOR_X; i++) \n    {{ \n        {inter['selector_name']}.values[i] = {inter['target'] + '_ptr'}[*{inter['name'] + '_ptr_current'} * DIM_INPUT_VECTOR_X + i]; \n    }} \n")
        else:
            # load the sparse matrix register
            selector_code.append(f"    int {inter['name'] + '_idx'} = tile_start_coord.y + nonzero_idx; \n")
            selector_code.append(f"    ValueT {inter['name']} = {inter['target'] + '_ptr'}[{inter['name'] + '_idx'}]; \n")
    
    # # debug print
    # for inter in selector_code:
    #     print(f"Selector code: {inter}")
    
    #  -------------------------------------------------------------
    #  Generate the CUDA kernel code (map operations)
    #  -------------------------------------------------------------
    map_code = []
    for op in map_operations:
        op_name = op['op']
        shape = op['shape']
        print(f"Operation: {op_name}")
        if op_name == 'add':
            map_code.append(f"    TensorT {op['output']} = {op['args'][0]} + {op['args'][1]}; \n")
        elif op_name == 'mul':
            if shape == (2,):
                map_code.append(f"    TensorT {op['output']} = {op['args'][0]} * {op['args'][1]}; \n")
            elif shape == (1,):
                map_code.append(f"    ValueT {op['output']} = {op['args'][0]} * {op['args'][1]}; \n")
            else:
                raise ValueError(f"Shape {shape} not supported")
        elif op_name == 'sub':
            if shape == (2,):
                map_code.append(f"    TensorT {op['output']} = {op['args'][0]} - {op['args'][1]}; \n")
            elif shape == (1,):
                map_code.append(f"    ValueT {op['output']} = {op['args'][0]} - {op['args'][1]}; \n")
            else:
                raise ValueError(f"Shape {shape} not supported")
        elif op_name == 'norm':
            map_code.append(f"    ValueT {op['output']} = {op['args'][0]}.l2Norm(); \n")
        elif op_name == 'truediv':
            map_code.append(f"    TensorT {op['output']} = {op['args'][0]} / {op['args'][1]}; \n")
        elif op_name == 'reducer':
            # add the register results to the SMEM
            map_code.append(f"    #pragma unroll\n")
            map_code.append(f"    for (int i = 0; i < DIM_INPUT_VECTOR_X; i++) \n")
            map_code.append(f"      s_tile_value_{op['output']}[nonzero_idx + i * TILE_ITEMS] = {op['args'][0]}.values[i]; \n")
        elif op_name == 'sum':
            map_code.append(f"    aggregator_reg_{op['output']} = aggregator_reg_{op['output']} + {op['args'][0]}; \n")
        else:
            # error
            raise ValueError(f"Operation {op_name} not supported")

    # Debug print to check kernel operations
    # print("Generated kernel operations:")
    # for op in map_code:
    #     print(op)

    #  -------------------------------------------------------------
    #  Generate the aggregator code
    #  -------------------------------------------------------------
    aggregator_code = []
    aggregator_reg_definitions = []
    # the shared memory can be reused for multiple blockReduce
    if aggregator_operations != []:
        aggregator_code.append(f"   // Allocate shared memory for BlockReduceT \n")
        aggregator_code.append(f"   __shared__ typename BlockReduceT::TempStorage temp_storage; \n")

    for op in aggregator_operations:
        aggregator_reg_definitions.append(f"   // each aggregator need a register to store the non-zero values \n")
        aggregator_reg_definitions.append(f"   TensorT aggregator_reg_{op['output']}; \n")
        if op['op'] == 'sum':
            aggregator_code.append(f"   // blockReduce \n")
            aggregator_code.append(f"   TensorT {op['output']} = BlockReduceT(temp_storage).Sum(aggregator_reg_{op['output']}); \n")
            aggregator_code.append(f"   if (threadIdx.x == 0) \n")
            aggregator_code.append(f"   {{ \n")
            aggregator_code.append(f"     #pragma unroll\n")
            aggregator_code.append(f"     for (int i = 0; i < DIM_INPUT_VECTOR_X; i++) \n")
            aggregator_code.append(f"      atomicAdd(spmv_params.output_y_{op['output']}_ptr + i, {op['output']}.values[i]); \n")
            aggregator_code.append(f"     }} \n")

    # debug print
    # for op in aggregator_reg_definitions:
    #     print(f"Aggregator reg definitions: {op}")
    # for op in aggregator_code:
    #     print(f"Aggregator code: {op}")

    #  -------------------------------------------------------------
    #  Generate the reducer code
    #  -------------------------------------------------------------
    reducer_smem_definitions = []
    reducer_code = []
    for op in reducer_operations:
        reducer_smem_definitions.append(f"   // each reducer need SMEM to store the non-zero values \n")
        reducer_smem_definitions.append(f"   __shared__ ValueT s_tile_value_{op['output']}[TILE_ITEMS * DIM_INPUT_VECTOR_X]; \n")
        reducer_code.append(f"   reduce( \n")
        reducer_code.append(f"                s_tile_value_{op['output']},          ///< [in, code gen] Shared memory array of non-zero values for the merge tile \n")
        reducer_code.append(f"                s_tile_row_end_offsets,         ///< [in, code gen] Shared memory array of row end offsets for the merge tile \n")
        reducer_code.append(f"                tile_start_coord,               ///< [in] Starting coordinate of the merge tile \n")
        reducer_code.append(f"                tile_end_coord,                 ///< [in] Ending coordinate of the merge tile \n")
        reducer_code.append(f"                tile_num_rows,                  ///< [in] Number of rows in the merge tile \n")
        reducer_code.append(f"                tile_num_nonzeros,               ///< [in] Number of non-zeros in the merge tile \n")
        reducer_code.append(f"                spmv_params.output_y_{op['output']}_ptr                  ///< [out] Output vector y \n")
        reducer_code.append(f"            ); \n")
        reducer_code.append(f"   CTA_SYNC(); \n")
    
    # # debug print
    # for op in reducer_smem_definitions:
    #     print(f"Reducer SMEM definitions: {op}")
    # for op in reducer_code:
    #     print(f"Reducer code: {op}")
        
    #  -------------------------------------------------------------
    #  Create directories for generated code if they don't exist
    #  -------------------------------------------------------------
    os.makedirs("include", exist_ok=True)
    
    #  -------------------------------------------------------------
    #  Read template files
    #  -------------------------------------------------------------
    with open("cuda_template/merged_agent_spmv_template.cuh", "r") as f:
        kernel_agent_template = f.read()
    
    with open("cuda_template/merged_utils_template.cuh", "r") as f:
        utils_template = f.read()
    
    #  -------------------------------------------------------------
    #  Apply templates using string.Template
    #  -------------------------------------------------------------
    input_declarations_str = ''.join(input_declarations_code)
    input_init_str = ''.join(input_init_code)
    selector_str = ''.join(selector_code)
    map_str = ''.join(map_code)
    reducer_smem_definitions_str = ''.join(reducer_smem_definitions)
    reducer_code_str = ''.join(reducer_code)
    aggregator_reg_definitions_str = ''.join(aggregator_reg_definitions)
    aggregator_code_str = ''.join(aggregator_code)
    agent_kernel_code = string.Template(kernel_agent_template).substitute(
        input_declarations_code=input_declarations_str,
        input_init_code=input_init_str,
        selector_code=selector_str,
        map_code=map_str,
        reducer_smem_definitions=reducer_smem_definitions_str,
        reducer_code=reducer_code_str,
        aggregator_reg_definitions=aggregator_reg_definitions_str,
        aggregator_code=aggregator_code_str
    )
    
    #  -------------------------------------------------------------
    #  Join the list of input declarations into a single string
    #  -------------------------------------------------------------
    input_declarations_utils_str = ''.join(input_declarations_utils_code)
    
    utils_code = string.Template(utils_template).substitute(
        input_declarations_utils_code=input_declarations_utils_str
    )
    
    #  -------------------------------------------------------------
    #  Write files
    #  -------------------------------------------------------------
    with open("include/merged_agent_spmv.cuh", "w") as f:
        f.write(agent_kernel_code)
    
    with open("include/merged_utils.cuh", "w") as f:
        f.write(utils_code)
    
    #  -------------------------------------------------------------
    print("CUDA code generated successfully!")
    #  -------------------------------------------------------------

def save_to_coo_format(spm_k, spm_l, row_end_offset, col_indices_i, col_indices_j, vector_x, output_dir="saved_data"):
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
    import os
    import numpy as np
    from scipy import sparse
    
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
        end = row_end_offset_np[i+1]
        row_indices[start:end] = i
    
    # Create COO matrices
    # 1. Matrix for spring constants (spm_k)
    coo_k = sparse.coo_matrix((spm_k_np.flatten(), (row_indices, col_indices_i_np)), 
                             shape=(num_rows, num_cols))
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
    coo_l = sparse.coo_matrix((spm_l_np.flatten(), (row_indices, col_indices_i_np)), 
                             shape=(num_rows, num_cols))
    
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
            f.write(f"{coo_matrix.shape[0]} {coo_matrix.shape[1]} {coo_matrix.nnz}\n")
            
            # Write data in coordinate format (1-indexed for Matrix Market)
            for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                # Use 1-indexed format for Matrix Market and format value as decimal
                f.write(f"{i+1} {j+1} {v:.6f}\n")
    
    # Save the matrices in Matrix Market format with decimal values
    write_mtx_file(os.path.join(output_dir, "spm_k_matrix.mtx"), coo_k, comment="Spring constant matrix (k)")
    write_mtx_file(os.path.join(output_dir, "spm_l_matrix.mtx"), coo_l, comment="Spring length matrix (l)")
    
    # Save vectors in a simple format that can be read in C++
    # For vector_x, save as a text file with shape information in the header
    with open(os.path.join(output_dir, "vector_x.txt"), 'w') as f:
        f.write(f"{vector_x_np.shape[0]} {vector_x_np.shape[1]}\n")  # Header: rows cols
        for i in range(vector_x_np.shape[0]):
            for j in range(vector_x_np.shape[1]):
                f.write(f"{vector_x_np[i, j]:.6f} ")
            f.write("\n")
    # Also save row indices, col_indices_i, and col_indices_j in a format readable by C++
    with open(os.path.join(output_dir, "indices.txt"), 'w') as f:
        f.write(f"{num_rows} {num_cols} {num_nnz}\n")  # Header: num_rows num_cols num_nnz
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

# Main function to run the entire pipeline
def main():

    torch.manual_seed(52)
    # data settings
    num_nnz = 20213
    num_cols = 1134
    num_rows = 542
    dim_x = 2
    spm_k = torch.rand(num_nnz, 1, device="cuda", dtype=torch.float32)  # (N, 1)
    # spm_l = torch.rand(num_nnz, 1, device="cuda", dtype=torch.float32)  # (N, 1)
    spm_l = spm_k.clone()
    vector_x = torch.rand(num_cols, dim_x, device="cuda", dtype=torch.float32)  # (N, D)
    output_y_reducer_i = torch.zeros(num_rows, dim_x, device="cuda", dtype=torch.float32)  # (M, D)
    output_y_reducer_j = torch.zeros(num_rows, dim_x, device="cuda", dtype=torch.float32)  # (M, D)
    row_end_offset = torch.sort(torch.randint(0, num_nnz + 1, (num_rows - 1, ), device="cuda", dtype=torch.int32))[0] 
    row_end_offset = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), row_end_offset])
    row_end_offset = torch.cat([row_end_offset, torch.tensor([num_nnz], device="cuda", dtype=torch.int32)]) # (M + 1)
    # print(f"row_end_offset: {row_end_offset}")
    # Ensure that i and j are different for each point
    # Generate col_indices_i and col_indices_j such that for each row, the number of nonzeros is determined by row_end_offset,
    # and for each nonzero, col_indices_j != col_indices_i.
    col_indices_i = torch.empty(num_nnz, dtype=torch.int32, device="cuda")
    col_indices_j = torch.empty(num_nnz, dtype=torch.int32, device="cuda")
    for row in range(num_rows):
        start = row_end_offset[row].item()
        end = row_end_offset[row + 1].item()
        nnz_in_row = end - start
        if nnz_in_row <= 0:
            continue
        # For each nonzero in this row, generate i and j
        # i_vals must be sampled without replacement
        if nnz_in_row > num_cols:
            raise ValueError(f"Cannot assign {nnz_in_row} unique i indices in row {row} with only {num_cols} columns.")
        # Sample i_vals without replacement
        i_vals = torch.randperm(num_cols, device="cuda")[:nnz_in_row].tolist()
        j_vals = []
        for i_val in i_vals:
            # pick a random j != i
            choices = torch.arange(num_cols, device="cuda")
            choices = choices[choices != i_val]
            j_val = choices[torch.randint(0, len(choices), (1,), device="cuda")].item()
            j_vals.append(j_val)
        # Sort by i, then by j
        sorted_pairs = sorted(zip(i_vals, j_vals), key=lambda x: (x[0], x[1]))
        for local_idx, (i_val, j_val) in enumerate(sorted_pairs):
            idx = start + local_idx
            col_indices_i[idx] = i_val
            col_indices_j[idx] = j_val
    
    # Save the sparse matrix and vectors in COO format
    save_to_coo_format(spm_k, spm_l, row_end_offset, col_indices_i, col_indices_j, vector_x)
    
    # # pytorch model for code generation
    # model = SpringMassSystem(
    #     spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows
    # )

    # # python model for verification
    # model_calculation = SpringMassSystem_calculation(
    #     spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows
    # )

    model_aggregator = AggregatorTest(
        spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows
    )

    model_aggregator_calculation = AggregatorTest_calculation(
        spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows
    )

    print("Step 1: Tracing model with torch.fx")
    traced_model, info_nodes = trace_model(model_aggregator)

    results_gold_1, results_gold_2 = model_aggregator_calculation.forward()
    print(f"Results gold 1: {results_gold_1}")
    print(f"Results gold 2: {results_gold_2}")
    
    # # run the model
    # results_gold_1, results_gold_2 = model_calculation.forward()
    # # results_gold_1 = model_calculation.forward()

    print("\nStep 2: Generating CUDA code from the graph")
    generate_cuda_code_from_graph(traced_model, info_nodes)

#     # check the results
#     # print(f"Results gold 1: {results_gold_1}")
#     # print(f"Results gold 2: {results_gold_2}")
#     results_gold_1 = results_gold_1.cpu()
#     results_gold_2 = results_gold_2.cpu()
    
# # Import our extension
#     import flex_spmv
#     comp_results_1, comp_results_2 = flex_spmv.flex_spmv(
#         spm_k, spm_l, row_end_offset, 
#         col_indices_i, col_indices_j, 
#         vector_x, 
#         output_y_reducer_i, 
#         output_y_reducer_j)


#     comp_results_1 = comp_results_1.cpu()
#     comp_results_2 = comp_results_2.cpu()
#     # print(f"Results flex_spmv 1: {comp_results_1}")
#     print(f"Results difference 1: {torch.norm(results_gold_1 - comp_results_1)}")
#     # print(f"Results flex_spmv 2: {comp_results_2}")
#     print(f"Results difference 2: {torch.norm(results_gold_2 - comp_results_2)}")
#     if torch.norm(results_gold_1 - comp_results_1) < 1e-6 and torch.norm(results_gold_2 - comp_results_2) < 1e-6:
#         print("Results are the same")
#     else:
#         print("Results are different") 

if __name__ == "__main__":
    main()
    