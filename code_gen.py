import torch
import torch.fx
import os
import numpy as np
import string

import easier as esr
from easier.core.jit import EasierTracer

# Part 1: Define spring-mass system model
class SpringMassSystem(esr.Module):
    def __init__(self, spm_k, spm_l, vector_x, vector_y, row_end_offset, col_indices_i, col_indices_j, num_rows):
        super().__init__()

        # data
        self.spm_k = spm_k  # (N, 1)
        self.spm_l = spm_l  # (N, 1)
        self.vector_x = vector_x  # (N, D)
        self.vector_y = vector_y  # (M, D)
        self.row_end_offset = row_end_offset # (M + 1)
        self.col_indices_i = col_indices_i # (N, 1)
        self.col_indices_j = col_indices_j # (N, 1)

        # Selector for i and j, and reducer
        self.selector_i = esr.Selector(self.col_indices_i)
        self.selector_j = esr.Selector(self.col_indices_j)
        self.reducer = esr.Reducer(self.row_end_offset, num_rows)

    def forward(self):
        # compute force
        r_i = self.selector_i(self.vector_x) # (N, 2)
        r_j = self.selector_j(self.vector_x) # (N, 2)
        r_ij = r_j - r_i # (N, 2)
        norm_r_ij = torch.norm(r_ij, dim=1, keepdim=True) # (N, 1)
        e_ij = r_ij / norm_r_ij # (N, 2)
        f_ij = self.spm_k * (norm_r_ij - self.spm_l) * e_ij # (N, 2)
        f_i = self.reducer(f_ij) # (M, 2)

        return f_i

class SpringMassSystem_calculation():
    def __init__(self, spm_k, spm_l, vector_x, vector_y, row_end_offset, col_indices_i, col_indices_j, num_rows):
        super().__init__()

        # data
        self.spm_k = spm_k  # (N, 1)
        self.spm_l = spm_l  # (N, 1)
        self.vector_x = vector_x  # (N, D)
        self.vector_y = vector_y  # (M, D)
        self.row_end_offset = row_end_offset # (M + 1)
        self.col_indices_i = col_indices_i # (N, 1)
        self.col_indices_j = col_indices_j # (N, 1)
        self.num_rows = num_rows

    def forward(self):
        # compute force
        r_i = self.vector_x[self.col_indices_i] # (N, 2)
        r_j = self.vector_x[self.col_indices_j] # (N, 2)
        r_ij = r_i - r_j # (N, 2)
        norm_r_ij = torch.norm(r_ij, dim=1, keepdim=True) # (N, 1)
        e_ij = r_ij / norm_r_ij # (N, 2)
        f_ij = self.spm_k * (norm_r_ij - self.spm_l) * e_ij # (N, 2)
        # reduce f_ij to f_i based on row_end_offset
        f_i = torch.zeros(self.num_rows, 2, device="cuda", dtype=f_ij.dtype)
        for i in range(self.num_rows):
            f_i[i] = torch.sum(f_ij[self.row_end_offset[i]:self.row_end_offset[i+1]], dim=0)
        return f_i

# Part 2: Trace the model with torch.fx
def trace_model(model):
    tracer = EasierTracer()
    traced_model = tracer.trace(model)
    print("FX Graph:")
    traced_model.print_tabular()    
    return traced_model

# Part 3: Generate CUDA code from the graph
def generate_cuda_code_from_graph(traced_model):
    # Create directory for generated code
    os.makedirs("cuda_code", exist_ok=True)
    
    # Analyze the graph and extract operations
    inputs = []
    input_keys = set()
    selector_register = []
    map_operations = []
    output_var = None

#---------------------------------
# Analyze the graph and extract operations, 1) input and output, 2) selector, 3) map
#---------------------------------

    # obtain the input and output
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
                        "name": node.target + "_idx",
                        "target": node.target,
                        "dtype": "int"
                    }
                )
        elif node.op == 'output':
            output_var = node.args[0].name
    
    # obtain the selector in intermediate register
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
    # obtain the map operations
    for i in range(len(traced_model.nodes)):
        node = list(traced_model.nodes)[i]
        if node.op == 'call_module':
            if 'selector' in str(node.target) or 'reducer' in str(node.target):
                continue
            else:
                map_operations.append({
                    'output': node.name,
                    'op': node.target,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                    'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None]
                })
        elif node.op == 'call_function' or node.op == 'call_method':
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
            
            map_operations.append({
                'output': node.name,
                'op': target_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None]
            })
        elif node.op == 'output':
            output_var = node.args[0].name
    
    # # debug print
    # for inp in inputs:
    #     print(f"Input: {inp}")
    for inter in selector_register:
        print(f"Selector register: {inter}")
    # for op in map_operations:
    #     print(f"Operation: {op}")
    # print(f"Output: {output_var}")

#---------------------------------
# Generate the CUDA code
#---------------------------------

    # Generate the input code
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
            # vector value and spm nnz
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
    
    # Generate the selector register code
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
    
    # Generate the CUDA kernel code (map operations)
    map_code = []
    for op in map_operations:
        if op['op'] == 'add':
            map_code.append(f"    TensorT {op['output']} = {op['args'][0]} + {op['args'][1]}; \n")
        elif op['op'] == 'mul':
            if op['output'] != 'mul':
                map_code.append(f"    TensorT {op['output']} = {op['args'][0]} * {op['args'][1]}; \n")
            else:
                map_code.append(f"    ValueT {op['output']} = {op['args'][0]} * {op['args'][1]}; \n")
        elif op['op'] == 'sub':
            if op['output'] == 'sub':
                map_code.append(f"    TensorT {op['output']} = {op['args'][0]} - {op['args'][1]}; \n")
            else:
                map_code.append(f"    ValueT {op['output']} = {op['args'][0]} - {op['args'][1]}; \n")
        elif op['op'] == 'norm':
            map_code.append(f"    ValueT {op['output']} = {op['args'][0]}.l2Norm(); \n")
        elif op['op'] == 'truediv':
            map_code.append(f"    TensorT {op['output']} = {op['args'][0]} / {op['args'][1]}; \n")
        else:
            # error
            raise ValueError(f"Operation {op['op']} not supported")
    # add the register results to the SMEM
    map_code.append(f"    #pragma unroll\n    for (int i = 0; i < DIM_INPUT_VECTOR_X; i++) \n    {{ \n        s_tile_value_nonzeros[nonzero_idx + i * TILE_ITEMS] = {op['output']}.values[i]; \n    }}")

    # # Debug print to check kernel operations
    # print("Generated kernel operations:")
    # for op in map_code:
    #     print(op)
        
    # Create directories for generated code if they don't exist
    os.makedirs("include-gen", exist_ok=True)
    
    # Read template files
    with open("cuda_template/merged_agent_spmv_template.cuh", "r") as f:
        kernel_agent_template = f.read()
    
    with open("cuda_template/merged_utils_template.cuh", "r") as f:
        utils_template = f.read()
    
    # Apply templates using string.Template
    # Join the lists into strings for template substitution
    input_declarations_str = ''.join(input_declarations_code)
    input_init_str = ''.join(input_init_code)
    selector_str = ''.join(selector_code)
    map_str = ''.join(map_code)
    
    agent_kernel_code = string.Template(kernel_agent_template).substitute(
        input_declarations_code=input_declarations_str,
        input_init_code=input_init_str,
        selector_code=selector_str,
        map_code=map_str
    )
    
    # Join the list of input declarations into a single string
    input_declarations_utils_str = ''.join(input_declarations_utils_code)
    
    utils_code = string.Template(utils_template).substitute(
        input_declarations_utils_code=input_declarations_utils_str
    )
    
    # Write files
    with open("include-gen/merged_agent_spmv.cuh", "w") as f:
        f.write(agent_kernel_code)
    
    with open("include-gen/merged_utils.cuh", "w") as f:
        f.write(utils_code)
    
    # print("CUDA code generated successfully!")

# Main function to run the entire pipeline
def main():

    # data settings
    num_nnz = 20
    num_cols = 10
    num_rows = 5
    dim_x = 2
    spm_k = torch.rand(num_nnz, 1, device="cuda")  # (N, 1)
    spm_l = torch.rand(num_nnz, 1, device="cuda")  # (N, 1)
    vector_x = torch.rand(num_cols, dim_x, device="cuda")  # (N, D)
    vector_y = torch.zeros(num_rows, dim_x, device="cuda")  # (M, D)
    row_end_offset = torch.sort(torch.randint(0, num_nnz + 1, (num_rows,), device="cuda"))[0] # (M + 1)
    row_end_offset = torch.cat([torch.zeros(1, device="cuda", dtype=row_end_offset.dtype), row_end_offset])
    row_end_offset = row_end_offset
    # Ensure that i and j are different for each point
    col_indices_i = torch.randint(0, num_cols, (num_nnz,), device="cuda")
    col_indices_j = torch.empty(num_nnz, dtype=torch.long, device="cuda")
    for idx in range(num_nnz):
        i_val = col_indices_i[idx].item()
        # pick a random j != i
        choices = torch.arange(num_cols, device="cuda")
        choices = choices[choices != i_val]
        j_val = choices[torch.randint(0, len(choices), (1,), device="cuda")].item()
        col_indices_j[idx] = j_val

    # pytorch model for code generation
    model = SpringMassSystem(
        spm_k, spm_l, vector_x, vector_y, row_end_offset, col_indices_i, col_indices_j, num_rows
    )
    # python model for verification
    model_calculation = SpringMassSystem_calculation(
        spm_k, spm_l, vector_x, vector_y, row_end_offset, col_indices_i, col_indices_j, num_rows
    )

    print("Step 1: Tracing model with torch.fx")
    traced_model = trace_model(model)
    
    # run the model
    f_i = model_calculation.forward()
    print(f"f_i: {f_i}")

    print("\nStep 2: Generating CUDA code from the graph")
    generate_cuda_code_from_graph(traced_model)

if __name__ == "__main__":
    main()
    