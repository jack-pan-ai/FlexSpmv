import torch
import torch.fx
import os
import numpy as np
import string

import easier as esr
from easier.core.jit import EasierTracer

# Part 1: Define Spmv model
class SpmvSystem(esr.Module):
    def __init__(self):
        super().__init__()

        # Selector for i and j
        self.num_nnz = 20
        self.num_cols = 10
        self.num_rows = 5
        vector_index = torch.randint(0, self.num_cols, (self.num_nnz,), device="cuda")
        row_end_offset = torch.randint(0, self.num_nnz, (self.num_rows,), device="cuda")
        self.selector_i = esr.Selector(vector_index)
        self.reduce_i = esr.Reducer(row_end_offset, self.num_rows)

        # spmv parameters
        self.spm_nnz = torch.rand(self.num_nnz, 1, device="cuda")  # (N, 1)
        self.vector = torch.rand(self.num_cols, 1, device="cuda")  # (N, 1)

    def forward(self):
        # compute force
        v = self.selector_i(self.vector)
        f_i = self.spm_nnz * v
        f_i = self.reduce_i(f_i)

        # compute force for another vector
        v1 = self.selector_i(self.vector)
        f_i1 = self.spm_nnz * v1
        f_i1 = self.reduce_i(f_i1)

        return [f_i, f_i1]

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
    operations = []
    inputs = []
    output_var = None
    
    for node in traced_model.nodes:
        if node.op == 'placeholder' or node.op == 'get_attr':
            inputs.append(
                {
                    "name": node.name,
                    "target": node.target,
                    "type": "scalar_t"
                }
            )
        elif node.op == 'call_module':
            if 'selector' in str(node.target):
                target_name = 'selector'
                inputs.append(
                    {
                        "name": node.target + "_idx",
                        "target": node.target,
                        "type": "int"
                    }
                )
                operations.append({
                    'output': node.name,
                    'op': target_name,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args]
                })
            elif 'reducer' in str(node.target):
                target_name = 'reducer'
                inputs.append(
                    {
                        "name": node.target + "_idx",
                        "target": node.target,
                        "type": "int"   
                    }
                )
                operations.append({
                    'output': node.name,
                    'op': target_name,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args] 
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
            
            operations.append({
                'output': node.name,
                'op': target_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args]
            })
        elif node.op == 'output':
            output_var = node.args[0].name
    
    print(f"Inputs: {inputs}")
    print(f"Operations: {operations}")
    print(f"Output: {output_var}")
    
    # Generate the CUDA kernel code
    kernel_operations = []
    for op in operations:
        if op['op'] == 'add':
            kernel_operations.append(f"    scalar_t {op['output']} = {op['args'][0]} + {op['args'][1]};")
        elif op['op'] == 'mul':
            kernel_operations.append(f"    scalar_t {op['output']} = {op['args'][0]} * {op['args'][1]};")
        elif op['op'] == 'sin':
            kernel_operations.append(f"    scalar_t {op['output']} = sinf({op['args'][0]});")
        elif op['op'] == 'selector':
            kernel_operations.append(f"    scalar_t {op['output']} = {op['args'][0]}[{op['output'] + '_idx'}[idx]];")
    
    # Debug print to check kernel operations
    print("Generated kernel operations:")
    for op in kernel_operations:
        print(op)
    
    # Read template files
    with open("cuda_template/merged_agent_spmv_template.cuh", "r") as f:
        kernel_agent_template = f.read()
    
    with open("cuda_template/merged_utils_template.cuh", "r") as f:
        utils_template = f.read()
    
    # Prepare template variables
    input_declarations = ",\n    ".join([f"const scalar_t* {inp['name']}" if inp['type'] == 'scalar_t' else f"const int* {inp['name']}" for inp in inputs])
    tensor_declarations = ",\n    ".join([f"torch::Tensor {inp['name']}" for inp in inputs])
    data_ptr_declarations = ",\n            ".join([f"{inp['name']}.data_ptr<scalar_t>()" if inp['type'] == 'scalar_t' else f"{inp['name']}.data_ptr<int>()" for inp in inputs])
    first_input = inputs[0]['name']
    
    # Apply templates using string.Template
    kernel_code = string.Template(kernel_agent_template).substitute(
        input_declarations=input_declarations,
        kernel_operations="\n".join(kernel_operations),
        output_var=output_var,
        tensor_declarations=tensor_declarations,
        data_ptr_declarations=data_ptr_declarations,
        first_input=first_input
    )
    
    cpp_code = string.Template(utils_template).substitute(
        tensor_declarations=", ".join([f"torch::Tensor {inp['name']}" for inp in inputs])
    )
    
    # Write files
    with open("include/merged_agent_spmv.cuh", "w") as f:
        f.write(kernel_code)
    
    with open("include/merged_utils.cuh", "w") as f:
        f.write(cpp_code)
    
    print("CUDA code generated successfully!")
    # return "cuda_code/merged_agent_spmv.cuh", "cuda_code/merged_utils.cuh"


# Main function to run the entire pipeline
def main():
    # pytorch model
    model = SpmvSystem()

    print("Step 1: Tracing model with torch.fx")
    traced_model = trace_model(model)
    
    # print("\nStep 2: Generating CUDA code from the graph")
    # generate_cuda_code_from_graph(traced_model)

if __name__ == "__main__":
    main()
    