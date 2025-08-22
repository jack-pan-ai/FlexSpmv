import torch
import torch.fx
import os
import numpy as np
import string

import easier as esr
from easier.core.jit import EasierTracer
import scipy.sparse
from scipy.io import mmwrite

from codegenDevice.reducer import reducer_diagnal_code_gen
from codegenDevice.utils import get_dim_length

    
info_nodes = {
    "vector_x": {"shape": (2,)},
    "vector_x_1": {"shape": (2,)},
    "selector_1": {"shape": (2,)},
    "selector_2": {"shape": (2,)},
    "spm_1": {"shape": (2,)},
    "spm_2": {"shape": (6,)},
    "add": {"shape": (2,)},
    "add_1": {"shape": (6,)},
    "reducer_1": {"shape": (2,)},
    "reducer_2": {"shape": (6,)},
    "sum_1": {"shape": (2,)},
    "sum_2": {"shape": (6,)}
}

# Part 1: Trace the model with torch.fx
def trace_model(model):
    tracer = EasierTracer()
    traced_model = tracer.trace(model)
    print("FX Graph:")
    traced_model.print_tabular()    
    return traced_model

# Part 2: Generate CUDA code from the graph
def generate_cuda_code_from_graph(traced_model):
    # Create directory for generated code
    os.makedirs("cuda_code", exist_ok=True)
    
    # Analyze the graph and extract operations
    inputs = []
    _input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    outputs = []

#  -------------------------------------------------------------
#  Analyze the graph and extract operations, 1) input and output, 2) selector, 3) map 4) reducer and aggregator
#  -------------------------------------------------------------

    #  -------------------------------------------------------------
    #  obtain the input and output, and the dimension of the input and output
    #  -------------------------------------------------------------
    for i in range(len(traced_model.nodes)):
        node = list(traced_model.nodes)[i]
        if node.op == 'get_attr':
            if node.target not in _input_keys:
                inputs.append(
                    {
                        "name": node.name,
                        "target": node.target,
                        "dtype": "scalar_t",
                        "shape": info_nodes[str(node.name)]['shape']
                    }
                )
                _input_keys.add(node.target)
        elif node.op == 'call_module':
            if 'selector' in str(node.target):
                target_name = 'selector'
                inputs.append(
                    {
                        "name": node.name,
                        "target": node.target,
                        "dtype": "int",
                        "shape": info_nodes[str(node.name)]['shape']
                    }
                )
        elif node.op == 'output':
            for arg in node.args[0]:
                outputs.append(
                    {
                        "name": arg.name,
                        "target": arg.target,
                        "dtype": "ValueT",
                        "shape": info_nodes[str(arg.name)]['shape']
                    }
                )
    
    # # debug print
    # for inp in inputs:
    #     print(f"Input: {inp}")
    # for out in outputs:
    #     print(f"Output: {out}")
    #  -------------------------------------------------------------
    #  obtain the selector in intermediate register, including the vector x and edge tensor
    #  here: obtain the data from HBM  to register, we need to consider the alignment of the data
    #  here: we will need to input dimension here for the pragma unroll
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
                        "dtype": "TensorKeyT",
                        "selector": 1, # represents the selector is used for vector x
                        "selector_name": node_forward.target,
                        "shape": info_nodes[node_current.target]['shape']
                    }
                )
            else:
                # tensor here is imitate the sparse matrix, not used for vector x
                selector_register.append(
                    {
                        "name": node_current.name,
                        "target": node_current.target,
                        "dtype": "TensorT",
                        "selector": 0, # represents the selector is for edge tensor, not used for vector x
                        "selector_name": node_current.name,
                        "shape": info_nodes[node_current.target]['shape']
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
                _target = 'reducer'
                map_operations.append({
                    'name': node.name,
                    'op': _target,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                    'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                    'shape': info_nodes[node.name]['shape']
                })
            else:
                # raise error
                raise ValueError(f"Operation {node.target} not supported")
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
                'name': node.name,
                'op': target_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                'shape': info_nodes[node.name]['shape']
            })
    # debug print
    # for op in map_operations:
    #     print(f"Map operation: {op}")
    #  -------------------------------------------------------------
    #  obtain the reducer and aggregator operations
    #  -------------------------------------------------------------
    for i in range(len(traced_model.nodes)):
        node = list(traced_model.nodes)[i]
        if 'reducer' in str(node.target):
            op_name = 'reducer'
            reducer_operations.append({
                'name': node.name,
                'op': op_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                'shape': info_nodes[node.name]['shape']
            })
        elif 'sum' in str(node.target):
            op_name = 'sum'
            aggregator_operations.append({
                'name': node.name,
                'op': op_name,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args],
                'kwargs': [node.kwargs if hasattr(node, 'kwargs') else None],
                'shape': info_nodes[node.name]['shape']
            })
    # debug print
    # for op in reducer_operations:
    #     print(f"Reducer operation: {op}")
    # for op in aggregator_operations:
    #     print(f"Aggregator operation: {op}")
    
#  -------------------------------------------------------------
#  Generate the CUDA code
#  -------------------------------------------------------------

    #  -------------------------------------------------------------
    #  Generate the input/output code
    #  -------------------------------------------------------------
    input_declarations_code = []
    input_init_code = []
    input_declarations_utils_code = []
    input_agent_tenosrs_code = []
    output_agent_tenosrs_code = []
    output_agent_SMEM_code = []
    output_agent_forloop_code = []
    offset_inside_forloop_code = []
    _tensor_names = set()
    for inp in inputs:
        if inp['dtype'] == 'int':
            # column indices
            input_declarations_utils_code.append(f"  OffsetT *{inp['name'] + '_ptr'}; \n")
            input_declarations_code.append(f"  ColumnIndicesIteratorT {inp['name'] + '_ptr'}; \n")
            input_init_code.append(f"    {inp['name'] + '_ptr'}(spmv_params.{inp['name'] + '_ptr'}), \n")
        else:
            # spm and vector x
            input_declarations_utils_code.append(f"  ValueT *{inp['name'] + '_ptr'}; \n")
            input_declarations_code.append(f"  VectorValueIteratorT {inp['name'] + '_ptr'}; \n")
            input_init_code.append(f"    {inp['name'] + '_ptr'}(spmv_params.{inp['name'] + '_ptr'}), \n")
            if inp['target'] not in _tensor_names:
                _dim = get_dim_length(inp['shape'])
                input_agent_tenosrs_code.append(f"  typedef Tensor<ValueT, {_dim}> TensorInput_{inp['target']}_T; \n")
                _tensor_names.add(inp['target'])

    
    # output code in the declarations utils file
    for out in outputs:
        input_declarations_utils_code.append(f"  ValueT *{"output_y_" + out['name'] + "_ptr"}; \n")
        _dim = get_dim_length(out['shape'])
        if 'reducer' in str(out['target']):
            # reducer outside the forloop
            output_agent_tenosrs_code.append(f"  // Tensor and TensorKey for reducers \n")
            output_agent_tenosrs_code.append(f"  typedef TensorKey<OffsetT, ValueT, {_dim}> TensorKeyOutput_{out['name']}_T; \n")
            output_agent_tenosrs_code.append(f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{out['name']}_T; \n")
            output_agent_tenosrs_code.append(f"  // Reduce-value-by-segment scan operator \n")
            output_agent_tenosrs_code.append(f"  typedef ReduceTensorByKeyOp<TensorKeyOutput_{out['name']}_T> ReduceBySegmentOp_{out['name']}_T; \n")
            output_agent_tenosrs_code.append(f"  typedef BlockScan< \n")
            output_agent_tenosrs_code.append(f"            TensorKeyOutput_{out['name']}_T, \n")
            output_agent_tenosrs_code.append(f"            BLOCK_THREADS, \n")
            output_agent_tenosrs_code.append(f"            AgentSpmvPolicyT::SCAN_ALGORITHM> \n")
            output_agent_tenosrs_code.append(f"            BlockScan_{out['name']}_T; \n")
            output_agent_SMEM_code.append(f"               SmemReuseReducer<{_dim}, BlockScan_{out['name']}_T> smem_{out['name']}; \n")
        elif 'sum' in str(out['target']):
            _name = out['name']
            output_agent_tenosrs_code.append(f"  // Tensor type and block reducefor output \n")
            output_agent_tenosrs_code.append(f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")
            output_agent_tenosrs_code.append(f"  typedef BlockReduce< \n")
            output_agent_tenosrs_code.append(f"            TensorOutput_{_name}_T, \n")
            output_agent_tenosrs_code.append(f"            BLOCK_THREADS, \n")
            output_agent_tenosrs_code.append(f"            BLOCK_REDUCE_WARP_REDUCTIONS> \n")
            output_agent_tenosrs_code.append(f"            BlockReduce_{_name}_T; \n")
            output_agent_SMEM_code.append(f"               typename BlockReduce_{_name}_T::TempStorage smem_{_name}; \n")
        else:
            # used for map output
            output_agent_tenosrs_code.append(f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{out['name']}_T; \n")
            # map inside the forloop
            output_agent_forloop_code.append(f"  #pragma unroll \n")
            output_agent_forloop_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            output_agent_forloop_code.append(f"  {{ \n")
            output_agent_forloop_code.append(f"    spmv_params.output_y_{out['name']}_ptr[(tile_start_coord.y + nonzero_idx) * {_dim} + i] = {out['name']}.values[i]; \n")
            output_agent_forloop_code.append(f"  }} \n")
        
    # is there is reducer, the offset is needed
    if reducer_operations != []:
        offset_inside_forloop_code.append(f"    loading_offsets(tile_num_rows, tile_start_coord); \n")
        output_agent_SMEM_code.append(f"    OffsetT s_tile_row_end_offsets[TILE_ITEMS]; \n")
        
    # debug print
    # for inp in input_declarations_utils_code:
    #     print(f"Input declarations utils: {inp}")
    # for inp in input_declarations_code:
    #     print(f"Input declarations: {inp}")
    # for inp in input_init_code:
    #     print(f"Input init code: {inp}")
    # for inp in input_agent_tenosrs_code:
    #     print(f"Input agent tenosrs code: {inp}")
    # for out in output_agent_tenosrs_code:
    #     print(f"Output agent tenosrs code: {out}")
    # for out in output_agent_SMEM_code:
    #     print(f"Output agent SMEM code: {out}")
    # for out in output_agent_forloop_code:
    #     print(f"Output agent forloop code: {out}")
    
    #  -------------------------------------------------------------
    #  Generate the selector register code
    #  -------------------------------------------------------------
    selector_code = []
    for inter in selector_register:
        #obtain the dimension of the selector
        _dim = get_dim_length(inter['shape'])
        _name = inter['name']
        _target = inter['target']
        _selector_name = inter['selector_name']
        if inter['selector'] == 1:
            # load the selector register
            selector_code.append(f"    ColumnIndicesIteratorT {_name + '_ptr_current'} = {_selector_name + '_ptr'} + tile_start_coord.y + nonzero_idx; \n")
            selector_code.append(f"    TensorInput_{_target}_T {_selector_name}({_target + '_ptr'} + *{_name + '_ptr_current'} * {_dim}); \n")
        else:
            # spm loading
            selector_code.append(f"    VectorValueIteratorT {_name + '_ptr_current'} = {_selector_name + '_ptr'} + (tile_start_coord.y + nonzero_idx) * {_dim}; \n")
            selector_code.append(f"    TensorInput_{_target}_T {_selector_name}({_target + '_ptr_current'}); \n")
    
    # # debug print
    # for inter in selector_code:
    #     print(f"Selector code: {inter}")
    
    #  -------------------------------------------------------------
    #  Generate the CUDA kernel code (map operations)
    #  -------------------------------------------------------------
    map_code = []
    for op in map_operations:
        _name = op['name']
        _op = op['op']
        _dim = get_dim_length(op['shape'])
        if _op == 'add':
            map_code.append(f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} + {op['args'][1]}; \n")
        elif _op == 'norm':
            map_code.append(f"    ValueT {_name} = {op['args'][0]}.l2Norm(); \n")
        elif _op == 'reducer':
            map_code.append(f"    temp_storage.smem_{_name}.s_tile_value_reducer[nonzero_idx] = {op['args'][0]}; \n")
        elif _op == 'sum':
            map_code.append(f"    {_name} = {_name} + {op['args'][0]}; \n")
        else:
            # error
            raise ValueError(f"Operation {op_name} not supported")

    # # Debug print to check kernel operations
    # print("Generated kernel operations:")
    # for op in map_code:
    #     print(op)
        
    #  -------------------------------------------------------------
    #  Generate the aggregator code
    #  -------------------------------------------------------------
    aggregator_code = []
    aggregator_code_dispatch = []
    aggregator_reg_definitions = []
    # the shared memory can be reused for multiple blockReduce
    if aggregator_operations != []:
        aggregator_code_dispatch.append(f"   // just padding parameters here \n")
        aggregator_code_dispatch.append(f"   CoordinateT tile_start_coord = {{-1, tile_idx * TILE_ITEMS}}; \n")
        aggregator_code_dispatch.append(f"   CoordinateT tile_end_coord = {{-1, min(tile_idx * TILE_ITEMS + TILE_ITEMS, spmv_params.num_nonzeros)}}; \n")

    for op in aggregator_operations:
        _name = op['name']
        _dim = get_dim_length(op['shape'])
        aggregator_reg_definitions.append(f"   // each aggregator need a register to store the non-zero values \n")
        aggregator_reg_definitions.append(f"   TensorOutput_{_name}_T {_name}; \n")
        if op['op'] == 'sum':
            aggregator_code.append(f"   // blockReduce \n")
            aggregator_code.append(f"   TensorOutput_{_name}_T {_name}_result = BlockReduce_{_name}_T(temp_storage.smem_{_name}).Sum({_name}); \n")
            aggregator_code.append(f"   if (threadIdx.x == 0) \n")
            aggregator_code.append(f"   {{ \n")
            aggregator_code.append(f"     #pragma unroll \n")
            aggregator_code.append(f"     for (int i = 0; i < {_dim}; i++) \n")
            aggregator_code.append(f"     {{ \n")
            aggregator_code.append(f"       atomicAdd(&spmv_params.output_y_{_name}_ptr[i], {_name}_result.values[i]); \n")
            aggregator_code.append(f"     }} \n")
            aggregator_code.append(f"   }} \n")

    # # debug print
    # for op in aggregator_reg_definitions:
    #     print(f"Aggregator reg definitions: {op}")
    # for op in aggregator_code:
    #     print(f"Aggregator code: {op}")

    #  -------------------------------------------------------------
    #  Generate the reducer code
    #  -------------------------------------------------------------
    reducer_code = []
    
    # generate the diagnoal code once
    if reducer_operations != []:
        reducer_diagonal_code_spmv, reducer_diagonal_code_spmv_agent, diagonal_code_spmv_agent_thread, offset_code_spmv_agent_dispatch = reducer_diagnal_code_gen()
    else:
        reducer_diagonal_code_spmv = []
        reducer_diagonal_code_spmv_agent = []
        diagonal_code_spmv_agent_thread = []
        offset_code_spmv_agent_dispatch = []
        
    # generate the reducer code for each reducer
    for op in reducer_operations:
        # get the dimension length of the shape
        _dim = get_dim_length(op['shape'])
        _name = op['name']
        # generate the SMEM definitions and the reducer code
        reducer_code.append(f"   reduce<{_dim}, BlockScan_{_name}_T, TensorOutput_{_name}_T, ReduceBySegmentOp_{_name}_T>( \n")
        reducer_code.append(f"                temp_storage.smem_{_name}.s_tile_value_reducer,          ///< [in, code gen] Shared memory array of non-zero values for the merge tile \n")
        reducer_code.append(f"                temp_storage.s_tile_row_end_offsets,         ///< [in, code gen] Shared memory array of row end offsets for the merge tile \n")
        reducer_code.append(f"                tile_start_coord,               ///< [in] Starting coordinate of the merge tile \n")
        reducer_code.append(f"                tile_end_coord,                 ///< [in] Ending coordinate of the merge tile \n")
        reducer_code.append(f"                thread_start_coord,             ///< [in] Starting coordinate of the thread \n")
        reducer_code.append(f"                tile_num_rows,                  ///< [in] Number of rows in the merge tile \n")
        reducer_code.append(f"                tile_num_nonzeros,               ///< [in] Number of non-zeros in the merge tile \n")
        reducer_code.append(f"                spmv_params.output_y_{_name}_ptr,                  ///< [out] Output vector y \n")
        reducer_code.append(f"                temp_storage.smem_{_name}.scan               ///< [in] Scan storage for BlockScanT \n")
        reducer_code.append(f"            ); \n")
        reducer_code.append(f"   CTA_SYNC(); \n")

    # # debug print
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

    with open("cuda_template/merged_spmv_template.cuh", "r") as f:
        kernel_spmv_template = f.read()
    
    #  -------------------------------------------------------------
    #  Apply templates using string.Template
    #  -------------------------------------------------------------
    def trans_str(code):
        if code == []:
            return ''
        else:
            return ''.join(code)
    
    input_declarations_str = trans_str(input_declarations_code)
    input_init_str = trans_str(input_init_code)
    input_agent_tenosrs_code_str = trans_str(input_agent_tenosrs_code)
    selector_str = trans_str(selector_code)
    map_str = trans_str(map_code)
    reducer_code_str = trans_str(reducer_code)
    aggregator_reg_definitions_str = trans_str(aggregator_reg_definitions)
    aggregator_code_str = trans_str(aggregator_code)
    reducer_diagonal_code_spmv_str = trans_str(reducer_diagonal_code_spmv)
    reducer_diagonal_code_spmv_agent_str = trans_str(reducer_diagonal_code_spmv_agent)
    output_agent_tenosrs_code_str = trans_str(output_agent_tenosrs_code)
    output_agent_SMEM_code_str = trans_str(output_agent_SMEM_code)
    diagonal_code_spmv_agent_thread_str = trans_str(diagonal_code_spmv_agent_thread)
    offset_code_spmv_agent_dispatch_str = trans_str(offset_code_spmv_agent_dispatch)
    output_agent_forloop_code_str = trans_str(output_agent_forloop_code)
    offset_inside_forloop_code_str = trans_str(offset_inside_forloop_code)
    aggregator_code_dispatch_str = trans_str(aggregator_code_dispatch)

    agent_kernel_code = string.Template(kernel_agent_template).substitute(
        input_declarations_code=input_declarations_str,
        input_init_code=input_init_str,
        input_agent_tenosrs_code=input_agent_tenosrs_code_str,
        selector_code=selector_str,
        map_code=map_str,
        reducer_code=reducer_code_str,
        aggregator_reg_definitions=aggregator_reg_definitions_str,
        aggregator_code=aggregator_code_str,
        reducer_diagonal_code_spmv_agent=reducer_diagonal_code_spmv_agent_str,
        output_agent_tenosrs_code=output_agent_tenosrs_code_str,
        output_agent_SMEM_code=output_agent_SMEM_code_str,
        diagonal_code_spmv_agent_thread=diagonal_code_spmv_agent_thread_str,
        output_agent_forloop_code=output_agent_forloop_code_str,
        offset_inside_forloop_code=offset_inside_forloop_code_str,
        aggregator_code_dispatch=aggregator_code_dispatch_str
    )

    spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
        reducer_diagonal_code_spmv=reducer_diagonal_code_spmv_str,
        offset_code_spmv_agent_dispatch=offset_code_spmv_agent_dispatch_str
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
    
    with open("include/merged_spmv.cuh", "w") as f:
        f.write(spmv_kernel_code)
    
    with open("include/merged_utils.cuh", "w") as f:
        f.write(utils_code)
    
    #  -------------------------------------------------------------
    print("CUDA code generated successfully!")
    #  -------------------------------------------------------------

    