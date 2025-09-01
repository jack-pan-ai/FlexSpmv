import torch
import torch.fx
import os
import numpy as np
import string

import easier as esr
from easier.core.jit import EasierTracer
import scipy.sparse
from scipy.io import mmwrite

from codegenDevice.utils import get_dim_length
from traceGraph.graph_trace import trace_graph


# Part 2: Generate CUDA code from the graph
def generate_cuda_code_from_graph(traced_model):
    
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

    inputs, outputs, selector_register, map_operations, reducer_operations, aggregator_operations = trace_graph(traced_model)
    
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
            raise ValueError(f"Operation {_op} not supported")

    # # Debug print to check kernel operations
    # print("Generated kernel operations:")
    # for op in map_code:
    #     print(op)
        
    #  -------------------------------------------------------------
    #  Generate the aggregator code
    #  -------------------------------------------------------------
    aggregator_code = []
    aggregator_reg_definitions = []

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

    # debug print
    # for op in aggregator_reg_definitions:
    #     print(f"Aggregator reg definitions: {op}")
    # for op in aggregator_code:
    #     print(f"Aggregator code: {op}")

    #  -------------------------------------------------------------
    #  Generate the reducer code
    #  -------------------------------------------------------------
    reducer_code = []
        
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
    _folder = 'reducer' if reducer_operations != [] else 'aggregator' # map only will use the aggregator template
    with open(f"cuda_template/{_folder}/merged_agent_spmv_template.cuh", "r") as f:
        kernel_agent_template = f.read()
    
    with open(f"cuda_template/merged_utils_template.cuh", "r") as f:
        utils_template = f.read()

    with open(f"cuda_template/{_folder}/merged_spmv_template.cuh", "r") as f:
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
    output_agent_tenosrs_code_str = trans_str(output_agent_tenosrs_code)
    output_agent_SMEM_code_str = trans_str(output_agent_SMEM_code)
    output_agent_forloop_code_str = trans_str(output_agent_forloop_code)

    agent_kernel_code = string.Template(kernel_agent_template).substitute(
        input_declarations_code=input_declarations_str,
        input_init_code=input_init_str,
        input_agent_tenosrs_code=input_agent_tenosrs_code_str,
        selector_code=selector_str,
        map_code=map_str,
        reducer_code=reducer_code_str,
        aggregator_reg_definitions=aggregator_reg_definitions_str,
        aggregator_code=aggregator_code_str,
        output_agent_tenosrs_code=output_agent_tenosrs_code_str,
        output_agent_SMEM_code=output_agent_SMEM_code_str,
        output_agent_forloop_code=output_agent_forloop_code_str
    )

    spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
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

    