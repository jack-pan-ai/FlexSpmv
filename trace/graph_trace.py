from itertools import zip_longest

import easier as esr
from easier.core.runtime.metadata import get_node_meta
from easier import Reducer, Selector
from easier.core.runtime.metadata import Role
from easier.core.passes.dataflow_fusion.node_group import \
    NodeGroup, get_node_group

def get_node_info(node, node_meta_dict, node_placeholder_meta_dict):
    '''
    Get the node information
    '''
    op = node.op
    name = node.name
    target = node.target
    args = node.args
    # the first dimension is the batch size
    if node.op == 'placeholder':
        # metadata of placeholder is saved in the traced_model
        node_shape = get_node_meta(node_placeholder_meta_dict[name]).shape[1:]
        node_dtype = get_node_meta(node_placeholder_meta_dict[name]).dtype
    else:
        # metadats of other nodes is saved in the node_group
        node_shape = get_node_meta(node_meta_dict[name]).shape[1:]
        node_dtype = get_node_meta(node_meta_dict[name]).dtype
    if node_shape == ():
        node_shape = (1,)
    return node_shape, op, name, target, args, node_dtype

# Part 2: Generate CUDA code from the graph
def trace_graph(submodule, traced_model):

    # Analyze the graph and extract operations
    inputs = []
    _input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    outputs = []

    # obtain the traced nodes list (exclude the output node)
    traced_nodes = list(submodule.graph.nodes)
    
    # dict here only provide the metadata info
    node_meta_dict = {node.name: node for node in get_node_group(submodule).nodes}
    # prepare the selector and selector tensor list
    selector_set = set([node.name for node in get_node_group(submodule).nodes \
        if node.op == 'call_module' and \
            list(node.meta['nn_module_stack'].values())[0][1] == Selector])
    selector_tensor_set = set([node.args[0].name for node in get_node_group(submodule).nodes \
        if node.name in selector_set])
    # the placeholder nodes here will contain the reducer nodes (which is not used)
    node_placeholder_meta_dict = {node.name: node \
        for node in traced_model.jit_engine.graph.nodes}
    # if node.op == 'get_attr' \
    #         or node.name in selector_tensor_set}
    print("selector_set: ", selector_set)
    print("selector_tensor_set: ", selector_tensor_set)
    print("node_placeholder_meta_dict: ", node_placeholder_meta_dict)
    print("node_meta_dict: ", node_meta_dict)
    
    # for node in node_group.nodes:
    #     print(node.op)
    #     # print(node.meta['nn_module_stack'])
    #     print(get_node_meta(node))
    # # print(node_group)

#  -------------------------------------------------------------
#  Analyze the graph and extract operations, 
#     1) input and output, 
#     2) selector, 
#     3) map 
#     4) reducer and aggregator
#  -------------------------------------------------------------


    # output in the fused submodule has two patterns:
    # 1) outputs 2) setitem 
    output_nodes = set()
    for node in traced_nodes:
        # trace the output node [TODO: check it for swe main]
        # if node.op == 'call_function' and "setitem" in str(node.target):
        #     output_nodes.add(
        #         (node.args[0].name)
        #     )
        if node.op == 'output':
            if isinstance(node.args[0], list):
                output_nodes.update([n.name if hasattr(n, 'name') \
                    else n for n in node.args[0]])
            else:
                output_nodes.add(node.args[0].name)
            # add outputs
            for _node in node.args[0]:
                node_shape, op, name, target, args, node_dtype = \
                    get_node_info(_node, node_meta_dict, node_placeholder_meta_dict)
                if op == 'call_module' and \
                list(node_meta_dict[name].meta["nn_module_stack"].values())[0][1] == Reducer:
                    target = 'reducer'
                if op == 'call_function' and "setitem" in str(target):
                    name = args[0]
                    target = args[-1]
                if op == 'call_function' and "sum" in str(target):
                    target = 'sum'
                outputs.append(
                    {
                        "op": op,
                        "name": name,
                        "target": target,
                        "dtype": "ValueT",
                        "shape": node_shape,
                        "args": args,
                        "dtype_data": node_dtype,
                    }
                )
    # debug print
    print("output_nodes: ", output_nodes)
    # exclude the output node
    traced_nodes = traced_nodes[:len(traced_nodes) - 1]

    #  -------------------------------------------------------------
    #  obtain the input and output, and the dimension of the input and output
    #  -------------------------------------------------------------
    for node in traced_nodes:
        node_shape, op, name, target, args, node_dtype = \
            get_node_info(node, node_meta_dict, node_placeholder_meta_dict)
        # if '.' in string of target, please replace it as '_'
        if isinstance(target, str) and '.' in target:
            target = target.replace('.', '_')
        if op == 'placeholder' and name not in output_nodes:
            # 1) obtain the input information, vertex and edge tensor
            inputs.append(
                {
                    "name": name,
                    "target": target,
                    "dtype": "scalar_t",
                    "shape": node_shape,
                    "op": op,
                    "args": args,
                    "dtype_data": node_dtype,
                }
            )
        elif op == 'call_module' and \
             list(node_meta_dict[name].meta["nn_module_stack"].values())[0][1] == Selector:
            # 2) obtain the selector index
            inputs.append(
                {
                    "name": name,
                    "target": target,
                    "dtype": "int",
                    "shape": node_shape,
                    "op": op,
                    "args": args,
                    "dtype_data": node_dtype,
                }
            )
    # debug print
    for inp in inputs:
        print(f"Input: {inp}")
    for out in outputs:
        print(f"Output: {out}")
    #  -------------------------------------------------------------
    #  obtain the selector in intermediate register, 
    #     including the vector x and edge tensor.
    #  1) the data from HBM  to register, we need to consider the alignment of the data
    #  2) we will need to input dimension here for the pragma unroll
    #  -------------------------------------------------------------
    
    for node in traced_nodes:
        
        node_shape, op, name, target, args, node_dtype = \
            get_node_info(node, node_meta_dict, node_placeholder_meta_dict)

        # edge tensor and vertex tensor traced from the graph
        # 1) edge tensor
        if op == 'placeholder' and name not in selector_tensor_set \
            and name not in output_nodes:
            selector_register.append(
                    {
                        "name": name,
                        "target": target,
                        "dtype": "TensorT",
                        "selector": 0, # edge tensor
                        "selector_name": name, # name and target are the same
                        "shape": node_shape,
                        "dtype_data": node_dtype,
                    }
                )
        # 2) vertex tensor
        if op == 'call_module' and name in selector_set:
            # if '.' in string of target, please replace it as '_'
            if isinstance(target, str) and '.' in target:
                target = target.replace('.', '_')
            selector_register.append(
                    {
                        "name": name,
                        "target": args[0].name,
                        "dtype": "TensorKeyT",
                        "selector": 1,  # vertex tensor
                        "selector_name": target, # name and target are not same
                        "shape": node_shape,
                        "dtype_data": node_dtype,
                    }
                )                
    # debug print
    for inter in selector_register:
        print(f"Selector register: {inter}")

    #  -------------------------------------------------------------
    #  obtain the map operations
    #  -------------------------------------------------------------
    for node in traced_nodes:

        node_shape, op, name, target, args, node_dtype = \
            get_node_info(node, node_meta_dict, node_placeholder_meta_dict)

        if op == 'call_module' and name not in selector_set:
            # obtain reducers
            map_operations.append({
                    'name': name,
                    'target': target,
                    'op': 'reducer', # a tag
                    'shape': node_shape,
                    'dtype_data': node_dtype,
                    'args': [arg.name if hasattr(arg, 'name') \
                        else arg for arg in args]
                })
        elif op == 'call_function' or op == 'call_method':
            target_name = str(target).split('.')[-1]
            if 'add' in str(target) and 'add_' not in str(target):
                target_name = 'add'
            elif 'mul' in str(target):
                target_name = 'mul'
            elif 'sub' in str(target):
                target_name = 'sub'
            elif 'norm' in str(target):
                target_name = 'norm'
            elif 'truediv' in str(target):
                target_name = 'truediv'
            elif 'norm' in str(target):
                target_name = 'norm'
                # [TODO]
            elif 'sum' in str(target):
                # aggregator
                target_name = 'sum'
            elif 'neg' in str(target):
                target_name = 'neg'
            elif 'add_' in str(target):
                target_name = 'add_'
            elif 'pow' in str(target):
                target_name = 'pow'
            elif 'setitem' in str(target):
                # intermediate map output
                target_name = 'setitem'
            elif 'getitem' in str(target):
                target_name = 'getitem'
            elif 'clone' in str(target):
                target_name = 'clone'
            elif 'copy_' in str(target):
                target_name = 'copy_'
            elif 'exp' in str(target):
                target_name = 'exp'
            elif 'abs' in str(target):
                target_name = 'abs'
            elif 'sign' in str(target):
                target_name = 'sign'
            elif ' lt' in f'{str(target)}':
                target_name = 'lt'
            elif ' gt' in f'{str(target)}':
                target_name = 'gt'
            elif ' where' in f'{str(target)}':                
                target_name = 'where'
            else:
                # error
                raise ValueError(f"Operation {target} not supported")
            
            # save the node information
            map_operations.append({
                'name': name,
                'target': target,
                'op': target_name,
                'shape': node_shape,
                'dtype_data': node_dtype,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in args]
            })
    # debug print
    for op in map_operations:
        print(f"Map operation: {op}")
    
    #  -------------------------------------------------------------
    #  obtain the reducer and aggregator operations
    #  -------------------------------------------------------------
    for node in traced_nodes:

        node_shape, op, name, target, args, node_dtype = \
            get_node_info(node, node_meta_dict, node_placeholder_meta_dict)

        if op == 'call_module' and name not in selector_set:
            # 1) obtain the reducer
            op_name = 'reducer'
            reducer_operations.append({
                'name': name,
                'op': op_name,
                'shape': node_shape,
                'dtype_data': node_dtype,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in args]
            })
        elif 'sum' in str(target):
            # 2) obtain the aggregator
            op_name = 'sum'
            
            aggregator_operations.append({
                'name': name,
                'op': op_name,
                'shape': node_shape,
                'dtype_data': node_dtype,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in args]
            })
            
    # debug print
    for op in reducer_operations:
        print(f"Reducer operation: {op}")
    for op in aggregator_operations:
        print(f"Aggregator operation: {op}")

    return inputs, outputs, selector_register, map_operations, \
        reducer_operations, aggregator_operations
