from itertools import zip_longest

import easier as esr
from easier.core.runtime.metadata import get_node_meta
from easier import Reducer, Selector
from easier.core.runtime.metadata import Role

# Part 2: Generate CUDA code from the graph
def trace_graph(traced_model):

    # Analyze the graph and extract operations
    inputs = []
    _input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    outputs = []

    # obtain the traced nodes list
    traced_nodes = list(traced_model.graph.nodes)
    

#  -------------------------------------------------------------
#  Analyze the graph and extract operations, 
#     1) input and output, 
#     2) selector, 
#     3) map 
#     4) reducer and aggregator
#  -------------------------------------------------------------


    # three get_attr involved: 1) egde tensor; 2) vertex tensor;
    #  3) output; here we need to tell them apart
    selector_vertices = set()
    output_nodes = set()
    for node in traced_nodes:
        # trace the selector input
        if node.op == 'call_module':
            _class_name = node.meta['nn_module_stack'][node.target][1]
            print(_class_name)
            print(_class_name == Selector)
            if _class_name == Selector:
                selector_vertices.add(
                    node.args[0].name
                )
        # trace the output node
        if node.op == 'call_function' and "setitem" in str(node.target):
            output_nodes.add(
                node.args[0].name
            )
    # debug print
    print(selector_vertices)
    print(output_nodes)

    #  -------------------------------------------------------------
    #  obtain the input and output, and the dimension of the input and output
    #  -------------------------------------------------------------
    _reducer_set = set()
    for node in traced_nodes:
        # obtain the shape of the node
        # the first dimension is the batch size
        node_shape = get_node_meta(node).shape[1:]
        op = node.op
        name = node.name
        target = node.target
        args = node.args
        
        if op == 'get_attr' and name not in output_nodes:
            # 1) obtain the input information, vertex and edge tensor
            inputs.append(
                {
                    "name": name,
                    "target": target,
                    "dtype": "scalar_t",
                    "shape": node_shape,
                }
            )
        elif op == 'call_module' and node.meta['nn_module_stack'][target][1] == Selector:
            # 2) obtain the selector index
            inputs.append(
                {
                    "name": name,
                    "target": target,
                    "dtype": "int",
                    "shape": node_shape,
                }
            )
        elif op == 'call_module' and node.meta['nn_module_stack'][target][1] == Reducer:
            _reducer_set.add(name)
        elif op == 'call_function' and "setitem" in str(target):
            # 3) obtain the output information
            # 3.1) Aggregator: function sum is tag name for code generation
            if get_node_meta(node).role == Role.REPLICATED:
                target = 'function sum'
                node_shape = get_node_meta(node).shape
            # 3.2) Reducer: the function reducer is tag name for code generation
            _reducer_tag_name = str(args[-1])
            if _reducer_tag_name in _reducer_set:
                target = 'function reducer'
            
            # save the output information
            outputs.append(
                {
                    "name": args[0].name,
                    "target": target,
                    "dtype": "ValueT",
                    "shape": node_shape,
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
        
        node_shape = get_node_meta(node).shape[1:]
        op = node.op
        name = node.name
        target = node.target
        args = node.args

        # edge tensor and vertex tensor traced from the graph
        # 1) edge tensor
        if op == 'get_attr' and name not in selector_vertices \
            and name not in output_nodes:
            selector_register.append(
                    {
                        "name": name,
                        "target": target,
                        "dtype": "TensorT",
                        "selector": 0, # edge tensor
                        "selector_name": name,
                        "shape": node_shape
                    }
                )
        # 2) vertex tensor
        if op == 'call_module' and node.meta['nn_module_stack'][target][1] == Selector:
            selector_register.append(
                    {
                        "name": name,
                        "target": args[0].target,
                        "dtype": "TensorKeyT",
                        "selector": 1,  # vertex tensor
                        "selector_name": target,
                        "shape": node_shape
                    }
                )                
    # debug print
    for inter in selector_register:
        print(f"Selector register: {inter}")

    #  -------------------------------------------------------------
    #  obtain the map operations
    #  -------------------------------------------------------------
    traced_nodes_iter = iter(traced_nodes) # used for the aggregator only
    for node in traced_nodes_iter:

        node_shape = get_node_meta(node).shape[1:]
        op = node.op
        name = node.name
        target = node.target
        args = node.args

        if op == 'call_module' and node.meta["nn_module_stack"][node.target][1] == Reducer:
            # obtain reducers
            map_operations.append({
                    'name': node.name,
                    'target': target,
                    'op': 'reducer', # a tag
                    'shape': node_shape,
                    'args': [arg.name if hasattr(arg, 'name') else arg for arg in node.args]
                })
        elif op == 'call_function' or op == 'call_method':
            target_name = str(target).split('.')[-1]
            if 'add' in str(target):
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
            elif 'sum' in str(target):
                # aggregator
                target_name = 'sum'
                args = args
                op = 'sum'
                for _ in range(3):
                    # skip the next 3 nodes which is used for communications
                    next(traced_nodes_iter, None)
                _node_agg = next(traced_nodes_iter, None)
                name = str(_node_agg.args[0])
            elif 'setitem' in str(target):
                # check the output name, replace it
                output_name_replace = str(args[0].name)
                output_name_original = str(args[-1].name)
                for op in map_operations:
                    if op['name'] == output_name_original:
                        op['name'] = output_name_replace
                        break
                continue
            else:
                # error
                raise ValueError(f"Operation {target} not supported")
            
            # save the node information
            map_operations.append({
                'name': name,
                'target': target,
                'op': target_name,
                'shape': node_shape,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in args]
            })
    # debug print
    for op in map_operations:
        print(f"Map operation: {op}")
    
    #  -------------------------------------------------------------
    #  obtain the reducer and aggregator operations
    #  -------------------------------------------------------------
    traced_nodes_iter = iter(traced_nodes) # used for the aggregator only
    for node in traced_nodes_iter:

        node_shape = get_node_meta(node).shape[1:]
        op = node.op
        name = node.name
        target = node.target
        args = node.args

        if op == 'call_module' and node.meta["nn_module_stack"][node.target][1] == Reducer:
            # 1) obtain the reducer
            op_name = 'reducer'
            reducer_operations.append({
                'name': name,
                'op': op_name,
                'shape': node_shape,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in args]
            })
        elif 'sum' in str(target):
            # 2) obtain the aggregator
            op_name = 'sum'
            for _ in range(3):
                # skip the next 3 nodes which is used for communications
                next(traced_nodes_iter, None)
            _node_agg = next(traced_nodes_iter, None)
            name = str(_node_agg.args[0])

            aggregator_operations.append({
                'name': name,
                'op': op_name,
                'shape': node_shape,
                'args': [arg.name if hasattr(arg, 'name') else arg for arg in args]
            })
        # replace the output name
        if op == 'call_function' and "setitem" in str(target):
            output_name_replace = str(args[0].name)
            output_name_original = str(args[-1].name)
            for op in reducer_operations:
                if op['name'] == output_name_original:
                    op['name'] = output_name_replace
                    break
            
    # debug print
    for op in reducer_operations:
        print(f"Reducer operation: {op}")
    for op in aggregator_operations:
        print(f"Aggregator operation: {op}")

    return inputs, outputs, selector_register, map_operations, reducer_operations, aggregator_operations
