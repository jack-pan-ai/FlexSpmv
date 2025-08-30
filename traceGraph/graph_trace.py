from itertools import zip_longest

import easier as esr
from easier.core.jit import EasierTracer

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
    # print("FX Graph:")
    traced_model.print_tabular()    
    return traced_model

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

#  -------------------------------------------------------------
#  Analyze the graph and extract operations, 1) input and output, 2) selector, 3) map 4) reducer and aggregator
#  -------------------------------------------------------------

    #  -------------------------------------------------------------
    #  obtain the input and output, and the dimension of the input and output
    #  -------------------------------------------------------------
    for node in traced_model.nodes:
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
    _nodes = list(traced_model.nodes)  # O(n), done once
    _n = len(_nodes)
    for i in range(_n):
        # current node and forward node 
        node_current = _nodes[i]
        node_forward = _nodes[i + 1] if i < _n - 1 else None

        # if the current node is a get_attr, and the forward node is a call_module and the target is a selector, then add the selector to the selector register
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

    # debug print
    # for inter in selector_register:
    #     print(f"Selector register: {inter}")

    #  -------------------------------------------------------------
    #  obtain the map operations
    #  -------------------------------------------------------------
    for node in traced_model.nodes:
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
    for node in traced_model.nodes:
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
    # # debug print
    # for op in reducer_operations:
    #     print(f"Reducer operation: {op}")
    # for op in aggregator_operations:
    #     print(f"Aggregator operation: {op}")

    return inputs, outputs, selector_register, map_operations, reducer_operations, aggregator_operations