# utils for easier

def analyze_tensor_distribution(traced_model):

    graph = traced_model.jit_engine.graph
    graph.print_tabular()
    submodules = []
    for node in graph.nodes:
        if node.op == 'call_module':
            submod = traced_model.get_submodule(node.target)
            submod.graph.print_tabular()
            submodules.append((submod, node))
    return submodules