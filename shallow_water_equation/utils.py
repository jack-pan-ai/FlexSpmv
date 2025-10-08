def get_submodules(eqn, run_to_collect=True):
    # get the graph of the eqn
    if run_to_collect:
        eqn()
    graph = eqn.jit_engine.graph
    graph.print_tabular()
    
    submodules = []
    for node in graph.nodes:
        if node.op == 'call_module' and node.name == 'easier5_select_reduce92':
            submod = eqn.get_submodule(node.target)
            submod.graph.print_tabular()
            submodules.append((submod, node))
    return submodules