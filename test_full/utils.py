# utils for easier

from shallow_water_equation.assemble_shallow_water import ShallowWaterMeshComponentsCollector, ShallowWaterInitializer
import easier as esr

def analyze_tensor_distribution(traced_model):

    graph = traced_model.jit_engine.graph
    graph.print_tabular()
    submodules = []
    for node in graph.nodes:
        if node.op == 'call_module':
            submod = traced_model.get_submodule(node.target)
            print(node.name)
            submod.graph.print_tabular()
            submodules.append((submod, node))
    return submodules


def get_submodules(eqn, run_to_collect=True):
    # get the call_module of the eqn
    if run_to_collect:
        eqn()
    graph = eqn.jit_engine.graph
    submodules = []
    nodes = []
    for node in graph.nodes:
        if node.op == 'call_module':
            submod = eqn.get_submodule(node.target)
            submodules.append(submod)
            nodes.append(node)
    return submodules, nodes


def assemble_shallow_water_test_model(mesh: str, shallow_water: str, device='gpu'):
    components = ShallowWaterMeshComponentsCollector(mesh)
    components.to(device)

    [components] = esr.compile(
        [components], 'torch', partition_mode='evenly'
    )  # type: ignore
    components: ShallowWaterMeshComponentsCollector
    components()
    
    return components #initializer


def assemble_shallow_water_initializer_test_model(mesh: str, shallow_water: str, device='gpu'):
    initializer = ShallowWaterInitializer(shallow_water, mesh)
    initializer.to(device)

    [initializer] = esr.compile(
        [initializer], 'torch', partition_mode='evenly'
    )  # type: ignore
    initializer: ShallowWaterInitializer
    initializer()

    return initializer
