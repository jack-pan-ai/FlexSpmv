# utils for easier

from shallow_water_equation.assemble_shallow_water import ShallowWaterMeshComponentsCollector, ShallowWaterInitializer
from poisson.assemble_poisson import PoissonMeshComponentsCollector, PoissonInitializer
from poisson.poisson_main import Poisson
import easier as esr
from easier.numeric.solver import CG, GMRES

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


def assemble_poisson_components_test_model(mesh: str, device='gpu'):
    components = PoissonMeshComponentsCollector(mesh)
    components.to(device)

    [components] = esr.compile(
        [components], 'torch', partition_mode='evenly'
    )  # type: ignore
    components: PoissonMeshComponentsCollector
    components()

    return components


def assemble_poisson_initializer_test_model(mesh: str, poisson: str, device='gpu'):
    initializer = PoissonInitializer(poisson, mesh)
    initializer.to(device)

    [initializer] = esr.compile(
        [initializer], 'torch', partition_mode='evenly'
    )  # type: ignore
    initializer: PoissonInitializer
    initializer()

    return initializer


def assemble_poisson_model_test_model(mesh: str, poisson: str, device='gpu', backend='torch', solver: str = 'cg'):
    eqn = Poisson(mesh, poisson, device)
    if solver == 'cg':
        sol = CG(eqn.A, eqn.b, eqn.x)
    else:
        sol = GMRES(eqn.A, eqn.b, eqn.x)

    eqn, sol = esr.compile([eqn, sol], backend)
    
    eqn()
    sol()
    return eqn, sol
