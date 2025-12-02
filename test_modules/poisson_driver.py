import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

import easier as esr

# Add the parent directory to Python path to find in-repo packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.merged_gen_gpu import generate_cuda_code_from_graph
from codegen.merged_gen_cpu import generate_cpu_code_from_graph
from python_wrapper.dynamic_replace_submodule import dynamic_replace_submodule

from poisson.assemble_poisson import PoissonInitializer
from test_modules.utils import (
    analyze_tensor_distribution,
    assemble_poisson_components_test_model,
    assemble_poisson_initializer_test_model,
    assemble_poisson_model_test_model,
)


def trace_poisson(mesh_path: str, poisson_path: str, device: str, backend: str, comm_backend: str, verify: str, target_name: str | None = None):
    esr.init(comm_backend)
    if verify == "initializer":
        compiled = assemble_poisson_initializer_test_model(mesh_path, poisson_path, device)
    else:  # component
        compiled = assemble_poisson_components_test_model(mesh_path, device)

    graph = compiled.jit_engine.graph
    graph.print_tabular()
    chosen = None
    fallback = None
    for node in graph.nodes:
        if node.op == 'call_module':
            submod = compiled.get_submodule(node.target)
            submod.graph.print_tabular()
            if fallback is None:
                fallback = (submod, node)
            if target_name is not None and node.name == target_name:
                chosen = (submod, node)
                break
    if chosen is None:
        if fallback is None:
            raise RuntimeError("No call_module nodes found in traced graph")
        chosen = fallback
    submodule, node = chosen
    submodule.graph.print_tabular()
    return compiled, submodule, node.name


def trace_poisson_model(mesh_path: str, poisson_path: str, device: str, backend: str, comm_backend: str, solver: str, target_name: str | None = None):
    esr.init(comm_backend)
    compiled_eqn, compiled_solver = assemble_poisson_model_test_model(mesh_path, poisson_path, device, backend, solver)


    graph_eqn = compiled_eqn.jit_engine.graph
    graph_solver = compiled_solver.jit_engine.graph
    graph_eqn.print_tabular()
    graph_solver.print_tabular()
    exit()
    chosen = None
    fallback = None
    for node in graph.nodes:
        if node.op == 'call_module':
            submod = compiled_eqn.get_submodule(node.target)
            if fallback is None:
                fallback = (submod, node)
            if target_name is not None and node.name == target_name:
                chosen = (submod, node)
                break
    if chosen is None:
        if fallback is None:
            raise RuntimeError("No call_module nodes found in traced graph")
        chosen = fallback
    submodule, node = chosen
    submodule.graph.print_tabular()
    return compiled_eqn, compiled_solver, submodule, node.name


def save_components_state(components):
    state = {}
    state['src_p'] = [components.src_p[i].collect().clone() for i in range(3)]
    state['dst_p'] = [components.dst_p[i].collect().clone() for i in range(3)]
    state['cells_p'] = [components.cells_p[i].collect().clone() for i in range(3)]
    state['bp'] = [components.bp[i].collect().clone() for i in range(2)]
    return state


def restore_components_state(components, state):
    for i in range(3):
        components.src_p[i].copy_(state['src_p'][i])
        components.dst_p[i].copy_(state['dst_p'][i])
        components.cells_p[i].copy_(state['cells_p'][i])
    for i in range(2):
        components.bp[i].copy_(state['bp'][i])


def save_initializer_state(initializer: PoissonInitializer):
    state = {}
    state['b'] = initializer.b.collect().clone()
    state['Ac'] = initializer.Ac.collect().clone()
    state['Af'] = initializer.Af.collect().clone()
    state['rho'] = initializer.rho.collect().clone()
    state['centroid'] = initializer.centroid.collect().clone()
    return state


def restore_initializer_state(initializer: PoissonInitializer, state):
    initializer.b.copy_(state['b'])
    initializer.Ac.copy_(state['Ac'])
    initializer.Af.copy_(state['Af'])
    initializer.rho.copy_(state['rho'])
    initializer.centroid.copy_(state['centroid'])


def save_model_state(eqn):
    state = {}
    state['x'] = eqn.x.collect().clone()
    return state


def restore_model_state(eqn, state):
    eqn.x.copy_(state['x'])


def run_poisson_model_solver(solver, atol=1e-5, maxiter=1000, debug_iter=10):
    info = solver.solve(atol=atol, maxiter=maxiter, debug_iter=debug_iter)
    return info


def compare_model_outputs(original_state, replaced_state, tolerance=1e-10):
    print("\n=== Correctness Comparison (Model: x) ===")
    a = original_state['x']
    b = replaced_state['x']
    if a.shape != b.shape:
        print(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    abs_diff = (a - b).abs()
    denom = a.abs() + 1e-15
    rel_diff = abs_diff / denom
    max_abs_diff = abs_diff.max().item() if a.numel() > 0 else 0.0
    max_rel_diff = rel_diff.max().item() if a.numel() > 0 else 0.0
    mean_abs_diff = abs_diff.mean().item() if a.numel() > 0 else 0.0
    mean_rel_diff = rel_diff.mean().item() if a.numel() > 0 else 0.0
    is_close = torch.allclose(a, b, rtol=tolerance, atol=tolerance)
    print(f"Max abs diff: {max_abs_diff:.2e}")
    print(f"Max rel diff: {max_rel_diff:.2e}")
    print(f"Mean abs diff: {mean_abs_diff:.2e}")
    print(f"Mean rel diff: {mean_rel_diff:.2e}")
    print(f"All close (tol={tolerance}): {is_close}")
    return is_close


def run_model_steps(model, num_steps=3):
    for step in range(num_steps):
        model()
        print(f"Step {step + 1}/{num_steps} completed")


def compare_components_outputs(original_state, replaced_state):
    print("\n=== Correctness Comparison (Components) ===")
    all_match = True

    def compare_int_tensor(name, idx, a, b):
        nonlocal all_match
        if a.dtype != b.dtype:
            print(f"\n{name}[{idx}]: dtype mismatch {a.dtype} vs {b.dtype}")
            all_match = False
            return
        if a.shape != b.shape:
            print(f"\n{name}[{idx}]: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            all_match = False
            return
        equal_mask = torch.eq(a, b)
        mismatches = torch.numel(equal_mask) - int(equal_mask.sum().item())
        max_abs_diff = int((a.to(torch.int64) - b.to(torch.int64)).abs().max().item()) if a.numel() > 0 else 0
        if mismatches == 0:
            print(f"{name}[{idx}]: OK (all equal)")
        else:
            frac = mismatches / a.numel() if a.numel() > 0 else 0.0
            print(f"{name}[{idx}]: {mismatches} mismatches ({frac:.2%}), max |diff|={max_abs_diff}")
            all_match = False

    for i in range(3):
        compare_int_tensor('SRC_P', i, original_state['src_p'][i], replaced_state['src_p'][i])
        compare_int_tensor('DST_P', i, original_state['dst_p'][i], replaced_state['dst_p'][i])
        compare_int_tensor('CELLS_P', i, original_state['cells_p'][i], replaced_state['cells_p'][i])

    for i in range(2):
        compare_int_tensor('BP', i, original_state['bp'][i], replaced_state['bp'][i])

    print(f"\n=== OVERALL RESULT (Components) ===")
    if all_match:
        print("✅ PASSED: Component outputs match exactly")
    else:
        print("❌ FAILED: Component outputs differ")

    return all_match


def compare_initializer_outputs(original_state, replaced_state, tolerance=1e-10):
    print("\n=== Correctness Comparison (Initializer) ===")
    keys = ['b', 'Ac', 'Af', 'rho', 'centroid']
    all_close = True
    for key in keys:
        a = original_state[key]
        b = replaced_state[key]
        if a.shape != b.shape:
            print(f"\n{key}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            all_close = False
            continue
        abs_diff = (a - b).abs()
        denom = a.abs() + 1e-15
        rel_diff = abs_diff / denom
        max_abs_diff = abs_diff.max().item() if a.numel() > 0 else 0.0
        max_rel_diff = rel_diff.max().item() if a.numel() > 0 else 0.0
        mean_abs_diff = abs_diff.mean().item() if a.numel() > 0 else 0.0
        mean_rel_diff = rel_diff.mean().item() if a.numel() > 0 else 0.0
        is_close = torch.allclose(a, b, rtol=tolerance, atol=tolerance)
        all_close = all_close and is_close

        print(f"\n{key.upper()}:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        print(f"  All close (tol={tolerance}): {is_close}")

    print(f"\n=== OVERALL RESULT (Initializer) ===")
    if all_close:
        print("✅ PASSED: Initializer outputs match within tolerance")
    else:
        print("❌ FAILED: Initializer outputs differ")

    return all_close


def build_extension(device: str):
    project_root = "/home/panq/dev/FlexSpmv"
    include_dirs = [project_root, os.path.join(project_root, "include")]

    from torch.utils.cpp_extension import load

    if device == "cuda":
        print("Building CUDA extension")
        src = os.path.join(project_root, "merged_binding.cu")
        ext = load(
            name="flex_spmv_ext",
            sources=[src],
            extra_include_paths=include_dirs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
            ],
            build_directory=tempfile.mkdtemp(prefix="easier_build_"),
            keep_intermediates=True,
            verbose=True,
        )
    else:
        print("Building CPU extension")
        src = os.path.join(project_root, "merged_binding_cpu.cpp")
        ext = load(
            name="flex_spmv_ext",
            sources=[src],
            extra_include_paths=include_dirs,
            extra_cflags=["-O3", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            build_directory=tempfile.mkdtemp(prefix="easier_build_"),
            keep_intermediates=True,
            verbose=True,
        )
    return ext


def main():
    parser = argparse.ArgumentParser(description="Generate and load CUDA/CPU code for a Poisson pipeline and test correctness")
    parser.add_argument("mesh", type=str, help="Path to mesh HDF5 file")
    parser.add_argument("poisson", type=str, help="Path to Poisson HDF5 file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default="torch", choices=["none", "torch", "cpu", "cuda"])
    parser.add_argument("--comm-backend", type=str, default="nccl", choices=["gloo", "nccl"])
    parser.add_argument("--keep", action="store_true", help="Keep generated files")
    parser.add_argument("--steps", type=int, default=3, help="Number of runs for testing")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Tolerance for correctness comparison")
    parser.add_argument("--verify", type=str, default="initializer", choices=["initializer", "component", "model"], help="Which pipeline to verify")
    parser.add_argument("--target-name", type=str, default=None, help="Optional target node name to replace; defaults to first call_module")
    parser.add_argument("--solver", type=str, default="cg", choices=["cg", "gmres"], help="Solver for model verification")

    args = parser.parse_args()

    print("=== Step 1: Tracing original model ===")
    if args.verify == "model":
        compiled, solver, submodule, target_node_name = trace_poisson_model(
            args.mesh,
            args.poisson,
            args.device,
            args.backend,
            args.comm_backend,
            args.solver,
            args.target_name,
        )
        # No direct call on eqn; solving is done via solver
    else:
        compiled, submodule, target_node_name = trace_poisson(
            args.mesh,
            args.poisson,
            args.device,
            args.backend,
            args.comm_backend,
            args.verify,
            args.target_name,
        )
        compiled()

    print("\n=== Step 2: Running original model ===")
    if args.verify == "model":
        initial_state = save_model_state(compiled)
        run_poisson_model_solver(solver)
        original_state = save_model_state(compiled)
    else:
        if args.verify == "initializer":
            initial_state = save_initializer_state(compiled)
        else:
            initial_state = save_components_state(compiled)
        run_model_steps(compiled, args.steps)
        if args.verify == "initializer":
            original_state = save_initializer_state(compiled)
        else:
            original_state = save_components_state(compiled)

    print("\n=== Step 3: Generating and building extension ===")
    if args.device == "cuda":
        generate_cuda_code_from_graph(submodule, compiled)
    else:
        generate_cpu_code_from_graph(submodule, compiled)
    extension = build_extension(args.device)

    print("\n=== Step 4: Replacing submodule with compiled extension ===")
    ok = dynamic_replace_submodule(compiled, extension, target_node_name)
    if not ok:
        raise RuntimeError(f"Failed to locate target submodule {target_node_name} for replacement")
    print("Extension compiled and registered successfully")

    print("\n=== Step 5: Running model with replaced submodule ===")
    if args.verify == "model":
        restore_model_state(compiled, initial_state)
        run_poisson_model_solver(solver)
        replaced_state = save_model_state(compiled)
    else:
        if args.verify == "initializer":
            restore_initializer_state(compiled, initial_state)
        else:
            restore_components_state(compiled, initial_state)
        run_model_steps(compiled, args.steps)
        if args.verify == "initializer":
            replaced_state = save_initializer_state(compiled)
        else:
            replaced_state = save_components_state(compiled)

    print("\n=== Step 6: Comparing outputs ===")
    if args.verify == "initializer":
        passed = compare_initializer_outputs(original_state, replaced_state, args.tolerance)
    elif args.verify == "component":
        passed = compare_components_outputs(original_state, replaced_state)
    else:
        passed = compare_model_outputs(original_state, replaced_state, args.tolerance)

    return 0 if passed else 1


if __name__ == "__main__":
    main()


