import argparse
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn
import operator

import easier as esr

from torch.utils.cpp_extension import load

# Add the parent directory to Python path to find in-repo packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.merged_gen_gpu import generate_cuda_code_from_graph
from codegen.merged_gen_cpu import generate_cpu_code_from_graph
from python_wrapper.dynamic_replace_submodule import dynamic_replace_submodule

from shallow_water_equation.swe_main import ShallowWaterEquation
from shallow_water_equation.assemble_shallow_water import ShallowWaterInitializer
from test_modules.utils import analyze_tensor_distribution, assemble_shallow_water_test_model, assemble_shallow_water_initializer_test_model


def trace_easier11(mesh_path: str, sw_path: str, device: str, backend: str, comm_backend: str, dt: float, verify: str):
    esr.init(comm_backend)
    if verify == "model":
        eqn = ShallowWaterEquation(mesh_path, sw_path, dt, device)
        [compiled_eqn] = esr.compile([eqn], backend)
        compiled_eqn()
    elif verify == "initializer":
        compiled_eqn = assemble_shallow_water_initializer_test_model(mesh_path, sw_path, device)
    else:  # component
        compiled_eqn = assemble_shallow_water_test_model(mesh_path, sw_path, device)

    
    # _ = analyze_tensor_distribution(compiled_eqn)
    # exit()
    
    # Collect call_module submodules
    graph = compiled_eqn.jit_engine.graph
    graph.print_tabular()
    chosen = None
    fallback = None
    for node in graph.nodes:
        if node.op == 'call_module':
            submod = compiled_eqn.get_submodule(node.target)
            if fallback is None:
                fallback = (submod, node)
            if node.name == 'easier0_select293':
                chosen = (submod, node)
                break
    # if chosen is None:
    #     chosen = fallback
    if chosen is None:
        raise RuntimeError("No call_module easier0_select293 nodes found in traced graph")
    submodule, node = chosen
    submodule.graph.print_tabular()
    return compiled_eqn, submodule, node.name


def save_model_state(model):
    """Save the current state of the model tensors"""
    state = {}
    state['h'] = model.h.collect().clone()
    state['uh'] = model.uh.collect().clone()
    state['vh'] = model.vh.collect().clone()
    return state


def restore_model_state(model, state):
    """Restore the model tensors to a saved state"""
    model.h.copy_(state['h'])
    model.uh.copy_(state['uh'])
    model.vh.copy_(state['vh'])


def save_components_state(components):
    """Save snapshot of ShallowWaterMeshComponentsCollector outputs."""
    state = {}
    state['src_p'] = [components.src_p[i].collect().clone() for i in range(3)]
    state['dst_p'] = [components.dst_p[i].collect().clone() for i in range(3)]
    state['cells_p'] = [components.cells_p[i].collect().clone() for i in range(3)]
    state['bp'] = [components.bp[i].collect().clone() for i in range(2)]
    return state


def restore_components_state(components, state):
    """Restore ShallowWaterMeshComponentsCollector outputs from a snapshot."""
    for i in range(3):
        components.src_p[i].copy_(state['src_p'][i])
        components.dst_p[i].copy_(state['dst_p'][i])
        components.cells_p[i].copy_(state['cells_p'][i])
    for i in range(2):
        components.bp[i].copy_(state['bp'][i])


def save_initializer_state(initializer):
    """Save snapshot of ShallowWaterInitializer outputs."""
    state = {}
    state['x'] = initializer.x.collect().clone()
    state['y'] = initializer.y.collect().clone()
    state['area'] = initializer.area.collect().clone()
    state['sx'] = initializer.sx.collect().clone()
    state['sy'] = initializer.sy.collect().clone()
    state['bsx'] = initializer.bsx.collect().clone()
    state['bsy'] = initializer.bsy.collect().clone()
    state['h'] = initializer.h.collect().clone()
    state['alpha'] = initializer.alpha.collect().clone()
    return state


def restore_initializer_state(initializer, state):
    """Restore ShallowWaterInitializer outputs from a snapshot."""
    initializer.x.copy_(state['x'])
    initializer.y.copy_(state['y'])
    initializer.area.copy_(state['area'])
    initializer.sx.copy_(state['sx'])
    initializer.sy.copy_(state['sy'])
    initializer.bsx.copy_(state['bsx'])
    initializer.bsy.copy_(state['bsy'])
    initializer.h.copy_(state['h'])
    initializer.alpha.copy_(state['alpha'])


def coo_rows_to_csr_ptr(row_ids: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Convert COO row indices to CSR row pointer (crow_indices) of length num_rows+1.
    Uses torch._convert_indices_from_coo_to_csr if available; falls back to bincount+cumsum.
    Returns int64 on CPU; caller can cast/move as needed.
    """
    if row_ids.dtype != torch.int64:
        row_ids = row_ids.to(torch.int64)
    # Prefer built-in if present
    conv = getattr(torch, "_convert_indices_from_coo_to_csr", None)
    if callable(conv):
        crow = conv(row_ids, size=num_rows)
        return crow
    # Fallback: bincount then cumsum
    counts = torch.bincount(row_ids, minlength=num_rows)
    crow = torch.empty(num_rows + 1, dtype=torch.int64)
    crow[0] = 0
    crow[1:] = torch.cumsum(counts, dim=0)
    return crow


def run_model_steps(model, num_steps=10, verify: str = "component"):
    """Run the model for a specified number of steps and capture final state"""
    for step in range(num_steps):
        model()
        print(f"Step {step + 1}/{num_steps} completed")

    if verify == "model":
        return save_model_state(model)
    if verify == "initializer":
        return save_initializer_state(model)
    return save_components_state(model)


def compare_outputs(original_state, replaced_state, tolerance=1e-10):
    """Compare outputs between original and replaced models"""
    print("\n=== Correctness Comparison ===")
    
    all_close = True
    for var in ['h', 'uh', 'vh']:
        orig_tensor = original_state[var]
        repl_tensor = replaced_state[var]
        
        # Compute differences
        abs_diff = torch.abs(orig_tensor - repl_tensor)
        rel_diff = abs_diff / (torch.abs(orig_tensor) + 1e-15)
        
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        is_close = torch.allclose(orig_tensor, repl_tensor, rtol=tolerance, atol=tolerance)
        all_close = all_close and is_close
        
        print(f"\n{var.upper()} Variable:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        print(f"  All close (tol={tolerance}): {is_close}")
    
    print(f"\n=== OVERALL RESULT ===")
    if all_close:
        print("✅ PASSED: Replaced model outputs match original model within tolerance")
    else:
        print("❌ FAILED: Replaced model outputs differ from original model")
    
    return all_close
def compare_components_outputs(original_state, replaced_state):
    """Compare outputs for ShallowWaterMeshComponentsCollector.
    Expects dict with lists of integer tensors: src_p[3], dst_p[3], cells_p[3], bp[2].
    Uses exact equality checks since tensors are integer indices.
    """
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
    """Compare outputs for ShallowWaterInitializer.
    Expects dict with float tensors: x, y, area, sx, sy, bsx, bsy, h, alpha.
    Uses allclose with the provided tolerance.
    """
    print("\n=== Correctness Comparison (Initializer) ===")

    keys = ['x', 'y', 'area', 'sx', 'sy', 'bsx', 'bsy', 'h', 'alpha']
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
    parser = argparse.ArgumentParser(description="Generate and load CUDA code for easier0_select293 and test correctness")
    parser.add_argument("mesh", type=str, help="Path to mesh HDF5 file")
    parser.add_argument("sw", type=str, help="Path to shallow water HDF5 file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default="torch", choices=["none", "torch", "cpu", "cuda"])
    parser.add_argument("--comm-backend", type=str, default="nccl", choices=["gloo", "nccl"])
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--keep", action="store_true", help="Keep generated files")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps to run for testing")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Tolerance for correctness comparison")
    parser.add_argument("--verify", type=str, default="model", choices=["model", "initializer", "component"], help="Which pipeline to verify")

    args = parser.parse_args()

        
    # Step 1: Trace and get the original model
    print("=== Step 1: Tracing original model ===")
    compiled_eqn, submodule, target_node_name = trace_easier11(
        args.mesh,
        args.sw,
        args.device,
        args.backend,
        args.comm_backend,
        args.dt,
        args.verify,
    )
    compiled_eqn()
    # Step 2: Save initial state and run original model
    print("\n=== Step 2: Running original model ===")
    if args.verify == "model":
        initial_state = save_model_state(compiled_eqn)
    elif args.verify == "initializer":
        initial_state = save_initializer_state(compiled_eqn)
    else:
        initial_state = save_components_state(compiled_eqn)
    original_final_state = run_model_steps(compiled_eqn, args.steps, args.verify)
    # print(f"Original model final state: {original_final_state}")
    
    # Step 3: Generate and build CUDA extension
    print("\n=== Step 3: Generating and building CUDA extension ===")
    if args.device == "cuda":
        generate_cuda_code_from_graph(submodule, compiled_eqn)
    else:
        generate_cpu_code_from_graph(submodule, compiled_eqn)
    extension = build_extension(args.device)
    
    # Step 4: Restore initial state and replace submodule
    print("\n=== Step 4: Replacing submodule with CUDA extension ===")
    ok = dynamic_replace_submodule(compiled_eqn, extension, target_node_name)
    if not ok:
        raise RuntimeError(f"Failed to locate target submodule {target_node_name} for replacement")
    print("CUDA extension compiled and registered successfully")
    
    # Step 5: Run model with replaced submodule
    print("\n=== Step 5: Running model with replaced submodule ===")
    if args.verify == "model":
        restore_model_state(compiled_eqn, initial_state)
    elif args.verify == "initializer":
        restore_initializer_state(compiled_eqn, initial_state)
    else:
        restore_components_state(compiled_eqn, initial_state)
    replaced_final_state = run_model_steps(compiled_eqn, args.steps, args.verify)
    # print(f"Replaced model final state: {replaced_final_state}")
    
    # Step 6: Compare outputs
    print("\n=== Step 6: Comparing outputs ===")
    if args.verify == "model":
        correctness_passed = compare_outputs(original_final_state, replaced_final_state, args.tolerance)
    elif args.verify == "initializer":
        correctness_passed = compare_initializer_outputs(original_final_state, replaced_final_state, args.tolerance)
    else:
        correctness_passed = compare_components_outputs(original_final_state, replaced_final_state)
    
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    main()




