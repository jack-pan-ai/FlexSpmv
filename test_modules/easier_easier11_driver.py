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

from shallow_water_equation.swe_main import ShallowWaterEquation
from shallow_water_equation.utils import get_submodules


def trace_easier11(mesh_path: str, sw_path: str, device: str, backend: str, comm_backend: str, dt: float):
    esr.init(comm_backend)
    eqn = ShallowWaterEquation(mesh_path, sw_path, dt, device)
    [compiled_eqn] = esr.compile([eqn], backend)
    compiled_eqn()
    submodules = get_submodules(compiled_eqn, run_to_collect=False)
    for (submodule, node) in submodules:
        if node.name == "easier5_select_reduce92":
            return compiled_eqn, submodule
    raise RuntimeError("easier5_select_reduce92 not found in traced graph")


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


def run_model_steps(model, num_steps=1):
    """Run the model for a specified number of steps and capture final state"""
    for step in range(num_steps):
        model()
        print(f"Step {step + 1}/{num_steps} completed")
    
    # Return final state
    final_state = save_model_state(model)
    return final_state


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
def build_extension():
    print("Building CUDA extension")
    project_root = "/home/panq/dev/FlexSpmv"
    src = os.path.join(project_root, "merged_binding.cu")
    include_dirs = [project_root, os.path.join(project_root, "include")]
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
        verbose=True,
    )
    return ext


def replace_submodule(
    compiled_eqn,
    ext,
    submodule_name: str = "easier5_select_reduce92",
    input_names = ("add_10", "bsx", "bsy"),
):
    print(f"Replacing submodule {submodule_name} with CUDA extension")
    # Find target GraphModule by name
    for (gm, node) in get_submodules(compiled_eqn, run_to_collect=False):
        if node.name != submodule_name:
            continue

        # Extract required static tensors from the original submodule
        # gather_b.idx and scatter_b.idx
        try:
            gather_idx = gm.get_submodule("gather_b").idx
            row_end_offsets = gm.get_submodule("scatter_b").idx
        except Exception as e:
            raise RuntimeError(f"Failed to access required indices from submodule '{submodule_name}': {e}")

        # Define a lightweight wrapper module that matches the original call signature
        class _Easier5Cuda(nn.Module):
            def __init__(self, ext_module, gather_idx_tensor, row_end_offsets_tensor):
                super().__init__()
                # Ensure dtypes/contiguity for indices expected by the kernel (int32 offsets)
                gather_idx_tensor = gather_idx_tensor.contiguous()
                row_end_offsets_tensor = row_end_offsets_tensor.contiguous()
                if gather_idx_tensor.dtype != torch.int32:
                    gather_idx_tensor = gather_idx_tensor.to(torch.int32)
                if row_end_offsets_tensor.dtype != torch.int32:
                    row_end_offsets_tensor = row_end_offsets_tensor.to(torch.int32)

                self.register_buffer("gather_b_1_idx", gather_idx_tensor, persistent=False)
                self.register_buffer("row_end_offsets", row_end_offsets_tensor, persistent=False)
                self._ext = ext_module

            def forward(self, add_10, bsx, bsy):
                # Keep original arg order: (add_10, bsx, bsy)
                device = add_10.device
                gi = self.gather_b_1_idx
                ro = self.row_end_offsets
                if gi.device != device:
                    gi = gi.to(device, non_blocking=True)
                if ro.device != device:
                    ro = ro.to(device, non_blocking=True)

                # Use CSR convention: ro length == num_rows + 1
                num_rows = int(ro.numel() - 1)

                # Ensure contiguity of runtime tensors
                add_10_c = add_10.contiguous()
                bsx_c = bsx.contiguous()
                bsy_c = bsy.contiguous()

                # Columns correspond to input vector length
                num_cols = int(add_10_c.numel())

                out1, out2 = self._ext.merged_spmv_launch(
                    bsx_c, bsy_c, gi, add_10_c, ro, num_rows, num_cols
                )
                # Original subgraph returns a list [scatter_b_2, scatter_b_3]
                return [out1, out2]

        # Replace the submodule on the parent with our CUDA-backed wrapper
        def _get_parent_and_name(root, fqname: str):
            parts = fqname.split(".")
            parent = root
            for p in parts[:-1]:
                parent = getattr(parent, p)
            return parent, parts[-1]

        parent, attr_name = _get_parent_and_name(compiled_eqn, submodule_name) # check
        setattr(parent, attr_name, _Easier5Cuda(ext, gather_idx, row_end_offsets))
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate and load CUDA code for easier5_select_reduce92 and test correctness")
    parser.add_argument("mesh", type=str, help="Path to mesh HDF5 file")
    parser.add_argument("sw", type=str, help="Path to shallow water HDF5 file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default="cuda", choices=["none", "torch", "cpu", "cuda"])
    parser.add_argument("--comm-backend", type=str, default="nccl", choices=["gloo", "nccl"])
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--keep", action="store_true", help="Keep generated files")
    parser.add_argument("--steps", type=int, default=1, help="Number of simulation steps to run for testing")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Tolerance for correctness comparison")

    args = parser.parse_args()

        
    # Step 1: Trace and get the original model
    print("=== Step 1: Tracing original model ===")
    compiled_eqn, submodule = trace_easier11(
        args.mesh,
        args.sw,
        args.device,
        args.backend,
        args.comm_backend,
        args.dt,
    )
    compiled_eqn()
    # Step 2: Save initial state and run original model
    print("\n=== Step 2: Running original model ===")
    initial_state = save_model_state(compiled_eqn)
    original_final_state = run_model_steps(compiled_eqn, args.steps)
    print(f"Original model final state: {original_final_state}")
    
    # Step 3: Generate and build CUDA extension
    print("\n=== Step 3: Generating and building CUDA extension ===")
    generate_cuda_code_from_graph(submodule, compiled_eqn)
    extension = build_extension()
    
    # Step 4: Restore initial state and replace submodule
    print("\n=== Step 4: Replacing submodule with CUDA extension ===")
    restore_model_state(compiled_eqn, initial_state)
    ok = replace_submodule(compiled_eqn, extension)
    if not ok:
        raise RuntimeError("Failed to locate target submodule easier5_select_reduce92 for replacement")
    print("CUDA extension compiled and registered successfully")
    
    # Step 5: Run model with replaced submodule
    print("\n=== Step 5: Running model with replaced submodule ===")
    replaced_final_state = run_model_steps(compiled_eqn, args.steps)
    print(f"Replaced model final state: {replaced_final_state}")
    
    # Step 6: Compare outputs
    print("\n=== Step 6: Comparing outputs ===")
    correctness_passed = compare_outputs(original_final_state, replaced_final_state, args.tolerance)
    
    # if args.keep:
    #     dst = Path.cwd() / "easier11_codegen"
    #     dst.mkdir(parents=True, exist_ok=True)
    #     for path in work_dir.rglob("*"):
    #         relative = path.relative_to(work_dir)
    #         if path.is_dir():
    #             (dst / relative).mkdir(parents=True, exist_ok=True)
    #         else:
    #             (dst / relative).write_bytes(path.read_bytes())
    #     print(f"\nGenerated files copied to {dst}")
    
    # # Final summary
    # print(f"\n=== FINAL SUMMARY ===")
    # print(f"Simulation steps: {args.steps}")
    # print(f"Tolerance: {args.tolerance}")
    # print(f"Correctness test: {'PASSED' if correctness_passed else 'FAILED'}")
    
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    main()




