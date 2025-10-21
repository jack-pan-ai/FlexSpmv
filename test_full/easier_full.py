import argparse
import os
import sys
import tempfile
import textwrap
from pathlib import Path
import shutil
import hashlib
from typing import Dict
import concurrent.futures

# Lightweight in-process cache to avoid reloading already-built identical extensions
EXT_CACHE: Dict[str, object] = {}


def _nvcc_threads_flag() -> str:
    try:
        default_threads = min(4, os.cpu_count() or 1)
        n = max(1, int(os.getenv("NVCC_THREADS", default_threads)))
    except Exception:
        n = min(4, os.cpu_count() or 1)
    return f"--threads={n}"

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
from test_full.utils import get_submodules, analyze_tensor_distribution, assemble_shallow_water_test_model, assemble_shallow_water_initializer_test_model


def trace_easier_full(mesh_path: str, sw_path: str, device: str, backend: str, comm_backend: str, dt: float, verify: str):
    esr.init(comm_backend)
    if verify == "model":
        eqn = ShallowWaterEquation(mesh_path, sw_path, dt, device)
        [compiled_eqn] = esr.compile([eqn], backend)
        compiled_eqn()
    elif verify == "initializer":
        compiled_eqn = assemble_shallow_water_initializer_test_model(mesh_path, sw_path, device)
    else:  # component
        compiled_eqn = assemble_shallow_water_test_model(mesh_path, sw_path, device)
    submodules, nodes = get_submodules(compiled_eqn, run_to_collect=False)
    if len(submodules) == 0:
        raise RuntimeError("No call_module nodes found in traced graph")
    return compiled_eqn, submodules, nodes


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


def run_model_steps(model, num_steps=1, verify: str = "component"):
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

def build_extension(name):
    print("Building CUDA extension")
    project_root = "/home/panq/dev/FlexSpmv"
    src = os.path.join(project_root, "merged_binding.cu")
    include_dirs = [project_root, os.path.join(project_root, "include")]
    # Configure environment for faster, cached builds
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        
    if "MAX_JOBS" not in os.environ and os.cpu_count():
        os.environ["MAX_JOBS"] = str(os.cpu_count())

    # Compute a stable hash of the generated CUDA source to enable 
    # cache reuse across iterations
    try:
        with open(src, "rb") as f:
            src_bytes = f.read()
        code_hash = hashlib.sha1(src_bytes).hexdigest()[:12]
    except Exception:
        code_hash = "nohash"
    ext_name = f"new_{name}_{code_hash}"

    # Return already loaded extension if source content is identical
    cached = EXT_CACHE.get(code_hash)
    if cached is not None:
        return cached

    # Use PyTorch's persistent extension cache (~/.cache/torch_extensions) by
    # not overriding build_directory; this allows incremental rebuilds across iterations
    ext = load(
        name=ext_name,
        sources=[src],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            _nvcc_threads_flag(),
            "-DNDEBUG",
        ],
        keep_intermediates=False,
        verbose=False,
    )
    EXT_CACHE[code_hash] = ext
    return ext


def build_extension_from_src(name, src_path, snapshot_include_dir, hash_key):
    print(f"Building CUDA extension from snapshot: {src_path}")
    project_root = "/home/panq/dev/FlexSpmv"
    # Prepend snapshot include dir to ensure headers match the generated source
    include_dirs = [snapshot_include_dir, project_root]

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    if shutil.which("sccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "sccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "sccache")
        os.environ.setdefault("CMAKE_CUDA_COMPILER_LAUNCHER", "sccache")
    elif shutil.which("ccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "ccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "ccache")
        os.environ.setdefault("CMAKE_CUDA_COMPILER_LAUNCHER", "ccache")
    if "MAX_JOBS" not in os.environ and os.cpu_count():
        # Throttle ninja parallelism to avoid oversubscription
        os.environ["MAX_JOBS"] = str(min(8, os.cpu_count()))

    # Use combined hash (source + headers) passed from snapshot
    code_hash = hash_key or "nohash"
    ext_name = f"new_{name}_{code_hash}"

    cached = EXT_CACHE.get(code_hash)
    if cached is not None:
        return cached

    ext = load(
        name=ext_name,
        sources=[src_path],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            _nvcc_threads_flag(),
            "-DNDEBUG",
        ],
        keep_intermediates=False,
        verbose=False,
    )
    EXT_CACHE[code_hash] = ext
    return ext


def _hash_directory_tree(root_dir: str) -> str:
    sha1 = hashlib.sha1()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        sha1.update(chunk)
            except Exception:
                continue
    return sha1.hexdigest()[:12]


def snapshot_generated_source(suffix: str) -> tuple[str, str, str]:
    project_root = "/home/panq/dev/FlexSpmv"
    src = os.path.join(project_root, "merged_binding.cu")
    inc_dir = os.path.join(project_root, "include")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    tmp_dir = tempfile.mkdtemp(prefix=f"easier_src_{suffix}_")
    dst_src = os.path.join(tmp_dir, f"merged_binding_{suffix}.cu")
    shutil.copyfile(src, dst_src)
    # Snapshot include directory to ensure consistency with generated source
    dst_inc = os.path.join(tmp_dir, "include")
    shutil.copytree(inc_dir, dst_inc)
    # Combined content hash (source + headers) to ensure unique cache key
    try:
        with open(dst_src, "rb") as f:
            src_bytes = f.read()
        src_hash = hashlib.sha1(src_bytes).hexdigest()[:12]
    except Exception:
        src_hash = "nohash"
    dir_hash = _hash_directory_tree(dst_inc)
    combined_hash = hashlib.sha1((src_hash + dir_hash).encode()).hexdigest()[:12]
    return dst_src, dst_inc, combined_hash


def snapshot_generated_source_cpu(suffix: str) -> tuple[str, str, str]:
    project_root = "/home/panq/dev/FlexSpmv"
    src = os.path.join(project_root, "merged_binding_cpu.cpp")
    gen_hdr = os.path.join(project_root, "merged_spmv.h")
    shared_hdr = os.path.join(project_root, "data_struct_shared.cuh")
    inc_dir = os.path.join(project_root, "include")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    tmp_dir = tempfile.mkdtemp(prefix=f"easier_src_cpu_{suffix}_")
    dst_src = os.path.join(tmp_dir, f"merged_binding_cpu_{suffix}.cpp")
    shutil.copyfile(src, dst_src)
    # Snapshot the generated header alongside the source to avoid races
    if os.path.exists(gen_hdr):
        shutil.copyfile(gen_hdr, os.path.join(tmp_dir, "merged_spmv.h"))
    # Also snapshot the shared CPU tensor header referenced by merged_spmv.h
    if os.path.exists(shared_hdr):
        shutil.copyfile(shared_hdr, os.path.join(tmp_dir, "data_struct_shared.cuh"))
    dst_inc = os.path.join(tmp_dir, "include")
    shutil.copytree(inc_dir, dst_inc)
    try:
        with open(dst_src, "rb") as f:
            src_bytes = f.read()
        src_hash = hashlib.sha1(src_bytes).hexdigest()[:12]
    except Exception:
        src_hash = "nohash"
    dir_hash = _hash_directory_tree(dst_inc)
    combined_hash = hashlib.sha1(("cpu:" + src_hash + dir_hash).encode()).hexdigest()[:12]
    return dst_src, dst_inc, combined_hash


def build_cpu_extension_from_src(name, src_path, snapshot_include_dir, hash_key):
    print(f"Building CPU extension from snapshot: {src_path}")
    project_root = "/home/panq/dev/FlexSpmv"
    # Prefer snapshot header (same directory as src) and snapshot include dir; add project_root as fallback
    include_dirs = [os.path.dirname(src_path), snapshot_include_dir, project_root]

    if shutil.which("sccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "sccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "sccache")
    elif shutil.which("ccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "ccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "ccache")
    if "MAX_JOBS" not in os.environ and os.cpu_count():
        os.environ["MAX_JOBS"] = str(min(8, os.cpu_count()))

    code_hash = hash_key or "nohash"
    cache_key = f"cpu:{code_hash}"
    ext_name = f"new_cpu_{name}_{code_hash}"

    cached = EXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ext = load(
        name=ext_name,
        sources=[src_path],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-fopenmp", "-DNDEBUG", "-iquote", os.path.dirname(src_path)],
        extra_ldflags=["-fopenmp"],
        keep_intermediates=False,
        verbose=False,
    )
    EXT_CACHE[cache_key] = ext
    return ext

def main():
    parser = argparse.ArgumentParser(description="Generate and load CUDA code for full and test correctness")
    parser.add_argument("mesh", type=str, help="Path to mesh HDF5 file")
    parser.add_argument("sw", type=str, help="Path to shallow water HDF5 file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default="torch", choices=["none", "torch", "cpu", "cuda"])
    parser.add_argument("--comm-backend", type=str, default="nccl", choices=["gloo", "nccl"])
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--keep", action="store_true", help="Keep generated files")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps to run for testing")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Tolerance for correctness comparison")
    parser.add_argument("--verify", type=str, default="component", choices=["model", "initializer", "component"], help="Which pipeline to verify")

    args = parser.parse_args()

        
    # Step 1: Trace and get the original model
    print("=== Step 1: Tracing original model ===")
    compiled_eqn, submodules, nodes = trace_easier_full(
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
    
    # Step 3: Generate device-specific sources for each submodule and snapshot (sequential to avoid overwrites)
    print("\n=== Step 3: Generating sources for each submodule (snapshotting) ===")
    build_jobs = []  # list of (module_name, snapshot_src, snapshot_inc, combined_hash)
    if args.device == "cuda":
        for submodule, node in zip(submodules, nodes):
            generate_cuda_code_from_graph(submodule, compiled_eqn)
            snap_src, snap_inc, combined_hash = snapshot_generated_source(node.name)
            build_jobs.append((node.name, snap_src, snap_inc, combined_hash))
    else:
        for submodule, node in zip(submodules, nodes):
            generate_cpu_code_from_graph(submodule, compiled_eqn)
            snap_src, snap_inc, combined_hash = snapshot_generated_source_cpu(node.name)
            build_jobs.append((node.name, snap_src, snap_inc, combined_hash))

    # Step 3b: Build all snapshots in parallel
    print("\n=== Step 3b: Building extensions in parallel ===")
    n_workers = min(len(build_jobs), max(1, (os.cpu_count() or 2) // 2))
    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for mod_name, snap_src, snap_inc, combined_hash in build_jobs:
            if args.device == "cuda":
                futures[mod_name] = executor.submit(
                    build_extension_from_src, mod_name, snap_src, snap_inc, combined_hash
                )
            else:
                futures[mod_name] = executor.submit(
                    build_cpu_extension_from_src, mod_name, snap_src, snap_inc, combined_hash
                )
        built_extensions = {name: fut.result() for name, fut in futures.items()}

    # Step 4: Restore initial state and replace submodules sequentially (to preserve semantics)
    print("\n=== Step 4: Replacing submodules with compiled extensions ===")
    for node in nodes:
        extension = built_extensions[node.name]
        ok = dynamic_replace_submodule(compiled_eqn, extension, node.name)
        if not ok:
            raise RuntimeError(f"Failed to locate target submodule {node.name} for replacement")
        print(f"Replaced submodule {node.name} successfully")
    
    # Step 5: Run model with replaced submodule
    print("\n=== Step 5: Running replaced model and Comparing outputs ===")

    if args.verify == "model":
        restore_model_state(compiled_eqn, initial_state)
        replaced_final_state = run_model_steps(compiled_eqn, args.steps, args.verify)
        correctness_passed = compare_outputs(original_final_state, replaced_final_state, args.tolerance)
    elif args.verify == "initializer":
        restore_initializer_state(compiled_eqn, initial_state)
        replaced_final_state = run_model_steps(compiled_eqn, args.steps, args.verify)
        correctness_passed = compare_initializer_outputs(original_final_state, replaced_final_state, args.tolerance)
    else:
        restore_components_state(compiled_eqn, initial_state)
        replaced_final_state = run_model_steps(compiled_eqn, args.steps, args.verify)
        correctness_passed = compare_components_outputs(original_final_state, replaced_final_state)
        
    return 0 if correctness_passed else 1

# torchrun ./test_full/easier_full.py ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 --device cpu --backend torch --comm-backend gloo --verify initializer
# torchrun ./test_full/easier_full.py ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 --device gpu --backend torch --comm-backend nccl --verify initializer

if __name__ == "__main__":
    main()




