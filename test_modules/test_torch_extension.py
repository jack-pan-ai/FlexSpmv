import os
import math
import torch
from torch.utils.cpp_extension import load


def build_extension():
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


def cpu_reference(add_10, bsx, bsy, gather_b_1_idx, row_end_offsets, num_rows):
    # add_10: [nv]
    # bsx, bsy: [ne]
    # gather_b_1_idx: [ne]
    # row_end_offsets: [num_rows + 1]
    ne = gather_b_1_idx.numel()
    reducer_1 = torch.zeros(num_rows, dtype=add_10.dtype)
    reducer_2 = torch.zeros(num_rows, dtype=add_10.dtype)
    for row in range(num_rows):
        row_start = int(row_end_offsets[row].item())
        row_end = int(row_end_offsets[row + 1].item())
        part1 = 0.0
        part2 = 0.0
        for i in range(row_start, row_end):
            sel_i = int(gather_b_1_idx[i].item())
            v_i = add_10[sel_i]
            pow_2 = v_i * v_i
            mul_32 = 0.5 * pow_2
            mul_36 = mul_32 * bsx[i]
            mul_40 = mul_32 * bsy[i]
            part1 += mul_36
            part2 += mul_40
        reducer_1[row] = part1
        reducer_2[row] = part2
    return reducer_1, reducer_2


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping run.")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Small random case
    num_rows = 5
    nv = 8
    ne = 12

    # Random monotonic row_end_offsets
    offsets = torch.randint(0, ne + 1, (num_rows - 1,), dtype=torch.int32)
    offsets, _ = torch.sort(offsets)
    row_end_offsets = torch.empty(num_rows + 1, dtype=torch.int32)
    row_end_offsets[0] = 0
    row_end_offsets[1:-1] = offsets
    row_end_offsets[-1] = ne

    gather_b_1_idx = torch.randint(0, nv, (ne,), dtype=torch.int32)

    add_10 = torch.rand(nv, dtype=torch.float32)
    bsx = torch.rand(ne, dtype=torch.float32)
    bsy = torch.rand(ne, dtype=torch.float32)

    # Move to CUDA
    add_10_cu = add_10.to(device)
    bsx_cu = bsx.to(device)
    bsy_cu = bsy.to(device)
    gather_b_1_idx_cu = gather_b_1_idx.to(device)
    row_end_offsets_cu = row_end_offsets.to(device)

    ext = build_extension()

    out1, out2 = ext.merged_spmv_launch(
        bsx_cu,
        bsy_cu,
        gather_b_1_idx_cu,
        add_10_cu,
        row_end_offsets_cu,
        int(num_rows),
        int(nv),
    )

    ref1, ref2 = cpu_reference(add_10, bsx, bsy, gather_b_1_idx, row_end_offsets, num_rows)
    out1_cpu = out1.view(num_rows).cpu()
    out2_cpu = out2.view(num_rows).cpu()

    max_diff1 = (out1_cpu - ref1).abs().max().item()
    max_diff2 = (out2_cpu - ref2).abs().max().item()

    print("reducer_1 max diff:", max_diff1)
    print("reducer_2 max diff:", max_diff2)

    assert max_diff1 < 1e-3, f"reducer_1 mismatch: {max_diff1}"
    assert max_diff2 < 1e-3, f"reducer_2 mismatch: {max_diff2}"
    print("OK")


if __name__ == "__main__":
    main()


