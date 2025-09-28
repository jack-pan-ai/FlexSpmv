import torch
import time
import argparse

import easier as esr

from easier.core.runtime.metadata import Role, get_node_meta, StructuredTensorMeta, get_node_view_src

from codegen.merged_gen_gpu import generate_cuda_code_from_graph
from codegen.merged_gen_cpu import generate_cpu_code_from_graph

from utils import analyze_tensor_distribution

# Part 1: Define model
class System(esr.Module):
    def __init__(
            self,
            spm_1,
            spm_2,
            output_y_map_1,
            output_y_map_2,
            num_rows):
        super().__init__()

        # Inputs
        self.spm_1 = esr.Tensor(spm_1, mode = 'partition')  # (N, 2)
        self.spm_2 = esr.Tensor(spm_2, mode = 'partition')  # (N, 2, 3) (N, 6)
        
        # Outputs
        self.y_add_1 = esr.Tensor(output_y_map_1, 
                                         mode = 'partition')  # (N, 2)
        self.y_add_2 = esr.Tensor(output_y_map_2, 
                                         mode = 'partition')  # (N, 6)
        
    def forward(self):
        add_1 = self.spm_1  + self.spm_1  # (N, 2)
        # add_2 = self.spm_2  + self.spm_2  # (N, 2, 3) (N, 6)

        self.y_add_1[:] = add_1  # (N, 2)
        # self.y_add_2[:] = add_2  # (N, 2, 3) (N, 6)

def system_test(
        num_nnz=20,
        num_cols=11,
        num_rows=11,
        device="cuda",
        precision="double",
        backend="torch",
        comm_backend="gloo"):

    torch.manual_seed(52)

    # Set dtype based on precision
    if precision == "single":
        dtype = torch.float32
        precision_name = "single (float32)"
    else:  # double
        dtype = torch.float64
        precision_name = "double (float64)"

    print(f"Running system test with num_nnz={num_nnz}, \
        num_cols={num_cols}, num_rows={num_rows}")
    print(f"Device: {device}")
    print(f"Precision: {precision_name}")
    # data settings
    dim_x = 2
    spm_1 = torch.rand(num_nnz, 2, 1, 
                       device=device, dtype=dtype)  # (N, 2, 1)
    spm_2 = torch.rand(num_nnz, 2, 3, 
                       device=device, dtype=dtype)  # (N, 2, 3)
    output_y_map_1 = torch.zeros(
        num_nnz,
        2, 1,
        device=device,
        dtype=dtype)  # (N, 2, 1)
    output_y_map_2 = torch.zeros(
        num_nnz, 
        2, 3, 
        device=device, dtype=dtype)  # (N, 2, 3)

    # pytorch model for code generation
    model = System(
        spm_1, spm_2,
        output_y_map_1, output_y_map_2, num_rows
    )

    print("Step 1: Tracing model with easier")
    start_time = time.time()
    esr.init(comm_backend)
    [traced_model] = esr.compile([model], backend)
    # print(traced_model.jit_engine.graph)
    # traced_model.jit_engine.graph.print_tabular()
    traced_model()
    submodule_node_pairs = analyze_tensor_distribution(traced_model)
    submodule, node_module = submodule_node_pairs[0]

    trace_time = time.time() - start_time
    print(f"Model tracing took: {trace_time:.6f} seconds")

    print("\nStep 2: Generating CUDA code from the graph")
    start_time = time.time()
    if device == "cuda":
        generate_cuda_code_from_graph(submodule, traced_model)
    else:
        generate_cpu_code_from_graph(submodule, traced_model)
    codegen_time = time.time() - start_time
    print(f"{device} code generation took: {codegen_time:.6f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run system test with \
            configurable matrix dimensions')
    parser.add_argument('--num_nonzeros', '--num_nnz', 
                        type=int, default=20,
                        help='Number of non-zero elements (default: 20)')
    parser.add_argument('--num_cols', type=int, default=11,
                        help='Number of columns (default: 11)')
    parser.add_argument('--num_rows', type=int, default=11,
                        help='Number of rows (default: 11)')
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='nccl'
    )
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "cuda"],
        default='torch'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=[
            'cpu',
            'cuda'],
        help='Device to run on: cpu or cuda (default: cuda)')
    parser.add_argument(
        '--precision',
        type=str,
        default='double',
        choices=[
            'single',
            'double'],
        help='Floating point precision: single (float32) \
            or double (float64) (default: double)')

    args = parser.parse_args()

    system_test(
        num_nnz=args.num_nonzeros,
        num_cols=args.num_cols,
        num_rows=args.num_rows,
        device=args.device,
        precision=args.precision,
        backend=args.backend,
        comm_backend=args.comm_backend)
