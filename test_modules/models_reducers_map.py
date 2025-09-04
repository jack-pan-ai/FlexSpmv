import torch
import time
import argparse

import easier as esr

from test_modules.utils import save_to_coo_format
from codegen.merged_gen_gpu import generate_cuda_code_from_graph
from codegen.merged_gen_cpu import generate_cpu_code_from_graph
from traceGraph.graph_trace import trace_model

# Part 1: Define model
class System(esr.Module):
    def __init__(self, spm_1, spm_2, vector_x, col_indices_1, col_indices_2, row_end_offset,
                        output_y_map_1, output_y_map_2, 
                        output_y_reducer_1, output_y_reducer_2, num_rows):
        super().__init__()

        # Inputs
        self.spm_1 = spm_1  # (N, 2)
        self.spm_2 = spm_2  # (N, 2, 3) (N, 6)
        self.vector_x = vector_x  # (N, 2)        
        self.col_indices_1 = col_indices_1 # (N, 1)
        self.col_indices_2 = col_indices_2 # (N, 1)
        self.row_end_offset = row_end_offset # (M + 1)
        
        # Outputs
        self.output_y_map_1 = output_y_map_1  # (N, 2)
        self.output_y_map_2 = output_y_map_2  # (N, 6)
        self.output_y_reducer_1 = output_y_reducer_1  # (M, 2)
        self.output_y_reducer_2 = output_y_reducer_2  # (M, 6)

        # Selectors and reducers
        self.selector_1 = esr.Selector(self.col_indices_1)
        self.selector_2 = esr.Selector(self.col_indices_2)
        self.reducer_1 = esr.Reducer(self.row_end_offset, num_rows)
        self.reducer_2 = esr.Reducer(self.row_end_offset, num_rows)

    def forward(self):
        
        r_1 = self.selector_1(self.vector_x) # (N, 2)
        r_2 = self.selector_2(self.vector_x) # (N, 2)

        output_y_map_1 = r_1 + self.spm_1 # (N, 2)
        output_y_map_2 = r_2 + self.spm_2 # (N, 2, 3) (N, 6)

        output_y_reducer_1 = self.reducer_1(output_y_map_1) # (M, 2)
        output_y_reducer_2 = self.reducer_2(output_y_map_2) # (M, 2, 3) (M, 6)
        

        return output_y_map_1, output_y_map_2, output_y_reducer_1, output_y_reducer_2

class System_calculation():
    def __init__(self, spm_1, spm_2, vector_x, col_indices_1, col_indices_2, row_end_offset,
                        output_y_map_1, output_y_map_2, 
                        output_y_reducer_1, output_y_reducer_2, num_rows):
        super().__init__()

        # Inputs
        self.spm_1 = spm_1  # (N, 2)
        self.spm_2 = spm_2  # (N, 2, 3) (N, 6)
        self.vector_x = vector_x  # (N, 2)
        self.col_indices_1 = col_indices_1 # (N, 1)
        self.col_indices_2 = col_indices_2 # (N, 1)
        self.row_end_offset = row_end_offset # (M + 1)
        # Outputs
        self.output_y_map_1 = output_y_map_1  # (N, 2)
        self.output_y_map_2 = output_y_map_2  # (N, 2, 3) (N, 6)
        self.output_y_reducer_1 = output_y_reducer_1  # (M, 2)
        self.output_y_reducer_2 = output_y_reducer_2  # (M, 2, 3) (M, 6)
        self.num_rows = num_rows

    def forward(self):
        # compute force
        r_1 = self.vector_x[self.col_indices_1] # (N, 2)
        r_2 = self.vector_x[self.col_indices_2] # (N, 2)

        self.output_y_map_1 = r_1 + self.spm_1 # (N, 2)
        self.output_y_map_2 = r_2 + self.spm_2 # (N, 2, 3) (N, 6)

        for i in range(self.num_rows):
            start = self.row_end_offset[i].item()
            end = self.row_end_offset[i + 1].item()
            partial_output_y_map_1 = 0.0
            partial_output_y_map_2 = 0.0
            for j in range(start, end):
                partial_output_y_map_1 += self.output_y_map_1[j]
                partial_output_y_map_2 += self.output_y_map_2[j]
            self.output_y_reducer_1[i] = partial_output_y_map_1
            self.output_y_reducer_2[i] = partial_output_y_map_2

        return self.output_y_map_1, self.output_y_map_2, self.output_y_reducer_1, self.output_y_reducer_2

def system_test(num_nnz=20, num_cols=11, num_rows=5, device="cuda", precision="double"):
    
    torch.manual_seed(52)
    
    # Set dtype based on precision
    if precision == "single":
        dtype = torch.float32
        precision_name = "single (float32)"
    else:  # double
        dtype = torch.float64
        precision_name = "double (float64)"
    
    print(f"Running system test with num_nnz={num_nnz}, num_cols={num_cols}, num_rows={num_rows}")
    print(f"Device: {device}")
    print(f"Precision: {precision_name}")
    print(f"Computational complexity: O({num_nnz}) operations")
    # data settings
    dim_x = 2
    dim_output_map1 = 2
    dim_output_map2 = 6
    spm_1 = torch.rand(num_nnz, 2, 1, device=device, dtype=dtype)  # (N, 2)
    spm_2 = torch.rand(num_nnz, 2, 3, device=device, dtype=dtype)  # (N, 6)
    vector_x = torch.rand(num_cols, dim_x, 1, device=device, dtype=dtype)  # (N, D)
    output_y_map_1 = torch.zeros(num_nnz, dim_output_map1, 1, device=device, dtype=dtype)  # (N, D)
    output_y_map_2 = torch.zeros(num_nnz, 2, 3, device=device, dtype=dtype)  # (N, D)
    output_y_reducer_1 = torch.zeros(num_rows, dim_output_map1, 1, device=device, dtype=dtype)  # (M, D)
    output_y_reducer_2 = torch.zeros(num_rows, 2, 3, device=device, dtype=dtype)  # (M, D)
    row_end_offset = torch.sort(torch.randint(0, num_nnz + 1, (num_rows - 1, ), device=device, dtype=torch.int64))[0] 
    row_end_offset = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), row_end_offset])
    row_end_offset = torch.cat([row_end_offset, torch.tensor([num_nnz], device=device, dtype=torch.int64)]) # (M + 1)
    # print(f"row_end_offset: {row_end_offset}")
    # Generate col_indices_1 and col_indices_2 randomly (can be the same or different)
    col_indices_1 = torch.randint(0, num_cols, (num_nnz,), dtype=torch.int64, device=device)
    col_indices_2 = torch.randint(0, num_cols, (num_nnz,), dtype=torch.int64, device=device)
    
    # # Save the sparse matrix and vectors in COO format
    # save_to_coo_format(spm_1, spm_2, row_end_offset, col_indices_1, col_indices_2, vector_x)
    
    # pytorch model for code generation
    model = System(
        spm_1, spm_2, vector_x, col_indices_1, col_indices_2, row_end_offset,
        output_y_map_1, output_y_map_2, 
        output_y_reducer_1, output_y_reducer_2, num_rows
    )

    # python model for verification
    model_calculation = System_calculation(
        spm_1, spm_2, vector_x, col_indices_1, col_indices_2, row_end_offset,
        output_y_map_1, output_y_map_2, 
        output_y_reducer_1, output_y_reducer_2, num_rows
    )


    print("Step 1: Tracing model with torch.fx")
    start_time = time.time()
    traced_model = trace_model(model)
    trace_time = time.time() - start_time
    print(f"Model tracing took: {trace_time:.6f} seconds")

    # run the model for verification
    print("\nStep 2: Running model forward calculation")
    
    # Warmup iterations to avoid initialization overhead
    if device == "cuda":
        print("Warming up GPU...")
    else:
        print("Warming up CPU...")
    for i in range(10):
        results_gold_1, results_gold_2, results_gold_3, results_gold_4 = model_calculation.forward()
    if device == "cuda":
        torch.cuda.synchronize()  # Wait for warmup to complete
    
    # Actual timing
    print("Running timed iterations...")
    
    if device == "cuda":
        # Use CUDA events for precise GPU timing
        torch.cuda.synchronize()  # Ensure all previous operations are complete
        
        # Create CUDA events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(100):
            results_gold_1, results_gold_2, results_gold_3, results_gold_4 = model_calculation.forward()
        end_event.record()
        
        torch.cuda.synchronize()  # Wait for all operations to complete
        forward_time_ms = start_event.elapsed_time(end_event)  # Returns time in milliseconds
        avg_time_per_iter = forward_time_ms / 100
    else:
        # Use regular timing for CPU
        start_time = time.time()
        for i in range(100):
            results_gold_1, results_gold_2, results_gold_3, results_gold_4 = model_calculation.forward()
        end_time = time.time()
        forward_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        avg_time_per_iter = forward_time_ms / 100
    
    # time_per_element = avg_time_per_iter / num_nnz * 1000  # microseconds per element
    print(f"Model forward calculation took: {avg_time_per_iter:.6f} ms per iteration")
    # print(f"Throughput: {num_nnz / avg_time_per_iter * 1000:.0f} elements/second")

    print("\nStep 3: Generating CUDA code from the graph")
    start_time = time.time()
    if device == "cuda":
        generate_cuda_code_from_graph(traced_model)
    else:
        generate_cpu_code_from_graph(traced_model)
    codegen_time = time.time() - start_time
    print(f"CUDA code generation took: {codegen_time:.6f} seconds")

    # # check the results
    # results_gold_1 = results_gold_1.cpu()
    # results_gold_2 = results_gold_2.cpu()
    # results_gold_3 = results_gold_3.cpu()
    # results_gold_4 = results_gold_4.cpu()
    # print(f"Results gold 1: {results_gold_1}")
    # print(f"Results gold 2: {results_gold_2}")
    # print(f"Results gold 3: {results_gold_3}")
    # print(f"Results gold 4: {results_gold_4}")
    
# # Import our extension
#     import flex_spmv
#     comp_results_1, comp_results_2 = flex_spmv.flex_spmv(
#         spm_k, spm_l, row_end_offset, 
#         col_indices_i, col_indices_j, 
#         vector_x, 
#         output_y_reducer_i, 
#         output_y_reducer_j)


    # comp_results_1 = comp_results_1.cpu()
    # comp_results_2 = comp_results_2.cpu()
    # # print(f"Results flex_spmv 1: {comp_results_1}")
    # print(f"Results difference 1: {torch.norm(results_gold_1 - comp_results_1)}")
    # # print(f"Results flex_spmv 2: {comp_results_2}")
    # print(f"Results difference 2: {torch.norm(results_gold_2 - comp_results_2)}")
    # if torch.norm(results_gold_1 - comp_results_1) < 1e-6 and torch.norm(results_gold_2 - comp_results_2) < 1e-6:
    #     print("Results are the same")
    # else:
    #     print("Results are different") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run system test with configurable matrix dimensions')
    parser.add_argument('--num_nonzeros', '--num_nnz', type=int, default=20, 
                        help='Number of non-zero elements (default: 20)')
    parser.add_argument('--num_cols', type=int, default=11,
                        help='Number of columns (default: 11)')
    parser.add_argument('--num_rows', type=int, default=5,
                        help='Number of rows (default: 5)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to run on: cpu or cuda (default: cuda)')
    parser.add_argument('--precision', type=str, default='double', choices=['single', 'double'],
                        help='Floating point precision: single (float32) or double (float64) (default: double)')
    
    args = parser.parse_args()
    system_test(num_nnz=args.num_nonzeros, num_cols=args.num_cols, num_rows=args.num_rows, 
                device=args.device, precision=args.precision)