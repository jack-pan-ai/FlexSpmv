import torch

import easier as esr

from test_modules.utils import save_to_coo_format
from codegenGPU.codegen import trace_model, generate_cuda_code_from_graph

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

def system_test():
    
    torch.manual_seed(52)
    # data settings
    num_nnz = 20
    num_cols = 11
    num_rows = 5
    dim_x = 2
    dim_output_map1 = 2
    dim_output_map2 = 6
    spm_1 = torch.rand(num_nnz, 2, 1, device="cuda", dtype=torch.float32)  # (N, 2)
    spm_2 = torch.rand(num_nnz, 2, 3, device="cuda", dtype=torch.float32)  # (N, 6)
    vector_x = torch.rand(num_cols, dim_x, 1, device="cuda", dtype=torch.float32)  # (N, D)
    output_y_map_1 = torch.zeros(num_nnz, dim_output_map1, 1, device="cuda", dtype=torch.float32)  # (N, D)
    output_y_map_2 = torch.zeros(num_nnz, 2, 3, device="cuda", dtype=torch.float32)  # (N, D)
    output_y_reducer_1 = torch.zeros(num_rows, dim_output_map1, 1, device="cuda", dtype=torch.float32)  # (M, D)
    output_y_reducer_2 = torch.zeros(num_rows, 2, 3, device="cuda", dtype=torch.float32)  # (M, D)
    row_end_offset = torch.sort(torch.randint(0, num_nnz + 1, (num_rows - 1, ), device="cuda", dtype=torch.int32))[0] 
    row_end_offset = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), row_end_offset])
    row_end_offset = torch.cat([row_end_offset, torch.tensor([num_nnz], device="cuda", dtype=torch.int32)]) # (M + 1)
    # print(f"row_end_offset: {row_end_offset}")
    # Generate col_indices_1 and col_indices_2 randomly (can be the same or different)
    col_indices_1 = torch.randint(0, num_cols, (num_nnz,), dtype=torch.int32, device="cuda")
    col_indices_2 = torch.randint(0, num_cols, (num_nnz,), dtype=torch.int32, device="cuda")
    
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
    traced_model = trace_model(model)

    # run the model for verification
    results_gold_1, results_gold_2, results_gold_3, results_gold_4 = model_calculation.forward()

    print("\nStep 2: Generating CUDA code from the graph")
    generate_cuda_code_from_graph(traced_model)

    # check the results
    # results_gold_1 = results_gold_1.cpu()
    # results_gold_2 = results_gold_2.cpu()
    # print(f"Results gold 1: {results_gold_1}")
    # print(f"Results gold 2: {results_gold_2}")
    
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
    system_test()