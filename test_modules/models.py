import torch

import easier as esr

# Part 1: Define spring-mass system model
class SpringMassSystem(esr.Module):
    def __init__(self, spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows):
        super().__init__()

        # data
        self.spm_k = spm_k  # (N, 1)
        self.spm_l = spm_l  # (N, 1)
        self.vector_x = vector_x  # (N, D)
        self.output_y_reducer_i = output_y_reducer_i  # (M, D)
        self.output_y_reducer_j = output_y_reducer_j  # (M, D)
        self.row_end_offset = row_end_offset # (M + 1)
        self.col_indices_i = col_indices_i # (N, 1)
        self.col_indices_j = col_indices_j # (N, 1)

        # Selector for i and j, and reducer
        self.selector_i = esr.Selector(self.col_indices_i)
        self.selector_j = esr.Selector(self.col_indices_j)
        self.reducer_i = esr.Reducer(self.row_end_offset, num_rows)
        self.reducer_j = esr.Reducer(self.row_end_offset, num_rows)

    def forward(self):
        # compute force
        r_i = self.selector_i(self.vector_x) # (N, 2)
        r_j = self.selector_j(self.vector_x) # (N, 2)
        r_ij = r_j - r_i # (N, 2)
        norm_r_ij = torch.norm(r_ij, dim=1, keepdim=True) # (N, 1)
        e_ij = r_ij / norm_r_ij # (N, 2)
        f_ij1 = self.spm_k * (norm_r_ij - self.spm_l) * e_ij # (N, 2)
        f_ij2 = self.spm_k * (norm_r_ij * self.spm_l) * e_ij # (N, 2)
        f_i1 = self.reducer_i(f_ij1) # (M, 2)
        f_i2 = self.reducer_j(f_ij2) # (M, 2)

        return f_i1, f_i2

# Part 1: Define spring-mass system model
class AggregatorTest(esr.Module):
    def __init__(self, spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows):
        super().__init__()

        # data
        self.spm_k = spm_k  # (N, 1)
        self.spm_l = spm_l  # (N, 1)
        self.vector_x = vector_x  # (N, D)
        self.output_y_reducer_i = output_y_reducer_i  # (M, D)
        self.output_y_reducer_j = output_y_reducer_j  # (M, D)
        self.row_end_offset = row_end_offset # (M + 1)
        self.col_indices_i = col_indices_i # (N, 1)
        self.col_indices_j = col_indices_j # (N, 1)

        # Selector for i and j, and reducer
        self.selector_i = esr.Selector(self.col_indices_i)
        self.selector_j = esr.Selector(self.col_indices_j)
        # self.reducer_i = esr.Reducer(self.row_end_offset, num_rows)
        # self.reducer_j = esr.Reducer(self.row_end_offset, num_rows)
        # self.aggregator_i = esr.Aggregator(self.row_end_offset, num_rows)
        # self.aggregator_j = esr.Aggregator(self.row_end_offset, num_rows)

    def forward(self):
        # compute force
        r_i = self.selector_i(self.vector_x) # (N, 2)
        r_j = self.selector_j(self.vector_x) # (N, 2)
        r_ij = r_j - r_i # (N, 2)
        norm_r_ij = torch.norm(r_ij, dim=1, keepdim=True) # (N, 1)
        e_ij = r_ij / norm_r_ij # (N, 2)
        f_ij1 = self.spm_k * (norm_r_ij - self.spm_l) * e_ij # (N, 2)
        f_ij2 = self.spm_k * (norm_r_ij * self.spm_l) * e_ij # (N, 2)
        f_i1 = esr.sum(f_ij1) # (M, 2)
        f_i2 = esr.sum(f_ij2) # (M, 2)

        return f_i1, f_i2

class SpringMassSystem_calculation():
    def __init__(self, spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows):
        super().__init__()

        # data
        self.spm_k = spm_k  # (N, 1)
        self.spm_l = spm_l  # (N, 1)
        self.vector_x = vector_x  # (N, D)
        self.output_y_reducer_i = output_y_reducer_i  # (M, D)
        self.output_y_reducer_j = output_y_reducer_j  # (M, D)
        self.row_end_offset = row_end_offset # (M + 1)
        self.col_indices_i = col_indices_i # (N, 1)
        self.col_indices_j = col_indices_j # (N, 1)
        self.num_rows = num_rows

    def forward(self):
        # compute force
        r_i = self.vector_x[self.col_indices_i] # (N, 2)
        r_j = self.vector_x[self.col_indices_j] # (N, 2)
        r_ij = r_j - r_i # (N, 2)
        norm_r_ij = torch.norm(r_ij, dim=1, keepdim=True) # (N, 1)
        e_ij = r_ij / norm_r_ij # (N, 2)
        f_ij1 = self.spm_k * (norm_r_ij - self.spm_l) * e_ij # (N, 2)
        f_ij2 = self.spm_k * (norm_r_ij * self.spm_l) * e_ij # (N, 2)
        f_i1 = torch.zeros(self.num_rows, 2, device="cuda", dtype=f_ij1.dtype)
        f_i2 = torch.zeros(self.num_rows, 2, device="cuda", dtype=f_ij2.dtype)

        print(f_ij2)
        # reduce f_ij to f_i based on row_end_offset
        for i in range(self.num_rows):
            f_i1[i] = torch.sum(f_ij1[self.row_end_offset[i]:self.row_end_offset[i+1]], dim=0)
            f_i2[i] = torch.sum(f_ij2[self.row_end_offset[i]:self.row_end_offset[i+1]], dim=0)
        # return f_i1, f_i2
        return f_i1, f_i2

class AggregatorTest_calculation():
    def __init__(self, spm_k, spm_l, vector_x, output_y_reducer_i, output_y_reducer_j, row_end_offset, col_indices_i, col_indices_j, num_rows):
        super().__init__()

        # data
        self.spm_k = spm_k  # (N, 1)
        self.spm_l = spm_l  # (N, 1)
        self.vector_x = vector_x  # (N, D)
        self.output_y_reducer_i = output_y_reducer_i  # (M, D)
        self.output_y_reducer_j = output_y_reducer_j  # (M, D)
        self.row_end_offset = row_end_offset # (M + 1)
        self.col_indices_i = col_indices_i # (N, 1)
        self.col_indices_j = col_indices_j # (N, 1)
        self.num_rows = num_rows

    def forward(self):
        # compute force
        r_i = self.vector_x[self.col_indices_i] # (N, 2)
        r_j = self.vector_x[self.col_indices_j] # (N, 2)
        r_ij = r_j - r_i # (N, 2)
        norm_r_ij = torch.norm(r_ij, dim=1, keepdim=True) # (N, 1)
        e_ij = r_ij / norm_r_ij # (N, 2)
        f_ij1 = self.spm_k * (norm_r_ij - self.spm_l) * e_ij # (N, 2)
        f_ij2 = self.spm_k * (norm_r_ij * self.spm_l) * e_ij # (N, 2)
        f_i1 = torch.zeros(2, device="cuda", dtype=f_ij1.dtype)
        f_i2 = torch.zeros(2, device="cuda", dtype=f_ij2.dtype)

        f_i1 = torch.sum(f_ij1, dim=0)
        f_i2 = torch.sum(f_ij2, dim=0)
        # print(f_ij2)
        # # reduce f_ij to f_i based on row_end_offset
        # for i in range(self.num_rows):
        #     f_i1[i] = torch.sum(f_ij1[self.row_end_offset[i]:self.row_end_offset[i+1]], dim=0)
        #     f_i2[i] = torch.sum(f_ij2[self.row_end_offset[i]:self.row_end_offset[i+1]], dim=0)
        # return f_i1, f_i2
        return f_i1, f_i2

