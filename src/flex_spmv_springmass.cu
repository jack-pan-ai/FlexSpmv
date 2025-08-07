/******************************************************************************
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIAeBILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//---------------------------------------------------------------------
// SpMV comparison tool
//---------------------------------------------------------------------

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "sparse_matrix.h"

using namespace cub;

#define INIT_KERNEL_THREADS 128 // INFO: this is from cub config
#define DIM_OUTPUT_VECTOR_Y 2            // Dimension of the output vector
#define DIM_INPUT_VECTOR_X 2    // Dimension of the input vector x
#define DIM_INPUT_MATRIX_A 1    // Dimension of the input matrix A


#include <cusparse.h>

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <utils.h>
#include <cub/util_debug.cuh>

#include "../include/merged_spmv.cuh"


//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet     = false;        // Whether to display stats in CSV format
bool                    g_verbose   = false;        // Whether to display output to console
bool                    g_verbose2  = false;        // Whether to display input to console
CachingDeviceAllocator  g_allocator(true);          // Caching allocator for device memory


//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
    typename ValueT,
    typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta)
{
    for (OffsetT row = 0; row < a.num_rows; ++row)
    {
        // ValueT partial = beta * vector_y_in[row];
        // printf("a.row_offsets[row + 1]: %d\n", a.row_offsets[row + 1]);
        ValueT partial[DIM_OUTPUT_VECTOR_Y] = {0.0f, 0.0f};
        for (
            OffsetT offset = a.row_offsets[row];
            offset < a.row_offsets[row + 1];
            ++offset)
        {
            // partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
            // flex spmv
            // partial += alpha * a.values[a.column_indices_A[offset]] * vector_x[a.column_indices[offset]];
            // spring mass
            // select
            ValueT ri[DIM_INPUT_VECTOR_X];
            ValueT rj[DIM_INPUT_VECTOR_X];
            ValueT k_ij[DIM_INPUT_MATRIX_A];
            ValueT l_ij[DIM_INPUT_MATRIX_A];
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                ri[i] = vector_x[a.column_indices_i[offset] * DIM_INPUT_VECTOR_X + i];
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                rj[i] = vector_x[a.column_indices_j[offset] * DIM_INPUT_VECTOR_X + i];
            for (int i = 0; i < DIM_INPUT_MATRIX_A; ++i)
                k_ij[i] = a.values[a.column_indices_k[offset] * DIM_INPUT_MATRIX_A + i];
            for (int i = 0; i < DIM_INPUT_MATRIX_A; ++i)
                l_ij[i] = a.values[a.column_indices_l[offset] * DIM_INPUT_MATRIX_A + i];

            // printf("ri: ");
            // for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
            //     printf("%f ", ri[i]);
            // printf("\n");
            
            // printf("rj: ");
            // for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
            //     printf("%f ", rj[i]);
            // printf("\n");
            
            // printf("k_ij: ");
            // for (int i = 0; i < DIM_INPUT_MATRIX_A; ++i)
            //     printf("%f ", k_ij[i]);
            // printf("\n");
            
            // printf("l_ij: ");
            // for (int i = 0; i < DIM_INPUT_MATRIX_A; ++i)
            //     printf("%f ", l_ij[i]);
            // printf("\n");

            
            // map
            ValueT norm_rij = 0.0f;
            ValueT unit_rij[DIM_INPUT_VECTOR_X];
            ValueT mass = 0.0f;
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                unit_rij[i] = ri[i] - rj[i];
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                norm_rij += unit_rij[i] * unit_rij[i];
            norm_rij = sqrt(norm_rij);
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                unit_rij[i] = unit_rij[i] / norm_rij;

            mass = - (norm_rij - l_ij[0]) * k_ij[0];
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                unit_rij[i] = unit_rij[i] * mass;

            // reduce
            for (int i = 0; i < DIM_INPUT_VECTOR_X; ++i)
                partial[i] += unit_rij[i];
        }
        for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; ++i)
            vector_y_out[row * DIM_OUTPUT_VECTOR_Y + i] = partial[i];
    }
}

//---------------------------------------------------------------------
// GPU Merge-based SpMV from scratch
//---------------------------------------------------------------------

/**
 * Run SpMV from scratch
 */
template <
    typename ValueT,
    typename OffsetT>
float LaunchSpMV(
    FlexParams<ValueT, OffsetT>&    params)
{
    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    
    // Get amount of temporary storage needed
    cudaError_t error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Warmup
    error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

}

/**
 * Run SpMV from scratch
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuMergeCsrmv_from_scratch(
    // ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    FlexParams<ValueT, OffsetT>&    params,
    int                             timing_iterations,
    float                           &setup_ms)
{
    setup_ms = 0.0;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    
    // Get amount of temporary storage needed
    cudaError_t error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    // CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(ValueT) * params.num_rows * DIM_OUTPUT_VECTOR_Y, cudaMemcpyHostToDevice));

    // Warmup
    error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.output_y_reducer_j, params.num_rows * DIM_OUTPUT_VECTOR_Y, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    GpuTimer timer;
    float elapsed_ms = 0.0;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
        CubDebugExit(error);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Display perf
 */
template <typename ValueT, typename OffsetT>
void DisplayPerf(
    float                           device_giga_bandwidth,
    double                          setup_ms,
    double                          avg_ms,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * (DIM_INPUT_VECTOR_X + 1) *2 + 4 * sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT) * DIM_OUTPUT_VECTOR_Y);

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_ms / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_ms / 1.0e6;

    if (!g_quiet)
        printf("fp%ld: %.4f setup ms, %.4f avg ms, %.5f gflops, %.3lf effective GB/s (%.2f%% peak)\n",
            sizeof(ValueT) * 8,
            setup_ms,
            avg_ms,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / device_giga_bandwidth * 100);
    else
        printf("%.5f, %.5f, %.6f, %.3lf, ",
            setup_ms,
            avg_ms,
            2 * nz_throughput,
            effective_bandwidth);

    fflush(stdout);
}



/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTest(
    ValueT                      alpha,
    ValueT                      beta,
    CooMatrix<ValueT, OffsetT>& coo_matrix,
    int                         timing_iterations,
    CommandLineArgs&            args)
{
    // Adaptive timing iterations: run 16 billion nonzeros through
    if (timing_iterations == -1)
        timing_iterations = std::min(50000ull, std::max(100ull, ((16ull << 30) / coo_matrix.num_nonzeros)));

    if (!g_quiet)
        printf("\t%d timing iterations\n", timing_iterations);

    // Convert to CSR
    CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
    if (!args.CheckCmdLineFlag("csrmv"))
        coo_matrix.Clear();

    // Display matrix info
    csr_matrix.Stats().Display(!g_quiet);
    if (!g_quiet)
    {
        printf("\n");
        csr_matrix.DisplayHistogram();
        printf("\n");
        if (g_verbose2)
            csr_matrix.Display();
        printf("\n");
    }
    fflush(stdout);

    // Allocate input and output vectors
    ValueT* vector_x        = new ValueT[csr_matrix.num_cols * DIM_INPUT_VECTOR_X];
    ValueT* vector_y_in     = new ValueT[csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y];
    ValueT* vector_y_out    = new ValueT[csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y];

    for (int col = 0; col < csr_matrix.num_cols * DIM_INPUT_VECTOR_X; ++col)
        // Initialize vector_x with seed
        vector_x[col] = static_cast<ValueT>(dis(gen) % 1000) / 1000.0f; // random number between 0 and 1

    for (int row = 0; row < csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y; ++row)
        vector_y_in[row] = 0.0f;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    float avg_ms, setup_ms;

    if (g_quiet) {
        printf("%s, %s, ", args.deviceProp.name, (sizeof(ValueT) > 4) ? "fp64" : "fp32"); fflush(stdout);
    }

    // Get GPU device bandwidth (GB/s)
    float device_giga_bandwidth = args.device_giga_bandwidth;

    // Allocate and initialize GPU problem
    // FlexSpmvParams<ValueT, OffsetT> params;
    FlexParams<ValueT, OffsetT> params;

    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.spm_k_ptr,          sizeof(ValueT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.spm_l_ptr,          sizeof(ValueT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets, sizeof(OffsetT) * (csr_matrix.num_rows + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.selector_i_ptr, sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.selector_j_ptr, sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.vector_x_ptr,        sizeof(ValueT) * csr_matrix.num_cols * DIM_INPUT_VECTOR_X));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.output_y_reducer_i,        sizeof(ValueT) * csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.output_y_reducer_j,        sizeof(ValueT) * csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y));
    params.num_rows         = csr_matrix.num_rows;
    params.num_cols         = csr_matrix.num_cols;
    params.num_nonzeros     = csr_matrix.num_nonzeros;

    CubDebugExit(cudaMemcpy((void*) params.spm_k_ptr,            (void*) csr_matrix.values,          sizeof(ValueT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.spm_l_ptr,            (void*) csr_matrix.values,          sizeof(ValueT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.d_row_end_offsets,   (void*) csr_matrix.row_offsets,     sizeof(OffsetT) * (csr_matrix.num_rows + 1), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.selector_i_ptr,      (void*) csr_matrix.column_indices_i, sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.selector_j_ptr,      (void*) csr_matrix.column_indices_j, sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.vector_x_ptr,        (void*) vector_x,                   sizeof(ValueT) * csr_matrix.num_cols * DIM_INPUT_VECTOR_X, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.output_y_reducer_i,  (void*) vector_y_in,                   sizeof(ValueT) * csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.output_y_reducer_j,  (void*) vector_y_in,                   sizeof(ValueT) * csr_matrix.num_rows * DIM_OUTPUT_VECTOR_Y, cudaMemcpyHostToDevice));

    // Merge-based from scratch
    if (!g_quiet) printf("\n\n");
    printf("Merge-based CsrMV from scratch, "); fflush(stdout);
    avg_ms = TestGpuMergeCsrmv_from_scratch(vector_y_out, params, timing_iterations, setup_ms);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);    
    
    // Cleanup
    if (params.spm_k_ptr)           CubDebugExit(g_allocator.DeviceFree(params.spm_k_ptr));
    if (params.spm_l_ptr)           CubDebugExit(g_allocator.DeviceFree(params.spm_l_ptr));
    if (params.d_row_end_offsets)   CubDebugExit(g_allocator.DeviceFree(params.d_row_end_offsets));
    if (params.selector_i_ptr)      CubDebugExit(g_allocator.DeviceFree(params.selector_i_ptr));
    if (params.selector_j_ptr)      CubDebugExit(g_allocator.DeviceFree(params.selector_j_ptr));
    if (params.vector_x_ptr)        CubDebugExit(g_allocator.DeviceFree(params.vector_x_ptr));
    if (params.output_y_reducer_i)  CubDebugExit(g_allocator.DeviceFree(params.output_y_reducer_i));
    if (params.output_y_reducer_j)  CubDebugExit(g_allocator.DeviceFree(params.output_y_reducer_j));

    if (vector_x)                   delete[] vector_x;
    if (vector_y_in)                delete[] vector_y_in;
    if (vector_y_out)               delete[] vector_y_out;

    // Check for any pending CUDA errors before final synchronization
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error before final sync: %s\n", cudaGetErrorString(error));
        fflush(stdout);
    }

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");
}

/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    ValueT              alpha,
    ValueT              beta,
    const std::string&  mtx_filename,
    int                 grid2d,
    int                 grid3d,
    int                 wheel,
    int                 dense,
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);

        if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
        {
            if (!g_quiet) printf("Trivial dataset\n");
            exit(0);
        }
        printf("%s, ", mtx_filename.c_str()); fflush(stdout);
    }
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        printf("grid2d_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        printf("grid3d_%d, ", grid3d); fflush(stdout);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        printf("wheel_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        // Generate dense graph
        OffsetT size = 1 << 24; // 16M nnz
        args.GetCmdLineArgument("size", size);

        OffsetT rows = size / dense;
        printf("dense_%d_x_%d, ", rows, dense); fflush(stdout);
        coo_matrix.InitDense(rows, dense);
    }
    else
    {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }

    RunTest(
        alpha,
        beta,
        coo_matrix,
        timing_iterations,
        args);
}



/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--csrmv | --hybmv | --bsrmv ] "
            "[--device=<device-id>] "
            "[--quiet] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp32] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n\t"
                "--dense=<cols>"
            "\n\t"
                "--grid2d=<width>"
            "\n\t"
                "--grid3d=<width>"
            "\n\t"
                "--wheel=<spokes>"
            "\n", argv[0]);
        exit(0);
    }

    bool                fp32;
    std::string         mtx_filename;
    int                 grid2d              = -1;
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 dense               = -1;
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp32 = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("wheel", wheel);
    args.GetCmdLineArgument("dense", dense);
    args.GetCmdLineArgument("alpha", alpha);
    args.GetCmdLineArgument("beta", beta);

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Run test(s)
    if (fp32)
    {
        RunTests<float, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }
    else
    {
        RunTests<double, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }

    // Check for any pending CUDA errors before final synchronization
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error before final sync: %s\n", cudaGetErrorString(error));
        fflush(stdout);
    }

    CubDebugExit(cudaDeviceSynchronize());

    fprintf(stderr, "The program is finished. \n");
    fflush(stderr);

    return 0;
}
