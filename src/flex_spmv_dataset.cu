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

#include "../include/device_flex_spmv.cuh"
#include "sparse_matrix.h"

using namespace cub;

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
        ValueT partial = beta * vector_y_in[row];
        for (
            OffsetT offset = a.row_offsets[row];
            offset < a.row_offsets[row + 1];
            ++offset)
        {
            // partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
            // flex spmv
            partial += alpha * a.values[a.column_indices_A[offset]] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}




//---------------------------------------------------------------------
// cuSparse HybMV
//---------------------------------------------------------------------

/**
 * Run cuSparse HYB SpMV (specialized for fp32)
 */
// template <
//     typename OffsetT>
// float TestCusparseHybmv(
//     float*                          vector_y_in,
//     float*                          reference_vector_y_out,
//     SpmvParams<float, OffsetT>&     params,
//     int                             timing_iterations,
//     float                           &setup_ms,
//     cusparseHandle_t                cusparse)
// {
//     CpuTimer cpu_timer;
//     cpu_timer.Start();

//     // Construct Hyb matrix
//     cusparseMatDescr_t mat_desc;
//     // cusparseHybMat_t hyb_desc;
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&mat_desc));
//     // AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateHybMat(&hyb_desc));
//     cusparseStatus_t status = cusparseScsr2hyb(
//         cusparse,
//         params.num_rows, params.num_cols,
//         mat_desc,
//         params.d_values, params.d_row_end_offsets, params.d_column_indices,
//         hyb_desc,
//         0,
//         CUSPARSE_HYB_PARTITION_AUTO);
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, status);

//     cudaDeviceSynchronize();
//     cpu_timer.Stop();
//     setup_ms = cpu_timer.ElapsedMillis();

//     // Reset input/output vector y
//     CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

//     // Warmup
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseShybmv(
//         cusparse,
//         CUSPARSE_OPERATION_NON_TRANSPOSE,
//         &params.alpha, mat_desc,
//         hyb_desc,
//         params.d_vector_x, &params.beta, params.d_vector_y));

//     if (!g_quiet)
//     {
//         int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
//         printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
//     }

//     // Timing
//     float elapsed_ms = 0.0;
//     GpuTimer timer;

//     timer.Start();
//     for(int it = 0; it < timing_iterations; ++it)
//     {
//         AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseShybmv(
//             cusparse,
//             CUSPARSE_OPERATION_NON_TRANSPOSE,
//             &params.alpha, mat_desc,
//             hyb_desc,
//             params.d_vector_x, &params.beta, params.d_vector_y));
//     }
//     timer.Stop();
//     elapsed_ms += timer.ElapsedMillis();

//     // Cleanup
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyHybMat(hyb_desc));
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(mat_desc));

//     return elapsed_ms / timing_iterations;
// }


// /**
//  * Run cuSparse HYB SpMV (specialized for fp64)
//  */
// template <
//     typename OffsetT>
// float TestCusparseHybmv(
//     double*                         vector_y_in,
//     double*                         reference_vector_y_out,
//     SpmvParams<double, OffsetT>&    params,
//     int                             timing_iterations,
//     float                           &setup_ms,
//     cusparseHandle_t                cusparse)
// {
//     CpuTimer cpu_timer;
//     cpu_timer.Start();

//     // Construct Hyb matrix
//     cusparseMatDescr_t mat_desc;
//     cusparseHybMat_t hyb_desc;
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&mat_desc));
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateHybMat(&hyb_desc));
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsr2hyb(
//         cusparse,
//         params.num_rows, params.num_cols,
//         mat_desc,
//         params.d_values, params.d_row_end_offsets, params.d_column_indices,
//         hyb_desc,
//         0,
//         CUSPARSE_HYB_PARTITION_AUTO));

//     cudaDeviceSynchronize();
//     cpu_timer.Stop();
//     setup_ms = cpu_timer.ElapsedMillis();

//     // Reset input/output vector y
//     CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

//     // Warmup
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDhybmv(
//         cusparse,
//         CUSPARSE_OPERATION_NON_TRANSPOSE,
//         &params.alpha, mat_desc,
//         hyb_desc,
//         params.d_vector_x, &params.beta, params.d_vector_y));

//     if (!g_quiet)
//     {
//         int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
//         printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
//     }

//     // Timing
//     float elapsed_ms = 0.0;
//     GpuTimer timer;

//     timer.Start();
//     for(int it = 0; it < timing_iterations; ++it)
//     {
//         AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDhybmv(
//             cusparse,
//             CUSPARSE_OPERATION_NON_TRANSPOSE,
//             &params.alpha, mat_desc,
//             hyb_desc,
//             params.d_vector_x, &params.beta, params.d_vector_y));
//     }
//     timer.Stop();
//     elapsed_ms += timer.ElapsedMillis();

//     // Cleanup
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyHybMat(hyb_desc));
//     AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(mat_desc));

//     return elapsed_ms / timing_iterations;
// }



//---------------------------------------------------------------------
// cuSparse CsrMV
//---------------------------------------------------------------------

/**
 * Run cuSparse SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float TestCusparseCsrmv(
    float*                          vector_y_in,
    float*                          reference_vector_y_out,
    SpmvParams<float, OffsetT>&     params,
    int                             timing_iterations,
    float                           &setup_ms,
    cusparseHandle_t                cusparse)
{
    setup_ms = 0.0;

    cusparseMatDescr_t desc;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&desc));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_ms    = 0.0;
    GpuTimer timer;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(desc));
    return elapsed_ms / timing_iterations;
}


/**
 * Run cuSparse SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float TestCusparseCsrmv(
    double*                         vector_y_in,
    double*                         reference_vector_y_out,
    SpmvParams<double, OffsetT>&    params,
    int                             timing_iterations,
    float                           &setup_ms,
    cusparseHandle_t                cusparse)
{
    cusparseMatDescr_t desc;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&desc));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_ms = 0.0;
    GpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));

    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(desc));
    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// GPU Merge-based SpMV
//---------------------------------------------------------------------

/**
 * Run CUB SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuMergeCsrmv(
    ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    FlexSpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations,
    float                           &setup_ms)
{
    setup_ms = 0.0;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Get amount of temporary storage needed
    CubDebugExit(DeviceFlexSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_column_indices_A,
        params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros));

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(ValueT) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    CubDebugExit(DeviceFlexSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_column_indices_A,
        params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    GpuTimer timer;
    float elapsed_ms = 0.0;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        CubDebugExit(DeviceFlexSpmv::CsrMV(
            d_temp_storage, temp_storage_bytes,
            params.d_values, params.d_column_indices_A,
            params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, params.d_vector_y,
            params.num_rows, params.num_cols, params.num_nonzeros));
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
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
float TestGpuMergeCsrmv_from_scratch(
    ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    FlexParams<ValueT, OffsetT>&    params_flex,
    int                             timing_iterations,
    float                           &setup_ms)
{
    setup_ms = 0.0;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    
    // Get amount of temporary storage needed
    cudaError_t error = merged::merged_spmv_launch<ValueT, OffsetT>(params_flex, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params_flex.d_vector_y, vector_y_in, sizeof(ValueT) * params_flex.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    error = merged::merged_spmv_launch<ValueT, OffsetT>(params_flex, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params_flex.d_vector_y, params_flex.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    GpuTimer timer;
    float elapsed_ms = 0.0;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        error = merged::merged_spmv_launch<ValueT, OffsetT>(params_flex, d_temp_storage, temp_storage_bytes);
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
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

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
    ValueT* vector_x        = new ValueT[csr_matrix.num_cols];
    ValueT* vector_y_in     = new ValueT[csr_matrix.num_rows];
    ValueT* vector_y_out    = new ValueT[csr_matrix.num_rows];

    for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    float avg_ms, setup_ms;

    if (g_quiet) {
        printf("%s, %s, ", args.deviceProp.name, (sizeof(ValueT) > 4) ? "fp64" : "fp32"); fflush(stdout);
    }

    // Get GPU device bandwidth (GB/s)
    float device_giga_bandwidth = args.device_giga_bandwidth;

    // Allocate and initialize GPU problem
    FlexSpmvParams<ValueT, OffsetT> params;
    FlexParams<ValueT, OffsetT> params_flex;

    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_values,          sizeof(ValueT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets, sizeof(OffsetT) * (csr_matrix.num_rows + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_column_indices_A, sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_x,        sizeof(ValueT) * csr_matrix.num_cols));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_y,        sizeof(ValueT) * csr_matrix.num_rows));
    params.num_rows         = csr_matrix.num_rows;
    params.num_cols         = csr_matrix.num_cols;
    params.num_nonzeros     = csr_matrix.num_nonzeros;
    params.alpha            = alpha;
    params.beta             = beta;
    // for merged spmv from scratch
    params_flex.d_spm_nnz          = params.d_values;
    params_flex.d_row_end_offsets = params.d_row_end_offsets;
    params_flex.d_column_indices  = params.d_column_indices;
    params_flex.d_column_indices_A = params.d_column_indices_A;
    params_flex.d_vector_x        = params.d_vector_x;
    params_flex.d_vector_y        = params.d_vector_y;
    params_flex.num_rows          = params.num_rows;
    params_flex.num_cols          = params.num_cols;
    params_flex.num_nonzeros      = params.num_nonzeros;
    params_flex.alpha             = alpha;
    params_flex.beta              = beta;

    CubDebugExit(cudaMemcpy((void*) params.d_values,            (void*) csr_matrix.values,          sizeof(ValueT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.d_row_end_offsets,   (void*) csr_matrix.row_offsets,     sizeof(OffsetT) * (csr_matrix.num_rows + 1), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.d_column_indices,    (void*) csr_matrix.column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.d_column_indices_A,  (void*) csr_matrix.column_indices_A, sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.d_vector_x,          (void*) vector_x,                   sizeof(ValueT) * csr_matrix.num_cols, cudaMemcpyHostToDevice));

    // Merge-based from scratch
    if (!g_quiet) printf("\n\n");
    printf("Merge-based CsrMV from scratch, "); fflush(stdout);
    avg_ms = TestGpuMergeCsrmv_from_scratch(vector_y_in, vector_y_out, params_flex, timing_iterations, setup_ms);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);
    
    // Merge-based
    if (!g_quiet) printf("\n\n");
    printf("Merge-based CsrMV, "); fflush(stdout);
    avg_ms = TestGpuMergeCsrmv(vector_y_in, vector_y_out, params, timing_iterations, setup_ms);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

    // // Initialize cuSparse (deprecated)
    // cusparseHandle_t cusparse;
    // AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreate(&cusparse));

	// // cuSPARSE CsrMV
    // if (!g_quiet) printf("\n\n");
    // printf("cuSPARSE CsrMV, "); fflush(stdout);
    // avg_ms = TestCusparseCsrmv(vector_y_in, vector_y_out, params, timing_iterations, setup_ms, cusparse);
    // DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

	// // cuSPARSE HybMV
    // if (!g_quiet) printf("\n\n");
    // printf("cuSPARSE HybMV, "); fflush(stdout);
    // avg_ms = TestCusparseHybmv(vector_y_in, vector_y_out, params, timing_iterations, setup_ms, cusparse);
    // DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

    // Cleanup
    if (params.d_values)            CubDebugExit(g_allocator.DeviceFree(params.d_values));
    if (params.d_row_end_offsets)   CubDebugExit(g_allocator.DeviceFree(params.d_row_end_offsets));
    if (params.d_column_indices)    CubDebugExit(g_allocator.DeviceFree(params.d_column_indices));
    if (params.d_column_indices_A)  CubDebugExit(g_allocator.DeviceFree(params.d_column_indices_A));
    if (params.d_vector_x)          CubDebugExit(g_allocator.DeviceFree(params.d_vector_x));
    if (params.d_vector_y)          CubDebugExit(g_allocator.DeviceFree(params.d_vector_y));

    if (vector_x)                   delete[] vector_x;
    if (vector_y_in)                delete[] vector_y_in;
    if (vector_y_out)               delete[] vector_y_out;
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

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");

    return 0;
}
