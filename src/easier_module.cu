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

using namespace cub;

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
    const ValueT*                         tensor_v, 
    const ValueT*                         tensor_spm1,
    const ValueT*                         tensor_spm2,
    const OffsetT*                         tensor_v1_idx,
    // const OffsetT*                         tensor_v2_idx,
    const OffsetT*                         offset,
    ValueT*                         reducer_1, // [out]
    ValueT*                         reducer_2, // [out]
    // ValueT*                         map_1, // [out]
    // ValueT*                         map_2, // [out]
    CommandLineArgs&                 args)
{
    for (OffsetT row = 0; row < args.num_rows; ++row)
    {
        // get the row start and end offset
        OffsetT row_start = offset[row];
        OffsetT row_end = offset[row + 1];

        // partial 
        ValueT* partial_1 = new ValueT[args.ne1_dim]();
        ValueT* partial_2 = new ValueT[args.ne2_dim]();

        // get the row non-zero values
        for (OffsetT i = row_start; i < row_end; ++i)
        {
            // selector
            OffsetT selector_i = tensor_v1_idx[i];
            // OffsetT selector_j = tensor_v2_idx[i];
            const ValueT* v_i = &tensor_v[selector_i * args.nv_dim];
            // const ValueT* v_j = &tensor_v[selector_j * args.nv_dim];
            const ValueT* spm_i = &tensor_spm1[i * args.ne1_dim];
            const ValueT* spm_j = &tensor_spm2[i * args.ne2_dim];

            // map
            ValueT gather_b_1 = v_i[0];
            ValueT pow_2 = pow(gather_b_1, 2);
            ValueT mul_32 = 0.5 * pow_2;
            ValueT mul_36 = mul_32 * spm_i[0];
            ValueT mul_40 = mul_32 * spm_j[0];

            // reducer
            for (int j = 0; j < args.ne1_dim; ++j)
                partial_1[j] += mul_36;
            for (int j = 0; j < args.ne2_dim; ++j)
                partial_2[j] += mul_40;
        }
        // reduce
        for (int j = 0; j < args.ne1_dim; ++j)
            reducer_1[row * args.ne1_dim + j] = partial_1[j];
        for (int j = 0; j < args.ne2_dim; ++j)
            reducer_2[row * args.ne2_dim + j] = partial_2[j];
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

template <typename ValueT>
int VerifyDeviceResults(
    ValueT*                         reducer_1_ref,
    ValueT*                         reducer_2_ref,
    // ValueT*                         map_1_ref,
    // ValueT*                         map_2_ref,
    ValueT*                         reducer_1_gpu,
    ValueT*                         reducer_2_gpu,
    // ValueT*                         map_1_gpu,
    // ValueT*                         map_2_gpu,
    bool                            verbose,
    CommandLineArgs&                args)
{
    // set epsilon
    ValueT epsilon = 1e-4;

    // Allocate array on host
    ValueT* reducer_1_host = new ValueT[args.num_rows * args.ne1_dim];
    ValueT* reducer_2_host = new ValueT[args.num_rows * args.ne2_dim];
    // ValueT* map_1_host = new ValueT[args.ne * args.ne1_dim];
    // ValueT* map_2_host = new ValueT[args.ne * args.ne2_dim];

    // Copy data back
    cudaMemcpy(reducer_1_host, reducer_1_gpu, sizeof(ValueT) * args.num_rows * args.ne1_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(reducer_2_host, reducer_2_gpu, sizeof(ValueT) * args.num_rows * args.ne2_dim, cudaMemcpyDeviceToHost);
    // cudaMemcpy(map_1_host, map_1_gpu, sizeof(ValueT) * args.ne * args.ne1_dim, cudaMemcpyDeviceToHost);
    // cudaMemcpy(map_2_host, map_2_gpu, sizeof(ValueT) * args.ne * args.ne2_dim, cudaMemcpyDeviceToHost);

    if (verbose)
    {
        printf("reducer_1:\n");
        for (int i = 0; i < args.num_rows; ++i)
        {
            printf("row %d: ", i);
            for (int j = 0; j < args.ne1_dim; ++j)
                printf("%f, ", reducer_1_ref[i * args.ne1_dim + j]);
            printf("|||");
            for (int j = 0; j < args.ne1_dim; ++j)
                printf("%f, ", reducer_1_host[i * args.ne1_dim + j]);
            printf("\n");
        }
        printf("\nreducer_2:\n");
        for (int i = 0; i < args.num_rows; ++i)
        {
            printf("row %d: ", i);
            for (int j = 0; j < args.ne2_dim; ++j)
                printf("%f, ", reducer_2_ref[i * args.ne2_dim + j]);
            printf("|||");
            for (int j = 0; j < args.ne2_dim; ++j)
                printf("%f, ", reducer_2_host[i * args.ne2_dim + j]);
            printf("\n");
        }
        // printf("\nmap_1:\n");
        // for (int i = 0; i < args.ne; ++i)
        // {
        //     printf("row %d: ", i);
        //     for (int j = 0; j < args.ne1_dim; ++j)
        //         printf("%f, ", map_1_ref[i * args.ne1_dim + j]);
        //     printf("|||");
        //     for (int j = 0; j < args.ne1_dim; ++j)
        //         printf("%f, ", map_1_host[i * args.ne1_dim + j]);
        //     printf("\n");
        // }
        // printf("\nmap_2:\n");
        // for (int i = 0; i < args.ne; ++i)
        // {
        //     printf("row %d: ", i);
        //     for (int j = 0; j < args.ne2_dim; ++j)
        //         printf("%f, ", map_2_ref[i * args.ne2_dim + j]);
        //     printf("|||");
        //     for (int j = 0; j < args.ne2_dim; ++j)
        //         printf("%f, ", map_2_host[i * args.ne2_dim + j]);
        //     printf("\n");
        // }
        printf("\n");
    }

    // Compare
    for (int i = 0; i < args.num_rows * args.ne1_dim; ++i)
        if (abs(reducer_1_host[i] - reducer_1_ref[i]) > epsilon) {
            printf("reducer_1_host[%d] = %f, reducer_1_ref[%d] = %f\n", i, reducer_1_host[i], i, reducer_1_ref[i]);
            return 1;
        }
    for (int i = 0; i < args.num_rows * args.ne2_dim; ++i)
        if (abs(reducer_2_host[i] - reducer_2_ref[i]) > epsilon) {
            printf("reducer_2_host[%d] = %f, reducer_2_ref[%d] = %f\n", i, reducer_2_host[i], i, reducer_2_ref[i]);
            return 1;
        }
    // for (int i = 0; i < args.ne * args.ne1_dim; ++i)
    //     if (abs(map_1_host[i] - map_1_ref[i]) > epsilon) {
    //         printf("map_1_host[%d] = %f, map_1_ref[%d] = %f\n", i, map_1_host[i], i, map_1_ref[i]);
    //         return 1;
    //     }
    // for (int i = 0; i < args.ne * args.ne2_dim; ++i)
    //     if (abs(map_2_host[i] - map_2_ref[i]) > epsilon) {
    //         printf("map_2_host[%d] = %f, map_2_ref[%d] = %f\n", i, map_2_host[i], i, map_2_ref[i]);
    //         return 1;
    //     }

    return 0;
}

/**
 * Run SpMV from scratch
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuMergeCsrmv_from_scratch(
    ValueT*                         reducer_1,
    ValueT*                         reducer_2,
    // ValueT*                         map_1,
    // ValueT*                         map_2,
    FlexParams<ValueT, OffsetT>&    params,
    int                             timing_iterations,
    float                           &setup_ms,
    CommandLineArgs&                args)
{
    setup_ms = 0.0;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    
    // Get amount of temporary storage needed
    cudaError_t error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
    CubDebugExit(error);

    // Allocate
    fprintf(stderr, "temp_storage_bytes: %zu\n", temp_storage_bytes);
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Warmup
    fprintf(stderr, "warmup\n");
    error = merged::merged_spmv_launch<ValueT, OffsetT>(params, d_temp_storage, temp_storage_bytes);
    fprintf(stderr, "warmup done\n");
    CubDebugExit(error);

    if (!g_quiet)
    {
        int compare = VerifyDeviceResults(
            reducer_1, 
            reducer_2,
            // map_1, map_2, 
            params.output_y_scatter_b_2_ptr, 
            params.output_y_scatter_b_3_ptr, 
            g_verbose,
            args
        );
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
    float                          setup_ms,
    float                          avg_ms,
    CommandLineArgs&                args)
{
    float nz_throughput, effective_bandwidth;
    size_t total_bytes = (
        args.ne * sizeof(ValueT) * args.ne1_dim *2 // read and write spm_i and map_i
        + args.ne * sizeof(ValueT) * args.ne2_dim *2 // read and write spm_j and map_j
        + args.num_rows * sizeof(OffsetT) * 2 // read and write row_end_offsets
        + args.ne * sizeof(OffsetT) * 2 // read selector_i and selector_j
        + args.ne * sizeof(ValueT) * args.nv_dim // read and write vector_x
        + args.num_rows * sizeof(ValueT) * args.ne1_dim // write reducer_1
        + args.num_rows * sizeof(ValueT) * args.ne2_dim // write reducer_2
    );

    nz_throughput       = float(args.ne) / avg_ms / 1.0e6;
    effective_bandwidth = float(total_bytes) / avg_ms / 1.0e6;

    if (!g_quiet)
        printf("fp%ld: %.4f setup ms, %.4f avg ms, %.5f gflops (position), %.3lf effective GB/s (%.2f%% peak)\n",
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
    int                         timing_iterations,
    CommandLineArgs&            args)
{
    std::mt19937 gen(49);
    std::uniform_int_distribution<> dis(0, 99999);

    // Allocate input and output vectors
    // Input tensor
    ValueT* tensor_v        = new ValueT[args.nv * args.nv_dim];
    ValueT* tensor_spm1     = new ValueT[args.ne * args.ne1_dim];
    ValueT* tensor_spm2     = new ValueT[args.ne * args.ne2_dim];
    OffsetT* tensor_v1_idx  = new OffsetT[args.ne];
    // OffsetT* tensor_v2_idx  = new OffsetT[args.ne];
    OffsetT* offset        = new OffsetT[args.num_rows + 1];

    // init tensors
    for (int i = 0; i < args.nv * args.nv_dim; ++i)
        tensor_v[i] = static_cast<ValueT>(dis(gen) % 1000) / 1000.0f;
    for (int i = 0; i < args.ne * args.ne1_dim; ++i)
        tensor_spm1[i] = static_cast<ValueT>(dis(gen) % 1000) / 1000.0f;
    for (int i = 0; i < args.ne * args.ne2_dim; ++i)
        tensor_spm2[i] = static_cast<ValueT>(dis(gen) % 1000) / 1000.0f;
    for (int i = 0; i < args.ne; ++i)
        tensor_v1_idx[i] = static_cast<OffsetT>(dis(gen) % args.nv);
    // for (int i = 0; i < args.ne; ++i)
    //     tensor_v2_idx[i] = static_cast<OffsetT>(dis(gen) % args.nv);
    // The offset array must start with 0, end with args.ne, and be sorted in non-decreasing order.
    offset[0] = 0;
    offset[args.num_rows] = args.ne;
    if (args.num_rows > 1) {
        // Generate (args.num_rows - 1) random offsets between 0 and args.ne
        std::vector<OffsetT> temp_offsets(args.num_rows - 1);
        for (int i = 0; i < args.num_rows - 1; ++i) {
            temp_offsets[i] = static_cast<OffsetT>(dis(gen) % (args.ne + 1));
        }
        std::sort(temp_offsets.begin(), temp_offsets.end());
        for (int i = 0; i < args.num_rows - 1; ++i) {
            offset[i + 1] = temp_offsets[i];
        }
    }

    // offset[2] = 4;

    // Print the offset array for debugging
    if (g_verbose) {
        printf("offset: ");
        for (int i = 0; i <= args.num_rows; ++i) {
            printf("%d ", static_cast<int>(offset[i]));
        }
        printf("\n");

        // Print the tensor_v1_idx and tensor_v2_idx arrays for debugging
        printf("tensor_v1_idx: ");
        for (int i = 0; i < args.ne; ++i) {
            printf("%d ", static_cast<int>(tensor_v1_idx[i]));
        }
        printf("\n");
        // printf("tensor_v2_idx: ");
        // for (int i = 0; i < args.ne; ++i) {
        //     printf("%d ", static_cast<int>(tensor_v2_idx[i]));
        // }
        // printf("\n");
        // Print the tensor_v array for debugging
        printf("tensor_v: ");
        for (int i = 0; i < args.nv * args.nv_dim; ++i) {
            for (int j = 0; j < args.nv_dim; ++j) {
                printf("%f ", tensor_v[i * args.nv_dim + j]);
            }
            printf("|");
        }
        printf("\n");

        // Print the tensor_spm1 and tensor_spm2 arrays for debugging
        printf("tensor_spm1: ");
        for (int i = 0; i < args.ne; ++i) {
            for (int j = 0; j < args.ne1_dim; ++j) {
                printf("%f ", tensor_spm1[i * args.ne1_dim + j]);
            }
            printf("|");
        }
        printf("\n");
        printf("tensor_spm2: ");
        for (int i = 0; i < args.ne; ++i) {
            for (int j = 0; j < args.ne2_dim; ++j) {
                printf("%f ", tensor_spm2[i * args.ne2_dim + j]);
            }
            printf("|");
        }
        printf("\n");
    }

    // Output vector
    ValueT* reducer_1    = new ValueT[args.num_rows * args.ne1_dim];
    ValueT* reducer_2    = new ValueT[args.num_rows * args.ne2_dim];
    // ValueT* map_1    = new ValueT[args.ne * args.ne1_dim];
    // ValueT* map_2    = new ValueT[args.ne * args.ne2_dim];

    // Compute reference answer
    SpmvGold(
        tensor_v,
        tensor_spm1,
        tensor_spm2,
        tensor_v1_idx,
        // tensor_v2_idx,
        offset,
        reducer_1,
        reducer_2,
        // map_1,
        // map_2,
        args
    );

    float avg_ms, setup_ms;

    if (g_quiet) {
        printf("%s, %s, ", args.deviceProp.name, (sizeof(ValueT) > 4) ? "fp64" : "fp32"); fflush(stdout);
    }

    // Get GPU device bandwidth (GB/s)
    float device_giga_bandwidth = args.device_giga_bandwidth;

    // Allocate and initialize GPU problem
    // FlexSpmvParams<ValueT, OffsetT> params;
    FlexParams<ValueT, OffsetT> params;

    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.bsx_ptr, sizeof(ValueT) * args.ne * args.ne1_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.bsy_ptr, sizeof(ValueT) * args.ne * args.ne2_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.gather_b_1_ptr,     sizeof(OffsetT) * args.ne));
    // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.selector_2_ptr,     sizeof(OffsetT) * args.ne));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.add_10_ptr,       sizeof(ValueT) * args.nv * args.nv_dim));

    // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.output_y_y_add_1_ptr,       sizeof(ValueT) * args.ne * args.ne1_dim));
    // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.output_y_y_add_2_ptr,       sizeof(ValueT) * args.ne * args.ne2_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.output_y_scatter_b_2_ptr,       sizeof(ValueT) * args.num_rows * args.ne1_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.output_y_scatter_b_3_ptr,       sizeof(ValueT) * args.num_rows * args.ne2_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets,  sizeof(OffsetT) * (args.num_rows + 1)));
    params.num_rows         = args.num_rows;
    params.num_cols         = args.num_cols;
    params.num_nonzeros     = args.ne;

    CubDebugExit(cudaMemcpy((void*) params.bsx_ptr,            (void*) tensor_spm1,          sizeof(ValueT) * args.ne * args.ne1_dim, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.bsy_ptr,            (void*) tensor_spm2,          sizeof(ValueT) * args.ne * args.ne2_dim, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.d_row_end_offsets,   (void*) offset,     sizeof(OffsetT) * (args.num_rows + 1), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.gather_b_1_ptr,       (void*) tensor_v1_idx, sizeof(OffsetT) * args.ne, cudaMemcpyHostToDevice));
    // CubDebugExit(cudaMemcpy((void*) params.selector_2_ptr,       (void*) tensor_v2_idx, sizeof(OffsetT) * args.ne, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void*) params.add_10_ptr,         (void*) tensor_v,                   sizeof(ValueT) * args.nv * args.nv_dim, cudaMemcpyHostToDevice));


    // Merge-based from scratch
    if (!g_quiet) printf("\n\n");
    printf("Merge-based CsrMV from scratch, "); fflush(stdout);
    avg_ms = TestGpuMergeCsrmv_from_scratch(
        reducer_1, 
        reducer_2,
        // map_1, map_2, 
        params, timing_iterations, 
        setup_ms, args);
    DisplayPerf<ValueT, OffsetT>(device_giga_bandwidth, setup_ms, avg_ms, args);    
    
    // Cleanup
    if (params.bsx_ptr)           CubDebugExit(g_allocator.DeviceFree(params.bsx_ptr));
    if (params.bsy_ptr)           CubDebugExit(g_allocator.DeviceFree(params.bsy_ptr));
    if (params.gather_b_1_ptr)      CubDebugExit(g_allocator.DeviceFree(params.gather_b_1_ptr));
    // if (params.selector_2_ptr)      CubDebugExit(g_allocator.DeviceFree(params.selector_2_ptr));
    if (params.add_10_ptr)        CubDebugExit(g_allocator.DeviceFree(params.add_10_ptr));
    // if (params.output_y_y_add_1_ptr)  CubDebugExit(g_allocator.DeviceFree(params.output_y_y_add_1_ptr));
    if (params.output_y_scatter_b_3_ptr)  CubDebugExit(g_allocator.DeviceFree(params.output_y_scatter_b_3_ptr));
    if (params.output_y_scatter_b_2_ptr)       CubDebugExit(g_allocator.DeviceFree(params.output_y_scatter_b_2_ptr));
    // if (params.output_y_y_reducer_2_ptr)       CubDebugExit(g_allocator.DeviceFree(params.output_y_y_reducer_2_ptr));
    if (params.d_row_end_offsets)   CubDebugExit(g_allocator.DeviceFree(params.d_row_end_offsets));

    if (tensor_v)                   delete[] tensor_v;
    if (tensor_spm1)                delete[] tensor_spm1;
    if (tensor_spm2)                delete[] tensor_spm2;
    if (tensor_v1_idx)              delete[] tensor_v1_idx;
    // if (tensor_v2_idx)              delete[] tensor_v2_idx;
    if (offset)                     delete[] offset;
    if (reducer_1)                  delete[] reducer_1;
    if (reducer_2)                  delete[] reducer_2;
    // if (map_1)                      delete[] map_1;
    // if (map_2)                      delete[] map_2;

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
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    RunTest<ValueT, OffsetT>(
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
            "[--rows=<rows>] "
            "[--cols=<cols>] "
            "[--nnz=<nnz>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n", argv[0]);
        exit(0);
    }

    bool                fp32;
    std::string         mtx_filename;
    int                 timing_iterations   = 100;
    // tesnor info
    args.nv = 1543;
    args.ne = 123134;
    args.num_rows = 12314;
    args.num_cols = 1543;
        // tesnor info
        // args.nv = 4;
        // args.ne = 12;
        // args.num_rows = 5;
        // args.num_cols = 4;
    args.nv_dim = 1;
    args.ne1_dim = 1;
    args.ne2_dim = 1;


    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp32 = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("rows", args.num_rows);
    args.GetCmdLineArgument("cols", args.num_cols);
    args.GetCmdLineArgument("nnz", args.ne);
    args.GetCmdLineArgument("nv", args.nv);
    args.GetCmdLineArgument("nv_dim", args.nv_dim);
    args.GetCmdLineArgument("ne1_dim", args.ne1_dim);
    args.GetCmdLineArgument("ne2_dim", args.ne2_dim);


    // if (g_verbose) {
        printf("args.num_rows: %ld\n", (long) args.num_rows);
        printf("args.num_cols: %ld\n", (long) args.num_cols);
        printf("args.ne: %ld\n", (long) args.ne);
        printf("args.nv: %ld\n", (long) args.nv);
        printf("args.nv_dim: %d\n", args.nv_dim);
        printf("args.ne1_dim: %d\n", args.ne1_dim);
        printf("args.ne2_dim: %d\n", args.ne2_dim);
        // for (int i = 0; i < args.nv_shape.size(); ++i) {
        //     printf("args.nv_shape[%d]: %ld\n", i, (long) args.nv_shape[i]);
        // }
        // for (int i = 0; i < args.ne1_shape.size(); ++i) {
        //     printf("args.ne1_shape[%d]: %ld\n", i, (long) args.ne1_shape[i]);
        // }
        // for (int i = 0; i < args.ne2_shape.size(); ++i) {
        //     printf("args.ne2_shape[%d]: %ld\n", i, (long) args.ne2_shape[i]);
        // }
    // }
    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Run test(s)
    if (fp32)
    {
        RunTests<float, int>(timing_iterations, args);
    }
    else
    {
        RunTests<double, int>(timing_iterations, args);
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
