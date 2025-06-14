/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <iterator>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/agent/agent_segment_fixup.cuh>
#include <cub/agent/agent_spmv_orig.cuh>
#include <cub/device/dispatch/dispatch_spmv_orig.cuh>
#include <cub/util_type.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/config.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "agent_flex_spmv.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/******************************************************************************
 * Flexible Spmv kernel entry points
 *****************************************************************************/

/**
 * Spmv search kernel. Identifies merge path starting coordinates for each tile.
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized SpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT>                    ///< Signed integer type for sequence offsets
__global__ void DeviceFlexSpmv1ColKernel(
    FlexSpmvParams<ValueT, OffsetT> spmv_params)                ///< [in] SpMV input parameter bundle
{
    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorT;

    VectorValueIteratorT wrapped_vector_x(spmv_params.d_vector_x);

    int row_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row_idx < spmv_params.num_rows)
    {
        OffsetT     end_nonzero_idx = spmv_params.d_row_end_offsets[row_idx];
        OffsetT     nonzero_idx = spmv_params.d_row_end_offsets[row_idx - 1];

        ValueT value = 0.0;
        if (end_nonzero_idx != nonzero_idx)
        {
            // value = spmv_params.d_values[nonzero_idx] * wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]];
            // // the dense matrix A 2D
            // value = spmv_params.d_dense_matrix[spmv_params.d_column_indices[nonzero_idx] + row_idx * spmv_params.dense_matrix_width] * wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]];
            // the dense matrix A 1D
            value = spmv_params.d_values[spmv_params.d_column_indices_A[nonzero_idx]] * wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]];
        }

        spmv_params.d_vector_y[row_idx] = value;
    }
}

/**
 * Flexible Spmv agent entry point
 */
template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
__launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__global__ void DeviceFlexSpmvKernel(
    FlexSpmvParams<ValueT, OffsetT>  spmv_params,                ///< [in] Flexible Spmv input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    int                             num_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                             num_segment_fixup_tiles)    ///< [in] Number of reduce-by-key tiles (fixup grid size)
{
    // Flexible Spmv agent type specialization
    typedef AgentFlexSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentFlexSpmvT;

    // Shared memory for AgentFlexSpmv
    __shared__ typename AgentFlexSpmvT::TempStorage temp_storage;

    AgentFlexSpmvT(temp_storage, spmv_params).ConsumeTile(
        d_tile_coordinates,
        d_tile_carry_pairs,
        num_tiles);

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

// Type definitions and policies moved outside
template <typename ValueT, typename OffsetT>
using SpmvParamsT = FlexSpmvParams<ValueT, OffsetT>;

template <typename OffsetT>
using CoordinateT = typename CubVector<OffsetT, 2>::Type;

template <typename ValueT, typename OffsetT>
using ScanTileStateT = ReduceByKeyScanTileState<ValueT, OffsetT>;

template <typename OffsetT, typename ValueT>
using KeyValuePairT = KeyValuePair<OffsetT, ValueT>;

// SM35
template <typename ValueT>
using Policy350SpmvPolicy = AgentSpmvPolicy<
    (sizeof(ValueT) > 4) ? 96 : 128,
    (sizeof(ValueT) > 4) ? 4 : 7,
    LOAD_LDG,
    LOAD_CA,
    LOAD_LDG,
    LOAD_LDG,
    LOAD_LDG,
    (sizeof(ValueT) > 4) ? true : false,
    BLOCK_SCAN_WARP_SCANS>;

using Policy350SegmentFixupPolicy = AgentSegmentFixupPolicy<
    128,
    3,
    BLOCK_LOAD_VECTORIZE,
    LOAD_LDG,
    BLOCK_SCAN_WARP_SCANS>;

// SM37
template <typename ValueT>
using Policy370SpmvPolicy = AgentSpmvPolicy<
    (sizeof(ValueT) > 4) ? 128 : 128,
    (sizeof(ValueT) > 4) ? 9 : 14,
    LOAD_LDG,
    LOAD_CA,
    LOAD_LDG,
    LOAD_LDG,
    LOAD_LDG,
    false,
    BLOCK_SCAN_WARP_SCANS>;

using Policy370SegmentFixupPolicy = AgentSegmentFixupPolicy<
    128,
    3,
    BLOCK_LOAD_VECTORIZE,
    LOAD_LDG,
    BLOCK_SCAN_WARP_SCANS>;

// SM50
template <typename ValueT>
using Policy500SpmvPolicy = AgentSpmvPolicy<
    (sizeof(ValueT) > 4) ? 64 : 128,
    (sizeof(ValueT) > 4) ? 6 : 7,
    LOAD_LDG,
    LOAD_DEFAULT,
    (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
    (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
    LOAD_LDG,
    (sizeof(ValueT) > 4) ? true : false,
    (sizeof(ValueT) > 4) ? BLOCK_SCAN_WARP_SCANS : BLOCK_SCAN_RAKING_MEMOIZE>;

using Policy500SegmentFixupPolicy = AgentSegmentFixupPolicy<
    128,
    3,
    BLOCK_LOAD_VECTORIZE,
    LOAD_LDG,
    BLOCK_SCAN_RAKING_MEMOIZE>;

// SM60
template <typename ValueT>
using Policy600SpmvPolicy = AgentSpmvPolicy<
    (sizeof(ValueT) > 4) ? 64 : 128,
    (sizeof(ValueT) > 4) ? 5 : 7,
    LOAD_DEFAULT,
    LOAD_DEFAULT,
    LOAD_DEFAULT,
    LOAD_DEFAULT,
    LOAD_DEFAULT,
    false,
    BLOCK_SCAN_WARP_SCANS>;

using Policy600SegmentFixupPolicy = AgentSegmentFixupPolicy<
    128,
    3,
    BLOCK_LOAD_DIRECT,
    LOAD_LDG,
    BLOCK_SCAN_WARP_SCANS>;

// Current PTX compiler pass policies
#if (CUB_PTX_ARCH >= 600)
template <typename ValueT>
using PtxSpmvPolicyT = Policy600SpmvPolicy<ValueT>;
using PtxSegmentFixupPolicy = Policy600SegmentFixupPolicy;
#elif (CUB_PTX_ARCH >= 500)
template <typename ValueT>
using PtxSpmvPolicyT = Policy500SpmvPolicy<ValueT>;
using PtxSegmentFixupPolicy = Policy500SegmentFixupPolicy;
#elif (CUB_PTX_ARCH >= 370)
template <typename ValueT>
using PtxSpmvPolicyT = Policy370SpmvPolicy<ValueT>;
using PtxSegmentFixupPolicy = Policy370SegmentFixupPolicy;
#else
template <typename ValueT>
using PtxSpmvPolicyT = Policy350SpmvPolicy<ValueT>;
using PtxSegmentFixupPolicy = Policy350SegmentFixupPolicy;
#endif

constexpr int INIT_KERNEL_THREADS = 128;


/**
 * Kernel kernel dispatch configuration.
 */
struct FlexSpmvKernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    void Init()
    {
        block_threads       = PolicyT::BLOCK_THREADS;
        items_per_thread    = PolicyT::ITEMS_PER_THREAD;
        tile_items          = block_threads * items_per_thread;
    }
};

/**
 * Flexible SpMV dispatch function
 */
template <typename ValueT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchFlexSpmv(
    void*                   d_temp_storage,                     ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    SpmvParamsT<ValueT, OffsetT>& spmv_params,                 ///< SpMV input parameter bundle
    cudaStream_t            stream                  = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                    debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
{
    cudaError error = cudaSuccess;
    do
    {
        // Get PTX version
        int ptx_version = 0;
        if (CubDebug(error = PtxVersion(ptx_version))) break;

        // Get kernel kernel dispatch configurations
        FlexSpmvKernelConfig spmv_config, segment_fixup_config;
        if (CUB_IS_DEVICE_CODE)
        {
            #if CUB_INCLUDE_DEVICE_CODE
                // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
                spmv_config.template Init<PtxSpmvPolicyT<ValueT>>();
                segment_fixup_config.template Init<PtxSegmentFixupPolicy>();
            #endif
        }
        else
        {
            #if CUB_INCLUDE_HOST_CODE
                // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
                if (ptx_version >= 600)
                {
                    spmv_config.template            Init<Policy600SpmvPolicy<ValueT>>();
                    segment_fixup_config.template   Init<Policy600SegmentFixupPolicy>();
                }
                else if (ptx_version >= 500)
                {
                    spmv_config.template            Init<Policy500SpmvPolicy<ValueT>>();
                    segment_fixup_config.template   Init<Policy500SegmentFixupPolicy>();
                }
                else if (ptx_version >= 370)
                {
                    spmv_config.template            Init<Policy370SpmvPolicy<ValueT>>();
                    segment_fixup_config.template   Init<Policy370SegmentFixupPolicy>();
                }
                else
                {
                    spmv_config.template            Init<Policy350SpmvPolicy<ValueT>>();
                    segment_fixup_config.template   Init<Policy350SegmentFixupPolicy>();
                }
            #endif
        }

        // 1. DeviceFlexSpmv1ColKernel
        // 2. DeviceSpmvSearchKernel
        // 3. DeviceFlexSpmvKernel
        // 4. DeviceSegmentFixupKernel
        
#ifndef CUB_RUNTIME_ENABLED
        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);
#else
        cudaError error = cudaSuccess;
        do
        {
            if (spmv_params.num_rows < 0 || spmv_params.num_cols < 0)
            {
                return cudaErrorInvalidValue;
            }

            if (spmv_params.num_rows == 0 || spmv_params.num_cols == 0)
            { // Empty problem, no-op.
                if (d_temp_storage == NULL)
                {
                    temp_storage_bytes = 1;
                }
                break;
            }

            if (spmv_params.num_cols == 1)
            {
                if (d_temp_storage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    temp_storage_bytes = 1;
                    break;
                }

                // Get search/init grid dims
                int degen_col_kernel_block_size = INIT_KERNEL_THREADS;
                int degen_col_kernel_grid_size = cub::DivideAndRoundUp(spmv_params.num_rows, degen_col_kernel_block_size);

                if (debug_synchronous) _CubLog("Invoking spmv_1col_kernel<<<%d, %d, 0, %lld>>>()\n",
                    degen_col_kernel_grid_size, degen_col_kernel_block_size, (long long) stream);

                // Invoke spmv_1col_kernel
                DeviceFlexSpmv1ColKernel<PtxSpmvPolicyT<ValueT>, ValueT, OffsetT><<<
                    degen_col_kernel_grid_size, degen_col_kernel_block_size, 0, stream
                >>>(spmv_params);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                break;
            }

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Total number of spmv work items
            int num_merge_items = spmv_params.num_rows + spmv_params.num_nonzeros;

            // Tile sizes of kernels
            int merge_tile_size              = spmv_config.block_threads * spmv_config.items_per_thread;
            int segment_fixup_tile_size     = segment_fixup_config.block_threads * segment_fixup_config.items_per_thread;

            // Number of tiles for kernels
            int num_merge_tiles            = cub::DivideAndRoundUp(num_merge_items, merge_tile_size);
            int num_segment_fixup_tiles    = cub::DivideAndRoundUp(num_merge_tiles, segment_fixup_tile_size);

            // Get SM occupancy for kernels
            int spmv_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                spmv_sm_occupancy,
                DeviceFlexSpmvKernel<PtxSpmvPolicyT<ValueT>, ScanTileStateT<ValueT, OffsetT>, ValueT, OffsetT, CoordinateT<OffsetT>, false, false>,
                spmv_config.block_threads))) break;

            int segment_fixup_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                segment_fixup_sm_occupancy,
                DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT<OffsetT, ValueT>*, ValueT*, OffsetT, ScanTileStateT<ValueT, OffsetT>>,
                segment_fixup_config.block_threads))) break;

            // Get grid dimensions
            dim3 spmv_grid_size(
                CUB_MIN(num_merge_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_merge_tiles, max_dim_x),
                1);

            dim3 segment_fixup_grid_size(
                CUB_MIN(num_segment_fixup_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_segment_fixup_tiles, max_dim_x),
                1);

            // Get the temporary storage allocation requirements
            size_t allocation_sizes[3];
            if (CubDebug(error = ScanTileStateT<ValueT, OffsetT>::AllocationSize(num_segment_fixup_tiles, allocation_sizes[0]))) break;    // bytes needed for reduce-by-key tile status descriptors
            allocation_sizes[1] = num_merge_tiles * sizeof(KeyValuePairT<OffsetT, ValueT>);       // bytes needed for block carry-out pairs
            allocation_sizes[2] = (num_merge_tiles + 1) * sizeof(CoordinateT<OffsetT>);   // bytes needed for tile starting coordinates

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void* allocations[3] = {};
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Construct the tile status interface
            ScanTileStateT<ValueT, OffsetT> tile_state;
            if (CubDebug(error = tile_state.Init(num_segment_fixup_tiles, allocations[0], allocation_sizes[0]))) break;

            // Alias the other allocations
            KeyValuePairT<OffsetT, ValueT>*  d_tile_carry_pairs      = (KeyValuePairT<OffsetT, ValueT>*) allocations[1];  // Agent carry-out pairs
            CoordinateT<OffsetT>*    d_tile_coordinates      = (CoordinateT<OffsetT>*) allocations[2];    // Agent starting coordinates

            // Get search/init grid dims
            int search_block_size   = INIT_KERNEL_THREADS;
            int search_grid_size    = cub::DivideAndRoundUp(num_merge_tiles + 1, search_block_size);

            if (search_grid_size < sm_count)
            {
                // Not enough spmv tiles to saturate the device: have spmv blocks search their own staring coords
                d_tile_coordinates = NULL;
            }
            else
            {
                // Use separate search kernel if we have enough spmv tiles to saturate the device

                // Log spmv_search_kernel configuration
                if (debug_synchronous) _CubLog("Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                    search_grid_size, search_block_size, (long long) stream);

                // Invoke spmv_search_kernel
                DeviceSpmvSearchKernel<PtxSpmvPolicyT<ValueT>, OffsetT, CoordinateT<OffsetT>, SpmvParamsT<ValueT, OffsetT>><<<
                    search_grid_size, search_block_size, 0, stream
                >>>(num_merge_tiles, d_tile_coordinates, spmv_params);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }

            // Log spmv_kernel configuration
            if (debug_synchronous) _CubLog("Invoking spmv_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                spmv_grid_size.x, spmv_grid_size.y, spmv_grid_size.z, spmv_config.block_threads, (long long) stream, spmv_config.items_per_thread, spmv_sm_occupancy);

            // Invoke spmv_kernel
            DeviceFlexSpmvKernel<PtxSpmvPolicyT<ValueT>, ScanTileStateT<ValueT, OffsetT>, ValueT, OffsetT, CoordinateT<OffsetT>, false, false><<<
                spmv_grid_size, spmv_config.block_threads, 0, stream
            >>>(spmv_params, d_tile_coordinates, d_tile_carry_pairs, num_merge_tiles, tile_state, num_segment_fixup_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Run reduce-by-key fixup if necessary
            if (num_merge_tiles > 1)
            {
                // Log segment_fixup_kernel configuration
                if (debug_synchronous) _CubLog("Invoking segment_fixup_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    segment_fixup_grid_size.x, segment_fixup_grid_size.y, segment_fixup_grid_size.z, segment_fixup_config.block_threads, (long long) stream, segment_fixup_config.items_per_thread, segment_fixup_sm_occupancy);

                // Invoke segment_fixup_kernel
                DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT<OffsetT, ValueT>*, ValueT*, OffsetT, ScanTileStateT<ValueT, OffsetT>><<<
                    segment_fixup_grid_size, segment_fixup_config.block_threads, 0, stream
                >>>(d_tile_carry_pairs, spmv_params.d_vector_y, num_merge_tiles, num_segment_fixup_tiles, tile_state);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);
#endif // CUB_RUNTIME_ENABLED

        if (CubDebug(error)) break;
    }
    while (0);

    return error;
}

} // namespace cub
CUB_NS_POSTFIX 
