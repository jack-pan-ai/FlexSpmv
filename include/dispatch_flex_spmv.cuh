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
    SpmvParams<ValueT, OffsetT> spmv_params)                ///< [in] SpMV input parameter bundle
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
            value = spmv_params.d_dense_matrix[spmv_params.d_column_indices[nonzero_idx] + row_idx * spmv_params.dense_matrix_width] * wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]];
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



/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceFlexSpmv
 */
template <
    typename ValueT,  ///< Matrix and vector value type
    typename OffsetT> ///< Signed integer type for sequence offsets
struct DispatchFlexSpmv : public DispatchSpmv<ValueT, OffsetT>
{
    // Override SpmvParams bundle type with FlexSpmvParams
    typedef FlexSpmvParams<ValueT, OffsetT> SpmvParamsT;

    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t FlexDispatch(
        void*                   d_temp_storage,                     ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
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
            KernelConfig spmv_config, segment_fixup_config;
            InitConfigs(ptx_version, spmv_config, segment_fixup_config);

            if (CubDebug(error = Dispatch(
                d_temp_storage, temp_storage_bytes, spmv_params, stream, debug_synchronous,
                DeviceFlexSpmv1ColKernel<PtxSpmvPolicyT, ValueT, OffsetT>,
                DeviceSpmvSearchKernel<PtxSpmvPolicyT, OffsetT, CoordinateT, SpmvParamsT>,
                DeviceFlexSpmvKernel<PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, false, false>,
                DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT*, ValueT*, OffsetT, ScanTileStateT>,
                spmv_config, segment_fixup_config))) break;

        }
        while (0);

        return error;
    }
};

} // namespace cub
CUB_NS_POSTFIX 
