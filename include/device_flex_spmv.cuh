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
#include <limits>

#include <cub/device/dispatch/dispatch_spmv_orig.cuh>
#include <cub/device/device_spmv.cuh>
#include <cub/util_namespace.cuh>
#include <cub/config.cuh>

// Include our custom dispatch implementation
#include "dispatch_flex_spmv.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {
    
/**
 * @brief DeviceFlexSpmv provides device-wide parallel operations for SpMV with a dense matrix and column indices
 * 
 * This implementation extends DeviceSpmv to support computing y = A[colIdxA[k]]*x[colIdxX[k]] where:
 * - A is a dense matrix
 * - colIdxA are column indices for matrix A
 * - colIdxX are column indices for vector x
 * - k is indexed from the row offset array
 */
struct DeviceFlexSpmv
{
    /**
     * @brief This function performs the matrix-vector operation y = A[colIdxA[k]]*x[colIdxX[k]]
     * Most of the implementation is the same as in DeviceSpmv, except for the sparse matrix parameters
     */
    template <
        typename ValueT,
        typename OffsetT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t CsrMV(
        void*               d_temp_storage,                     ///< [in] Device-accessible allocation of temporary storage
        size_t&             temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of d_temp_storage allocation
        ValueT*             d_dense_matrix,                     ///< [in] Pointer to the dense matrix A
        OffsetT*            d_column_indices_A,                 ///< [in] Pointer to the column indices for matrix A
        int                 dense_matrix_width,                 ///< [in] Width of the dense matrix
        OffsetT*            d_row_offsets,                      ///< [in] Pointer to the array of m + 1 row offsets
        OffsetT*            d_column_indices,                   ///< [in] Pointer to the array of column indices
        ValueT*             d_vector_x,                         ///< [in] Pointer to the dense input vector x
        ValueT*             d_vector_y,                         ///< [out] Pointer to the dense output vector y
        int                 num_rows,                           ///< [in] Number of rows of matrix A
        int                 num_cols,                           ///< [in] Number of columns of matrix A
        int                 num_nonzeros,                       ///< [in] Number of nonzero elements
        ValueT              alpha              = 1.0,           ///< [in] Alpha multiplicand
        ValueT              beta               = 0.0,           ///< [in] Beta addend-multiplicand
        cudaStream_t        stream             = 0,             ///< [in] CUDA stream to launch kernels within
        bool                debug_synchronous  = false)         ///< [in] Whether to synchronize after each kernel launch
    {
        // Set up the FlexSpmvParams
        FlexSpmvParams<ValueT, OffsetT> spmv_params;
        spmv_params.d_values             = nullptr;  // Not used in our implementation
        spmv_params.d_row_end_offsets    = d_row_offsets + 1;
        spmv_params.d_column_indices     = d_column_indices;
        spmv_params.d_vector_x           = d_vector_x;
        spmv_params.d_vector_y           = d_vector_y;
        spmv_params.num_rows             = num_rows;
        spmv_params.num_cols             = num_cols;
        spmv_params.num_nonzeros         = num_nonzeros;
        spmv_params.alpha                = alpha;
        spmv_params.beta                 = beta;
        
        // Additional parameters for the dense matrix
        spmv_params.d_dense_matrix       = d_dense_matrix;
        spmv_params.d_column_indices_A   = d_column_indices_A;
        spmv_params.dense_matrix_width   = dense_matrix_width;

        // Dispatch to our custom implementation
        return DispatchFlexSpmv<ValueT, OffsetT>::FlexDispatch(
            d_temp_storage,
            temp_storage_bytes,
            spmv_params, // spmv_params is a FlexSpmvParams struct
            stream,
            debug_synchronous);
    }
};

} // namespace cub
CUB_NS_POSTFIX 