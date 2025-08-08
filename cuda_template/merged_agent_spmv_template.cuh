/**
 * @file merged_agent_flex_spmv.cuh
 * @brief Extension of CUB's AgentSpmv
 */

#pragma once

#include <iterator>

#include <cub/agent/agent_spmv_orig.cuh>
#include <cub/util_type.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <cub/util_namespace.cuh>

// Add this include to get FlexParams and dimension macros
#include "merged_utils.cuh"
#include "merged_spmv_kernels.cuh"

/// CUB namespace
namespace merged
{
    // Import CUB namespace to avoid having to prefix every CUB function
    using namespace cub;

    // Reduce tensor by key op for tensor type
    template <typename TensorT>
    struct ReduceTensorByKeyOp
    {

        /// Constructor
        __host__ __device__ __forceinline__ ReduceTensorByKeyOp() {}

        /// Scan operator
        __host__ __device__ __forceinline__ TensorT operator()(
            const TensorT &first,  ///< First partial reduction
            const TensorT &second) ///< Second partial reduction
        {
            TensorT retval = second;

            if (first.key == second.key)
            {
#pragma unroll
                for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                {
                    retval.values[i] = first.values[i] + retval.values[i];
                }
            }

            return retval;
        }
    };

    /**
     * @brief AgentFlexSpmv implements SpMV using a matrix A and vector x
     */
    template <
        typename AgentSpmvPolicyT,   ///< Parameterized AgentSpmvPolicy tuning policy type
        typename ValueT,             ///< Matrix and vector value type
        typename OffsetT,            ///< Signed integer type for sequence offsets
        typename TensorT,            ///< Tensor type
        int PTX_ARCH = CUB_PTX_ARCH> ///< PTX compute capability
    struct AgentFlexSpmv
    {
        //---------------------------------------------------------------------
        // Types and constants
        //---------------------------------------------------------------------

        /// Constants
        enum
        {
            BLOCK_THREADS = AgentSpmvPolicyT::BLOCK_THREADS,
            ITEMS_PER_THREAD = AgentSpmvPolicyT::ITEMS_PER_THREAD,
            TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
        };

        /// 2D merge path coordinate type
        typedef typename cub::CubVector<OffsetT, 2>::Type CoordinateT;

        /// Input iterator wrapper types (for applying cache modifiers)
        typedef cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            RowOffsetsSearchIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            RowOffsetsIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            ColumnIndicesIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
            SpmValueIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
            VectorValueIteratorT;

        // Reduce-value-by-segment scan operator
        typedef ReduceTensorByKeyOp<TensorT> ReduceBySegmentOpT;

        // BlockReduce specialization
        typedef BlockReduce<
            ValueT,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
            BlockReduceT;

        // BlockScan specialization
        typedef BlockScan<
            TensorT,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>
            BlockScanT;

        // BlockScan specialization
        typedef BlockScan<
            ValueT,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>
            BlockPrefixSumT;

        // BlockExchange specialization
        typedef BlockExchange<
            ValueT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
            BlockExchangeT;

        /// Merge item type (either a non-zero value or a row-end offset)
        union MergeItem
        {
            OffsetT row_end_offset;
            ValueT nonzero;
            ValueT value_vector_x;
        };

        /// Shared memory type required by this thread block
        struct _TempStorage
        {
            CoordinateT tile_coords[2];

            union Aliasable
            {
                // Smem needed for tile of merge items
                // the merged items include:
                // 1. row_end_offset,
                // 2. nonzero,
                // 3. value_vector_x (result of the dot product will be reused in side nonzeros)
                // Dimension will vary for each of them
                // TILE_ITEMS is the number of item for result y, without considering the dimension of the tensor
                // MergeItem merge_items[(ITEMS_PER_THREAD + TILE_ITEMS) * (DIM_INPUT_VECTOR_X * NUM_INPUT_VECTOR_X + DIM_INPUT_MATRIX_A * NUM_INPUT_MATRIX_A) + 1];
                MergeItem merge_items[TILE_ITEMS];
                // OffsetT row_end_offset[TILE_ITEMS];

                // Smem needed for block exchange
                typename BlockExchangeT::TempStorage exchange;

                // Smem needed for block-wide reduction
                typename BlockReduceT::TempStorage reduce;

                // Smem needed for tile scanning
                typename BlockScanT::TempStorage scan;

                // Smem needed for tile prefix sum
                typename BlockPrefixSumT::TempStorage prefix_sum;
            } aliasable;
        };

        /// Temporary storage type (unionable)
        struct TempStorage : Uninitialized<_TempStorage>
        {
        };

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        _TempStorage &temp_storage; /// Reference to temp_storage

        FlexParams<ValueT, OffsetT> &spmv_params;

        // [code generation]
        RowOffsetsIteratorT wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
        ${input_declarations_code}

        //---------------------------------------------------------------------
        // Constructor
        //---------------------------------------------------------------------

        /**
         * Constructor // [code generation]
         */
        __device__ __forceinline__
        AgentFlexSpmv(
            TempStorage &temp_storage,                ///< Reference to temp_storage
            FlexParams<ValueT, OffsetT> &spmv_params) ///< SpMV input parameter bundle
            : temp_storage(temp_storage.Alias()),
              spmv_params(spmv_params),
              ${input_init_code}
              wd_row_end_offsets(spmv_params.d_row_end_offsets)
        {
        }

        //---------------------------------------------------------------------
        // Tile processing
        //---------------------------------------------------------------------

        __device__ __forceinline__ void reduce(
            ValueT *s_tile_value_nonzeros,      ///< [in, code gen] Shared memory array of non-zero values for the merge tile
            OffsetT *s_tile_row_end_offsets,    ///< [in, code gen] Shared memory array of row end offsets for the merge tile
            CoordinateT tile_start_coord,       ///< [in] Starting coordinate of the merge tile
            CoordinateT tile_end_coord,         ///< [in] Ending coordinate of the merge tile
            int tile_num_rows,                  ///< [in] Number of rows in the merge tile
            int tile_num_nonzeros,               ///< [in] Number of non-zeros in the merge tile
            ValueT *output_vector_y              ///< [out] Output vector y
        )
        {
                        // Search for the thread's starting coordinate within the merge tile
            CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);
            CoordinateT thread_start_coord;

            MergePathSearch(
                OffsetT(threadIdx.x * ITEMS_PER_THREAD), // Diagonal
                s_tile_row_end_offsets,                  // List A
                tile_nonzero_indices,                    // List B
                tile_num_rows,
                tile_num_nonzeros,
                thread_start_coord);

            CTA_SYNC(); // Perf-sync

            // Compute the thread's merge path segment
            CoordinateT thread_current_coord = thread_start_coord;
            TensorT scan_segment[ITEMS_PER_THREAD];
            ValueT running_total[DIM_OUTPUT_VECTOR_Y];
            #pragma unroll
            for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
            {
                running_total[i] = 0.0;
            }
            

            OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
            ValueT *nonzero = s_tile_value_nonzeros + thread_current_coord.y;

// Reduce
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
                {
// Move down (accumulate)
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        scan_segment[ITEM].values[i] = nonzero[i * TILE_ITEMS];
                        running_total[i] += nonzero[i * TILE_ITEMS];
                    }
                    ++thread_current_coord.y;
                    nonzero = s_tile_value_nonzeros + thread_current_coord.y;
                }
                else
                {
// Move right (reset)
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        scan_segment[ITEM].values[i] = 0.0;
                        running_total[i] = 0.0;
                    }
                    ++thread_current_coord.x;
                    row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
                }

                scan_segment[ITEM].key = thread_current_coord.x;
            }

            CTA_SYNC();

            // Block-wide reduce-value-by-segment
            TensorT tile_carry;
            ReduceBySegmentOpT scan_op;
            TensorT scan_item;

#pragma unroll
            for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
            {
                scan_item.values[i] = running_total[i];
            }
            scan_item.key = thread_current_coord.x;

            BlockScanT(temp_storage.aliasable.scan).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

            if (threadIdx.x == 0)
            {
                scan_item.key = thread_start_coord.x;
#pragma unroll
                for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                {
                    scan_item.values[i] = 0.0;
                }
            }

            if (tile_num_rows > 0)
            {

                CTA_SYNC();
                // Scan downsweep and scatter
                // memory reuse for the partial results 
                // TILE_ITEMS is used to avoid bank conflict
                // ValueT *s_partials = &temp_storage.aliasable.merge_items[0].nonzero;
                ValueT *s_partials = s_tile_value_nonzeros;

                if (scan_item.key != scan_segment[0].key)
                {
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; ++i)
                    {
                        s_partials[scan_item.key + i * TILE_ITEMS] = scan_item.values[i];
                    }
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; ++i)
                    {
                        scan_segment[0].values[i] += scan_item.values[i];
                    }
                }

#pragma unroll
                for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    if (scan_segment[ITEM - 1].key != scan_segment[ITEM].key)
                    {
#pragma unroll
                        for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; ++i)
                        {
                            s_partials[scan_segment[ITEM - 1].key + i * TILE_ITEMS] = scan_segment[ITEM - 1].values[i];
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; ++i)
                        {
                            scan_segment[ITEM].values[i] += scan_segment[ITEM - 1].values[i];
                        }
                    }
                }

                CTA_SYNC();

// memory coalescing for writing the output vector y
#pragma unroll 1
                for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
                {
                    #pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        atomicAdd(
                            &output_vector_y[tile_start_coord.x * DIM_OUTPUT_VECTOR_Y + item * DIM_OUTPUT_VECTOR_Y + i],
                            s_partials[item + i * TILE_ITEMS]
                        );
                        // output_vector_y[tile_start_coord.x * DIM_OUTPUT_VECTOR_Y + item * DIM_OUTPUT_VECTOR_Y + i] = s_partials[item + i * TILE_ITEMS];
                    }
                }
            }

            CTA_SYNC();

            // atomic add the residual sum, the tile's carry-out, to the Global memory
            if (threadIdx.x == 0)
            {
                tile_carry.key += tile_start_coord.x;
                if (tile_carry.key < spmv_params.num_rows)
                {
                    #pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        atomicAdd(
                            &output_vector_y[tile_carry.key * DIM_OUTPUT_VECTOR_Y + i],
                            tile_carry.values[i]);
                    }
                };
            }
        }

        /**
         * Consume a merge tile, specialized for direct load of nonzeros
         */
        __device__ __forceinline__ void ConsumeTile(
            int tile_idx,
            CoordinateT tile_start_coord,
            CoordinateT tile_end_coord,
            Int2Type<true> is_direct_load) ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
        {
            int tile_num_rows = tile_end_coord.x - tile_start_coord.x;
            int tile_num_nonzeros = tile_end_coord.y - tile_start_coord.y;

            //shared memory reused is disabled 
            // OffsetT *s_tile_row_end_offsets = &temp_storage.aliasable.merge_items[0].row_end_offset;
            __shared__ OffsetT s_tile_row_end_offsets[TILE_ITEMS];

// Gather the row end-offsets for the merge tile into shared memory
#pragma unroll 1
            for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
            {
                const OffsetT offset =
                    (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item),
                               static_cast<OffsetT>(spmv_params.num_rows - 1));
                s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
            }

            CTA_SYNC();

            // [code generation]
            ${reducer_smem_definitions}

// Select
// Gather the nonzeros for the merge tile into shared memory
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (nonzero_idx < tile_num_nonzeros)
                {
                    // [code generation]
                    ${selector_code}
                    ${map_code}
                }
            }

            CTA_SYNC();
            // reduce the intermeidate computations
            // [code generation]
            ${reducer_code}
        }



        /**
         * Process a merge tile
         */
        __device__ __forceinline__ void ConsumeTile(
            CoordinateT *d_tile_coordinates, ///< [in] Pointer to the temporary array of tile starting coordinates
            // TensorT *d_tile_carry_pairs,     ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
            int num_merge_tiles             ///< [in] Total number of merge tiles
        )
        {
            int tile_idx = (blockIdx.y * gridDim.x) + blockIdx.x;

            if (tile_idx >= num_merge_tiles)
                return;

            // Read our starting coordinates
            if (threadIdx.x < 2)
            {
                if (d_tile_coordinates == NULL)
                {
                    // Search our starting coordinates
                    OffsetT diagonal = (tile_idx + threadIdx.x) * TILE_ITEMS;
                    CoordinateT tile_coord;
                    CountingInputIterator<OffsetT> nonzero_indices(0);

                    // Search the merge path
                    MergePathSearch(
                        diagonal,
                        RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets),
                        nonzero_indices,
                        spmv_params.num_rows,
                        spmv_params.num_nonzeros,
                        tile_coord);

                    temp_storage.tile_coords[threadIdx.x] = tile_coord;
                }
                else
                {
                    temp_storage.tile_coords[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x];
                }
            }

            CTA_SYNC();

            CoordinateT tile_start_coord = temp_storage.tile_coords[0];
            CoordinateT tile_end_coord = temp_storage.tile_coords[1];

            ConsumeTile(
                tile_idx,
                tile_start_coord,
                tile_end_coord,
                Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>()); // PTX >=520 use the indirect load of nonzeros
        }
    };

} // namespace merged