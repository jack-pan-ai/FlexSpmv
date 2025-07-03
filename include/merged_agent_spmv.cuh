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
        bool HAS_ALPHA,              ///< Whether the input parameter \p alpha is 1
        bool HAS_BETA,               ///< Whether the input parameter \p beta is 0
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
        typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

        /// Input iterator wrapper types (for applying cache modifiers)
        typedef CacheModifiedInputIterator<
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
                MergeItem merge_items[(ITEMS_PER_THREAD + TILE_ITEMS) * (DIM_INPUT_VECTOR_X + DIM_INPUT_MATRIX_A) + 1];

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

        RowOffsetsIteratorT wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
        ColumnIndicesIteratorT wd_column_indices; ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
        VectorValueIteratorT wd_vector_x;         ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
        VectorValueIteratorT wd_vector_y;         ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>

        // <Wrapped pointer to the array of indexing A>
        ColumnIndicesIteratorT wd_column_indices_A; ///< Wrapped pointer to column indices for matrix A
        SpmValueIteratorT wd_spm_nnz;       ///< Wrapped pointer to sparse matrix A

        //---------------------------------------------------------------------
        // Constructor
        //---------------------------------------------------------------------

        /**
         * Constructor
         */
        __device__ __forceinline__
        AgentFlexSpmv(
            TempStorage &temp_storage,                ///< Reference to temp_storage
            FlexParams<ValueT, OffsetT> &spmv_params) ///< SpMV input parameter bundle
            : temp_storage(temp_storage.Alias()),
              spmv_params(spmv_params),
              wd_row_end_offsets(spmv_params.d_row_end_offsets),
              wd_column_indices(spmv_params.d_column_indices),
              wd_column_indices_A(spmv_params.d_column_indices_A),
              wd_spm_nnz(spmv_params.d_spm_nnz),
              wd_vector_x(spmv_params.d_vector_x),
              wd_vector_y(spmv_params.d_vector_y)
        {
        }

        //---------------------------------------------------------------------
        // Tile processing
        //---------------------------------------------------------------------
        /**
         * Consume a merge tile, specialized for indirect load of nonzeros
         */
        __device__ __forceinline__ TensorT ConsumeTile(
            int tile_idx,
            CoordinateT tile_start_coord,
            CoordinateT tile_end_coord,
            Int2Type<false> is_direct_load) ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
        {
            int tile_num_rows = tile_end_coord.x - tile_start_coord.x;
            int tile_num_nonzeros = tile_end_coord.y - tile_start_coord.y;

            // [code generation] here we need treat it carefully, because the dimension of the tensor is variable
            OffsetT *s_tile_row_end_offsets = &temp_storage.aliasable.merge_items[0].row_end_offset;
            ValueT *s_tile_nonzeros = &temp_storage.aliasable.merge_items[tile_num_rows + ITEMS_PER_THREAD].nonzero;
            ValueT *s_tile_value_vector_x = &temp_storage.aliasable.merge_items[tile_num_rows + ITEMS_PER_THREAD + TILE_ITEMS * DIM_INPUT_MATRIX_A].value_vector_x;

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

// Gather the nonzeros for the merge tile into shared memory
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD * DIM_INPUT_MATRIX_A; ++ITEM)
            {
                int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (nonzero_idx < tile_num_nonzeros * DIM_INPUT_MATRIX_A)
                {

                    ColumnIndicesIteratorT ci_A = wd_column_indices_A + tile_start_coord.y + nonzero_idx;
                    ValueT *s = s_tile_nonzeros + nonzero_idx;

                    // load the nonzeros from the sparse matrix A into the shared memory
                    *s = wd_spm_nnz[*ci_A];
                }
            }

            CTA_SYNC();

// Gather the vector x for the merge tile into shared memory
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD * DIM_INPUT_VECTOR_X; ++ITEM)
            {
                int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (nonzero_idx < tile_num_nonzeros * DIM_INPUT_VECTOR_X)
                {

                    ColumnIndicesIteratorT ci_x = wd_column_indices + tile_start_coord.y + nonzero_idx;
                    ValueT *s_x = s_tile_value_vector_x + nonzero_idx;

                    // load the nonzeros from the sparse matrix A into the shared memory
                    *s_x = wd_vector_x[*ci_x];
                }
            }

            CTA_SYNC();

// compute the dot product of the nonzeros from the sparse matrix A and the vector x and store the result in the shared memory
// this case is very special case where DIM_OUTPUT_VECTOR_Y=DIM_INPUT_MATRIX_A=DIM_INPUT_VECTOR_X
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD * DIM_OUTPUT_VECTOR_Y; ++ITEM)
            {
                int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (nonzero_idx < tile_num_nonzeros * DIM_INPUT_MATRIX_A)
                {
                    ValueT *s_A_value = s_tile_nonzeros + nonzero_idx;
                    ValueT *s_x_value = s_tile_value_vector_x + nonzero_idx;
                    ValueT result = (*s_A_value) * (*s_x_value);
                    // load the nonzeros from the sparse matrix A into the shared memory
                    *s_A_value = result; // the result is the dot product of the nonzero and the vector x
                }
            }

            CTA_SYNC();

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
            for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
            {
                running_total[i] = 0.0;
            }
            

            OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
            ValueT *nonzero = s_tile_nonzeros + thread_current_coord.y * DIM_OUTPUT_VECTOR_Y;

#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
                {
// Move down (accumulate)
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        scan_segment[ITEM].values[i] = nonzero[i];
                        running_total[i] += nonzero[i];
                    }
                    ++thread_current_coord.y;
                    nonzero = s_tile_nonzeros + thread_current_coord.y * DIM_OUTPUT_VECTOR_Y;
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
                // random access for the partial results (bank conflict may happen)
                ValueT *s_partials = &temp_storage.aliasable.merge_items[0].nonzero;

                if (scan_item.key != scan_segment[0].key)
                {
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; ++i)
                    {
                        s_partials[scan_item.key * DIM_OUTPUT_VECTOR_Y + i] = scan_item.values[i];
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
                            s_partials[scan_segment[ITEM - 1].key * DIM_OUTPUT_VECTOR_Y + i] = scan_segment[ITEM - 1].values[i];
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
                for (int item = threadIdx.x; item < tile_num_rows * DIM_OUTPUT_VECTOR_Y; item += BLOCK_THREADS)
                {
                    spmv_params.d_vector_y[tile_start_coord.x * DIM_OUTPUT_VECTOR_Y + item] = s_partials[item];
                }
            }

            // Return the tile's running carry-out
            return tile_carry;
        }

        /**
         * Process a merge tile
         */
        __device__ __forceinline__ void ConsumeTile(
            CoordinateT *d_tile_coordinates, ///< [in] Pointer to the temporary array of tile starting coordinates
            TensorT *d_tile_carry_pairs,     ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
            int num_merge_tiles)             ///< [in] Total number of merge tiles
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

            TensorT tile_carry = ConsumeTile(
                tile_idx,
                tile_start_coord,
                tile_end_coord,
                Int2Type<false>()); // PTX >=520 use the indirect load of nonzeros

            // Output the tile's carry-out
            if (threadIdx.x == 0)
            {
                if (HAS_ALPHA)
                {
                    // overload the operator * for tensor type
                    tile_carry = tile_carry * spmv_params.alpha;
                }

                tile_carry.key += tile_start_coord.x;
                if (tile_carry.key >= spmv_params.num_rows)
                {
                    // FIXME: This works around an invalid memory access in the
                    // fixup kernel. The underlying issue needs to be debugged and
                    // properly fixed, but this hack prevents writes to
                    // out-of-bounds addresses. It doesn't appear to have an effect
                    // on the validity of the results, since this only affects the
                    // carry-over from last tile in the input.
                    tile_carry.key = spmv_params.num_rows - 1;
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        tile_carry.values[i] = ValueT{};
                    }
                };

                d_tile_carry_pairs[tile_idx] = tile_carry;
            }
        }
    };

    /******************************************************************************
     * Thread block abstractions
     ******************************************************************************/

    /**
     * \brief AgentSegmentFixup implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key
     */
    template <
        typename AgentSegmentFixupPolicyT,  ///< Parameterized AgentSegmentFixupPolicy tuning policy type
        typename PairsInputIteratorT,       ///< Random-access input iterator type for keys
        typename AggregatesOutputIteratorT, ///< Random-access output iterator type for values
        typename OffsetT>                   ///< Signed integer type for global offsets
    struct AgentSegmentFixup
    {
        //---------------------------------------------------------------------
        // Types and constants
        //---------------------------------------------------------------------

        // Data type of key-value input iterator
        typedef typename std::iterator_traits<PairsInputIteratorT>::value_type TensorKeyPairT;

        // Value type
        typedef typename TensorKeyPairT::Value ValueT;

        // Constants
        enum
        {
            BLOCK_THREADS = AgentSegmentFixupPolicyT::BLOCK_THREADS,
            ITEMS_PER_THREAD = AgentSegmentFixupPolicyT::ITEMS_PER_THREAD,
            TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
            // Whether or not do fixup using RLE + global atomics
            USE_ATOMIC_FIXUP = true,
        };

        // Cache-modified Input iterator wrapper type (for applying cache modifier) for keys
        typedef typename If<IsPointer<PairsInputIteratorT>::VALUE,
                            CacheModifiedInputIterator<AgentSegmentFixupPolicyT::LOAD_MODIFIER, TensorKeyPairT, OffsetT>, // Wrap the native input pointer with CacheModifiedValuesInputIterator
                            PairsInputIteratorT>::Type                                                                    // Directly use the supplied input iterator type
            WrappedPairsInputIteratorT;

        // Parameterized BlockLoad type for pairs
        typedef BlockLoad<
            TensorKeyPairT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentSegmentFixupPolicyT::LOAD_ALGORITHM>
            BlockLoadPairs;

        // Shared memory type for this thread block
        union _TempStorage
        {
            // Smem needed for loading keys
            typename BlockLoadPairs::TempStorage load_pairs;
        };

        // Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage>
        {
        };

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        _TempStorage &temp_storage;                 ///< Reference to temp_storage
        WrappedPairsInputIteratorT d_pairs_in;      ///< Input keys
        AggregatesOutputIteratorT d_aggregates_out; ///< Output value aggregates

        //---------------------------------------------------------------------
        // Constructor
        //---------------------------------------------------------------------

        // Constructor
        __device__ __forceinline__
        AgentSegmentFixup(
            TempStorage &temp_storage,                  ///< Reference to temp_storage
            PairsInputIteratorT d_pairs_in,             ///< Input keys
            AggregatesOutputIteratorT d_aggregates_out) ///< Output value aggregates
            : temp_storage(temp_storage.Alias()),
              d_pairs_in(d_pairs_in),
              d_aggregates_out(d_aggregates_out)
        {
        }

        //---------------------------------------------------------------------
        // Cooperatively scan a device-wide sequence of tiles with other CTAs
        //---------------------------------------------------------------------

        /**
         * Process input tile.  Specialized for atomic-fixup
         */
        template <bool IS_LAST_TILE>
        __device__ __forceinline__ void ConsumeTile(
            OffsetT num_remaining,           ///< Number of global input items remaining (including this tile)
            int tile_idx,                    ///< Tile index
            OffsetT tile_offset,             ///< Tile offset
            Int2Type<true> use_atomic_fixup) ///< Marker whether to use atomicAdd (instead of reduce-by-key)
        {
            TensorKeyPairT pairs[ITEMS_PER_THREAD];

            // Load pairs
            TensorKeyPairT oob_pair;
            oob_pair.key = -1;

            if (IS_LAST_TILE)
                BlockLoadPairs(temp_storage.load_pairs).Load(d_pairs_in + tile_offset, pairs, num_remaining, oob_pair);
            else
                BlockLoadPairs(temp_storage.load_pairs).Load(d_pairs_in + tile_offset, pairs);

// RLE
#pragma unroll
            for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                ValueT *d_scatter = d_aggregates_out + pairs[ITEM - 1].key * DIM_OUTPUT_VECTOR_Y;
                if (pairs[ITEM].key != pairs[ITEM - 1].key)
                {
#pragma unroll
                    for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                    {
                        atomicAdd(d_scatter + i, pairs[ITEM - 1].values[i]);
                    }
                }
                else
                    pairs[ITEM] = pairs[ITEM - 1] + pairs[ITEM];
            }

            // Flush last item if valid
            ValueT *d_scatter = d_aggregates_out + pairs[ITEMS_PER_THREAD - 1].key * DIM_OUTPUT_VECTOR_Y;
            if ((!IS_LAST_TILE) || (pairs[ITEMS_PER_THREAD - 1].key >= 0))
            {
#pragma unroll
                for (int i = 0; i < DIM_OUTPUT_VECTOR_Y; i++)
                {
                    atomicAdd(d_scatter + i, pairs[ITEMS_PER_THREAD - 1].values[i]);
                }
            }
        }

        /**
         * Scan tiles of items as part of a dynamic chained scan
         */
        __device__ __forceinline__ void ConsumeRange(
            OffsetT num_items, ///< Total number of input items
            int num_tiles)     ///< Total number of input tiles
        {
            // Blocks are launched in increasing order, so just assign one tile per block
            int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y; // Current tile index
            OffsetT tile_offset = tile_idx * TILE_ITEMS;          // Global offset for the current tile
            OffsetT num_remaining = num_items - tile_offset;      // Remaining items (including this tile)

            if (num_remaining > TILE_ITEMS)
            {
                // Not the last tile (full)
                ConsumeTile<false>(num_remaining, tile_idx, tile_offset, Int2Type<USE_ATOMIC_FIXUP>());
            }
            else if (num_remaining > 0)
            {
                // The last tile (possibly partially-full)
                ConsumeTile<true>(num_remaining, tile_idx, tile_offset, Int2Type<USE_ATOMIC_FIXUP>());
            }
        }
    };

} // namespace merged