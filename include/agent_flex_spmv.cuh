/**
 * @file agent_flex_spmv.cuh
 * @brief Extension of CUB's AgentSpmv to use flexible sparse matrix with column indices
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

/// Optional outer namespace(s)
CUB_NS_PREFIX

#define DIMENSION 2

/// CUB namespace
namespace cub {

/**
 * @brief FlexSpmvParams extends SpmvParams to include a flexible matrix and indices inputs
 */
template <
    typename ValueT,              ///< Matrix and vector value type
    typename OffsetT>             ///< Signed integer type for sequence offsets
struct FlexSpmvParams : public SpmvParams<ValueT, OffsetT>
{
    // flexible matrix and indices inputs (code gen) removed if using spmv from scratchss
    ValueT*     d_spm_A;           ///< Pointer to the sparse matrix A
    ValueT*     d_spm_B;           ///< Pointer to the sparse matrix B
    OffsetT*    d_column_indices_A;       ///< Pointer to the column indices for matrix A
    OffsetT*    d_column_indices_1;       ///< Pointer to the column indices for matrix A
    OffsetT*    d_column_indices_2;       ///< Pointer to the column indices for matrix A
    
    // Base params remain the same:
    // ValueT*   d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*  d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    // OffsetT*  d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*   d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;            ///< Number of rows of matrix <b>A</b>.
    int             num_cols;            ///< Number of columns of matrix <b>A</b>.
    int             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          alpha;               ///< Alpha multiplicand
    ValueT          beta;                ///< Beta addend-multiplicand
};


/**
 * @brief AgentFlexSpmv implements SpMV using a flexible matrix and indices inputs
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    bool        HAS_ALPHA,                  ///< Whether the input parameter \p alpha is 1
    bool        HAS_BETA,                   ///< Whether the input parameter \p beta is 0
    int         PTX_ARCH = CUB_PTX_ARCH>    ///< PTX compute capability
struct AgentFlexSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
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
        ValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        SpmIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;

    // Reduce-value-by-segment scan operator
    typedef ReduceByKeyOp<cub::Sum> ReduceBySegmentOpT;

    // BlockReduce specialization
    typedef BlockReduce<
            ValueT,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    // BlockScan specialization
    typedef BlockScan<
            KeyValuePairT,
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
        // Value type to pair with index type OffsetT (NullType if loading values directly during merge)
        typedef typename If<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS, NullType, ValueT>::Type MergeValueT;

        OffsetT     row_end_offset;
        MergeValueT nonzero;
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        CoordinateT tile_coords[2];

        union Aliasable
        {
            // Smem needed for tile of merge items
            MergeItem merge_items[(ITEMS_PER_THREAD + TILE_ITEMS + 1) * DIMENSION];

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
    struct TempStorage : Uninitialized<_TempStorage> {};

     //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------


    _TempStorage&                   temp_storage;         /// Reference to temp_storage

    FlexSpmvParams<ValueT, OffsetT>&    spmv_params;

    // ValueIteratorT                  wd_values;            ///< Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT             wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    // ColumnIndicesIteratorT          wd_column_indices;    ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorT            wd_vector_x;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    VectorValueIteratorT            wd_vector_y;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    
    // <Wrapped pointer to the array of indexing A> (code gen)
    ColumnIndicesIteratorT          wd_column_indices_A;  ///< Wrapped pointer to column indices for flexible matrix
    SpmIteratorT                    wd_spm_A;      ///< Wrapped pointer to flexible matrix array
    SpmIteratorT                    wd_spm_B;      ///< Wrapped pointer to flexible matrix array
    ColumnIndicesIteratorT          wd_column_indices_1;  ///< Wrapped pointer to column indices for flexible matrix
    ColumnIndicesIteratorT          wd_column_indices_2;  ///< Wrapped pointer to column indices for flexible matrix

    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    AgentFlexSpmv(
        TempStorage&                    temp_storage,      ///< Reference to temp_storage
        FlexSpmvParams<ValueT, OffsetT>&  spmv_params)       ///< SpMV input parameter bundle
    :
        temp_storage(temp_storage.Alias()),
        spmv_params(spmv_params),
        wd_row_end_offsets(spmv_params.d_row_end_offsets),
        // wd_column_indices(spmv_params.d_column_indices),
        wd_vector_x(spmv_params.d_vector_x),
        wd_vector_y(spmv_params.d_vector_y),

        // flexible matrix and indices inputs (code gen) removed if using spmv from scratch
        wd_column_indices_A(spmv_params.d_column_indices_A),
        wd_spm_A(spmv_params.d_spm_A),
        wd_spm_B(spmv_params.d_spm_B),
        wd_column_indices_1(spmv_params.d_column_indices_1),
        wd_column_indices_2(spmv_params.d_column_indices_2)
    {}

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------

    // /**
    //  * Consume a merge tile, specialized for direct-load of nonzeros (to be modified later)
    //  */
    // __device__ __forceinline__ KeyValuePairT ConsumeTile(
    //     int             tile_idx,
    //     CoordinateT     tile_start_coord,
    //     CoordinateT     tile_end_coord,
    //     Int2Type<true>  is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    // {

    //     int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
    //     int         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;
    //     OffsetT*    s_tile_row_end_offsets  = &temp_storage.aliasable.merge_items[0].row_end_offset;

    //     // Gather the row end-offsets for the merge tile into shared memory
    //     for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
    //     {
    //         const OffsetT offset =
    //           (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item),
    //                      static_cast<OffsetT>(spmv_params.num_rows - 1));
    //         s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
    //     }

    //     __syncthreads();

    //     // Search for the thread's starting coordinate within the merge tile
    //     CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start_coord.y);
    //     CoordinateT                     thread_start_coord;

    //     MergePathSearch(
    //         OffsetT(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
    //         s_tile_row_end_offsets,                     // List A
    //         tile_nonzero_indices,                       // List B
    //         tile_num_rows,
    //         tile_num_nonzeros,
    //         thread_start_coord);

    //     __syncthreads();            // Perf-sync

    //     // Compute the thread's merge path segment
    //     CoordinateT     thread_current_coord = thread_start_coord;
    //     KeyValuePairT   scan_segment[ITEMS_PER_THREAD];

    //     ValueT          running_total = 0.0;

    //     #pragma unroll
    //     for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    //     {
    //         OffsetT nonzero_idx         = CUB_MIN(tile_nonzero_indices[thread_current_coord.y], spmv_params.num_nonzeros - 1);
    //         OffsetT column_idx          = wd_column_indices[nonzero_idx];
    //         OffsetT column_idx_A        = wd_column_indices_A[nonzero_idx];
            
    //         ValueT  value               = wd_spm_A[column_idx_A]; // one dimension for this spm vector
    //         ValueT  vector_value        = wd_vector_x[column_idx];
    //         ValueT  nonzero             = value * vector_value;

    //         OffsetT row_end_offset      = s_tile_row_end_offsets[thread_current_coord.x];

    //         if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
    //         {
    //             // Move down (accumulate)
    //             running_total += nonzero;
    //             scan_segment[ITEM].value    = running_total;
    //             scan_segment[ITEM].key      = tile_num_rows;
    //             ++thread_current_coord.y;
    //         }
    //         else
    //         {
    //             // Move right (reset)
    //             scan_segment[ITEM].value    = running_total;
    //             scan_segment[ITEM].key      = thread_current_coord.x;
    //             running_total               = 0.0;
    //             ++thread_current_coord.x;
    //         }
    //     }

    //     __syncthreads();

    //     // Block-wide reduce-value-by-segment
    //     KeyValuePairT       tile_carry;
    //     ReduceBySegmentOpT  scan_op;
    //     KeyValuePairT       scan_item;

    //     scan_item.value = running_total;
    //     scan_item.key   = thread_current_coord.x;

    //     BlockScanT(temp_storage.aliasable.scan).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

    //     if (tile_num_rows > 0)
    //     {
    //         if (threadIdx.x == 0)
    //             scan_item.key = -1;

    //         // Direct scatter
    //         #pragma unroll
    //         for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    //         {
    //             if (scan_segment[ITEM].key < tile_num_rows)
    //             {
    //                 if (scan_item.key == scan_segment[ITEM].key)
    //                     scan_segment[ITEM].value = scan_item.value + scan_segment[ITEM].value;

    //                 if (HAS_ALPHA)
    //                 {
    //                     scan_segment[ITEM].value *= spmv_params.alpha;
    //                 }

    //                 if (HAS_BETA)
    //                 {
    //                     // Update the output vector element
    //                     ValueT addend = spmv_params.beta * wd_vector_y[tile_start_coord.x + scan_segment[ITEM].key];
    //                     scan_segment[ITEM].value += addend;
    //                 }

    //                 // Set the output vector element
    //                 spmv_params.d_vector_y[tile_start_coord.x + scan_segment[ITEM].key] = scan_segment[ITEM].value;
    //             }
    //         }
    //     }

    //     // Return the tile's running carry-out
    //     return tile_carry;
    // }

        /**
     * Consume a merge tile, specialized for indirect load of nonzeros
     */
    __device__ __forceinline__ void ConsumeTile(
        int             tile_idx,
        CoordinateT     tile_start_coord,
        CoordinateT     tile_end_coord,
        KeyValuePairT   tile_carry[DIMENSION],        // <out> tile carry-out
        Int2Type<false> is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    {
        int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;

        OffsetT*    s_tile_row_end_offsets  = &temp_storage.aliasable.merge_items[0].row_end_offset;
        ValueT*     s_tile_nonzeros[DIMENSION];
        for (int k = 0; k < DIMENSION; ++k) {
            s_tile_nonzeros[k] = &temp_storage.aliasable.merge_items[tile_num_rows + ITEMS_PER_THREAD + k * tile_num_nonzeros].nonzero;
        }

        // put all intermediate results in register, considering the edge-wise operation is private to each thread (code gen), the r_ij can be reused 
        ValueT r_i[DIMENSION];
        ValueT r_j[DIMENSION];
        ValueT r_ij[DIMENSION];
        ValueT e_ij[DIMENSION];
        ValueT f_ij[DIMENSION];
        ValueT r_ij_norm=0.0;

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

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                     // List A
            tile_nonzero_indices,                       // List B
            tile_num_rows,
            tile_num_nonzeros,
            thread_start_coord);

        CTA_SYNC();            // Perf-sync

#if (CUB_PTX_ARCH >= 520)
        
        // Gather the nonzeros for the merge tile into shared memory
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            int nonzero_idx                 = threadIdx.x + (ITEM * BLOCK_THREADS);

            // (code gen) for CUB_PTX_ARCH >= 520
            // ValueIteratorT a                = wd_values + tile_start_coord.y + nonzero_idx;
            // ColumnIndicesIteratorT ci       = wd_column_indices + tile_start_coord.y + nonzero_idx;
            ColumnIndicesIteratorT ci_A     = wd_column_indices_A + tile_start_coord.y + nonzero_idx;
            ColumnIndicesIteratorT ci_1     = wd_column_indices_1 + tile_start_coord.y + nonzero_idx;
            ColumnIndicesIteratorT ci_2     = wd_column_indices_2 + tile_start_coord.y + nonzero_idx;

            ValueT* s[DIMENSION];
            for (int k = 0; k < DIMENSION; ++k) {
                s[k] = s_tile_nonzeros[k] + nonzero_idx;
            }

            if (nonzero_idx < tile_num_nonzeros)
            {
                // index for A and X, I_A[i] and I_X[i]
                // OffsetT column_idx              = *ci;
                OffsetT column_idx_A            = *ci_A;
                OffsetT column_idx_1            = *ci_1;
                OffsetT column_idx_2            = *ci_2;

                // select ri and rj from vector x
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    r_i[k] = wd_vector_x[column_idx_1 * DIMENSION + k];
                    r_j[k] = wd_vector_x[column_idx_2 * DIMENSION + k];
                }

                // compute r_ij
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    r_ij[k] = r_i[k] - r_j[k];
                }

                // compute r_ij_norm
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    r_ij_norm += r_ij[k] * r_ij[k];
                }
                r_ij_norm = std::sqrt(r_ij_norm);

                // compute e_ij
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    e_ij[k] = r_ij[k] / r_ij_norm;
                }

                // compute f_ij
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    f_ij[k] = - wd_spm_B[column_idx_A] * (r_ij_norm - wd_spm_A[column_idx_A]) * e_ij[k];
                }

                // assign f_ij to s, arrangement of f_ij is [f_00[0], f_01[0], f_02[0], f_03[0], ...]
                // this is to avoid the bank conflict
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    *s[k] = f_ij[k];
                }

                // // value for X and A, X[I_X[i]] and A[I_A[i]]
                // ValueT vector_value                    = wd_vector_x[column_idx];
                // ValueT vector_value_A                  = wd_spm_A[column_idx_A]; // row-major store

                // ValueT nonzero                  = vector_value * vector_value_A;

                // *s    = nonzero;
            }
        }
#else

        // // Gather the nonzeros for the merge tile into shared memory
        // if (tile_num_nonzeros > 0)
        // {
        //     #pragma unroll
        //     for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        //     {
        //         int     nonzero_idx             = threadIdx.x + (ITEM * BLOCK_THREADS);
        //         nonzero_idx                     = CUB_MIN(nonzero_idx, tile_num_nonzeros - 1);

        //         OffsetT column_idx              = wd_column_indices[tile_start_coord.y + nonzero_idx];
        //         OffsetT column_idx_A            = wd_column_indices_A[tile_start_coord.y + nonzero_idx];

        //         ValueT  vector_value            = wd_vector_x[column_idx];
        //         ValueT  vector_value_A          = wd_spm_A[column_idx_A]; // same position to vector X
        //         ValueT  nonzero                 = vector_value * vector_value_A;

        //         s_tile_nonzeros[nonzero_idx]    = nonzero;
        //     }
        // }

#endif

        CTA_SYNC();

        // Compute the thread's merge path segment
        CoordinateT     thread_current_coord = thread_start_coord;
        KeyValuePairT   scan_segment[DIMENSION][ITEMS_PER_THREAD];
        ValueT          running_total[DIMENSION];
        #pragma unroll
        for (int k = 0; k < DIMENSION; ++k) {
            running_total[k] = 0.0;
        }

        OffsetT row_end_offset  = s_tile_row_end_offsets[thread_current_coord.x];
        ValueT  nonzero[DIMENSION];
        #pragma unroll
        for (int k = 0; k < DIMENSION; ++k) {
            nonzero[k] = s_tile_nonzeros[k][thread_current_coord.y];
        }

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
            {
                // Move down (accumulate)
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    scan_segment[k][ITEM].value    = nonzero[k];
                    running_total[k]               += nonzero[k];
                }
                ++thread_current_coord.y;
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    nonzero[k] = s_tile_nonzeros[k][thread_current_coord.y];
                }
            }
            else
            {
                // Move right (reset)
                #pragma unroll
                for (int k = 0; k < DIMENSION; ++k) {
                    scan_segment[k][ITEM].value    = 0.0;
                    running_total[k]               = 0.0;
                }
                ++thread_current_coord.x;
                row_end_offset              = s_tile_row_end_offsets[thread_current_coord.x];
            }

            #pragma unroll
            for (int k = 0; k < DIMENSION; ++k) {
                scan_segment[k][ITEM].key = thread_current_coord.x;
            }
        }

        CTA_SYNC();

        // Block-wide reduce-value-by-segment
        ReduceBySegmentOpT  scan_op;
        KeyValuePairT       scan_item[DIMENSION];

        #pragma unroll
        for (int k = 0; k < DIMENSION; ++k) {
            scan_item[k].value = running_total[k];
            scan_item[k].key = thread_current_coord.x;
        }

        #pragma unroll
        for (int k = 0; k < DIMENSION; ++k) {
            BlockScanT(temp_storage.aliasable.scan).ExclusiveScan(scan_item[k], scan_item[k], scan_op, tile_carry[k]);
        }

        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int k = 0; k < DIMENSION; ++k) {
                scan_item[k].key = thread_start_coord.x;
                scan_item[k].value = 0.0;
            }
        }

        if (tile_num_rows > 0)
        {
            #pragma unroll
            for (int k = 0; k < DIMENSION; ++k) {

                CTA_SYNC();

                // Scan downsweep and scatter
                ValueT* s_partials = &temp_storage.aliasable.merge_items[0].nonzero;

                if (scan_item[k].key != scan_segment[k][0].key)
                {
                    s_partials[scan_item[k].key] = scan_item[k].value;
                }
                else
                {
                    scan_segment[k][0].value += scan_item[k].value;
                }

                #pragma unroll
                for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    if (scan_segment[k][ITEM - 1].key != scan_segment[k][ITEM].key)
                    {
                        s_partials[scan_segment[k][ITEM - 1].key] = scan_segment[k][ITEM - 1].value;
                    }
                    else
                    {
                        scan_segment[k][ITEM].value += scan_segment[k][ITEM - 1].value;
                    }
                }

                CTA_SYNC();

                #pragma unroll 1
                for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
                {
                    spmv_params.d_vector_y[(tile_start_coord.x + item) * DIMENSION + k] = s_partials[item];
                }
            }
        }

        // Return the tile's running carry-out
        // return tile_carry;
    }




    /**
     * Process a merge tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
        KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        int                             num_merge_tiles)            ///< [in] Total number of merge tiles
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
                OffsetT                         diagonal = (tile_idx + threadIdx.x) * TILE_ITEMS;
                CoordinateT                     tile_coord;
                CountingInputIterator<OffsetT>  nonzero_indices(0);

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
        KeyValuePairT tile_carry[DIMENSION];

        ConsumeTile(
            tile_idx,
            tile_start_coord,
            tile_end_coord,
            tile_carry,
            Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>());

        // Output the tile's carry-out
        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int k = 0; k < DIMENSION; ++k) {
                if (HAS_ALPHA)
                {
                    tile_carry[k].value *= spmv_params.alpha;
                }

                tile_carry[k].key += tile_start_coord.x;
                if (tile_carry[k].key >= spmv_params.num_rows)
                {
                    // FIXME: This works around an invalid memory access in the
                    // fixup kernel. The underlying issue needs to be debugged and
                    // properly fixed, but this hack prevents writes to
                    // out-of-bounds addresses. It doesn't appear to have an effect
                    // on the validity of the results, since this only affects the
                    // carry-over from last tile in the input.
                    tile_carry[k].key = spmv_params.num_rows - 1;
                    tile_carry[k].value = ValueT{};
                };

                d_tile_carry_pairs[tile_idx + k * num_merge_tiles]    = tile_carry[k];
            }
        }
    }
};

} // namespace cub
CUB_NS_POSTFIX 