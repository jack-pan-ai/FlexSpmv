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
                retval = first + retval;
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

        // Smem for intermediate results and scan
        template <int DIM_REDUCER, typename BlockScanT>
        union SmemReuseReducer
        {
            typedef Tensor<ValueT, DIM_REDUCER> TensorT;
            // Smem for intermediate results
            TensorT s_tile_value_reducer[TILE_ITEMS]; 
            // Smem needed for tile scanning
            typename BlockScanT::TempStorage scan;
        };

        // Tensor and TensorKey for input vector x
          typedef Tensor<ValueT, 1> TensorInput_add_10_T; 
  typedef Tensor<ValueT, 1> TensorInput_bsx_T; 
  typedef Tensor<ValueT, 1> TensorInput_bsy_T; 


        // Tensor and TensorKey for map 
          typedef Tensor<ValueT, 1> TensorOutput_pow_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_32_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_36_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_40_T; 

        // Tensor and TensorKey for reducers 
          // Tensor and TensorKey for reducers 
  typedef TensorKey<OffsetT, ValueT, 1>                     TensorKeyOutput_scatter_b_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_scatter_b_2_T; 
  // Reduce-value-by-segment scan operator 
  typedef ReduceTensorByKeyOp<TensorKeyOutput_scatter_b_2_T>                        ReduceBySegmentOp_scatter_b_2_T; 
  typedef BlockScan< 
            TensorKeyOutput_scatter_b_2_T, 
            BLOCK_THREADS, 
            AgentSpmvPolicyT::SCAN_ALGORITHM> 
            BlockScan_scatter_b_2_T; 
  // Tensor and TensorKey for reducers 
  typedef TensorKey<OffsetT, ValueT, 1>                     TensorKeyOutput_scatter_b_3_T; 
  typedef Tensor<ValueT, 1> TensorOutput_scatter_b_3_T; 
  // Reduce-value-by-segment scan operator 
  typedef ReduceTensorByKeyOp<TensorKeyOutput_scatter_b_3_T>                        ReduceBySegmentOp_scatter_b_3_T; 
  typedef BlockScan< 
            TensorKeyOutput_scatter_b_3_T, 
            BLOCK_THREADS, 
            AgentSpmvPolicyT::SCAN_ALGORITHM> 
            BlockScan_scatter_b_3_T; 


        /// Shared memory type required by this thread block
        struct _TempStorage
        {
            // tile coordinates for blocks
            CoordinateT tile_coords[2];
            // smem for intermediate results and scan
                           SmemReuseReducer<1,                     BlockScan_scatter_b_2_T> smem_scatter_b_2; 
               SmemReuseReducer<1,                     BlockScan_scatter_b_3_T> smem_scatter_b_3; 

            OffsetT s_tile_row_end_offsets[TILE_ITEMS];
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
        RowOffsetsIteratorT wd_row_end_offsets;

        // [code generation] wrapper pointers for loading the data
          VectorValueIteratorT add_10_ptr; 
  VectorValueIteratorT bsx_ptr; 
  VectorValueIteratorT bsy_ptr; 
  ColumnIndicesIteratorT gather_b_1_ptr; 


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
                wd_row_end_offsets(spmv_params.d_row_end_offsets),
                  add_10_ptr(spmv_params.add_10_ptr), 
    bsx_ptr(spmv_params.bsx_ptr), 
    bsy_ptr(spmv_params.bsy_ptr), 
    gather_b_1_ptr(spmv_params.gather_b_1_ptr), 

              spmv_params(spmv_params)
        {
        }

        //---------------------------------------------------------------------
        // Tile processing
        //---------------------------------------------------------------------

        __device__ __forceinline__ void loading_offsets(
            int tile_num_rows, 
            CoordinateT tile_start_coord){
            // Gather the row end-offsets for the merge tile into shared memory
            #pragma unroll 1
            for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
            {
                const OffsetT offset =
                    (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item),
                            static_cast<OffsetT>(spmv_params.num_rows - 1));
                temp_storage.s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
            }

            CTA_SYNC();
        }

        __device__ __forceinline__ void search_thread_start_coord(
            OffsetT *s_tile_row_end_offsets, // [in] Shared memory array of row end offsets for the merge tile
            CoordinateT tile_start_coord, // [in] Starting coordinate of the merge tile
            int tile_num_rows, // [in] Number of rows in the merge tile
            int tile_num_nonzeros, // [in] Number of non-zeros in the merge tile
            CoordinateT &thread_start_coord // [out] Starting coordinate of the thread
        )
        {
            CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);

            MergePathSearch(
                OffsetT(threadIdx.x * ITEMS_PER_THREAD), // Diagonal
                s_tile_row_end_offsets,                  // List A
                tile_nonzero_indices,                    // List B
                tile_num_rows,
                tile_num_nonzeros,
                thread_start_coord);

            CTA_SYNC(); // Perf-sync
        }


        template <int DimReducer, typename BlockScanT, typename TensorT, typename ReduceBySegmentOpT>
        __device__ __forceinline__ void reduce(
            TensorT *s_tile_value_nonzeros,      ///< [in, code gen] Shared memory array of non-zero values for the merge tile
            OffsetT *s_tile_row_end_offsets,    ///< [in, code gen] Shared memory array of row end offsets for the merge tile
            CoordinateT tile_start_coord,       ///< [in] Starting coordinate of the merge tile
            CoordinateT tile_end_coord,         ///< [in] Ending coordinate of the merge tile
            CoordinateT thread_start_coord,     ///< [in] Starting coordinate of the thread
            int tile_num_rows,                  ///< [in] Number of rows in the merge tile
            int tile_num_nonzeros,               ///< [in] Number of non-zeros in the merge tile
            ValueT *output_vector_y,             ///< [out] Output vector y
            typename BlockScanT::TempStorage &scan_storage ///< [in] Scan storage for BlockScanT operations
        )
        {
            typedef TensorKey<OffsetT, ValueT, DimReducer> TensorKeyT;
            // Compute the thread's merge path segment
            CoordinateT thread_current_coord = thread_start_coord;
            TensorKeyT scan_segment[ITEMS_PER_THREAD];
            TensorT running_total;
            CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);

            OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
            TensorT nonzero = s_tile_value_nonzeros[thread_current_coord.y];

// Reduce
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
                {
// Move down (accumulate)
                    scan_segment[ITEM].set(nonzero.values);
                    running_total = running_total + nonzero;
                    ++thread_current_coord.y;
                    nonzero = s_tile_value_nonzeros[thread_current_coord.y];
                }
                else
                {
// Move right (reset)
                    scan_segment[ITEM].set(0.0);
                    running_total.set(0.0);
                    ++thread_current_coord.x;
                    row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
                }
                scan_segment[ITEM].key = thread_current_coord.x;
            }

            CTA_SYNC();

            // Block-wide reduce-value-by-segment
            TensorKeyT tile_carry;
            ReduceBySegmentOpT scan_op;
            TensorKeyT scan_item(running_total.values);
            scan_item.key = thread_current_coord.x;

            BlockScanT(scan_storage).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

            if (threadIdx.x == 0)
            {
                scan_item.key = thread_start_coord.x;
                scan_item.set(0.0);
            }

            if (tile_num_rows > 0)
            {

                CTA_SYNC();
                // Scan downsweep and scatter
                // memory reuse for the partial results 
                // TILE_ITEMS is used to avoid bank conflict
                TensorT *s_partials = s_tile_value_nonzeros;

                if (scan_item.key != scan_segment[0].key)
                {
                    s_partials[scan_item.key].set(scan_item.values);
                }
                else
                {
                    scan_segment[0] = scan_segment[0] + scan_item;
                }

#pragma unroll
                for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    if (scan_segment[ITEM - 1].key != scan_segment[ITEM].key)
                    {
                        s_partials[scan_segment[ITEM - 1].key].set(scan_segment[ITEM - 1].values);
                    }
                    else
                    {
                        scan_segment[ITEM] = scan_segment[ITEM] + scan_segment[ITEM - 1];
                    }
                }

                CTA_SYNC();

// memory coalescing for writing the output vector y
#pragma unroll 1
                for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
                {
                    #pragma unroll
                    for (int i = 0; i < DimReducer; i++)
                    {
                        atomicAdd(
                            &output_vector_y[(tile_start_coord.x + item) * DimReducer + i],
                            s_partials[item].values[i]
                        );
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
                    for (int i = 0; i < DimReducer; i++)
                    {
                        atomicAdd(
                            &output_vector_y[tile_carry.key * DimReducer + i],
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

            loading_offsets(tile_num_rows, tile_start_coord);

// Select
// Gather the nonzeros for the merge tile into shared memory
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (nonzero_idx < tile_num_nonzeros)
                {
                    // [code generation]
                        VectorValueIteratorT bsx_ptr_current = bsx_ptr +                     (tile_start_coord.y + nonzero_idx) * 1; 
    TensorInput_bsx_T bsx(bsx_ptr_current); 
    VectorValueIteratorT bsy_ptr_current = bsy_ptr +                     (tile_start_coord.y + nonzero_idx) * 1; 
    TensorInput_bsy_T bsy(bsy_ptr_current); 
    ColumnIndicesIteratorT gather_b_1_ptr_current =                     gather_b_1_ptr + tile_start_coord.y + nonzero_idx; 
    TensorInput_add_10_T                     gather_b_1(add_10_ptr + *gather_b_1_ptr_current * 1); 


                    // map
                        TensorOutput_pow_2_T pow_2 = gather_b_1 ^                     2; 
    TensorOutput_mul_32_T mul_32 = 0.5 *                     pow_2; 
    TensorOutput_mul_36_T mul_36 = mul_32 *                     bsx; 
    temp_storage.smem_scatter_b_2.s_tile_value_reducer[nonzero_idx] =                     mul_36; 
    TensorOutput_mul_40_T mul_40 = mul_32 *                     bsy; 
    temp_storage.smem_scatter_b_3.s_tile_value_reducer[nonzero_idx] =                     mul_40; 


                    //output for map
                    
                }
            }
            CTA_SYNC();

            // reduce the intermeidate computations 
            // all reducers share the same row end offsets 
            // Search for the thread's starting coordinate within the merge tile 
            CoordinateT thread_start_coord; 
            search_thread_start_coord( 
                temp_storage.s_tile_row_end_offsets, 
                tile_start_coord, 
                tile_num_rows, 
                tile_num_nonzeros, 
                thread_start_coord); 
            // [code generation]
               reduce<1, BlockScan_scatter_b_2_T, TensorOutput_scatter_b_2_T,                 ReduceBySegmentOp_scatter_b_2_T>( 
                temp_storage.smem_scatter_b_2.s_tile_value_reducer,                          ///< [in, code gen] Shared memory array of non-zero values for the merge tile 
                temp_storage.s_tile_row_end_offsets,                         ///< [in, code gen] Shared memory array of row end offsets for the merge tile 
                tile_start_coord,                               ///< [in] Starting coordinate of the merge tile 
                tile_end_coord,                                 ///< [in] Ending coordinate of the merge tile 
                thread_start_coord,                             ///< [in] Starting coordinate of the thread 
                tile_num_rows,                                  ///< [in] Number of rows in the merge tile 
                tile_num_nonzeros,                               ///< [in] Number of non-zeros in the merge tile 
                spmv_params.output_y_scatter_b_2_ptr,                       ///< [out] Output vector y 
                temp_storage.smem_scatter_b_2.scan                         ///< [in] Scan storage for BlockScanT 
            ); 
   CTA_SYNC(); 
   reduce<1, BlockScan_scatter_b_3_T, TensorOutput_scatter_b_3_T,                 ReduceBySegmentOp_scatter_b_3_T>( 
                temp_storage.smem_scatter_b_3.s_tile_value_reducer,                          ///< [in, code gen] Shared memory array of non-zero values for the merge tile 
                temp_storage.s_tile_row_end_offsets,                         ///< [in, code gen] Shared memory array of row end offsets for the merge tile 
                tile_start_coord,                               ///< [in] Starting coordinate of the merge tile 
                tile_end_coord,                                 ///< [in] Ending coordinate of the merge tile 
                thread_start_coord,                             ///< [in] Starting coordinate of the thread 
                tile_num_rows,                                  ///< [in] Number of rows in the merge tile 
                tile_num_nonzeros,                               ///< [in] Number of non-zeros in the merge tile 
                spmv_params.output_y_scatter_b_3_ptr,                       ///< [out] Output vector y 
                temp_storage.smem_scatter_b_3.scan                         ///< [in] Scan storage for BlockScanT 
            ); 
   CTA_SYNC(); 

        }



        /**
         * Process a merge tile
         */
        __device__ __forceinline__ void ConsumeTile(
            CoordinateT *d_tile_coordinates, ///< [in] Pointer to the temporary array of tile starting coordinates
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