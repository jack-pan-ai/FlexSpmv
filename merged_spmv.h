#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "data_struct_shared.cuh"

// Minimal CPU helpers
template <typename OffsetT> struct CountingInputIterator {
  OffsetT start;
  inline CountingInputIterator(OffsetT s) : start(s) {}
  inline OffsetT operator[](OffsetT idx) const { return start + idx; }
};

struct int2 {
  int x;
  int y;
};

/**
 * Computes the begin offsets into A and B for the specific diagonal from CUB
 */
template <typename AIteratorT, typename BIteratorT, typename OffsetT,
          typename CoordinateT>
inline void MergePathSearch(
    OffsetT diagonal,             ///< [in]The diagonal to search
    AIteratorT a,                 ///< [in]List A
    BIteratorT b,                 ///< [in]List B
    OffsetT a_len,                ///< [in]Length of A
    OffsetT b_len,                ///< [in]Length of B
    CoordinateT &path_coordinate) ///< [out] (x,y) coordinate where diagonal
                                  ///< intersects the merge path
{
  OffsetT x_min = std::max(diagonal - b_len, 0);
  OffsetT x_max = std::min(diagonal, a_len);

  while (x_min < x_max) {
    OffsetT x_pivot = (x_min + x_max) >> 1;
    if (a[x_pivot] <= b[diagonal - x_pivot - 1])
      x_min = x_pivot + 1; // Contract range up A (down B)
    else
      x_max = x_pivot; // Contract range down A (up B)
  }

  path_coordinate.x = std::min(x_min, a_len);
  path_coordinate.y = diagonal - x_min;
}

/**
 * Apply carry-out fix-up for rows spanning multiple threads for reducers
 */
template <typename ValueT, typename OffsetT, int dim>
void ApplyCarryOutFixup(int num_threads, int num_rows,
                        OffsetT *row_carry_out_reducer,
                        Tensor<ValueT, dim> *value_carry_out_reducer,
                        ValueT *output_y_reducer_ptr) {

  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out_reducer[tid] < num_rows) {
      for (int i = 0; i < dim; i++) {
        output_y_reducer_ptr[row_carry_out_reducer[tid] * dim + i] +=
            value_carry_out_reducer[tid].values[i];
      }
    }
  }
}

/**
 * Apply carry-out fix-up for rows spanning multiple threads for aggregators
 */
template <typename ValueT, typename OffsetT, int dim>
void ApplyCarryOutFixup(int num_threads,
                        Tensor<ValueT, dim> *value_carry_out_sum,
                        ValueT *output_y_sum_ptr) {
  typedef Tensor<ValueT, dim> TensorOutput_sum_T;
  TensorOutput_sum_T sum_result;
  for (int tid = 0; tid < num_threads; ++tid) {
    sum_result += value_carry_out_sum[tid];
  }
  for (int i = 0; i < dim; i++) {
    output_y_sum_ptr[i] = sum_result.values[i];
  }
}

/**
 * OpenMP CPU merge-based SpMV from CUB
 */
template <typename ValueT, typename OffsetT>
void OmpMergeSystem(
    int num_threads, OffsetT *__restrict row_end_offsets,
    // [code generation]
      ValueT *__restrict vector_x_ptr, 
  OffsetT *__restrict selector_1_ptr, 
  OffsetT *__restrict selector_2_ptr, 
  ValueT *__restrict spm_1_ptr, 
  ValueT *__restrict spm_2_ptr, 
  ValueT *__restrict output_y_y_add_1_ptr, 
  ValueT *__restrict output_y_y_add_2_ptr, 
  ValueT *__restrict output_y_y_reducer_1_ptr, 
  ValueT *__restrict output_y_y_reducer_2_ptr, 
 int num_rows, int num_nonzeros) {
  // [code generation]
  // input and output tensors types
    typedef Tensor<ValueT, 2> TensorInput_vector_x_T; 
  typedef Tensor<ValueT, 2> TensorInput_spm_1_T; 
  typedef Tensor<ValueT, 6> TensorInput_spm_2_T; 
   typedef Tensor<ValueT, 2> TensorOutput_y_add_1_T; 
  typedef Tensor<ValueT, 6> TensorOutput_y_add_2_T; 
  typedef Tensor<ValueT, 2> TensorOutput_y_reducer_1_T; 
 OffsetT row_carry_out_y_reducer_1[256]; 
 TensorOutput_y_reducer_1_T value_carry_out_y_reducer_1[256]; 
  typedef Tensor<ValueT, 6> TensorOutput_y_reducer_2_T; 
 OffsetT row_carry_out_y_reducer_2[256]; 
 TensorOutput_y_reducer_2_T value_carry_out_y_reducer_2[256]; 


#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int tid = 0; tid < num_threads; tid++) {
    OffsetT num_merge_items =
        num_rows + num_nonzeros; // Merge path total length
    OffsetT items_per_thread = (num_merge_items + num_threads - 1) /
                               num_threads; // Merge items per thread

    // Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for
    // [code generation]
    // Merge list B (NZ indices)
    CountingInputIterator<OffsetT> nonzero_indices(0);
    int2 thread_coord;
    int2 thread_coord_end;
    int start_diagonal = std::min(items_per_thread * tid, num_merge_items);
    int end_diagonal =
        std::min(start_diagonal + items_per_thread, num_merge_items);
    MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, num_rows,
                    num_nonzeros, thread_coord);
    MergePathSearch(end_diagonal, row_end_offsets, nonzero_indices, num_rows,
                    num_nonzeros, thread_coord_end);

    // Consume whole rows
    for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x) {

         TensorOutput_y_reducer_1_T reducer_y_reducer_1_running_total; 
   TensorOutput_y_reducer_2_T reducer_y_reducer_2_running_total; 

      for (; thread_coord.y < row_end_offsets[thread_coord.x];
           ++thread_coord.y) {
        // selector
          OffsetT column_indices_selector_1 = selector_1_ptr[thread_coord.y]; 
  TensorInput_vector_x_T selector_1(vector_x_ptr +                     column_indices_selector_1 * 2); 
  OffsetT column_indices_selector_2 = selector_2_ptr[thread_coord.y]; 
  TensorInput_vector_x_T selector_2(vector_x_ptr +                     column_indices_selector_2 * 2); 
  TensorInput_spm_1_T spm_1(spm_1_ptr +                     thread_coord.y * 2); 
  TensorInput_spm_2_T spm_2(spm_2_ptr +                     thread_coord.y * 6); 


        // mapping
            TensorOutput_y_add_1_T y_add_1 = selector_1 +                     spm_1; 
    TensorOutput_y_add_2_T y_add_2 = selector_2 +                     spm_2; 


        // output for reducer
           reducer_y_reducer_1_running_total += y_add_1; 
   reducer_y_reducer_2_running_total += y_add_2; 


        // output for map
          for (int i = 0; i < 2; i++) 
  { 
    output_y_y_add_1_ptr[thread_coord.y * 2 + i] = y_add_1.values[i]; 
  } 
  for (int i = 0; i < 6; i++) 
  { 
    output_y_y_add_2_ptr[thread_coord.y * 6 + i] = y_add_2.values[i]; 
  } 

      }

      // output for reducer
         for (int i = 0; i < 2; i++) 
   { 
     output_y_y_reducer_1_ptr[thread_coord.x * 2 + i] =                 reducer_y_reducer_1_running_total.values[i]; 
   } 
   for (int i = 0; i < 6; i++) 
   { 
     output_y_y_reducer_2_ptr[thread_coord.x * 6 + i] =                 reducer_y_reducer_2_running_total.values[i]; 
   } 

    }

    // Consume partial portion of thread's last row
       TensorOutput_y_reducer_1_T y_reducer_1_running_total_partial; 
   TensorOutput_y_reducer_2_T y_reducer_2_running_total_partial; 


    for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y) {
        OffsetT column_indices_selector_1 = selector_1_ptr[thread_coord.y]; 
  TensorInput_vector_x_T selector_1(vector_x_ptr +                     column_indices_selector_1 * 2); 
  OffsetT column_indices_selector_2 = selector_2_ptr[thread_coord.y]; 
  TensorInput_vector_x_T selector_2(vector_x_ptr +                     column_indices_selector_2 * 2); 
  TensorInput_spm_1_T spm_1(spm_1_ptr +                     thread_coord.y * 2); 
  TensorInput_spm_2_T spm_2(spm_2_ptr +                     thread_coord.y * 6); 


      // mapping
          TensorOutput_y_add_1_T y_add_1 = selector_1 +                     spm_1; 
    TensorOutput_y_add_2_T y_add_2 = selector_2 +                     spm_2; 


      // output for reducer
         y_reducer_1_running_total_partial += y_add_1; 
   y_reducer_2_running_total_partial += y_add_2; 


      // output for map
        for (int i = 0; i < 2; i++) 
  { 
    output_y_y_add_1_ptr[thread_coord.y * 2 + i] = y_add_1.values[i]; 
  } 
  for (int i = 0; i < 6; i++) 
  { 
    output_y_y_add_2_ptr[thread_coord.y * 6 + i] = y_add_2.values[i]; 
  } 

    
    }

    // Save carry-outs
       row_carry_out_y_reducer_1[tid] = thread_coord_end.x; 
   value_carry_out_y_reducer_1[tid] = y_reducer_1_running_total_partial; 
   row_carry_out_y_reducer_2[tid] = thread_coord_end.x; 
   value_carry_out_y_reducer_2[tid] = y_reducer_2_running_total_partial; 

  }

  // Carry-out fix-up (rows spanning multiple threads)
     ApplyCarryOutFixup<ValueT, OffsetT, 2>( 
     num_threads, num_rows, row_carry_out_y_reducer_1,                 value_carry_out_y_reducer_1, output_y_y_reducer_1_ptr); 
   ApplyCarryOutFixup<ValueT, OffsetT, 6>( 
     num_threads, num_rows, row_carry_out_y_reducer_2,                 value_carry_out_y_reducer_2, output_y_y_reducer_2_ptr); 

}
