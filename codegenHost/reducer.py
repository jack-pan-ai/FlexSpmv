# reducer.py for cpu

# generate the diagnoal code once
# for the spmv kernel and agent search kernel
def reducer_diagnal_code_gen():
    # generate the diagnoal code once
    # for the spmv kernel and agent search kernel
    diagonal_code_search = []
    forloop_code_reducers_consume_special_front = []
    forloop_code_reducers_consume_special_end = []
    forloop_code_reducers_partial_special_front = []
    forloop_code_reducers_partial_special_end = []

    # generate the diagnoal code once
    forloop_code_reducers_consume_special_front.append(f"    for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x) \n")
    forloop_code_reducers_consume_special_front.append(f"    {{ \n")
    forloop_code_reducers_consume_special_end.append(f"    }} \n")
    forloop_code_reducers_partial_special_front.append(f"    for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y) \n")
    forloop_code_reducers_partial_special_front.append(f"    {{ \n")
    forloop_code_reducers_partial_special_end.append(f"    }} \n")

    # generate the diagnoal code once
    diagonal_code_search.append(f"    CountingInputIterator<OffsetT> nonzero_indices(0); \n")
    diagonal_code_search.append(f"    int2 thread_coord; \n")
    diagonal_code_search.append(f"    int2 thread_coord_end; \n")
    diagonal_code_search.append(f"    int start_diagonal = std::min(items_per_thread * tid, num_merge_items); \n")
    diagonal_code_search.append(f"    int end_diagonal = std::min(start_diagonal + items_per_thread, num_merge_items); \n")
    diagonal_code_search.append(f"    MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, num_rows, \n")
    diagonal_code_search.append(f"                    num_nonzeros, thread_coord); \n")
    diagonal_code_search.append(f"    MergePathSearch(end_diagonal, row_end_offsets, nonzero_indices, num_rows, \n")
    diagonal_code_search.append(f"                    num_nonzeros, thread_coord_end); \n")
    

    return diagonal_code_search, forloop_code_reducers_consume_special_front, forloop_code_reducers_consume_special_end, forloop_code_reducers_partial_special_front, forloop_code_reducers_partial_special_end