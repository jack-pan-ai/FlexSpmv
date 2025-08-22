# reducer.py

# generate the diagnoal code once
# for the spmv kernel and agent search kernel
def reducer_diagnal_code_gen():
    # generate the diagnoal code once
    # for the spmv kernel and agent search kernel
    diagonal_code_spmv = []
    diagonal_code_spmv_agent = []
    diagonal_code_spmv_agent_thread = []
    offset_code_spmv_agent_dispatch = []

    diagonal_code_spmv.append(f"    if (search_grid_size < sm_count) \n")
    diagonal_code_spmv.append(f"    {{ \n")
    diagonal_code_spmv.append(f"        d_tile_coordinates = NULL; \n")
    diagonal_code_spmv.append(f"    }} \n")
    diagonal_code_spmv.append(f"    else \n")
    diagonal_code_spmv.append(f"    {{ \n")
    diagonal_code_spmv.append(f"        // Use separate search kernel if we have enough spmv tiles to saturate the device \n")
    diagonal_code_spmv.append(f"        // Log spmv_search_kernel configuration \n")
    diagonal_code_spmv.append(f"        if (debug_synchronous) \n")
    diagonal_code_spmv.append(f"        {{ \n")
    diagonal_code_spmv.append(f"            _CubLog(\"Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\\n\", \n")
    diagonal_code_spmv.append(f"                    search_grid_size, search_block_size, (long long)stream); \n")
    diagonal_code_spmv.append(f"        }} \n")
    diagonal_code_spmv.append(f"        // Invoke spmv_search_kernel \n")
    diagonal_code_spmv.append(f"        spmv_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>(num_merge_tiles, d_tile_coordinates, spmv_params); \n")
    diagonal_code_spmv.append(f"        // Check for failure to launch \n")
    diagonal_code_spmv.append(f"        if (CubDebug(error = cudaPeekAtLastError())) \n")
    diagonal_code_spmv.append(f"            break; \n")
    diagonal_code_spmv.append(f"        // Sync the stream if specified to flush runtime errors \n")
    diagonal_code_spmv.append(f"        if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) \n")
    diagonal_code_spmv.append(f"            break; \n")
    diagonal_code_spmv.append(f"    }} \n")

    # agent search code
    diagonal_code_spmv_agent.append(f"          // Read our starting coordinates \n")
    diagonal_code_spmv_agent.append(f"          if (threadIdx.x < 2) \n")
    diagonal_code_spmv_agent.append(f"          {{ \n")
    diagonal_code_spmv_agent.append(f"            if (d_tile_coordinates == NULL) \n")
    diagonal_code_spmv_agent.append(f"            {{ \n")
    diagonal_code_spmv_agent.append(f"                  // Search our starting coordinates \n")
    diagonal_code_spmv_agent.append(f"                  OffsetT diagonal = (tile_idx + threadIdx.x) * TILE_ITEMS; \n")
    diagonal_code_spmv_agent.append(f"                  CoordinateT tile_coord; \n")
    diagonal_code_spmv_agent.append(f"                  CountingInputIterator<OffsetT> nonzero_indices(0); \n")
    diagonal_code_spmv_agent.append(f"                  // Search the merge path \n")
    diagonal_code_spmv_agent.append(f"                  MergePathSearch( \n")
    diagonal_code_spmv_agent.append(f"                      diagonal, \n")
    diagonal_code_spmv_agent.append(f"                      RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets), \n")
    diagonal_code_spmv_agent.append(f"                      nonzero_indices, \n")
    diagonal_code_spmv_agent.append(f"                      spmv_params.num_rows, \n")
    diagonal_code_spmv_agent.append(f"                      spmv_params.num_nonzeros, \n")
    diagonal_code_spmv_agent.append(f"                      tile_coord); \n")
    diagonal_code_spmv_agent.append(f"                  temp_storage.tile_coords[threadIdx.x] = tile_coord; \n")
    diagonal_code_spmv_agent.append(f"              }} \n")
    diagonal_code_spmv_agent.append(f"              else \n")
    diagonal_code_spmv_agent.append(f"              {{ \n")
    diagonal_code_spmv_agent.append(f"                  temp_storage.tile_coords[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x]; \n")
    diagonal_code_spmv_agent.append(f"               }} \n")
    diagonal_code_spmv_agent.append(f"            }} \n")
    diagonal_code_spmv_agent.append(f"            CTA_SYNC(); \n")
    diagonal_code_spmv_agent.append(f"            CoordinateT tile_start_coord = temp_storage.tile_coords[0]; \n")
    diagonal_code_spmv_agent.append(f"            CoordinateT tile_end_coord = temp_storage.tile_coords[1]; \n")

    # thread code
    diagonal_code_spmv_agent_thread.append(f"    // reduce the intermeidate computations \n")
    diagonal_code_spmv_agent_thread.append(f"    // all reducers share the same row end offsets \n")
    diagonal_code_spmv_agent_thread.append(f"    // Search for the thread's starting coordinate within the merge tile \n")
    diagonal_code_spmv_agent_thread.append(f"    CoordinateT thread_start_coord; \n")
    diagonal_code_spmv_agent_thread.append(f"    search_thread_start_coord( \n")
    diagonal_code_spmv_agent_thread.append(f"        temp_storage.s_tile_row_end_offsets, \n")
    diagonal_code_spmv_agent_thread.append(f"        tile_start_coord, \n")
    diagonal_code_spmv_agent_thread.append(f"        tile_num_rows, \n")
    diagonal_code_spmv_agent_thread.append(f"        tile_num_nonzeros, \n")
    diagonal_code_spmv_agent_thread.append(f"        thread_start_coord); \n")

    # spmv offset code
    offset_code_spmv_agent_dispatch.append(f"    // [INFO] the row_end_offsets is shifted by 1, \n")
    offset_code_spmv_agent_dispatch.append(f"    spmv_params.d_row_end_offsets = spmv_params.d_row_end_offsets + 1; \n")

    return diagonal_code_spmv, diagonal_code_spmv_agent, diagonal_code_spmv_agent_thread, offset_code_spmv_agent_dispatch