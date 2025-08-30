# This file is used to generate the CPU code from the graph
import string
import easier as esr

from traceGraph.graph_trace import trace_graph
from codegenHost.merged_gen_core import declarations_gen, reducer_gen, map_gen, aggregator_gen

# debug print
def debug_print(code, code_name):
    if code != []:
        for op in code:
            print(f"{code_name}: {op}")
    else:
        print(f"{code_name} is empty")

# Generate CUDA code from the graph
def generate_cpu_code_from_graph(traced_model):
    
    # Analyze the graph and extract operations
    inputs = []
    _input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    outputs = []

    inputs, outputs, selector_register, map_operations, reducer_operations, aggregator_operations = trace_graph(traced_model)
    
#  -------------------------------------------------------------
#  Generate the code
#  -------------------------------------------------------------
    #  -------------------------------------------------------------
    #  Generate the code for input and output and selector
    #  -------------------------------------------------------------
    input_parameters_code = []
    input_agent_tenosrs_code = []
    output_agent_tenosrs_code = []
    output_agent_forloop_code_main = []
    output_agent_forloop_code_last_row = []
    selector_code = []

    input_parameters_code, input_agent_tenosrs_code, output_agent_tenosrs_code, output_agent_forloop_code_main, output_agent_forloop_code_last_row, selector_code = declarations_gen(inputs, outputs, selector_register, reducer_operations, aggregator_operations)

    # used for the last row
    if reducer_operations != []:
        selector_code_last_row = selector_code
    else:
        selector_code_last_row = []
    
    debug_print(input_parameters_code, "input_parameters_code")
    debug_print(input_agent_tenosrs_code, "input_agent_tenosrs_code")
    debug_print(output_agent_tenosrs_code, "output_agent_tenosrs_code")
    debug_print(output_agent_forloop_code_main, "output_agent_forloop_code_main")
    debug_print(output_agent_forloop_code_last_row, "output_agent_forloop_code_last_row")
    debug_print(selector_code, "selector_code")

    #  -------------------------------------------------------------
    #  Generate the code for map
    #  -------------------------------------------------------------
    map_code = map_gen(map_operations)
    debug_print(map_code, "map_code")

    # used for the last row
    if reducer_operations != []:
        map_code_last_row = map_code
    else:
        map_code_last_row = []
    debug_print(map_code_last_row, "map_code_last_row")

    # offset tag for reduce and aggregator
    forloop_code_row_end_offsets_tag = []
    if reducer_operations != []:
        forloop_code_row_end_offsets_tag.append(f"    row_end_offsets[thread_coord.x]")
    elif aggregator_operations != []:
        forloop_code_row_end_offsets_tag.append(f"    thread_coord_end.y")
    else:
        # only mapping
        forloop_code_row_end_offsets_tag.append(f"    row_end_offsets[thread_coord.y]")
    debug_print(forloop_code_row_end_offsets_tag, "forloop_code_row_end_offsets_tag")
    #  -------------------------------------------------------------
    #  Generate the code for reducers and aggregators
    #  -------------------------------------------------------------

    reducer_consume_init_code, reducer_consume_forloop_code, reducer_consume_forloop_add_code, reducer_partial_init_code, reducer_partial_forloop_code, reducer_partial_carry_code, reducer_partial_carry_fixup_code, reducer_diagonal_code_search, forloop_code_reducers_consume_special_front, forloop_code_reducers_consume_special_end, forloop_code_reducers_partial_special_front, forloop_code_reducers_partial_special_end = reducer_gen(reducer_operations)
    debug_print(reducer_consume_init_code, "reducer_consume_init_code")
    debug_print(reducer_consume_forloop_code, "reducer_consume_forloop_code")
    debug_print(reducer_consume_forloop_add_code, "reducer_consume_forloop_add_code")
    debug_print(reducer_partial_init_code, "reducer_partial_init_code")
    debug_print(reducer_partial_forloop_code, "reducer_partial_forloop_code")
    debug_print(reducer_partial_carry_code, "reducer_partial_carry_code")
    debug_print(reducer_partial_carry_fixup_code, "reducer_partial_carry_fixup_code")
    debug_print(reducer_diagonal_code_search, "reducer_diagonal_code_search")
    debug_print(forloop_code_reducers_consume_special_front, "forloop_code_reducers_consume_special_front")
    debug_print(forloop_code_reducers_consume_special_end, "forloop_code_reducers_consume_special_end")
    debug_print(forloop_code_reducers_partial_special_front, "forloop_code_reducers_partial_special_front")
    debug_print(forloop_code_reducers_partial_special_end, "forloop_code_reducers_partial_special_end")
    debug_print(forloop_code_row_end_offsets_tag, "forloop_code_row_end_offsets_tag")
    # Aggregators
    # debug
    debug_print(aggregator_operations, "aggregator_operations")
    aggregator_partial_carry_fixup_code, aggregator_partial_forloop_code, aggregator_diagonal_code_search, aggregator_tenosrs_carry_out_code = aggregator_gen(aggregator_operations)
    debug_print(aggregator_partial_carry_fixup_code, "aggregator_partial_carry_fixup_code")
    debug_print(aggregator_partial_forloop_code, "aggregator_partial_forloop_code")
    debug_print(aggregator_diagonal_code_search, "aggregator_diagonal_code_search")
    debug_print(aggregator_tenosrs_carry_out_code, "aggregator_tenosrs_carry_out_code")
    #  -------------------------------------------------------------
    #  Read template files
    #  -------------------------------------------------------------
    with open("cpu_template/merged_spmv_template_cpu.h", "r") as f:
        kernel_spmv_template = f.read()
    
    #  -------------------------------------------------------------
    #  Apply templates using string.Template
    #  -------------------------------------------------------------
    def trans_str(code):
        if code == []:
            return ''
        else:
            return ''.join(code)
    
    input_parameters_str = trans_str(input_parameters_code)
    input_agent_tenosrs_str = trans_str(input_agent_tenosrs_code)
    output_agent_tenosrs_str = trans_str(output_agent_tenosrs_code)
    output_agent_forloop_str = trans_str(output_agent_forloop_code_main)
    output_agent_forloop_last_row_str = trans_str(output_agent_forloop_code_last_row)
    selector_str = trans_str(selector_code)
    map_str = trans_str(map_code)
    map_code_last_row_str = trans_str(map_code_last_row)
    selector_code_last_row_str = trans_str(selector_code_last_row)
    reducer_consume_init_str = trans_str(reducer_consume_init_code)
    reducer_consume_forloop_str = trans_str(reducer_consume_forloop_code)
    reducer_consume_forloop_add_str = trans_str(reducer_consume_forloop_add_code)
    reducer_partial_init_str = trans_str(reducer_partial_init_code)
    reducer_partial_forloop_str = trans_str(reducer_partial_forloop_code)
    reducer_partial_carry_str = trans_str(reducer_partial_carry_code)
    reducer_partial_carry_fixup_str = trans_str(reducer_partial_carry_fixup_code)
    reducer_diagonal_code_search_str = trans_str(reducer_diagonal_code_search)
    forloop_code_reducers_consume_special_front_str = trans_str(forloop_code_reducers_consume_special_front)
    forloop_code_reducers_consume_special_end_str = trans_str(forloop_code_reducers_consume_special_end)
    forloop_code_reducers_partial_special_front_str = trans_str(forloop_code_reducers_partial_special_front)
    forloop_code_reducers_partial_special_end_str = trans_str(forloop_code_reducers_partial_special_end)
    forloop_code_row_end_offsets_tag_str = trans_str(forloop_code_row_end_offsets_tag)
    aggregator_partial_carry_fixup_code_str = trans_str(aggregator_partial_carry_fixup_code)
    aggregator_partial_forloop_code_str = trans_str(aggregator_partial_forloop_code)
    aggregator_diagonal_code_search_str = trans_str(aggregator_diagonal_code_search)
    aggregator_tenosrs_carry_out_code_str = trans_str(aggregator_tenosrs_carry_out_code)
    spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
        input_parameters_code=input_parameters_str,
        input_agent_tenosrs_code=input_agent_tenosrs_str,
        output_agent_tenosrs_code=output_agent_tenosrs_str,
        output_agent_forloop_code_main=output_agent_forloop_str,
        output_agent_forloop_code_last_row=output_agent_forloop_last_row_str,
        selector_code=selector_str,
        map_code=map_str,
        map_code_last_row=map_code_last_row_str,
        selector_code_last_row=selector_code_last_row_str,
        reducer_consume_init_code=reducer_consume_init_str,
        reducer_consume_forloop_code=reducer_consume_forloop_str,
        reducer_consume_forloop_add_code=reducer_consume_forloop_add_str,
        reducer_partial_init_code=reducer_partial_init_str,
        reducer_partial_forloop_code=reducer_partial_forloop_str,
        reducer_partial_carry_code=reducer_partial_carry_str,
        reducer_partial_carry_fixup_code=reducer_partial_carry_fixup_str,
        reducer_diagonal_code_search=reducer_diagonal_code_search_str,
        forloop_code_reducers_consume_special_front=forloop_code_reducers_consume_special_front_str,
        forloop_code_reducers_consume_special_end=forloop_code_reducers_consume_special_end_str,
        forloop_code_reducers_partial_special_front=forloop_code_reducers_partial_special_front_str,
        forloop_code_reducers_partial_special_end=forloop_code_reducers_partial_special_end_str,
        forloop_code_row_end_offsets_tag=forloop_code_row_end_offsets_tag_str,
        aggregator_partial_carry_fixup_code=aggregator_partial_carry_fixup_code_str,
        aggregator_partial_forloop_code=aggregator_partial_forloop_code_str,
        aggregator_diagonal_code_search=aggregator_diagonal_code_search_str,
        aggregator_tenosrs_carry_out_code=aggregator_tenosrs_carry_out_code_str,
    )

    
    # #  -------------------------------------------------------------
    # #  Write files
    # #  -------------------------------------------------------------
    with open("cpu/merged_spmv.h", "w") as f:
        f.write(spmv_kernel_code)
    # with open("trash/merged_spmv.h", "w") as f:
    #     f.write(spmv_kernel_code)
    
    #  -------------------------------------------------------------
    print("CPU code generated successfully!")
    #  -------------------------------------------------------------

    