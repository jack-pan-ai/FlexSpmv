import os
from string import Template as StrTemplate

from codegen.utils import get_dim_length


def generate_binding_code(
        project_root, inputs, outputs, selector_register
    ):
    """
    Generate the CUDA/Torch extension binding (merged_binding.cu) from graph metadata.

    Args:
        project_root: absolute path to repo root
        inputs: list of input node dicts from trace_graph
        outputs: list of output node dicts from trace_graph
        selector_register: list of selector metadata from trace_graph
    """

    # Infer argument ordering compatible with current driver:
    #  - value inputs excluding any selector target(s)
    #  - selector index tensors
    #  - selector target value tensors (gather sources)
    #  - row_end_offsets, num_rows, num_cols
    gather_target_names = set(
        inter["target"] for inter in selector_register if inter.get("selector") == 1
    )

    value_inputs = [inp for inp in inputs if inp["dtype"] != "int"]
    selector_inputs = [inp for inp in inputs if inp["dtype"] == "int"]

    non_gather_value_inputs = [
        inp for inp in value_inputs if inp["name"] not in gather_target_names
    ]
    gather_value_inputs = [
        inp for inp in value_inputs if inp["name"] in gather_target_names
    ]

    # Pick a representative value tensor for dtype dispatch
    dispatch_tensor_name = (
        non_gather_value_inputs[0]["name"]
        if non_gather_value_inputs
        else (gather_value_inputs[0]["name"] \
            if gather_value_inputs else value_inputs[0]["name"])  # type: ignore[index]
    )

    # Build parameter list for the binding function
    bind_params = []
    for vi in non_gather_value_inputs:
        bind_params.append((vi["name"], "value"))
    for gi in gather_value_inputs:
        bind_params.append((gi["name"], "value"))
    for si in selector_inputs:
        bind_params.append((si["target"] + "_idx", "index"))
    # Static CSR pointer and sizes
    bind_params.extend([
        ("row_end_offsets", "index"), 
        ("num_rows", "size"), 
        ("num_cols", "size")
    ])

    def make_tensor_checks(name, kind):
        lines = []
        if kind in ("value", "index"):
            lines.append(f"  TORCH_CHECK({name}.is_cuda(), \"{name} must be a CUDA tensor\");\n")
            lines.append(f"  TORCH_CHECK({name}.is_contiguous(), \"{name} must be contiguous\");\n")
        return lines

    # Load binding template
    template_binding_path = os.path.join(project_root, "cuda_template", "merged_binding_template.cu")
    with open(template_binding_path, "r") as f:
        binding_template = f.read()

    # Prepare substitutions: outputs (reducers, aggregators, maps)
    tuple_types = []
    output_allocations = []
    output_tuple_returns_list = []
    params_output_ptrs = []

    def add_output(out_name: str, dim: int, size_expr: str, idx: int):
        var_name = f"out_{idx}_{out_name}"
        output_allocations.append(
            f"  torch::Tensor {var_name} = torch::zeros({{{size_expr}}}, options_val);\n"
        )
        output_tuple_returns_list.append(var_name)
        params_output_ptrs.append(
            f"  params.output_y_{out_name}_ptr = reinterpret_cast<ValueT*>({var_name}.data_ptr());\n"
        )
        tuple_types.append("torch::Tensor")

    for idx, o in enumerate(outputs):
        dim = get_dim_length(o["shape"])
        out_name = o["name"]
        if str(o["target"]) == "reducer":
            add_output(out_name, dim, f"num_rows * {dim}", idx)
        elif "sum" in str(o["target"]):
            add_output(out_name, dim, f"{dim}", idx)
        else:
            add_output(out_name, dim, f"ne * {dim}", idx)

    output_tuple_returns = ", ".join(output_tuple_returns_list)
    tuple_type_list = ", ".join(tuple_types) if tuple_types else "void"

    # Function params
    def ctype_of(kind: str) -> str:
        return "torch::Tensor" if kind != "size" else "int64_t"

    function_params = ",\n".join([f"    {ctype_of(k)} {n}" for n, k in bind_params])
    function_call_args = ", ".join([n for n, _ in bind_params])

    # Checks
    basic_checks = []
    for n, k in bind_params:
        if k in ("value", "index"):
            basic_checks.extend(make_tensor_checks(n, k))
    dtype_checks = []
    for vi in non_gather_value_inputs + gather_value_inputs:
        n = vi["name"]
        dtype_checks.append(
            f"  TORCH_CHECK({n}.scalar_type() == \
            c10::CppTypeToScalarType<ValueT>::value, \"{n} dtype mismatch\");\n"
        )
    for si in selector_inputs:
        n = si["target"] + "_idx"
        dtype_checks.append(
            f"  TORCH_CHECK({n}.scalar_type() == \
            c10::CppTypeToScalarType<OffsetT>::value, \"{n} dtype mismatch\");\n"
        )
    dtype_checks.append(
        "  TORCH_CHECK(row_end_offsets.scalar_type() == \
            c10::CppTypeToScalarType<OffsetT>::value, \"row_end_offsets dtype mismatch\");\n"
    )

    # ne expr and bsx/bsy size checks
    if selector_inputs:
        gi0 = selector_inputs[0]["target"] + "_idx"
        ne_expr = f"{gi0}.numel()"
    else:
        ne_expr = "1"
    ne_multiple_checks = []
    if any(v["name"] == "bsx" for v in non_gather_value_inputs + gather_value_inputs):
        ne_multiple_checks.append(
            "  TORCH_CHECK(bsx.numel() % ne == \
            0, \"bsx.numel() must be a multiple of ne\");\n"
        )
    if any(v["name"] == "bsy" for v in non_gather_value_inputs + gather_value_inputs):
        ne_multiple_checks.append(
            "  TORCH_CHECK(bsy.numel() % ne == \
            0, \"bsy.numel() must be a multiple of ne\");\n"
        )

    # Params mapping
    params_value_ptrs = [
        f"  params.{vi['name']}_ptr = \
        reinterpret_cast<ValueT*>({vi['name']}.data_ptr());\n"
        for vi in (non_gather_value_inputs + gather_value_inputs)
    ]
    params_index_ptrs = [
        f"  params.{si['target']}_ptr = \
        reinterpret_cast<OffsetT*>(({si['target']}_idx).data_ptr());\n"
        for si in selector_inputs
    ]

    # pybind args
    pybind_args = "\n".join([f"      , py::arg(\"{n}\")" for n, _ in bind_params])

    binding_code = StrTemplate(binding_template).substitute(
        tuple_type_list=tuple_type_list,
        function_params=function_params,
        function_call_args=function_call_args,
        basic_checks="".join(basic_checks),
        dtype_checks="".join(dtype_checks),
        ne_expr=ne_expr,
        ne_multiple_checks="".join(ne_multiple_checks),
        dispatch_tensor=dispatch_tensor_name,
        output_allocations="".join(output_allocations),
        params_value_ptrs="".join(params_value_ptrs),
        params_index_ptrs="".join(params_index_ptrs),
        params_output_ptrs="".join(params_output_ptrs),
        output_tuple_returns=output_tuple_returns,
        pybind_args=pybind_args,
    )

    # Write binding file at project root
    binding_path = os.path.join(project_root, "merged_binding.cu")
    with open(binding_path, "w") as f:
        f.write(binding_code)
    print(f"Binding code generated successfully at {binding_path}!")


