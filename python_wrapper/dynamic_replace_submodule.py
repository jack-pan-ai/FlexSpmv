import torch
import torch.nn as nn

from easier import Selector, Reducer
from typing import List, Tuple, Dict, Optional, Iterable

from shallow_water_equation.utils import get_submodules


def _coo_rows_to_csr_ptr(row_ids: torch.Tensor, num_rows: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Convert COO row indices to CSR crow_indices and return (crow_indices, num_rows).

    If num_rows is not provided, it is inferred as max(row_ids) + 1.
    Returns crow_indices as int64 on the same device as ``row_ids``; caller
    may cast/move as needed.
    """
    if row_ids.dtype != torch.int64:
        row_ids = row_ids.to(torch.int64)
    if num_rows is None:
        if row_ids.numel() == 0:
            raise RuntimeError("row_ids is empty; cannot infer num_rows")
        num_rows = int(row_ids.max().item()) + 1
    conv = getattr(torch, "_convert_indices_from_coo_to_csr", None)
    if callable(conv):
        crow = conv(row_ids, size=num_rows)
        return crow, num_rows
    counts = torch.bincount(row_ids, minlength=num_rows)
    crow = torch.empty(num_rows + 1, dtype=torch.int64, device=row_ids.device)
    crow[0] = 0
    crow[1:] = torch.cumsum(counts, dim=0)
    return crow, num_rows


def _iter_placeholders_in_order(gm) -> List[str]:
    names: List[str] = []
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            names.append(n.name)
    return names


def _collect_selector_modules(gm) -> Tuple[List[Tuple[str, nn.Module]], Optional[nn.Module]]:
    """Collect selector-like modules and a reducer-like module.

    Heuristics:
    - Any submodule having attribute `idx` is considered a selector-like module.
    - A module that also has attribute `n` is considered a reducer metadata source
      (providing row ids and num_rows).
    Returns (selector_modules, reducer_module_or_none) where selector_modules is a list
    of (qualified_name, module) in a stable order of appearance by traversal of gm.graph.
    """
    selector_modules: List[Tuple[str, nn.Module]] = []
    reducer_mod: Optional[nn.Module] = None

    for _name, _module in gm.named_modules():
        if _name == "":
            continue
        if type(_module) == Selector:
            selector_modules.append((_name, _module))
        if type(_module) == Reducer:
            reducer_mod = _module
            break

    return selector_modules, reducer_mod


def _infer_gather_value_inputs(gm, selector_qnames: Iterable[str]) -> List[str]:
    """Infer which placeholder tensors are the gather value targets.

    Strategy: For each call_module node whose target matches a selector qualified name,
    inspect node.args and collect any placeholder names referenced as value tensors.
    """
    gather_inputs: List[str] = []
    selector_set = set(selector_qnames)

    # Build quick map from node to placeholder names
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        if node.target not in selector_set:
            continue

        def _walk_args(a):
            if isinstance(a, tuple) or isinstance(a, list):
                for x in a:
                    yield from _walk_args(x)
            elif hasattr(a, "op") and a.op == "placeholder":
                yield a.name

        for pname in _walk_args(node.args):
            if pname not in gather_inputs:
                gather_inputs.append(pname)

    return gather_inputs


class _DynamicCudaWrapper(nn.Module):
    def __init__(
        self,
        ext_module,
        input_param_names: List[str],
        non_gather_value_inputs: List[str],
        gather_value_inputs: List[str],
        selector_idx_tensors: List[torch.Tensor],
        row_end_offsets: torch.Tensor,
    ) -> None:
        super().__init__()
        # Cast and register selector indices as buffers (int32)
        buf_names: List[str] = []
        for i, t in enumerate(selector_idx_tensors):
            t = t.contiguous()
            # [TODO:] Currently, the selector indices are always int32, \
            # so we don't need to check the dtype, othewise we could \
            # set to int64 if needed
            if t.dtype != torch.int32:
                t = t.to(torch.int32)
            buf_name = f"selector_{i}_idx"
            self.register_buffer(buf_name, t, persistent=False)
            buf_names.append(buf_name)
        # CSR row_end_offsets as int32 buffer
        reo = row_end_offsets.contiguous()
        if reo.dtype != torch.int32:
            reo = reo.to(torch.int32)
        self.register_buffer("row_end_offsets", reo, persistent=False)

        self._ext = ext_module
        self._input_param_names = list(input_param_names)
        self._non_gather_value_inputs = list(non_gather_value_inputs)
        self._gather_value_inputs = list(gather_value_inputs)
        self._selector_buf_names = buf_names

    def __call__(self, *args):
        # Map incoming args by placeholder name (preserve contiguity)
        arg_map: Dict[str, torch.Tensor] = {
            name: arg.contiguous() for name, arg in zip(self._input_param_names, args)
        }
        device = next(iter(arg_map.values())).device

        ro = self.row_end_offsets
        if ro.device != device:
            ro = ro.to(device, non_blocking=True)

        gi_list: List[torch.Tensor] = []
        for buf_name in self._selector_buf_names:
            gi = getattr(self, buf_name)
            if gi.device != device:
                gi = gi.to(device, non_blocking=True)
            gi_list.append(gi)

        num_rows = int(ro.numel() - 1)

        # Determine gather source for num_cols
        if len(self._gather_value_inputs) > 0:
            gv_name = self._gather_value_inputs[0]
        else:
            gv_name = self._non_gather_value_inputs[0]
        num_cols = int(arg_map[gv_name].numel())

        call_args: List[torch.Tensor] = []
        # 1) non-gather value inputs
        for n in self._non_gather_value_inputs:
            call_args.append(arg_map[n])
        # 2) gather value inputs
        for n in self._gather_value_inputs:
            call_args.append(arg_map[n])
        # 3) selector index tensors
        for gi in gi_list:
            call_args.append(gi)
        # 4) CSR and sizes
        call_args.extend([ro, num_rows, num_cols])

        outs = self._ext.merged_spmv_launch(*call_args)
        if isinstance(outs, tuple):
            return list(outs)
        return [outs]


def dynamic_replace_submodule(
    compiled_eqn: nn.Module,
    ext,
    submodule_name: str = "easier5_select_reduce92",
) -> bool:
    """Replace a fused FX submodule with a dynamic CUDA wrapper.

    This detects input ordering, selector indices, and reducer metadata directly
    from the FX graph and its submodules, then constructs a wrapper that calls
    the CUDA extension following the binding parameter convention.
    """
    for (gm, node) in get_submodules(compiled_eqn, run_to_collect=False):
        if node.name != submodule_name:
            continue

        input_param_names = _iter_placeholders_in_order(gm)

        selector_modules, reducer_mod = _collect_selector_modules(gm)
        selector_qnames = [q for q, _ in selector_modules]

        # Gather selector idx tensors in graph order (excluding reducer if duplicated later)
        selector_idx_tensors: List[torch.Tensor] = []
        for qname, mod in selector_modules:
            selector_idx_tensors.append(mod.idx)

        # Reducer metadata: row ids and num_rows
        scatter_row_ids: Optional[torch.Tensor] = None
        inferred_num_rows: Optional[int] = None
        if reducer_mod is not None:
            scatter_row_ids = reducer_mod.idx
            if hasattr(reducer_mod, "n"):
                inferred_num_rows = int(reducer_mod.n)

        if scatter_row_ids is None:
            # Fallback: if no explicit reducer module found, try first selector's idx
            if len(selector_idx_tensors) == 0:
                raise RuntimeError("Failed to detect any selector indices inside the submodule")
            scatter_row_ids = selector_idx_tensors[0]

        row_end_offsets, num_rows = _coo_rows_to_csr_ptr(scatter_row_ids, inferred_num_rows)

        # Infer which inputs are gather value sources (targets of selector modules)
        gather_value_inputs = _infer_gather_value_inputs(gm, selector_qnames)
        # non-gather are the remaining placeholders in original order
        non_gather_value_inputs = [n for n in input_param_names if n not in gather_value_inputs]

        wrapper = _DynamicCudaWrapper(
            ext_module=ext,
            input_param_names=input_param_names,
            non_gather_value_inputs=non_gather_value_inputs,
            gather_value_inputs=gather_value_inputs,
            selector_idx_tensors=selector_idx_tensors,
            row_end_offsets=row_end_offsets,
        )

        setattr(compiled_eqn, submodule_name, wrapper)
        return True

    return False


