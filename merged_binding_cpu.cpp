#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "merged_spmv.h" // generated CPU header

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_cpu_typed(
    torch::Tensor bpoints,
    torch::Tensor bp_1,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(bpoints.device().is_cpu(), "bpoints must be a CPU tensor");
  TORCH_CHECK(bpoints.is_contiguous(), "bpoints must be contiguous");
  TORCH_CHECK(bp_1.device().is_cpu(), "bp_1 must be a CPU tensor");
  TORCH_CHECK(bp_1.is_contiguous(), "bp_1 must be contiguous");
  TORCH_CHECK(row_end_offsets.device().is_cpu(), "row_end_offsets must be a CPU tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(bpoints.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "bpoints dtype mismatch");
  TORCH_CHECK(bp_1.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "bp_1 dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = num_cols;
  TORCH_CHECK(ne > 0, "selector index must be non-empty (ne > 0)");


  auto options_val = torch::TensorOptions().dtype(bpoints.scalar_type()).device(bpoints.device());


  // Build raw pointers in OmpMergeSystem parameter order




  const int num_threads = at::get_num_threads();

  // Invoke generated kernel
  OmpMergeSystem<ValueT, OffsetT>(
    num_threads, reinterpret_cast<ValueT*>(bpoints.data_ptr()), reinterpret_cast<ValueT*>(bp_1.data_ptr()), static_cast<int>(num_rows), static_cast<int>(ne));


  
}

void merged_spmv_launch_bind_cpu(
    torch::Tensor bpoints,
    torch::Tensor bp_1,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(bpoints.device().is_cpu(), "CPU device required");
  switch (bpoints.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_cpu_typed<float, int>(bpoints, bp_1, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_cpu_typed<double, int>(bpoints, bp_1, row_end_offsets, num_rows, num_cols);
        case torch::kLong:
      return merged_spmv_launch_cpu_typed<long, int>(bpoints, bp_1, row_end_offsets, num_rows, num_cols);

    default:
      TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind_cpu,
      "Run merged SpMV CPU kernel"
      , py::arg("bpoints")
      , py::arg("bp_1")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


