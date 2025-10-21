//---------------------------------------------------------------------
// CPU code generation
//---------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

#include "merged_spmv.h"

// --------------------------------------------------------------------
// Reference CPU implementation Aggregators
// --------------------------------------------------------------------
template <typename ValueT, typename OffsetT>
void SpmvGoldCPU_map(const ValueT *tensor_v, const ValueT *tensor_spm1,
                            //  const ValueT *tensor_spm2,
                             const OffsetT *tensor_v1_idx,
                            //  const OffsetT *tensor_v2_idx, 
                             int nv_dim, int ne1_dim, int ne2_dim, int nnz,
                             ValueT *map_1)
{
  for (OffsetT i = 0; i < nnz; ++i) {
    OffsetT selector_i = tensor_v1_idx[i];
    // OffsetT selector_j = tensor_v2_idx[i];
    // const ValueT *v_i = &tensor_v[selector_i * nv_dim];
    // const ValueT *v_j = &tensor_v[selector_j * nv_dim];
    const ValueT *spm_i = &tensor_spm1[i * ne1_dim];
    // const ValueT *spm_j = &tensor_spm2[i * ne2_dim];

    std::vector<ValueT> map_1_row(ne1_dim);
    // std::vector<ValueT> map_2_row(ne2_dim);
    for (int j = 0; j < ne1_dim; ++j)
      map_1_row[j] = spm_i[j] + spm_i[j];
    // for (int j = 0; j < ne2_dim; ++j)
    //   map_2_row[j] =  spm_j[j] + spm_j[j];

    for (int j = 0; j < ne1_dim; ++j)
      map_1[i * ne1_dim + j] = map_1_row[j];
    // for (int j = 0; j < ne2_dim; ++j)
    //   map_2[i * ne2_dim + j] = map_2_row[j];

    // for (int j = 0; j < ne1_dim; ++j)
    //   aggregator_1[j] += map_1_row[j];
    // for (int j = 0; j < ne2_dim; ++j)
    //   aggregator_2[j] += map_2_row[j];
  }
}

// --------------------------------------------------------------------
// Deterministic data generation for testing
// --------------------------------------------------------------------
template <typename ValueT, typename OffsetT>
void GenerateRandomSystem(int seed, int num_rows, int num_cols, int nnz,
                          std::vector<OffsetT> &row_end_offsets,
                          std::vector<OffsetT> &selector_1,
                          // std::vector<OffsetT> &selector_2,
                          std::vector<ValueT> &vector_x, // len num_cols * 2
                          std::vector<ValueT> &spm_1,    // len num_nnz * 2
                          std::vector<ValueT> &spm_2)    // len num_nnz * 6
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> col_dist(0, num_cols - 1);
  std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
  std::uniform_int_distribution<int> offset_dist(0, nnz - 1);

  // Build row_end_offsets (cumulative non-zeros up to and including each row)
  row_end_offsets.resize(num_rows + 1);
  // The offset array must start with 0, end with args.ne, and be sorted in
  // non-decreasing order.
  row_end_offsets[0] = 0;
  row_end_offsets[num_rows] = nnz;
  if (num_rows > 1) {
    // Generate (num_rows - 1) random offsets between 0 and nnz
    std::vector<OffsetT> temp_offsets(num_rows - 1);
    for (int i = 0; i < num_rows - 1; ++i) {
      temp_offsets[i] = static_cast<OffsetT>(offset_dist(rng));
    }
    std::sort(temp_offsets.begin(), temp_offsets.end());
    for (int i = 0; i < num_rows - 1; ++i) {
      row_end_offsets[i + 1] = temp_offsets[i];
    }
  }

  selector_1.resize(nnz);
  // selector_2.resize(nnz);
  for (int i = 0; i < nnz; ++i) {
    // Ensure selectors are within valid column range [0, num_cols-1]
    selector_1[i] = static_cast<OffsetT>(col_dist(rng));
    // selector_2[i] = static_cast<OffsetT>(col_dist(rng));
  }

  vector_x.resize(num_cols);
  for (int c = 0; c < num_cols; ++c)
    vector_x[c] = static_cast<ValueT>(val_dist(rng));

  spm_1.resize(nnz);
  for (int i = 0; i < nnz; ++i)
    spm_1[i] = static_cast<ValueT>(val_dist(rng));

  spm_2.resize(nnz);
  for (int i = 0; i < nnz; ++i)
    spm_2[i] = static_cast<ValueT>(val_dist(rng));
}

// --------------------------------------------------------------------
// Test harness: run OmpMergeSystem and compare with reference
// --------------------------------------------------------------------
template <typename ValueT, typename OffsetT>
bool VerifyOmpMergeSystem(int seed, int num_rows, int num_cols, int nnz,
                          bool verbose = false, bool verbose2 = false) {
  std::vector<OffsetT> row_end_offsets;
  std::vector<OffsetT> selector_1;
  std::vector<OffsetT> selector_2;
  std::vector<ValueT> vector_x;
  std::vector<ValueT> spm_1;
  std::vector<ValueT> spm_2;

  GenerateRandomSystem<ValueT, OffsetT>(seed, num_rows, num_cols, nnz,
                                        row_end_offsets, selector_1, // selector_2,
                                        vector_x, spm_1, spm_2);

  const int nv_dim = 2, ne1_dim = 2, ne2_dim = 6;

  if (verbose) {
    std::cout << "Generated system: " << num_rows << " rows, " << num_cols
              << " cols, " << nnz << " nnz\n";
  }

  // Outputs
  std::vector<ValueT> out_map1_ref(nnz * ne1_dim);
  std::vector<ValueT> out_map2_ref(nnz * ne2_dim);


  std::vector<ValueT> out_agg1_ref(ne1_dim);
  std::vector<ValueT> out_agg2_ref(ne2_dim);
  SpmvGoldCPU_map<ValueT, OffsetT>(
      vector_x.data(), 
      spm_1.data(),
      //  spm_2.data(),
       selector_1.data(),
      //  selector_2.data(),
        nv_dim, ne1_dim, ne2_dim, nnz, 
      //  out_agg2_ref.data(),
       out_map1_ref.data());

  std::vector<ValueT> out_map1(nnz * ne1_dim);
  std::vector<ValueT> out_map2(nnz * ne2_dim);


  int threads = omp_get_max_threads();

  OmpMergeSystem<ValueT, OffsetT>(
      threads,
      spm_1.data(),
      //  spm_2.data(),
       out_map1.data(),
      //  out_map2.data(),
      num_rows, nnz);

  auto almost_equal = [](ValueT a, ValueT b) {
    ValueT eps = static_cast<ValueT>(1e-5);
    return std::abs(a - b) <= eps;
  };

    if (verbose2)
    {
      // print out_map1_ref
      std::cout << "map1_ref: ";
      for (size_t i = 0; i < out_map1_ref.size(); ++i)
        std::cout << out_map1_ref[i] << " ";
      std::cout << std::endl;
      // print out_map1
      std::cout << "map1: ";
      for (size_t i = 0; i < out_map1.size(); ++i)
        std::cout << out_map1[i] << " ";
      std::cout << std::endl;
      // print out_map2_ref
      std::cout << "map2_ref: ";
      // for (size_t i = 0; i < out_map2_ref.size(); ++i)
      //   std::cout << out_map2_ref[i] << " ";
      // std::cout << std::endl;
      // // print out_map2
      // std::cout << "map2: ";
      // for (size_t i = 0; i < out_map2.size(); ++i)
      //   std::cout << out_map2[i] << " ";
      // std::cout << std::endl;
    }

  bool ok = true;
  for (size_t i = 0; i < out_map1.size(); ++i)
    if (!almost_equal(out_map1[i], out_map1_ref[i])) {
      ok = false;
      if (verbose)
        std::cout << "map1 mismatch at " << i << "\n";
      break;
    }
  // for (size_t i = 0; i < out_map2.size() && ok; ++i)
  //   if (!almost_equal(out_map2[i], out_map2_ref[i])) {
  //     ok = false;
  //     if (verbose)
  //       std::cout << "map2 mismatch at " << i << "\n";
  //     break;
  //   }

  if (verbose)
    std::cout << (ok ? "OmpMergeSystem verification PASSED\n"
                     : "OmpMergeSystem verification FAILED\n");
  return ok;
}

// Convenience quick test
inline bool RunOmpMergeSystemQuickTest() {
  return VerifyOmpMergeSystem<float, int>(/*seed=*/123, /*rows=*/8, /*cols=*/16,
                                          /*nnz=*/20,
                                          /*verbose=*/true);
}

// Main function with command-line argument parsing
int main(int argc, char **argv) {
  // Default parameters
  bool use_double = false;
  int num_rows = 1231;
  int num_cols = 5435;
  int nnz = 1231432;
  int seed = 123;
  bool verbose = true;
  bool verbose2 = false;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--double" || arg == "-d") {
      use_double = true;
    } else if (arg == "--float" || arg == "-f") {
      use_double = false;
    } else if (arg == "--rows" || arg == "-r") {
      if (i + 1 < argc) {
        num_rows = std::atoi(argv[++i]);
      }
    } else if (arg == "--cols" || arg == "-c") {
      if (i + 1 < argc) {
        num_cols = std::atoi(argv[++i]);
      }
    } else if (arg == "--nnz" || arg == "-n") {
      if (i + 1 < argc) {
        nnz = std::atoi(argv[++i]);
      }
    } else if (arg == "--seed" || arg == "-s") {
      if (i + 1 < argc) {
        seed = std::atoi(argv[++i]);
      }
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--verbose2" || arg == "-v2") {
      verbose2 = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout
          << "  --double, -d         Use double precision (default: float)\n";
      std::cout << "  --float, -f          Use single precision\n";
      std::cout << "  --rows, -r NUM       Number of rows (default: 8)\n";
      std::cout << "  --cols, -c NUM       Number of columns (default: 16)\n";
      std::cout << "  --nnz, -n NUM        Target number of non-zeros "
                   "(default: 32)\n";
      std::cout << "  --seed, -s NUM       Random seed (default: 123)\n";
      std::cout << "  --verbose, -v        Increase output verbosity\n";
      std::cout << "  --verbose2, -v2        Increase output verbosity\n";
      std::cout << "  --help, -h           Show this help message\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::cerr << "Use --help for usage information.\n";
      return 1;
    }
  }

  if (verbose) {
    std::cout << "=== OmpMergeSystem Test ===\n";
    std::cout << "Precision: " << (use_double ? "double" : "float") << "\n";
    std::cout << "Matrix size: " << num_rows << " x " << num_cols << "\n";
    std::cout << "Target nnz: " << nnz << "\n";
    std::cout << "Seed: " << seed << "\n";
    std::cout << "Threads: " << omp_get_max_threads() << "\n\n";
  }

  bool success = false;

  if (use_double) {
    success = VerifyOmpMergeSystem<double, int>(seed, num_rows, num_cols, nnz,
                                                verbose, verbose2);
  } else {
    success = VerifyOmpMergeSystem<float, int>(seed, num_rows, num_cols, nnz,
                                               verbose, verbose2);
  }

  if (verbose) {
    std::cout << "\n=== Test Result ===\n";
    std::cout << (success ? "PASSED" : "FAILED") << "\n";
  }

  return success ? 0 : 1;
}