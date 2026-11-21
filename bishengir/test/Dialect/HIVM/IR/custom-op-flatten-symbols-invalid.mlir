// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

// -----

// flatten_symbols present but flatten_ranks missing.
func.func @custom_flatten_symbols_without_ranks(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op Either flatten_ranks or flatten_symbols missing}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d",
       flatten_symbols = ["k_1d"]}
      "user.no_ranks"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

// flatten_ranks and flatten_symbols must pair up one to one.
func.func @custom_flatten_symbols_size_mismatch(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks and flatten_symbols size mismatch}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_3d",
       flatten_ranks = [1 : i64, 2 : i64],
       flatten_symbols = ["k_1d"]}
      "user.mismatch"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

// A rank must be a positive integer.
func.func @custom_flatten_symbols_zero_rank(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks must be an array of positive integers}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d",
       flatten_ranks = [0 : i64],
       flatten_symbols = ["k_0d"]}
      "user.zero_rank"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

// A rank must be an integer attribute.
func.func @custom_flatten_symbols_non_integer_rank(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks must be an array of positive integers}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d",
       flatten_ranks = ["1"],
       flatten_symbols = ["k_1d"]}
      "user.string_rank"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

// Each rank may only be declared once.
func.func @custom_flatten_symbols_duplicate_rank(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks contains duplicate rank 1}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d",
       flatten_ranks = [1 : i64, 1 : i64],
       flatten_symbols = ["k_1d", "k_1d_bis"]}
      "user.duplicate"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

// A symbol must be a non-empty string.
func.func @custom_flatten_symbols_empty_symbol(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_symbols must be an array of non-empty strings}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d",
       flatten_ranks = [1 : i64],
       flatten_symbols = [""]}
      "user.empty_symbol"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

// The same checks apply to hivm.hir.custom_macro.
func.func @custom_macro_flatten_symbols_non_string_symbol(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_symbols must be an array of non-empty strings}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_macro_2d",
       flatten_ranks = [1 : i64],
       flatten_symbols = [1 : i64]}
      "user.macro_int_symbol"
      ins(%arg0 : memref<2x2xf32>)
      outs(%arg1 : memref<2x2xf32>)
  return
}

// -----

func.func @custom_non_array_pair(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks and flatten_symbols must be arrays}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = "bad", flatten_symbols = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_non_array_ranks(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks and flatten_symbols must be arrays}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = "bad", flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_non_array_symbols(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks and flatten_symbols must be arrays}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [1 : i64], flatten_symbols = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_unpaired_non_array_ranks(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op Either flatten_ranks or flatten_symbols missing}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_unpaired_non_array_symbols(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op Either flatten_ranks or flatten_symbols missing}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_symbols = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_wide_truncating_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks ranks must fit in a signed 64-bit integer}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [18446744073709551617 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_wide_out_of_range_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks ranks must fit in a signed 64-bit integer}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775808 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_wide_negative_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks must be an array of positive integers}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [-18446744073709551616 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_mixed_width_duplicate(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks contains duplicate rank 1}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [1 : i64, 1 : i128], flatten_symbols = ["k_1d", "other"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_non_array_pair(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks and flatten_symbols must be arrays}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = "bad", flatten_symbols = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_non_array_ranks(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks and flatten_symbols must be arrays}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = "bad", flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_non_array_symbols(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks and flatten_symbols must be arrays}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [1 : i64], flatten_symbols = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_unpaired_non_array_ranks(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op Either flatten_ranks or flatten_symbols missing}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_unpaired_non_array_symbols(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op Either flatten_ranks or flatten_symbols missing}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_symbols = "bad"}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_wide_truncating_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks ranks must fit in a signed 64-bit integer}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [18446744073709551617 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_wide_out_of_range_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks ranks must fit in a signed 64-bit integer}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775808 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_wide_negative_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks must be an array of positive integers}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [-18446744073709551616 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_mixed_width_duplicate(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks contains duplicate rank 1}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [1 : i64, 1 : i128], flatten_symbols = ["k_1d", "other"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_duplicate_max_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks contains duplicate rank 9223372036854775807}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775807 : i64, 9223372036854775807 : i128],
       flatten_symbols = ["first", "duplicate"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_duplicate_max_minus_one_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom' op flatten_ranks contains duplicate rank 9223372036854775806}}
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775806 : i64, 9223372036854775806 : i128],
       flatten_symbols = ["first", "duplicate"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_duplicate_max_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks contains duplicate rank 9223372036854775807}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775807 : i64, 9223372036854775807 : i128],
       flatten_symbols = ["first", "duplicate"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

func.func @custom_macro_duplicate_max_minus_one_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  // expected-error @+1 {{'hivm.hir.custom_macro' op flatten_ranks contains duplicate rank 9223372036854775806}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775806 : i64, 9223372036854775806 : i128],
       flatten_symbols = ["first", "duplicate"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}
