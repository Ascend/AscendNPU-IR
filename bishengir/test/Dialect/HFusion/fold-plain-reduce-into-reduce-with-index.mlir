// RUN: bishengir-opt %s -canonicalize -split-input-file | FileCheck %s
//
// Triton lowers `chunk_max = tl.max(x); chunk_arg = tl.argmax(x)` into a plain
// value reduction plus a reduce_with_index that produces the same value:
//
//   %v    = linalg.reduce (maximumf) ins(%src) outs(fill -inf)     // value used
//   %wi:2 = hfusion.reduce_with_index {already_initialize_init} <max>
//             ins(%src, %idx) outs(...)                            // index used
//
// `%wi#0` already is that value, so `FoldPlainReduceIntoReduceWithIndex` drops
// the plain reduction and the kernel keeps a single full-width pass over the
// tile.
//
// The reduce_with_index has to be written in the generic form here: its custom
// syntax carries no region, and the pattern needs the combine region to compare
// combiners and to check that the reduction init is the identity.  The
// `already_initialize_init` discardable attribute therefore goes in the generic
// attribute dictionary, after the region list.

// CHECK-LABEL: func.func @fold_plain_max_reduce_into_reduce_with_index
// CHECK-NOT:     linalg.reduce
// CHECK:         %[[WI:.*]]:2 = hfusion.reduce_with_index
// CHECK-SAME:    <max> ins(%[[SRC:.*]], %[[IDX:.*]] : tensor<32x64xf32>, tensor<32x64xi32>)
// CHECK:         return %[[WI]]#0, %[[WI]]#1
func.func @fold_plain_max_reduce_into_reduce_with_index(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maxnumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%neg_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maxnumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// The plain reduction may also be emitted before the reduce_with_index (that is
// the order Triton produces).  Everything involved is side-effect free, so the
// pattern hoists the reduce_with_index -- and the pure producers of its operands
// -- in front of the plain reduction and folds it the same way.

// CHECK-LABEL: func.func @fold_when_plain_reduce_comes_first
// CHECK-NOT:     linalg.reduce
// CHECK:         %[[WI:.*]]:2 = hfusion.reduce_with_index
// CHECK:         return %[[WI]]#0, %[[WI]]#1
func.func @fold_when_plain_reduce_comes_first(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  // The buffers the hoist may move are created up front; only the fills and the
  // reduce_with_index itself sit after the plain reduction.
  %v_init = tensor.empty() : tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%neg_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maxnumf %in, %init : f32
        linalg.yield %n : f32
      }

  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maxnumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// The min form folds the same way; its identity init is +inf.

// CHECK-LABEL: func.func @fold_plain_min_reduce_into_reduce_with_index
// CHECK-NOT:     linalg.reduce
// CHECK:         %[[WI:.*]]:2 = hfusion.reduce_with_index
// CHECK-SAME:    <min>
// CHECK:         return %[[WI]]#0, %[[WI]]#1
func.func @fold_plain_min_reduce_into_reduce_with_index(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %pos_inf = arith.constant 0x7F800000 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%pos_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<min>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.minnumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%pos_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.minnumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// Negative: the plain reduction does not start from the combiner's identity (a
// zero fill instead of -inf), so it does not compute the value the
// reduce_with_index produces and both reductions must stay.

// CHECK-LABEL: func.func @no_fold_when_init_is_not_identity
// CHECK-DAG:     linalg.reduce
// CHECK-DAG:     hfusion.reduce_with_index
func.func @no_fold_when_init_is_not_identity(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maxnumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%zero : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maxnumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// Negative: a different combiner (a sum) is not the value that the
// reduce_with_index already produces.

// CHECK-LABEL: func.func @no_fold_for_a_different_combiner
// CHECK-DAG:     linalg.reduce
// CHECK-DAG:     hfusion.reduce_with_index
func.func @no_fold_for_a_different_combiner(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maxnumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%zero : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.addf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// The pipeline reaches this point with `maximumf` (the NaN normalization
// rewrites the plain reduction's `maxnumf`), so the fold has to fire for that
// combiner as well.

// CHECK-LABEL: func.func @fold_plain_max_reduce_with_maximumf
// CHECK-NOT:     linalg.reduce
// CHECK:         %[[WI:.*]]:2 = hfusion.reduce_with_index
// CHECK:         return %[[WI]]#0, %[[WI]]#1
func.func @fold_plain_max_reduce_with_maximumf(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maximumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%neg_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maximumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// Negative: the body post-processes the reduced value before yielding it, so
// `%wi#0` is no longer the max and must not replace the plain reduction.

// CHECK-LABEL: func.func @no_fold_when_value_is_post_processed
// CHECK-DAG:     linalg.reduce
// CHECK-DAG:     hfusion.reduce_with_index
func.func @no_fold_when_value_is_post_processed(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %one = arith.constant 1.000000e+00 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maxnumf %a, %b : f32
    %bumped = arith.addf %m, %one : f32
    "linalg.yield"(%bumped, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%neg_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maxnumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// Negative: an op is fused between the reduced element and the combiner, so the
// body no longer reduces the source directly.

// CHECK-LABEL: func.func @no_fold_when_combiner_operand_is_fused
// CHECK-DAG:     linalg.reduce
// CHECK-DAG:     hfusion.reduce_with_index
func.func @no_fold_when_combiner_operand_is_fused(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<max>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %neg = arith.negf %a : f32
    %m = arith.maxnumf %neg, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%neg_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maxnumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}

// -----

// Negative: the declared reduce_kind disagrees with the combiner found in the
// body, so the value result cannot be trusted to be a max.

// CHECK-LABEL: func.func @no_fold_when_body_disagrees_with_reduce_kind
// CHECK-DAG:     linalg.reduce
// CHECK-DAG:     hfusion.reduce_with_index
func.func @no_fold_when_body_disagrees_with_reduce_kind(
    %src: tensor<32x64xf32>, %idx: tensor<32x64xi32>
) -> (tensor<32xf32>, tensor<32xi32>) {
  %neg_inf = arith.constant 0xFF800000 : f32
  %v_init = tensor.empty() : tensor<32xf32>
  %v_fill = linalg.fill ins(%neg_inf : f32) outs(%v_init : tensor<32xf32>) -> tensor<32xf32>
  %i_init = tensor.empty() : tensor<32xi32>
  %wi:2 = "hfusion.reduce_with_index"(%src, %idx, %v_fill, %i_init)
      <{reduce_kind = #hfusion.reduce_with_index_kind<min>, unsigned_src = false,
         tie_break_left = true, dimensions = array<i64: 1>,
         operandSegmentSizes = array<i32: 2, 2>}> ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %m = arith.maxnumf %a, %b : f32
    "linalg.yield"(%m, %ai) : (f32, i32) -> ()
  }) {already_initialize_init}
      : (tensor<32x64xf32>, tensor<32x64xi32>, tensor<32xf32>, tensor<32xi32>)
        -> (tensor<32xf32>, tensor<32xi32>)

  %plain_init = tensor.empty() : tensor<32xf32>
  %plain_fill = linalg.fill ins(%neg_inf : f32) outs(%plain_init : tensor<32xf32>) -> tensor<32xf32>
  %plain = linalg.reduce ins(%src : tensor<32x64xf32>) outs(%plain_fill : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %n = arith.maxnumf %in, %init : f32
        linalg.yield %n : f32
      }
  return %plain, %wi#1 : tensor<32xf32>, tensor<32xi32>
}
