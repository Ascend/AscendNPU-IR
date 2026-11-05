// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -verify-diagnostics | FileCheck %s --check-prefix=NORMALIZE

// maxnumf/minnumf rewrite to `select(cmp | isnan(src1), src0, src1)`, so only
// src1 needs the NaN check. A non-NaN splat in that position drops both the
// check and the `vor`, leaving just the compare and the select.

// NORMALIZE-LABEL: func.func @maxnum_positive_zero_fill
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%arg0, %[[FILL]]
// NORMALIZE-NOT: binary_fn<vor>
// NORMALIZE: hfusion.select ins(%[[CMP]], %arg0, %[[FILL]]
func.func @maxnum_positive_zero_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %zero = arith.constant 0.0 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @maxnum_one_fill
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%arg0, %[[FILL]]
// NORMALIZE-NOT: binary_fn<vor>
// NORMALIZE: hfusion.select ins(%[[CMP]], %arg0, %[[FILL]]
func.func @maxnum_one_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %one = arith.constant 1.0 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%one : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A NaN splat in the src1 position can fire the check, so it is kept.

// NORMALIZE-LABEL: func.func @minnum_nan_fill_second
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%arg0, %[[FILL]]
// NORMALIZE: %[[OR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[CMP]],
// NORMALIZE: hfusion.select ins(%[[OR]], %arg0, %[[FILL]]
func.func @minnum_nan_fill_second(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %nan = arith.constant 0x7FC00000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%nan : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A splat in the src0 position says nothing about src1, so the check stays.

// NORMALIZE-LABEL: func.func @minnum_negative_zero_fill_first
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%[[FILL]], %arg0
// NORMALIZE: %[[OR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[CMP]],
// NORMALIZE: hfusion.select ins(%[[OR]], %[[FILL]], %arg0
func.func @minnum_negative_zero_fill_first(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %negative_zero = arith.constant 0x80000000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%negative_zero : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%fill, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// With both operands non-NaN splats the src1 splat alone drops the check, so
// a splat in the src0 position neither helps nor blocks it.

// NORMALIZE-LABEL: func.func @minnum_both_non_nan_fills
// NORMALIZE: %[[FILL0:.*]] = linalg.fill
// NORMALIZE: %[[FILL1:.*]] = linalg.fill
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%[[FILL0]], %[[FILL1]]
// NORMALIZE-NOT: binary_fn<vor>
// NORMALIZE: hfusion.select ins(%[[CMP]], %[[FILL0]], %[[FILL1]]
func.func @minnum_both_non_nan_fills() -> tensor<512xf32> {
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %fill0_init = tensor.empty() : tensor<512xf32>
  %fill0 = linalg.fill ins(%one : f32) outs(%fill0_init : tensor<512xf32>) -> tensor<512xf32>
  %fill1_init = tensor.empty() : tensor<512xf32>
  %fill1 = linalg.fill ins(%two : f32) outs(%fill1_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%fill0, %fill1 : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A scalar constant settles src1 the same way a fill does, so the check and
// the vor drop out. It also has to: hfusion.isnan takes a ranked tensor, so
// building one over a scalar operand would be malformed.

// NORMALIZE-LABEL: func.func @minnum_non_nan_scalar_src1
// NORMALIZE: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%arg0, %[[CST]]
// NORMALIZE-NOT: binary_fn<vor>
// NORMALIZE: hfusion.select ins(%[[CMP]], %arg0, %[[CST]]
func.func @minnum_non_nan_scalar_src1(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %two = arith.constant 2.0 : f32
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %two : tensor<512xf32>, f32) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A splat constant tensor is uniform without going through linalg.fill, so one
// element settles every lane and the check drops out as well.

// NORMALIZE-LABEL: func.func @minnum_non_nan_splat_const_src1
// NORMALIZE: %[[CST:.*]] = arith.constant dense<2.000000e+00>
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%arg0, %[[CST]]
// NORMALIZE-NOT: binary_fn<vor>
// NORMALIZE: hfusion.select ins(%[[CMP]], %arg0, %[[CST]]
func.func @minnum_non_nan_splat_const_src1(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %cst = arith.constant dense<2.0> : tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %cst : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// Uniform is not enough on its own: a splat of NaN keeps the check, because
// the result then has to be NaN rather than src1.

// NORMALIZE-LABEL: func.func @minnum_nan_splat_const_src1
// NORMALIZE: binary_fn<vor>
func.func @minnum_nan_splat_const_src1(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %cst = arith.constant dense<0x7FC00000> : tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %cst : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A constant tensor whose elements disagree is not settled by any one of them,
// so nothing is proven and the check stays.

// NORMALIZE-LABEL: func.func @minnum_non_splat_const_src1
// NORMALIZE: binary_fn<vor>
func.func @minnum_non_splat_const_src1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %out = tensor.empty() : tensor<4xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %cst : tensor<4xf32>, tensor<4xf32>) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// A conversion from an integer cannot produce NaN -- an out-of-range integer
// becomes an infinity -- so src1 needs no check even though it is a runtime
// value rather than a constant.

// NORMALIZE-LABEL: func.func @minnum_int_cast_src1
// NORMALIZE: %[[CAST:.*]] = hfusion.cast
// NORMALIZE: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%arg0, %[[CAST]]
// NORMALIZE-NOT: binary_fn<vor>
// NORMALIZE: hfusion.select ins(%[[CMP]], %arg0, %[[CAST]]
func.func @minnum_int_cast_src1(%arg0: tensor<512xf32>, %arg1: tensor<512xi32>) -> tensor<512xf32> {
  %e = tensor.empty() : tensor<512xf32>
  %f = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<512xi32>) outs(%e : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %f : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A float-to-float conversion does propagate NaN, so the check stays.

// NORMALIZE-LABEL: func.func @minnum_float_cast_src1
// NORMALIZE: binary_fn<vor>
func.func @minnum_float_cast_src1(%arg0: tensor<512xf32>, %arg1: tensor<512xf16>) -> tensor<512xf32> {
  %e = tensor.empty() : tensor<512xf32>
  %f = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<512xf16>) outs(%e : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %f : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}
