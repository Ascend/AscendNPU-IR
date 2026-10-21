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
