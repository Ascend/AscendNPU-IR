// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -verify-diagnostics | FileCheck %s --check-prefix=NORMALIZE --implicit-check-not=hfusion.select

// NORMALIZE-LABEL: func.func @maxnum_positive_zero_fill(
// NORMALIZE-SAME: %[[ARG:.*]]: tensor<512xf32>
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[ARG]], %[[FILL]]
// NORMALIZE: hfusion.bitcast ins(%[[ARG]] : tensor<512xf32>)
// NORMALIZE: %[[IS_NAN:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// NORMALIZE: %[[RESULT:.*]] = hfusion.select ins(%[[IS_NAN]], %[[FILL]], %[[BASE]]
// NORMALIZE: return %[[RESULT]] : tensor<512xf32>
func.func @maxnum_positive_zero_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %zero = arith.constant 0.0 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @minnum_negative_zero_fill_first(
// NORMALIZE-SAME: %[[ARG:.*]]: tensor<512xf32>
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[FILL]], %[[ARG]]
// NORMALIZE: hfusion.bitcast ins(%[[ARG]] : tensor<512xf32>)
// NORMALIZE: %[[IS_NAN:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// NORMALIZE: %[[RESULT:.*]] = hfusion.select ins(%[[IS_NAN]], %[[FILL]], %[[BASE]]
// NORMALIZE: return %[[RESULT]] : tensor<512xf32>
func.func @minnum_negative_zero_fill_first(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %negative_zero = arith.constant 0x80000000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%negative_zero : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%fill, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @maxnum_one_fill(
// NORMALIZE-SAME: %[[ARG:.*]]: tensor<512xf32>
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[ARG]], %[[FILL]]
// NORMALIZE: hfusion.bitcast ins(%[[ARG]] : tensor<512xf32>)
// NORMALIZE: %[[IS_NAN:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// NORMALIZE: %[[RESULT:.*]] = hfusion.select ins(%[[IS_NAN]], %[[FILL]], %[[BASE]]
// NORMALIZE: return %[[RESULT]] : tensor<512xf32>
func.func @maxnum_one_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %one = arith.constant 1.0 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%one : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// A NaN fill must retain both checks, including the final selection of the
// original fill when the other operand is also NaN.
// NORMALIZE-LABEL: func.func @minnum_nan_fill(
// NORMALIZE-SAME: %[[ARG:.*]]: tensor<512xf32>
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[FILL]], %[[ARG]]
// NORMALIZE: %[[SELECT_FILL:.*]] = hfusion.select ins(%{{.*}}, %[[ARG]], %[[BASE]]
// NORMALIZE: %[[RESULT:.*]] = hfusion.select ins(%{{.*}}, %[[FILL]], %[[SELECT_FILL]]
// NORMALIZE: return %[[RESULT]] : tensor<512xf32>
func.func @minnum_nan_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %nan = arith.constant 0x7FC00000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%nan : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%fill, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @maxnum_nan_fill_rhs(
// NORMALIZE-SAME: %[[ARG:.*]]: tensor<512xf32>
// NORMALIZE: %[[FILL:.*]] = linalg.fill
// NORMALIZE: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[ARG]], %[[FILL]]
// NORMALIZE: %[[SELECT_ARG:.*]] = hfusion.select ins(%{{.*}}, %[[FILL]], %[[BASE]]
// NORMALIZE: %[[RESULT:.*]] = hfusion.select ins(%{{.*}}, %[[ARG]], %[[SELECT_ARG]]
// NORMALIZE: return %[[RESULT]] : tensor<512xf32>
func.func @maxnum_nan_fill_rhs(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %nan = arith.constant 0x7FC00000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%nan : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// Two known non-NaN fills need no NaN checks or selections, including infinity.
// NORMALIZE-LABEL: func.func @maxnum_non_nan_fills
// NORMALIZE-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// NORMALIZE-DAG: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// NORMALIZE: %[[INF_FILL:.*]] = linalg.fill ins(%[[INF]] : f32)
// NORMALIZE: %[[ZERO_FILL:.*]] = linalg.fill ins(%[[ZERO]] : f32)
// NORMALIZE-NOT: hfusion.bitcast
// NORMALIZE: %[[RESULT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[INF_FILL]], %[[ZERO_FILL]]
// NORMALIZE-NOT: hfusion.bitcast
// NORMALIZE: return %[[RESULT]] : tensor<512xf32>
func.func @maxnum_non_nan_fills() -> tensor<512xf32> {
  %inf = arith.constant 0x7F800000 : f32
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<512xf32>
  %inf_fill = linalg.fill ins(%inf : f32) outs(%init : tensor<512xf32>) -> tensor<512xf32>
  %zero_fill = linalg.fill ins(%zero : f32) outs(%init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%inf_fill, %zero_fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}
