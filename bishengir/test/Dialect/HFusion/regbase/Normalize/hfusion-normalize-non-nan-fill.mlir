// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -verify-diagnostics | FileCheck %s --check-prefix=NORMALIZE

// NORMALIZE-LABEL: func.func @maxnum_positive_zero_fill
// NORMALIZE: %[[POS_ZERO_FILL:.*]] = linalg.fill
// NORMALIZE: %[[MASKED_ARG0:.*]] = hfusion.select
// NORMALIZE-NOT: hfusion.select
// NORMALIZE: hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MASKED_ARG0]], %[[POS_ZERO_FILL]]
func.func @maxnum_positive_zero_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %zero = arith.constant 0.0 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @minnum_negative_zero_fill_first
// NORMALIZE: %[[NEG_ZERO_FILL:.*]] = linalg.fill
// NORMALIZE: %[[MASKED_ARG1:.*]] = hfusion.select
// NORMALIZE-NOT: hfusion.select
// NORMALIZE: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[NEG_ZERO_FILL]], %[[MASKED_ARG1]]
func.func @minnum_negative_zero_fill_first(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %negative_zero = arith.constant 0x80000000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%negative_zero : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%fill, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @maxnum_one_fill
// NORMALIZE: %[[ONE_FILL:.*]] = linalg.fill
// NORMALIZE: %[[MASKED_ARG2:.*]] = hfusion.select
// NORMALIZE-NOT: hfusion.select
// NORMALIZE: hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MASKED_ARG2]], %[[ONE_FILL]]
func.func @maxnum_one_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %one = arith.constant 1.0 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%one : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %fill : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// NORMALIZE-LABEL: func.func @minnum_nan_fill
// NORMALIZE-COUNT-2: hfusion.select
// NORMALIZE-NOT: hfusion.select
// NORMALIZE: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
func.func @minnum_nan_fill(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %nan = arith.constant 0x7FC00000 : f32
  %fill_init = tensor.empty() : tensor<512xf32>
  %fill = linalg.fill ins(%nan : f32) outs(%fill_init : tensor<512xf32>) -> tensor<512xf32>
  %out = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%fill, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}
