// RUN: bishengir-opt -hfusion-cache-io-for-return-arg %s -cse -canonicalize-ext -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_single_reshape
// CHECK-SAME: %[[arg0:.*]]: tensor<16x16xf32> {hacc.cached_io}
// CHECK-SAME: -> (tensor<64x4xf32>, tensor<16x16xf32> {hacc.cached_io})
// CHECK: %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK: %[[load:.*]] = hfusion.load ins(%[[expanded]]
// CHECK: %[[store:.*]] = hfusion.store {{.*}} ins(%[[load]]
// CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[store]]
// CHECK: return %{{.*}}, %[[collapsed]] : tensor<64x4xf32>, tensor<16x16xf32>
func.func @test_single_reshape(%arg0: tensor<16x16xf32>) -> (tensor<64x4xf32>, tensor<16x16xf32>) {
  %asu = tensor.empty() : tensor<16x4x4xf32>
  %v1 = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [16, 4, 4] : tensor<16x16xf32> into tensor<16x4x4xf32>
  %v2 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%v1 : tensor<16x4x4xf32>) outs(%asu : tensor <16x4x4xf32>) -> tensor<16x4x4xf32>
  %v3 = tensor.collapse_shape %v2 [[0, 1], [2]] : tensor<16x4x4xf32> into tensor<64x4xf32>
  return %v3, %arg0 : tensor<64x4xf32>, tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_multiple_reshape(
// CHECK: %[[expanded:.*]] = tensor.expand_shape %arg0 {{\[}}[0], [1, 2]] output_shape [16, 8, 2]
// CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[expanded]] {{\[}}[0, 1], [2]]
// CHECK: %[[load:.*]] = hfusion.load ins(%[[collapsed]]
// CHECK: %[[store:.*]] = hfusion.store {{.*}} ins(%[[load]]
// CHECK: %[[expanded_1:.*]] = tensor.expand_shape %[[store]]
// CHECK: %[[collapse:.*]] = tensor.collapse_shape %[[expanded_1]]
// CHECK: return %{{.*}}, %[[collapse]] : tensor<2x64x2xf32>, tensor<16x16xf32>
module {
  func.func @test_multiple_reshape(%arg0: tensor<16x16xf32>) -> (tensor<2x64x2xf32>, tensor<16x16xf32>) {
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [16, 8, 2] : tensor<16x16xf32> into tensor<16x8x2xf32>
    %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] : tensor<16x8x2xf32> into tensor<128x2xf32>
    %0 = tensor.empty() : tensor<128x2xf32>
    %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%collapsed : tensor<128x2xf32>) outs(%0 : tensor<128x2xf32>) -> tensor<128x2xf32>
    %expanded_0 = tensor.expand_shape %1 [[0, 1], [2]] output_shape [2, 64, 2] : tensor<128x2xf32> into tensor<2x64x2xf32>
    return %expanded_0, %arg0 : tensor<2x64x2xf32>, tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_return_single_argument(
// CHECK-NOT: hfusion.load
// CHECK-NOT: hfusion.store
func.func @test_return_single_argument(%arg0: tensor<?x32x128xbf16>) -> tensor<?x32x128xbf16> {
  return %arg0 : tensor<?x32x128xbf16>
}