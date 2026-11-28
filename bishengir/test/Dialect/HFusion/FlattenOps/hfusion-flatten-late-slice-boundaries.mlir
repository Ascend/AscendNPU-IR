// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" | FileCheck %s
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false use-preorder-walk-traversal=false" | FileCheck %s

// Both visitation orders must preserve boundaries introduced by another user
// of the source, even when the dimensions at that boundary are dropped.
// CHECK-LABEL: func.func @slice_before_transpose
// CHECK: %[[SLICE:.*]] = tensor.extract_slice {{.*}} : tensor<2x1x2xi32> to tensor<2x2xi32>
// CHECK: return %[[SLICE]],
func.func @slice_before_transpose(%arg: tensor<2x1x1x2xi32>) -> (tensor<2x2xi32>, tensor<1x2x2x1xi32>) {
  %slice = tensor.extract_slice %arg[0, 0, 0, 0] [2, 1, 1, 2] [1, 1, 1, 1] : tensor<2x1x1x2xi32> to tensor<2x2xi32>
  %empty = tensor.empty() : tensor<1x2x2x1xi32>
  %transpose = linalg.transpose ins(%arg : tensor<2x1x1x2xi32>) outs(%empty : tensor<1x2x2x1xi32>) permutation = [2, 3, 0, 1]
  return %slice, %transpose : tensor<2x2xi32>, tensor<1x2x2x1xi32>
}

// CHECK-LABEL: func.func @transpose_before_slice
// CHECK: %[[SLICE:.*]] = tensor.extract_slice {{.*}} : tensor<2x1x2xi32> to tensor<2x2xi32>
// CHECK: return %[[SLICE]],
func.func @transpose_before_slice(%arg: tensor<2x1x1x2xi32>) -> (tensor<2x2xi32>, tensor<1x2x2x1xi32>) {
  %empty = tensor.empty() : tensor<1x2x2x1xi32>
  %transpose = linalg.transpose ins(%arg : tensor<2x1x1x2xi32>) outs(%empty : tensor<1x2x2x1xi32>) permutation = [2, 3, 0, 1]
  %slice = tensor.extract_slice %arg[0, 0, 0, 0] [2, 1, 1, 2] [1, 1, 1, 1] : tensor<2x1x1x2xi32> to tensor<2x2xi32>
  return %slice, %transpose : tensor<2x2xi32>, tensor<1x2x2x1xi32>
}
