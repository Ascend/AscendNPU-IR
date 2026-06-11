// RUN: bishengir-opt --one-shot-bufferize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @scope_single_tensor(
// CHECK: scope.return %{{.*}} : memref<128x128xf32>
func.func @scope_single_tensor(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = scope.scope : () -> tensor<128x128xf32> {
    %1 = tensor.empty() : tensor<128x128xf32>
    scope.return %1 : tensor<128x128xf32>
  }
  return %0 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func @scope_multi_tensor(
// CHECK: scope.return %{{.*}}, %{{.*}} : memref<128x128xf32>, memref<64xf32>
func.func @scope_multi_tensor(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<64xf32>) {
  %0, %1 = scope.scope : () -> (tensor<128x128xf32>, tensor<64xf32>) {
    %t1 = tensor.empty() : tensor<128x128xf32>
    %t2 = tensor.empty() : tensor<64xf32>
    scope.return %t1, %t2 : tensor<128x128xf32>, tensor<64xf32>
  }
  return %0, %1 : tensor<128x128xf32>, tensor<64xf32>
}

// -----

// CHECK-LABEL: func @scope_mixed_types(
// CHECK: scope.return %{{.*}}, %{{.*}} : memref<128xf32>, index
func.func @scope_mixed_types(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>) -> (tensor<128xf32>, index) {
  %c0 = arith.constant 0 : index
  %0, %1 = scope.scope : () -> (tensor<128xf32>, index) {
    %t = tensor.empty() : tensor<128xf32>
    scope.return %t, %c0 : tensor<128xf32>, index
  }
  return %0, %1 : tensor<128xf32>, index
}

// -----

// CHECK-LABEL: func @scope_hir_ops(
// CHECK: scope.return %{{.*}} : memref<128x128xf32>
func.func @scope_hir_ops(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128xf32> {
  %0 = scope.scope : () -> tensor<128x128xf32> {
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %empty1 = tensor.empty() : tensor<128x128xf32>
    %vmul = hivm.hir.vmul ins(%arg0, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%empty1 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
    %empty2 = tensor.empty() : tensor<128x128xf32>
    %vadd = hivm.hir.vadd ins(%vmul, %arg0 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%empty2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scope.return %vadd : tensor<128x128xf32>
  }
  return %0 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func @scope_empty(
// CHECK: scope.scope
// CHECK: scope.return
func.func @scope_empty() -> () {
  scope.scope : () -> () {
    scope.return
  }
  return
}
