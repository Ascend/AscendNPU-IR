// RUN: bishengir-opt %s -hivm-decompose-alloc-tensor-copy -split-input-file | FileCheck %s

// All discardable attributes of the `alloc_tensor` (e.g., the `to_be_replaced`
// marker inserted by `normalize-to-tensor`) are carried over to the
// `hivm.hir.copy`. A dynamically shaped source gets a `tensor.dim` and a
// sized `alloc_tensor`.
// CHECK-LABEL: func.func @test_copy_attrs_dynamic_shape
func.func @test_copy_attrs_dynamic_shape(%arg0: tensor<?x20x16x16xf16>) -> tensor<?x20x16x16xf16> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {to_be_replaced} : tensor<?x20x16x16xf16>
  // CHECK: %[[DIM:.*]] = tensor.dim %arg0, %{{.*}} : tensor<?x20x16x16xf16>
  // CHECK-NEXT: %[[ALLOC_ATTRS:.*]] = bufferization.alloc_tensor(%[[DIM]]) : tensor<?x20x16x16xf16>
  // CHECK-NEXT: %[[CP_ATTRS:.*]] = hivm.hir.copy ins(%arg0 : tensor<?x20x16x16xf16>) outs(%[[ALLOC_ATTRS]] : tensor<?x20x16x16xf16>) {to_be_replaced} -> tensor<?x20x16x16xf16>
  // CHECK-NEXT: return %[[CP_ATTRS]] : tensor<?x20x16x16xf16>
  return %0 : tensor<?x20x16x16xf16>
}

// -----

// A statically shaped source keeps a plain `alloc_tensor` with no dynamic
// sizes and no `tensor.dim`.
// CHECK-LABEL: func.func @test_copy_attrs_static_shape
// CHECK-NOT: tensor.dim
func.func @test_copy_attrs_static_shape(%arg0: tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {to_be_replaced} : tensor<5x20x16x16xf16>
  // CHECK: %[[ALLOC_STATIC:.*]] = bufferization.alloc_tensor() : tensor<5x20x16x16xf16>
  // CHECK-NEXT: %[[CP_STATIC:.*]] = hivm.hir.copy ins(%arg0 : tensor<5x20x16x16xf16>) outs(%[[ALLOC_STATIC]] : tensor<5x20x16x16xf16>) {to_be_replaced} -> tensor<5x20x16x16xf16>
  // CHECK-NEXT: return %[[CP_STATIC]] : tensor<5x20x16x16xf16>
  return %0 : tensor<5x20x16x16xf16>
}
