// RUN: bishengir-opt %s -hivm-expose-memref-write-to-tensor -split-input-file -verify-diagnostics | FileCheck %s

// Test: a writable to_tensor whose memref was DPS-written before is exposed as
// copy{to_be_replaced} into a fresh tensor.empty. For a dynamic-typed
// to_tensor the empty must be created with dim sizes (a bare tensor.empty
// would be invalid IR), and the helper dim/empty must keep referencing the
// to_tensor result (retargeting them to the later copy breaks dominance).

// CHECK-LABEL: func.func @dynamic_to_tensor(
// CHECK: %[[T:.+]] = bufferization.to_tensor %{{.+}} restrict writable : memref<?x20x16x16xf16>
// CHECK: %[[DIM:.+]] = tensor.dim %[[T]], %{{.+}} : tensor<?x20x16x16xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?x20x16x16xf16>
// CHECK: %[[CP:.+]] = hivm.hir.copy ins(%[[T]] : tensor<?x20x16x16xf16>) outs(%[[EMPTY]] : tensor<?x20x16x16xf16>) {to_be_replaced}
// CHECK: hivm.hir.vadd ins(%[[CP]], %[[CP]] :
func.func @dynamic_to_tensor(%arg0: memref<?x20x16x16xf16>, %arg1: index) {
  %alloc = memref.alloc(%arg1) : memref<?x20x16x16xf16>
  hivm.hir.load ins(%arg0 : memref<?x20x16x16xf16>) outs(%alloc : memref<?x20x16x16xf16>)
  %t = bufferization.to_tensor %alloc restrict writable : memref<?x20x16x16xf16>
  %e = tensor.empty(%arg1) : tensor<?x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<?x20x16x16xf16>, tensor<?x20x16x16xf16>) outs(%e : tensor<?x20x16x16xf16>) -> tensor<?x20x16x16xf16>
  return
}

// -----

// Static shapes keep the original form (no tensor.dim needed).

// CHECK-LABEL: func.func @static_to_tensor(
// CHECK: %[[T:.+]] = bufferization.to_tensor %{{.+}} restrict writable : memref<5x20x16x16xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<5x20x16x16xf16>
// CHECK: hivm.hir.copy ins(%[[T]] : tensor<5x20x16x16xf16>) outs(%[[EMPTY]] : tensor<5x20x16x16xf16>) {to_be_replaced}
// CHECK-NOT: tensor.dim
func.func @static_to_tensor(%arg0: memref<5x20x16x16xf16>) {
  %alloc = memref.alloc() : memref<5x20x16x16xf16>
  hivm.hir.load ins(%arg0 : memref<5x20x16x16xf16>) outs(%alloc : memref<5x20x16x16xf16>)
  %t = bufferization.to_tensor %alloc restrict writable : memref<5x20x16x16xf16>
  %e = tensor.empty() : tensor<5x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<5x20x16x16xf16>, tensor<5x20x16x16xf16>) outs(%e : tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16>
  return
}

// -----

// Negative: no DPS write before to_tensor -> nothing is exposed.

// CHECK-LABEL: func.func @no_prior_write(
// CHECK-NOT: to_be_replaced
func.func @no_prior_write(%arg0: memref<5x20x16x16xf16>) {
  %alloc = memref.alloc() : memref<5x20x16x16xf16>
  %t = bufferization.to_tensor %alloc restrict writable : memref<5x20x16x16xf16>
  %e = tensor.empty() : tensor<5x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<5x20x16x16xf16>, tensor<5x20x16x16xf16>) outs(%e : tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16>
  return
}
