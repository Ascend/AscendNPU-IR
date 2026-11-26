// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s
module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
// CHECK-LABEL: func.func @a3(
// CHECK-NOT: broadcast_axes
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[T:.*]] = bufferization.to_tensor
// CHECK: %[[E:.*]] = tensor.empty()
// CHECK: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>) outs(%[[E]] : tensor<4x32xf32>) dimensions = [0]
// CHECK-NEXT: return %[[B]] : tensor<4x32xf32>
func.func @a3(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950">} {
// CHECK-LABEL: func.func @a5(
// CHECK-NOT: broadcast_axes
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[T:.*]] = bufferization.to_tensor
// CHECK: %[[E:.*]] = tensor.empty()
// CHECK: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>) outs(%[[E]] : tensor<4x32xf32>) dimensions = [0]
// CHECK-NEXT: return %[[B]] : tensor<4x32xf32>
func.func @a5(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
module {
// CHECK-LABEL: func.func @absent(
// CHECK-NOT: broadcast_axes
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[T:.*]] = bufferization.to_tensor
// CHECK: %[[E:.*]] = tensor.empty()
// CHECK: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>) outs(%[[E]] : tensor<4x32xf32>) dimensions = [0]
// CHECK-NEXT: return %[[B]] : tensor<4x32xf32>
func.func @absent(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
module attributes {hacc.target = #hacc.target<"Ascend950">} {
// CHECK-LABEL: func.func @nested_a5(
// CHECK-NOT: broadcast_axes
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[T:.*]] = bufferization.to_tensor
// CHECK: %[[E:.*]] = tensor.empty()
// CHECK: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>) outs(%[[E]] : tensor<4x32xf32>) dimensions = [0]
// CHECK-NEXT: return %[[B]] : tensor<4x32xf32>
func.func @nested_a5(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
module attributes {hacc.target = #hacc.target<"Ascend960">} {
// CHECK-LABEL: func.func @nested_a6(
// CHECK-NOT: linalg.broadcast
// CHECK: hivm.hir.load {{.*}}broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: %[[R:.*]] = bufferization.to_tensor
// CHECK-NEXT: return %[[R]] : tensor<4x32xf32>
func.func @nested_a6(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

}

// -----
// Folding live arithmetic is not part of broadcast fusion; dead ops may go.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  // CHECK-LABEL: func.func @no_fold_live_arithmetic(
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[SUM:.*]] = arith.addi %[[ZERO]], %[[ONE]] : index
  // CHECK-NOT: arith.muli
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: linalg.broadcast
  // CHECK: return %[[SUM]] : index
  func.func @no_fold_live_arithmetic(%row: tensor<32xf32>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %fold = arith.addi %c0, %c1 : index
    %dead = arith.muli %fold, %c1 : index
    %empty = tensor.empty() : tensor<4x32xf32>
    %b = linalg.broadcast ins(%row : tensor<32xf32>) outs(%empty : tensor<4x32xf32>) dimensions = [0]
    return %fold : index
  }
}
