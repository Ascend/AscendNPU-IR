// RUN: bishengir-opt %s -split-input-file > %t.before
// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s

// Representative element widths: 8/16/32-bit elements fuse, while 1/64-bit
// elements retain the original chain. LoadOp owns concrete dtype admissibility.

// CHECK-LABEL: func.func @broadcast_i8(
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_i8(%src: memref<128xi8> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi8> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi8>
  memref.copy %src, %row : memref<128xi8> to memref<128xi8>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi8>
  %e = tensor.empty() : tensor<4x128xi8>
  %b = linalg.broadcast ins(%t : tensor<128xi8>) outs(%e : tensor<4x128xi8>) dimensions = [0]
  return %b : tensor<4x128xi8>
}
}

// -----

// CHECK-LABEL: func.func @broadcast_i16(
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_i16(%src: memref<128xi16> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi16> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi16>
  memref.copy %src, %row : memref<128xi16> to memref<128xi16>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi16>
  %e = tensor.empty() : tensor<4x128xi16>
  %b = linalg.broadcast ins(%t : tensor<128xi16>) outs(%e : tensor<4x128xi16>) dimensions = [0]
  return %b : tensor<4x128xi16>
}
}

// -----

// CHECK-LABEL: func.func @broadcast_i32(
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_i32(%src: memref<128xi32> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi32>
  memref.copy %src, %row : memref<128xi32> to memref<128xi32>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi32>
  %e = tensor.empty() : tensor<4x128xi32>
  %b = linalg.broadcast ins(%t : tensor<128xi32>) outs(%e : tensor<4x128xi32>) dimensions = [0]
  return %b : tensor<4x128xi32>
}
}

// -----

// CHECK-LABEL: func.func @broadcast_i1(
// CHECK-NEXT: %[[ROW:.*]] = memref.alloc() : memref<128xi1>
// CHECK-NEXT: memref.copy %{{.*}}, %[[ROW]] : memref<128xi1> to memref<128xi1>
// CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[ROW]] restrict writable : memref<128xi1>
// CHECK-NEXT: %[[E:.*]] = tensor.empty() : tensor<4x128xi1>
// CHECK-NEXT: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<128xi1>) outs(%[[E]] : tensor<4x128xi1>) dimensions = [0]
// CHECK-NEXT: return %[[B]] : tensor<4x128xi1>
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_i1(%src: memref<128xi1> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi1> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi1>
  memref.copy %src, %row : memref<128xi1> to memref<128xi1>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi1>
  %e = tensor.empty() : tensor<4x128xi1>
  %b = linalg.broadcast ins(%t : tensor<128xi1>) outs(%e : tensor<4x128xi1>) dimensions = [0]
  return %b : tensor<4x128xi1>
}
}

// -----

// CHECK-LABEL: func.func @broadcast_i64(
// CHECK-NEXT: %[[ROW:.*]] = memref.alloc() : memref<128xi64>
// CHECK-NEXT: memref.copy %{{.*}}, %[[ROW]] : memref<128xi64> to memref<128xi64>
// CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[ROW]] restrict writable : memref<128xi64>
// CHECK-NEXT: %[[E:.*]] = tensor.empty() : tensor<4x128xi64>
// CHECK-NEXT: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<128xi64>) outs(%[[E]] : tensor<4x128xi64>) dimensions = [0]
// CHECK-NEXT: return %[[B]] : tensor<4x128xi64>
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_i64(%src: memref<128xi64> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi64> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi64>
  memref.copy %src, %row : memref<128xi64> to memref<128xi64>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi64>
  %e = tensor.empty() : tensor<4x128xi64>
  %b = linalg.broadcast ins(%t : tensor<128xi64>) outs(%e : tensor<4x128xi64>) dimensions = [0]
  return %b : tensor<4x128xi64>
}
}
