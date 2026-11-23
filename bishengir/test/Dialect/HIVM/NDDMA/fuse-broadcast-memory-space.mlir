// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s

// CHECK-LABEL: func.func @temporary_cbuf
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
// CHECK: memref.copy %{{.*}}, %[[ROW]]
// CHECK: bufferization.to_tensor %[[ROW]] restrict writable
// CHECK: linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @temporary_cbuf(%src: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
  memref.copy %src, %row : memref<32xf32, #hivm.address_space<gm>> to memref<32xf32, #hivm.address_space<cbuf>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32, #hivm.address_space<cbuf>>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @temporary_cc
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<cc>>
// CHECK: memref.copy %{{.*}}, %[[ROW]]
// CHECK: bufferization.to_tensor %[[ROW]] restrict writable
// CHECK: linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @temporary_cc(%src: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32, #hivm.address_space<cc>>
  memref.copy %src, %row : memref<32xf32, #hivm.address_space<gm>> to memref<32xf32, #hivm.address_space<cc>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32, #hivm.address_space<cc>>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @temporary_ssbuf
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<ssbuf>>
// CHECK: memref.copy %{{.*}}, %[[ROW]]
// CHECK: bufferization.to_tensor %[[ROW]] restrict writable
// CHECK: linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @temporary_ssbuf(%src: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32, #hivm.address_space<ssbuf>>
  memref.copy %src, %row : memref<32xf32, #hivm.address_space<gm>> to memref<32xf32, #hivm.address_space<ssbuf>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32, #hivm.address_space<ssbuf>>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @temporary_ub
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @temporary_ub(%src: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  memref.copy %src, %row : memref<32xf32, #hivm.address_space<gm>> to memref<32xf32, #hivm.address_space<ub>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32, #hivm.address_space<ub>>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @temporary_unclassified
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @temporary_unclassified(%src: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32>
  memref.copy %src, %row : memref<32xf32, #hivm.address_space<gm>> to memref<32xf32>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}
