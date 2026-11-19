// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s
// Driver cleanup may move constants; rejected live broadcast chains stay.

// CHECK-LABEL: func.func @negative_offset
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] :
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @negative_offset(%src: memref<32xf32, strided<[1], offset: -1>>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32, strided<[1], offset: -1>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}


// -----

// CHECK-LABEL: func.func @noncontiguous_tail
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] :
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @noncontiguous_tail(%src: memref<32xf32, strided<[2]>>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32, strided<[2]>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}


// -----

// CHECK-LABEL: func.func @unknown_stride
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] :
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @unknown_stride(%src: memref<32xf32, strided<[?]>>) -> tensor<4x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32, strided<[?]>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}


// -----

// CHECK-LABEL: func.func @aliased_row(
// CHECK-SAME: %[[SRC:[^:]+]]: memref<32xf32>)
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32>
// CHECK: hivm.hir.load ins(%[[SRC]] : memref<32xf32>) outs(%[[ROW]] : memref<32xf32>)
// CHECK: %[[READ:.*]] = memref.load %[[ROW]][
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[ROW]] restrict writable : memref<32xf32>
// CHECK: %[[BCAST:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>)
// CHECK: return %[[BCAST]], %[[READ]] : tensor<4x32xf32>, f32
module attributes {hacc.target = #hacc.target<"Ascend960">} {
// Keep the extra read live: a dead read may be removed before matching.
func.func @aliased_row(%src: memref<32xf32>) -> (tensor<4x32xf32>, f32) attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %c0 = arith.constant 0 : index
  %v = memref.load %row[%c0] : memref<32xf32>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b, %v : tensor<4x32xf32>, f32
}
}


// -----

// CHECK-LABEL: func.func @dynamic_output
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] :
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @dynamic_output(%src: memref<32xf32>, %m: index) -> tensor<?x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty(%m) : tensor<?x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<?x32xf32>) dimensions = [0]
  return %b : tensor<?x32xf32>
}
}


// -----

// CHECK-LABEL: func.func @singleton_overlap
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] :
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @singleton_overlap(%src: memref<1x32xf32, strided<[1, 1]>>) -> tensor<4x1x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<1x32xf32>
  hivm.hir.load ins(%src : memref<1x32xf32, strided<[1, 1]>>) outs(%row : memref<1x32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<1x32xf32>
  %e = tensor.empty() : tensor<4x1x32xf32>
  %b = linalg.broadcast ins(%t : tensor<1x32xf32>) outs(%e : tensor<4x1x32xf32>) dimensions = [0]
  return %b : tensor<4x1x32xf32>
}
}


// -----
// CHECK-LABEL: func.func @temporary_cbuf
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
// CHECK: memref.copy %{{.*}}, %[[ROW]]
// CHECK: bufferization.to_tensor %[[ROW]] restrict writable
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return %[[BCAST]] :
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
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return %[[BCAST]] :
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
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return %[[BCAST]] :
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
