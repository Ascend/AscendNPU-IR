// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s
// Driver DCE and constant CSE are allowed; check the live non-target chains.
// No padding/fill optimization and no cross-call signature changes.
// CHECK-LABEL: func.func private @fused(
// CHECK-SAME: tensor<32xf32>
// CHECK: linalg.broadcast
// CHECK: return
// CHECK-LABEL: func.func @entry
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: call @fused
// CHECK-SAME: (tensor<32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func private @fused(%row: tensor<32xf32>, %init: tensor<32x32xf32>, %matrix: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %b = linalg.broadcast ins(%row : tensor<32xf32>) outs(%init : tensor<32x32xf32>) dimensions = [0]
    %r = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%matrix, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
  func.func @entry(%src: memref<32xf32> {tt.tensor_kind = 0 : i32}, %matrix: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %row = memref.alloc() : memref<32xf32>
    memref.copy %src, %row : memref<32xf32> to memref<32xf32>
    %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
    %init = tensor.empty() : tensor<32x32xf32>
    %r = call @fused(%t, %init, %matrix) : (tensor<32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}


// -----
// CHECK-LABEL: func.func @dynamic_pad(
// CHECK-SAME: %[[SRC:[^:]+]]: memref<32xf32>{{.*}}, %[[LEN:[^:]+]]: index, %[[PAD:[^:]+]]: f32)
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[N:.*]] = arith.minui %[[LEN]], %[[C32]] : index
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32>
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[N]], %[[C32]] : index
// CHECK: scf.if %[[COND]]
// CHECK: linalg.fill ins(%[[PAD]] : f32) outs(%[[ROW]] : memref<32xf32>)
// CHECK: %[[SV:.*]] = memref.subview %[[SRC]][0] [%[[N]]] [1]
// CHECK: %[[DV:.*]] = memref.subview %[[ROW]][0] [%[[N]]] [1]
// CHECK: memref.copy %[[SV]], %[[DV]] :
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[ROW]] restrict writable : memref<32xf32>
// CHECK: %[[BCAST:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>)
// CHECK: return %[[BCAST]] : tensor<4x32xf32>
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @dynamic_pad(%src: memref<32xf32> {tt.tensor_kind = 0 : i32}, %len: index, %pad: f32) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %z = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %n = arith.minui %len, %c32 : index
  %row = memref.alloc() : memref<32xf32>
  %lt = arith.cmpi slt, %n, %c32 : index
  scf.if %lt {
    linalg.fill ins(%pad : f32) outs(%row : memref<32xf32>)
  }
  %sv = memref.subview %src[0] [%n] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
  %dv = memref.subview %row[0] [%n] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
  memref.copy %sv, %dv : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}


// -----
// CHECK-LABEL: func.func @partial_pad
// CHECK: %[[COND:.*]] = arith.cmpi slt,
// CHECK: scf.if %[[COND]]
// CHECK: linalg.fill
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] : tensor<4x32xf32>
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @partial_pad(%src: memref<32xf32> {tt.tensor_kind = 0 : i32}, %len: index, %pad: f32) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %z = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %n = arith.constant 17 : index
  %row = memref.alloc() : memref<32xf32>
  %lt = arith.cmpi slt, %n, %c32 : index
  scf.if %lt {
    linalg.fill ins(%pad : f32) outs(%row : memref<32xf32>)
  }
  %sv = memref.subview %src[0] [17] [1] : memref<32xf32> to memref<17xf32, strided<[1]>>
  %dv = memref.subview %row[0] [17] [1] : memref<32xf32> to memref<17xf32, strided<[1]>>
  memref.copy %sv, %dv : memref<17xf32, strided<[1]>> to memref<17xf32, strided<[1]>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}


// -----
// CHECK-LABEL: func.func @full_pad
// CHECK: %[[COND:.*]] = arith.cmpi slt,
// CHECK: scf.if %[[COND]]
// CHECK: linalg.fill
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: return %[[BCAST]] : tensor<4x32xf32>
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @full_pad(%src: memref<32xf32> {tt.tensor_kind = 0 : i32}, %len: index, %pad: f32) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %z = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %n = arith.constant 32 : index
  %row = memref.alloc() : memref<32xf32>
  %lt = arith.cmpi slt, %n, %c32 : index
  scf.if %lt {
    linalg.fill ins(%pad : f32) outs(%row : memref<32xf32>)
  }
  %sv = memref.subview %src[0] [32] [1] : memref<32xf32> to memref<32xf32, strided<[1]>>
  %dv = memref.subview %row[0] [32] [1] : memref<32xf32> to memref<32xf32, strided<[1]>>
  memref.copy %sv, %dv : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}
