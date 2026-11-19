// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s

// Single-output UB capacity in bits, including alignment and target defaults.
// CHECK-LABEL: func.func @equal(
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">, dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"UB_SIZE", 4096 : i64>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i64>>>} {
func.func @equal(%src: memref<128xi8> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi8> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi8>
  memref.copy %src, %row : memref<128xi8> to memref<128xi8>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi8>
  %e = tensor.empty() : tensor<4x128xi8>
  %b = linalg.broadcast ins(%t : tensor<128xi8>) outs(%e : tensor<4x128xi8>) dimensions = [0]
  return %b : tensor<4x128xi8>
}
}

// -----

// CHECK-LABEL: func.func @over(
// CHECK-NOT: hivm.hir.load
// CHECK: memref.copy
// CHECK: linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">, dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"UB_SIZE", 4096 : i64>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i64>>>} {
func.func @over(%src: memref<128xi8> {tt.tensor_kind = 0 : i32}) -> tensor<5x128xi8> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi8>
  memref.copy %src, %row : memref<128xi8> to memref<128xi8>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi8>
  %e = tensor.empty() : tensor<5x128xi8>
  %b = linalg.broadcast ins(%t : tensor<128xi8>) outs(%e : tensor<5x128xi8>) dimensions = [0]
  return %b : tensor<5x128xi8>
}
}

// -----

// CHECK-LABEL: func.func @alignment_over(
// CHECK-NOT: hivm.hir.load
// CHECK: memref.copy
// CHECK: linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">, dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"UB_SIZE", 5120 : i64>, #dlti.dl_entry<"UB_ALIGN_SIZE", 2048 : i64>>>} {
func.func @alignment_over(%src: memref<128xi8> {tt.tensor_kind = 0 : i32}) -> tensor<5x128xi8> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi8>
  memref.copy %src, %row : memref<128xi8> to memref<128xi8>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi8>
  %e = tensor.empty() : tensor<5x128xi8>
  %b = linalg.broadcast ins(%t : tensor<128xi8>) outs(%e : tensor<5x128xi8>) dimensions = [0]
  return %b : tensor<5x128xi8>
}
}

// -----

// CHECK-LABEL: func.func @fallback_equal(
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @fallback_equal(%src: memref<128xi8> {tt.tensor_kind = 0 : i32}) -> tensor<2496x128xi8> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi8>
  memref.copy %src, %row : memref<128xi8> to memref<128xi8>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi8>
  %e = tensor.empty() : tensor<2496x128xi8>
  %b = linalg.broadcast ins(%t : tensor<128xi8>) outs(%e : tensor<2496x128xi8>) dimensions = [0]
  return %b : tensor<2496x128xi8>
}
}

// -----

// CHECK-LABEL: func.func @missing_size(
// CHECK-NOT: hivm.hir.load
// CHECK: memref.copy
// CHECK: linalg.broadcast
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">, dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i64>>>} {
func.func @missing_size(%src: memref<128xi8> {tt.tensor_kind = 0 : i32}) -> tensor<4x128xi8> attributes {hacc.entry} {
  %row = memref.alloc() : memref<128xi8>
  memref.copy %src, %row : memref<128xi8> to memref<128xi8>
  %t = bufferization.to_tensor %row restrict writable : memref<128xi8>
  %e = tensor.empty() : tensor<4x128xi8>
  %b = linalg.broadcast ins(%t : tensor<128xi8>) outs(%e : tensor<4x128xi8>) dimensions = [0]
  return %b : tensor<4x128xi8>
}
}
