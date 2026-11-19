// RUN: bishengir-opt %s -split-input-file -hivm-fuse-op-into-load | FileCheck %s
// CHECK-LABEL: func.func @row_broadcast
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @row_broadcast(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
// The expanded row is too narrow for the broadcast NDDMA fast path.
// CHECK-LABEL: func.func @column_broadcast
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<32xf32>
// CHECK-NEXT: hivm.hir.load ins(%{{.*}} : memref<32xf32>) outs(%[[ROW]] : memref<32xf32>){{$}}
// CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[ROW]] restrict writable : memref<32xf32>
// CHECK-NEXT: %[[E:.*]] = tensor.empty() : tensor<32x4xf32>
// CHECK-NEXT: %[[B:.*]] = linalg.broadcast ins(%[[T]] : tensor<32xf32>) outs(%[[E]] : tensor<32x4xf32>) dimensions = [1]
// CHECK-NEXT: return %[[B]] : tensor<32x4xf32>
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @column_broadcast(%src: memref<32xf32>) -> tensor<32x4xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<32x4xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<32x4xf32>) dimensions = [1]
  return %b : tensor<32x4xf32>
}
}

// -----
// Same column broadcast with a 128-byte expanded row: fusion is admitted.
// CHECK-LABEL: func.func @column_broadcast_aligned
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [1]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @column_broadcast_aligned(%src: memref<32xf32>) -> tensor<32x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<32x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<32x32xf32>) dimensions = [1]
  return %b : tensor<32x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @load_policy
// CHECK: hivm.hir.load
// CHECK-SAME: eviction_policy = <EvictFirst>
// CHECK-SAME: broadcast_axes = [0]
// CHECK-SAME: core_type = <VECTOR>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @load_policy(%src: memref<32xf32>) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%src : memref<32xf32>) outs(%row : memref<32xf32>) eviction_policy = <EvictFirst> core_type = <VECTOR>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @singleton_nonoverlap
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @singleton_nonoverlap(%src: memref<1x32xf32, strided<[32, 1]>>) -> tensor<4x1x32xf32> attributes {hacc.entry} {
  %row = memref.alloc() : memref<1x32xf32>
  hivm.hir.load ins(%src : memref<1x32xf32, strided<[32, 1]>>) outs(%row : memref<1x32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<1x32xf32>
  %e = tensor.empty() : tensor<4x1x32xf32>
  %b = linalg.broadcast ins(%t : tensor<1x32xf32>) outs(%e : tensor<4x1x32xf32>) dimensions = [0]
  return %b : tensor<4x1x32xf32>
}
}

// -----
// CHECK-LABEL: func.func @copy_broadcast
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @copy_broadcast(%src: memref<32xf32> {tt.tensor_kind = 0 : i32}) -> tensor<4x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %row = memref.alloc() : memref<32xf32>
  memref.copy %src, %row : memref<32xf32> to memref<32xf32>
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %e = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%e : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
}
}
