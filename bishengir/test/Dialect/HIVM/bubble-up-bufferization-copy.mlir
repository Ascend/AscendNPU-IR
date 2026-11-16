// RUN: bishengir-opt -test-bubble-up-bufferization %s | FileCheck %s

// A full tile uses local coordinates even when its original offset is dynamic.
// CHECK-LABEL: func.func @rebased_view_dynamic_offset(
// CHECK: %[[RING:.*]] = memref.alloc() : memref<32xf32>
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<32xf32>) outs(%[[RING]] : memref<32xf32>)
// CHECK: %[[VIEW:.*]] = memref.subview %[[RING]][0] [32] [1]
// CHECK: annotation.mark %[[VIEW]] : memref<32xf32, strided<[1]>>
// CHECK: return
func.func @rebased_view_dynamic_offset(%input: tensor<32xf32>, %offset: index) {
  %ring = memref.alloc() : memref<64xf32>
  %up = builtin.unrealized_conversion_cast %ring, %offset : memref<64xf32>, index to memref<32xf32>
      {bubble_up_propagate_up, tiling_dim_info = [0 : index, -9223372036854775808 : index, 32 : index]}
  hivm.hir.copy ins(%input : tensor<32xf32>) outs(%up : memref<32xf32>)
  %view = memref.subview %ring[%offset] [32] [1]
      : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
  annotation.mark %view : memref<32xf32, strided<[1], offset: ?>>
  return
}
