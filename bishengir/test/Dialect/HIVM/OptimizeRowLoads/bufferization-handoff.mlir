// RUN: bishengir-opt %s -test-bubble-up-bufferization -hivm-optimize-row-loads -canonicalize | FileCheck %s
// RUN: bishengir-opt %s -test-bubble-up-bufferization -canonicalize -hivm-optimize-row-loads -canonicalize | FileCheck %s

// Run bufferization tiling propagation before optimizing row loads.
// Nonzero global lower and canonicalized clipping must retain paired DMA.
// CHECK-LABEL: func.func @row_tile(
// CHECK: memref.alloc() : memref<2x8xf32>
// CHECK-NOT: scf.for
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK-NOT: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @row_tile(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<2x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    %tile = tensor.extract_slice %result[2, 0] [2, 8] [1, 1] {to_be_bubbled_slice = 0 : i64}
      : tensor<4x8xf32> to tensor<2x8xf32>
    bufferization.materialize_in_destination %tile in writable %out
      : (tensor<2x8xf32>, memref<2x8xf32>) -> ()
    return
  }

  // This slice cannot propagate through the select before mask sinking.
  // Optimize handles the current four-row gather and retains the final slice,
  // filling only the actual gather destination once.
  // CHECK-LABEL: func.func @masked_row_tile(
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[TILE:.*]] = memref.alloc() : memref<4x8xf32>
  // CHECK: hivm.hir.vbrc ins(%[[ZERO]] : f32) outs(%[[TILE]] : memref<4x8xf32>)
  // CHECK: scf.if
  // CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
  // CHECK: hivm.hir.load
  // CHECK: } else {
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK-NOT: arith.select
  // CHECK: return
  func.func @masked_row_tile(%base: memref<?xf32>, %mask: tensor<4xi1>,
      %out: memref<2x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%row : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    %mask_init = tensor.empty() : tensor<4x8xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x8xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x8xf32>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x8xi1>, tensor<4x8xf32>
    %tile = tensor.extract_slice %selected[2, 0] [2, 8] [1, 1] {to_be_bubbled_slice = 0 : i64}
      : tensor<4x8xf32> to tensor<2x8xf32>
    bufferization.materialize_in_destination %tile in writable %out
      : (tensor<2x8xf32>, memref<2x8xf32>) -> ()
    return
  }
}
