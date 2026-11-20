// RUN: bishengir-opt %s -hivm-optimize-row-loads -split-input-file -verify-each | FileCheck %s

// Casts and equivalent source geometry need no preparatory canonicalization.
// Unused arithmetic must not prevent pairing, and unused single-row descriptors
// must not be emitted on the unconditional paired path.
// CHECK-LABEL: func.func @cast_geometry_dead(
// CHECK: scf.for {{.*}} step %c2
// CHECK-NOT: memref.reinterpret_cast
// CHECK-NOT: memref.cast
// CHECK-NOT: arith.addf
// CHECK: memref.reinterpret_cast {{.*}} sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK-NOT: sizes: [1, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_geometry_dead(%base: memref<?xf32>, %f: f32)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %c4 step %c1 {
      %dead = arith.addf %f, %f : f32
      %width = arith.addi %c8, %c0 : index
      %unit = arith.subi %c1, %c0 : index
      %offset = arith.muli %i, %c8 : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, %width], strides: [8, %unit]
        : memref<?xf32> to memref<1x?xf32, strided<[8, ?], offset: ?>>
      %source = memref.cast %src : memref<1x?xf32, strided<[8, ?], offset: ?>> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %target = memref.cast %row : memref<1x8xf32, strided<[8, 1], offset: ?>> to memref<?x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%source : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%target : memref<?x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    return
  }
}

// -----
// Odd tails still replay the original descriptor and transparent casts.
// CHECK-LABEL: func.func @cast_tail(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: } {hivm.parallel_loop}
// CHECK: sizes: [1, 8]
// CHECK: memref.cast
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_tail(%base: memref<?xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %dst = memref.alloc() : memref<3x8xf32>
    scf.for %i = %c0 to %c3 step %c1 {
      %offset = arith.muli %i, %c8 : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %source = memref.cast %src : memref<1x8xf32, strided<[8, 1], offset: ?>> to memref<?x8xf32, strided<[8, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<3x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%source : memref<?x8xf32, strided<[8, 1], offset: ?>>)
        outs(%row : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    return
  }
}

// -----
// Broadcast DMA is valid only on A6; the A5 row-load pass must leave it intact.
// CHECK-LABEL: func.func @broadcast_axes(
// CHECK-NOT: hivm.hir.vbrc
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK-NOT: scf.if
// CHECK: broadcast_axes = [1]
// CHECK-NOT: scf.if
// CHECK: arith.select
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @broadcast_axes(%base: memref<?xf32, #hivm.address_space<gm>>, %mask: tensor<4xi1>) -> tensor<4x8xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dst = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c4 step %c1 {
      %src = memref.reinterpret_cast %base to offset: [%i], sizes: [1, 1], strides: [1, 1]
        : memref<?xf32, #hivm.address_space<gm>> to memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %row = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%row : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) broadcast_axes = [1] core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst : memref<4x8xf32, #hivm.address_space<ub>>
    %mask_init = tensor.empty() : tensor<4x8xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>) outs(%mask_init : tensor<4x8xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x8xf32>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x8xi1>, tensor<4x8xf32>
    return %selected : tensor<4x8xf32>
  }
}

// -----
// A source subview needs no reinterpret descriptor for mask sinking. A private
// padded destination need not be dense. Live metadata observations through a
// transparent tile cast do not read data or defeat private ownership.
// CHECK-LABEL: func.func @metadata_mask_without_descriptor(
// CHECK: hivm.hir.vbrc
// CHECK: scf.for {{.*}} step %c1
// CHECK: tensor.extract
// CHECK-NOT: tensor.extract
// CHECK: scf.if
// CHECK: hivm.hir.load
// CHECK-NOT: tensor.extract
// CHECK-NOT: arith.select
// CHECK: %[[DIM:.*]] = tensor.dim
// CHECK-NOT: arith.select
// CHECK: return {{.*}}, %[[DIM]] : tensor<4x8xf32>, index
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @metadata_mask_without_descriptor(%base: memref<?x8xf32>, %ids: tensor<4xi32>) -> (tensor<4x8xf32>, index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dst = memref.alloc() : memref<4x8xf32, strided<[16, 1]>>
    %tile = memref.cast %dst : memref<4x8xf32, strided<[16, 1]>> to memref<4x8xf32, strided<[?, 1], offset: ?>>
    scf.for %i = %c0 to %c4 step %c1 {
      %row = memref.subview %tile[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32, strided<[?, 1], offset: ?>> to memref<1x8xf32, strided<[?, 1], offset: ?>>
      %dim = memref.dim %row, %c0 : memref<1x8xf32, strided<[?, 1], offset: ?>>
      %same = arith.addi %i, %c0 : index
      %id = tensor.extract %ids[%same] : tensor<4xi32>
      %index = arith.index_cast %id : i32 to index
      %src = memref.subview %base[%index, 0] [1, 8] [1, 1]
        : memref<?x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%row : memref<1x8xf32, strided<[?, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %metadata = bufferization.to_tensor %dst : memref<4x8xf32, strided<[16, 1]>>
    %shape = tensor.cast %metadata : tensor<4x8xf32> to tensor<?x8xf32>
    %dim = tensor.dim %shape, %c0 : tensor<?x8xf32>
    %loaded = bufferization.to_tensor %dst : memref<4x8xf32, strided<[16, 1]>>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x8xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>) outs(%mask_init : tensor<4x8xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x8xf32>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x8xi1>, tensor<4x8xf32>
    return %selected, %dim : tensor<4x8xf32>, index
  }
}

// -----
// The loop covers its entire four-row tile, not its eight-row allocation.
// Observing that wider allocation must not authorize erasing its select.
// CHECK-LABEL: func.func @wider_observation(
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: arith.select
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @wider_observation(%base: memref<?xf32>, %mask: tensor<8xi1>) -> tensor<8x8xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %root = memref.alloc() : memref<8x8xf32>
    %dst = memref.subview %root[0, 0] [4, 8] [1, 1] : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1]>>
    scf.for %i = %c0 to %c4 step %c1 {
      %src = memref.reinterpret_cast %base to offset: [0], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1]>>
      %row = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32, strided<[8, 1]>> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1]>>)
        outs(%row : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %root : memref<8x8xf32>
    %mask_init = tensor.empty() : tensor<8x8xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<8xi1>) outs(%mask_init : tensor<8x8xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<8x8xf32>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<8x8xi1>, tensor<8x8xf32>
    return %selected : tensor<8x8xf32>
  }
}
