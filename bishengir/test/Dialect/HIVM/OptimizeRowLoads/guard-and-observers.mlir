// RUN: bishengir-opt %s -hivm-optimize-row-loads="fallback-loop=false" -canonicalize -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hivm-optimize-row-loads="fallback-loop=true" -canonicalize -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hivm-optimize-row-loads="fallback-loop=true" -verify-each -split-input-file -o /dev/null

// Prepared full-tile contract. The guard has a provably regular address, so
// omitting the guard eligibility condition would issue an unconditional DMA.
// Loop-local zero padding also exercises lane fallback operand remapping.

// CHECK-LABEL: func.func @guarded_regular(
// CHECK: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x8xf32>)
// CHECK: scf.for {{.*}} step %c2
// CHECK: arith.andi
// CHECK: sizes: [2, 8]
// CHECK: } else {
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @guarded_regular(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
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
      %valid = tensor.extract %mask[%i] : tensor<8xi1>
      %padding = arith.constant 0 : index
      scf.if %valid {
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) left_padding_num = %padding : index core_type = <VECTOR>
      } else {
        hivm.hir.vbrc ins(%fallback : f32) outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>)
      }
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}


// -----
// CHECK-LABEL: func.func @unguarded_regular(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unguarded_regular(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
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
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}


// -----
// CHECK-LABEL: func.func @nonzero_lower_guard(
// CHECK: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x8xf32>)
// CHECK: scf.for {{.*}} step %c2
// CHECK: arith.andi
// CHECK: sizes: [2, 8]
// CHECK: } else {
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nonzero_lower_guard(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 7 : index
    %lower = arith.constant 3 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %lower to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %local = arith.subi %i, %lower : index
      %slice = memref.subview %dst[%local, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %valid = tensor.extract %mask[%i] : tensor<8xi1>
      %padding = arith.constant 0 : index
      scf.if %valid {
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) left_padding_num = %padding : index core_type = <VECTOR>
      } else {
        hivm.hir.vbrc ins(%fallback : f32) outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>)
      }
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}


// -----
// CHECK-LABEL: func.func @multi_index_read(
// CHECK: scf.for {{.*}} step %c2
// CHECK: tensor.extract {{.*}}[%c1, %{{.*}}] : tensor<2x5xi32>
// CHECK: tensor.extract {{.*}}[%c1, %{{.*}}] : tensor<2x5xi32>
// CHECK: sizes: [2, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @multi_index_read(%base: memref<?xf32>, %ids: tensor<2x5xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
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
      %next = arith.addi %i, %c1 : index
      %id = tensor.extract %ids[%c1, %next] : tensor<2x5xi32>
      %idIndex = arith.index_cast %id : i32 to index
      %scaled = arith.muli %idIndex, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}


// -----
// CHECK-LABEL: func.func @no_observer(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @no_observer(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
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
    return
  }
}


// -----
// CHECK-LABEL: func.func @dealloc_without_observer(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: memref.dealloc
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dealloc_without_observer(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
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
    memref.dealloc %dst : memref<4x8xf32>
    return
  }
}


// -----
// CHECK-LABEL: func.func @dealloc_with_observer(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: memref.dealloc
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dealloc_with_observer(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %mask: tensor<8xi1>, %fallback: f32)
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
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    memref.dealloc %dst : memref<4x8xf32>
    return
  }
}
