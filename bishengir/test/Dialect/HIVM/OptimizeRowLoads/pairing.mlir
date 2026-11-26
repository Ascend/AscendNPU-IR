// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -split-input-file | FileCheck %s --check-prefix=SCALAR --implicit-check-not=hivm.hir.custom --implicit-check-not=hivm.hir.varange
// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -one-shot-bufferize="bufferize-function-boundaries" -hivm-infer-mem-scope -convert-hivm-to-std -split-input-file | FileCheck %s --check-prefix=LIB
// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -one-shot-bufferize="bufferize-function-boundaries" -hivm-infer-mem-scope -hivm-graph-sync-solver -split-input-file | FileCheck %s --check-prefix=SYNC
// RUN: bishengir-opt %s -hivm-optimize-row-loads="enabled=false" -split-input-file | FileCheck %s --check-prefix=OFF
// RUN: bishengir-opt %s -hivm-optimize-row-loads="enabled=false" -canonicalize -split-input-file | FileCheck %s --check-prefix=UNPAIRED

// AIV row gathers are matched directly without an earlier classification pass.

// -----

// SCALAR-LABEL: func.func @regular
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR: return
// OFF-LABEL: func.func @regular
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @regular
// LIB: call @load_gm_to_ubuf_2d_float
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @regular(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}

// -----

// The signed-stride paired load must issue without MTE2 barriers.
// The UNPAIRED run disables the pass to retain the single-row baseline.
// UNPAIRED-LABEL: func.func @plain_ids(
// UNPAIRED: scf.for {{.*}} step %c1
// UNPAIRED: sizes: [1, 8]
// UNPAIRED-NOT: sizes: [2, 8]
// UNPAIRED: return
// SYNC-LABEL: func.func @plain_ids(
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
// SCALAR-LABEL: func.func @plain_ids
// SCALAR: step %c2
// SCALAR: arith.subi
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR-NOT: hivm.hir.load
// SCALAR: } {hivm.parallel_loop}
// SCALAR: return
// OFF-LABEL: func.func @plain_ids
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @plain_ids
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB-NOT: call @load_gm_to_ubuf_2d_float
// LIB: } {hivm.parallel_loop}
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @plain_ids(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
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

// The signed-stride paired load must issue without MTE2 barriers.
// SYNC-LABEL: func.func @masked_ids(
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
// SCALAR-LABEL: func.func @masked_ids
// SCALAR: step %c2
// SCALAR: arith.cmpi sge
// SCALAR: arith.andi
// SCALAR: arith.select
// SCALAR: arith.subi
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR-NOT: hivm.hir.load
// SCALAR: } {hivm.parallel_loop}
// SCALAR: return
// OFF-LABEL: func.func @masked_ids
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @masked_ids
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB-NOT: call @load_gm_to_ubuf_2d_float
// LIB: } {hivm.parallel_loop}
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @masked_ids(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %iv = arith.index_cast %i : index to i32
      %pos = arith.addi %position, %iv : i32
      %bounds = arith.cmpi slt, %pos, %limit : i32
      %positive = arith.cmpi sge, %id, %zero : i32
      %valid = arith.andi %positive, %bounds : i1
      %safe = arith.select %valid, %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
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

// The signed-stride paired load must issue without MTE2 barriers.
// SYNC-LABEL: func.func @wrapped_i32_product(
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
// SCALAR-LABEL: func.func @wrapped_i32_product
// SCALAR: step %c2
// SCALAR: %[[WRAPPED:.*]] = arith.muli {{.*}} : i32
// SCALAR: arith.index_cast %[[WRAPPED]] : i32 to index
// SCALAR: arith.subi
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR-NOT: hivm.hir.load
// SCALAR: } {hivm.parallel_loop}
// SCALAR: return
// OFF-LABEL: func.func @wrapped_i32_product
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @wrapped_i32_product
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB-NOT: call @load_gm_to_ubuf_2d_float
// LIB: } {hivm.parallel_loop}
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @wrapped_i32_product(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %scaled32 = arith.muli %safe, %pitch32 : i32
      %wide = arith.extsi %scaled32 : i32 to i64
      %offset = arith.index_cast %wide : i64 to index
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

// SCALAR-LABEL: func.func @odd_rows
// SCALAR: arith.subi
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR: hivm.hir.load
// SCALAR: return
// OFF-LABEL: func.func @odd_rows
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @odd_rows
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB: call @load_gm_to_ubuf_2d_float
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @odd_rows(%base: memref<?xf32>, %ids: tensor<3xi32>,
      %out: memref<3x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 3 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<3x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<3xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<3x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<3x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<3x8xf32>, memref<3x8xf32>) -> ()
    return
  }
}

// -----

// SCALAR-LABEL: func.func @precomputed_offsets
// SCALAR: scf.for {{.*}} step %c2
// SCALAR-NOT: scf.if
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR: return
// OFF-LABEL: func.func @precomputed_offsets
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @precomputed_offsets
// LIB: scf.for {{.*}} step %c2
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @precomputed_offsets(%base: memref<?xf32>, %ids: tensor<4xi64>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi64>
      %offset = arith.index_cast %id : i64 to index
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

// SCALAR-LABEL: func.func @external_destination
// SCALAR: scf.for
// SCALAR-NOT: scf.if
// SCALAR: hivm.hir.load
// SCALAR: return
// OFF-LABEL: func.func @external_destination
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @external_destination
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @external_destination(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32, #hivm.address_space<ub>>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %out[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    return
  }
}

// -----

// SCALAR-LABEL: func.func @side_effect
// SCALAR: scf.for
// SCALAR-NOT: scf.if
// SCALAR: hivm.hir.load
// SCALAR: return
// OFF-LABEL: func.func @side_effect
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @side_effect
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @side_effect(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
      memref.store %id, %side[%c0] : memref<1xi32>
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

// SCALAR-LABEL: func.func @unaligned_rows
// SCALAR: scf.for
// SCALAR-NOT: scf.if
// SCALAR: hivm.hir.load
// SCALAR: return
// OFF-LABEL: func.func @unaligned_rows
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @unaligned_rows
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unaligned_rows(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x3xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x3xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 3], strides: [3, 1]
        : memref<?xf32> to memref<1x3xf32, strided<[3, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 3] [1, 1]
        : memref<4x3xf32> to memref<1x3xf32, strided<[3, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x3xf32, strided<[3, 1], offset: ?>>)
        outs(%slice : memref<1x3xf32, strided<[3, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x3xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x3xf32>, memref<4x3xf32>) -> ()
    return
  }
}

// -----

// SCALAR-LABEL: func.func @a3_unchanged
// SCALAR: scf.for
// SCALAR-NOT: scf.if
// SCALAR: hivm.hir.load
// SCALAR: return
// LIB-LABEL: func.func @a3_unchanged
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: return
// OFF-LABEL: func.func @a3_unchanged
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_unchanged(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
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

// A non-entry AIV function, such as an outlined helper, stays unchanged.
// SCALAR-LABEL: func.func @non_entry_unchanged
// SCALAR: scf.for
// SCALAR-NOT: scf.if
// SCALAR: hivm.hir.load
// SCALAR: return
// LIB-LABEL: func.func @non_entry_unchanged
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: return
// OFF-LABEL: func.func @non_entry_unchanged
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @non_entry_unchanged(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
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
// The signed-stride paired load must issue without MTE2 barriers.
// SYNC-LABEL: func.func @bitwise_ids(
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
// SCALAR-LABEL: func.func @bitwise_ids
// SCALAR: step %c2
// SCALAR: arith.andi {{.*}} : i32
// SCALAR: arith.subi
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR-NOT: hivm.hir.load
// SCALAR: } {hivm.parallel_loop}
// SCALAR: return
// OFF-LABEL: func.func @bitwise_ids
// OFF: scf.for
// OFF-NOT: scf.if
// OFF: return
// LIB-LABEL: func.func @bitwise_ids
// LIB: scf.for
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB-NOT: call @load_gm_to_ubuf_2d_float
// LIB: } {hivm.parallel_loop}
// LIB: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @bitwise_ids(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %mask = arith.constant 3 : i32
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %masked = arith.andi %id, %mask : i32
      %safe = arith.subi %masked, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
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
// Keep the real load-to-consumer dependency while removing false WAW edges.
// SYNC-LABEL: func.func @independent_rows_with_consumer(
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// SYNC-NEXT: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// SYNC-NEXT: hivm.hir.store
// SYNC: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @independent_rows_with_consumer(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
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
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.addi %id, %zero : i32
      %wide = arith.extsi %safe : i32 to i64
      %scaled = arith.muli %wide, %pitch64 : i64
      %offset = arith.index_cast %scaled : i64 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    hivm.hir.store ins(%result : tensor<4x8xf32>)
      outs(%out : memref<4x8xf32>)
    return
  }
}

// -----
// Multi-buffer metadata must survive the rewrite without counting as a
// separate memory access.
// SCALAR-LABEL: func.func @multibuffer_mark(
// SCALAR: annotation.mark {{.*}} {hivm.multi_buffer = 2 : i32}
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8]
// SYNC-LABEL: func.func @multibuffer_mark(
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @multibuffer_mark(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index

    %dst = memref.alloc() : memref<4x8xf32>
    annotation.mark %dst {hivm.multi_buffer = 2 : i32} : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %product = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %product : i32 to index
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
// Metadata outside the loop is preserved and does not prevent pairing.
// SCALAR-LABEL: func.func @other_annotation(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR-NOT: scf.if
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @other_annotation(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index

    %dst = memref.alloc() : memref<4x8xf32>
    annotation.mark %dst {hivm.multi_buffer = 2 : i32, effects = ["read", "write"]} : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %product = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %product : i32 to index
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
// SCALAR-LABEL: func.func @reverse_regular
// SCALAR: scf.for {{.*}} step %c2
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR-NOT: hivm.hir.load
// SCALAR: return
// LIB-LABEL: func.func @reverse_regular
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB-NOT: call @load_gm_to_ubuf_2d_float
// LIB: return
// SYNC-LABEL: func.func @reverse_regular
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
// OFF-LABEL: func.func @reverse_regular
// OFF: scf.for {{.*}} step %c1
// OFF: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @reverse_regular(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant -16 : index
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
// SCALAR-LABEL: func.func @broadcast_rows
// SCALAR: scf.for {{.*}} step %c2
// SCALAR-NOT: scf.if
// SCALAR-NOT: scf.for
// SCALAR: strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR-NOT: hivm.hir.load
// SCALAR: return
// LIB-LABEL: func.func @broadcast_rows
// LIB-NOT: scf.if
// LIB: call @load_gm_to_ubuf_2d_float
// LIB-NOT: call @load_gm_to_ubuf_2d_float
// LIB: return
// SYNC-LABEL: func.func @broadcast_rows
// SYNC-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC: return
// OFF-LABEL: func.func @broadcast_rows
// OFF: scf.for {{.*}} step %c1
// OFF: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @broadcast_rows(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 0 : index
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
