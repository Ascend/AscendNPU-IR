// RUN: bishengir-opt %s -hivm-optimize-row-loads -split-input-file | FileCheck %s

// RUN: bishengir-opt %s -canonicalize -hivm-optimize-row-loads -split-input-file | FileCheck %s --check-prefix=GENERAL-DMA

// Pair the rows actually executed by partial and dynamic loop windows.
// Check before canonicalization so empty-loop handling is covered too.
// unknown_upper assumes valid <= 4 for legal input tensor accesses.

// CHECK-LABEL: func.func @dynamic_prefix(
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_prefix(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index
    %bounded = arith.minsi %valid, %rows : index
    %nonnegative = arith.maxsi %bounded, %c0 : index
    %upper = arith.minsi %nonnegative, %rows : index
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %upper step %c1 {
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
// CHECK-LABEL: func.func @dynamic_regular(
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_regular(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index
    %bounded = arith.minsi %valid, %rows : index
    %nonnegative = arith.maxsi %bounded, %c0 : index
    %upper = arith.minsi %nonnegative, %rows : index
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %upper step %c1 {
      %offset = arith.muli %i, %pitch : index
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
// CHECK-LABEL: func.func @short_constant_prefix(
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @short_constant_prefix(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index
    %upper = arith.constant 3 : index
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %upper step %c1 {
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
// CHECK-LABEL: func.func @empty_prefix(
// CHECK: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_prefix(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index

    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %c0 step %c1 {
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
// CHECK-LABEL: func.func @unknown_upper(
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unknown_upper(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index
    %upper = arith.maxsi %valid, %c0 : index
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %upper step %c1 {
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
// CHECK-LABEL: func.func @unknown_lower(
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unknown_lower(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %valid: index, %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %pitch = arith.constant 16 : index
    %upper = arith.minsi %valid, %rows : index
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %upper step %c1 {
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
// CHECK-LABEL: func.func @dynamic_subblocks(
// CHECK: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_subblocks(%base: memref<?xf32>, %ids: tensor<8xi32>,
      %valid: index, %stride: i32, %out: memref<8x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0 : i32
    %bounded = arith.minsi %valid, %c8 : index
    %active = arith.maxsi %bounded, %c0 : index
    scf.for %s = %c0 to %c2 step %c1 {
      %lower = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%s]
      %end = arith.addi %lower, %c4 : index
      %upper = arith.minsi %end, %active : index
      %dst = memref.alloc() : memref<4x8xf32>
      annotation.mark %dst {hivm.multi_buffer = 2 : i32} : memref<4x8xf32>
      scf.for %i = %lower to %upper step %c1 {
        %id = tensor.extract %ids[%i] : tensor<8xi32>
        %safe = arith.maxsi %id, %zero : i32
        %product = arith.muli %safe, %stride : i32
        %offset = arith.index_cast %product : i32 to index
        %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
          : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
        %next = arith.addi %i, %c1 : index
        %first = arith.maxsi %i, %lower : index
        %last = arith.minsi %next, %end : index
        %clipped = arith.maxsi %first, %last : index
        %size = arith.subi %clipped, %first : index
        %local = arith.subi %first, %lower : index
        %slice = memref.subview %dst[%local, 0] [%size, 8] [1, 1]
          : memref<4x8xf32> to memref<?x8xf32, strided<[8, 1], offset: ?>>
        hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
          outs(%slice : memref<?x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
      } {hivm.parallel_loop}
      %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
      %target = memref.subview %out[%lower, 0] [4, 8] [1, 1]
        : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
      bufferization.materialize_in_destination %result in writable %target
        : (tensor<4x8xf32>, memref<4x8xf32, strided<[8, 1], offset: ?>>) -> ()
    }
    return
  }
}

// -----
// A reinterpret_cast addresses base storage independently of its rank.
// GENERAL-DMA-LABEL: func.func @ranked_gm_base(
// GENERAL-DMA: sizes: [2, 8], strides:
// GENERAL-DMA: return
// CHECK-LABEL: func.func @ranked_gm_base(
// CHECK: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @ranked_gm_base(%base: memref<64x16xf32>, %ids: tensor<4xi32>,
      %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %product = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %product : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<64x16xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
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
// An invariant source descriptor may be hoisted out of the row loop.
// GENERAL-DMA-LABEL: func.func @hoisted_row_descriptor(
// GENERAL-DMA: sizes: [2, 8], strides:
// GENERAL-DMA: return
// CHECK-LABEL: func.func @hoisted_row_descriptor(
// CHECK: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @hoisted_row_descriptor(%base: memref<?xf32>, %offset: index,
      %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %dst = memref.alloc() : memref<4x8xf32>

    %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
      : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>

    scf.for %i = %c0 to %rows step %c1 {
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
// Relaxing base rank must retain the GM storage requirement.
// CHECK-LABEL: func.func @ranked_local_base(
// CHECK: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @ranked_local_base(%base: memref<64x16xf32, #hivm.address_space<ub>>, %ids: tensor<4xi32>,
      %stride: i32, %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x8xf32>

    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %product = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %product : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<64x16xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}

// -----
// Canonicalization may turn a hoisted descriptor offset into an attribute.
// GENERAL-DMA-LABEL: func.func @hoisted_static_row_descriptor(
// GENERAL-DMA: sizes: [2, 8], strides:
// GENERAL-DMA: return
// CHECK-LABEL: func.func @hoisted_static_row_descriptor(
// CHECK: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @hoisted_static_row_descriptor(%base: memref<?xf32>,
      %out: memref<4x8xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %dst = memref.alloc() : memref<4x8xf32>

    %src = memref.reinterpret_cast %base to offset: [16], sizes: [1, 8], strides: [8, 1]
      : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: 16>>

    scf.for %i = %c0 to %rows step %c1 {
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: 16>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}
