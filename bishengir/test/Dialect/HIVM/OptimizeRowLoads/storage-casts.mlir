// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -split-input-file | FileCheck %s --check-prefix=SCALAR

// Recover tile storage through descriptor casts and reject non-UB roots
// and unsupported alias views directly in the row-load optimization.

// -----
// SCALAR-LABEL: func.func @ub_space_cast(
// SCALAR: memref.memory_space_cast
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @ub_space_cast(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [%c1, %c1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[?, ?], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[?, ?], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}

// -----
// SCALAR-LABEL: func.func @refined_cast(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]

// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @refined_cast(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %dynamic = memref.alloc(%c8) : memref<4x?xf32, #hivm.address_space<ub>>
    %root = memref.cast %dynamic : memref<4x?xf32, #hivm.address_space<ub>> to memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
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
// SCALAR-LABEL: func.func @ub_cast_chain(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @ub_cast_chain(%base: memref<?xf32>, %out: memref<4x8xf32, #hivm.address_space<ub>>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %a = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    %b = memref.cast %a : memref<4x8xf32> to memref<4x8xf32>
    %dst = memref.memory_space_cast %b : memref<4x8xf32> to memref<4x8xf32, #hivm.address_space<ub>>

    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32, #hivm.address_space<ub>>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32, #hivm.address_space<ub>>) -> ()
    return
  }
}

// -----
// SCALAR-LABEL: func.func @cast_root_read(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_root_read(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    %read = memref.load %root[%c0, %c0] : memref<4x8xf32, #hivm.address_space<ub>>
    memref.store %read, %out[%c0, %c0] : memref<4x8xf32>
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
// SCALAR-LABEL: func.func @cast_view_write(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_view_write(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    %v = arith.constant 0.0 : f32
    memref.store %v, %dst[%c0, %c0] : memref<4x8xf32>
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
// SCALAR-LABEL: func.func @cast_root_escape(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @consume_root(memref<4x8xf32, #hivm.address_space<ub>>)
  func.func @cast_root_escape(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    func.call @consume_root(%root) : (memref<4x8xf32, #hivm.address_space<ub>>) -> ()
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
// SCALAR-LABEL: func.func @cast_sibling_alias(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_sibling_alias(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    %other = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    %peek = memref.load %other[%c0, %c0] : memref<4x8xf32>
    memref.store %peek, %out[%c0, %c0] : memref<4x8xf32>
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
// SCALAR-LABEL: func.func @cast_wrong_space(
// SCALAR: scf.for {{.*}} step %c1
// SCALAR-NOT: sizes: [2, 8]
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_wrong_space(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<gm>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<gm>> to memref<4x8xf32>
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
// A default-space local allocation may be explicitly refined to UB.
// SCALAR-LABEL: func.func @cast_unknown_space(
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8]
// SCALAR: hivm.hir.load
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_unknown_space(%base: memref<?xf32>, %out: memref<4x8xf32, #hivm.address_space<ub>>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32>
    %dst = memref.memory_space_cast %root : memref<4x8xf32> to memref<4x8xf32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32, #hivm.address_space<ub>>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32, #hivm.address_space<ub>>) -> ()
    return
  }
}

// -----
// Pairing preserves the current dense descriptor, not the root's padded layout.
// SCALAR-LABEL: func.func @cast_layout_change(
// SCALAR: %[[DENSE:.*]] = memref.reinterpret_cast
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8]
// SCALAR: memref.subview %[[DENSE]]
// SCALAR: hivm.hir.load
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_layout_change(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %tmp = memref.memory_space_cast %root : memref<4x8xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<4x8xf32, strided<[16, 1]>>
    %dst = memref.reinterpret_cast %tmp to offset: [0], sizes: [4, 8], strides: [8, 1]
      : memref<4x8xf32, strided<[16, 1]>> to memref<4x8xf32>
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
// SCALAR-LABEL: func.func @ub_indirect_space_cast(
// SCALAR: memref.memory_space_cast
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: arith.subi
// SCALAR: sizes: [2, 8], strides: [%{{.*}}, 1]
// SCALAR: memref.subview %{{.*}}[%{{.*}}, 0] [2, 8] [1, 1] : memref<4x8xf32>
// SCALAR: hivm.hir.load
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @ub_indirect_space_cast(%base: memref<?xf32>, %ids: tensor<4xi32>, %out: memref<4x8xf32>, %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %root = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %idx = arith.index_cast %id : i32 to index
      %scaled = arith.muli %idx, %pitch : index
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
// A prepared tile may cross an outlining boundary. Lowering needs its current
// descriptor, not an alloc defining op in the same function.

// SCALAR-LABEL: func.func @prepared_ub_argument(
// SCALAR: memref.memory_space_cast
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: sizes: [2, 8], strides: [{{.*}}, 1]
// SCALAR: hivm.hir.load
// SCALAR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @prepared_ub_argument(%base: memref<?xf32>, %out: memref<4x8xf32>, %batch: index, %root: memref<4x8xf32, #hivm.address_space<ub>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.memory_space_cast %root : memref<4x8xf32, #hivm.address_space<ub>> to memref<4x8xf32>
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
