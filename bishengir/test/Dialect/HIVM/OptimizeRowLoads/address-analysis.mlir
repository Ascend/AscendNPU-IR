// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @subtract_invariant(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @subtract_invariant(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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

// CHECK-LABEL: func.func @cancel_invariant(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cancel_invariant(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %shifted = arith.addi %i, %batch : index
      %local = arith.subi %shifted, %batch : index
      %offset = arith.muli %local, %pitch : index
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

// CHECK-LABEL: func.func @affine_subtract(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @affine_subtract(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %offset = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - s0)>(%i)[%batch]
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

// CHECK-LABEL: func.func @variable_stride(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @variable_stride(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %offset = arith.muli %i, %batch : index
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

// CHECK-LABEL: func.func @nonlinear_stride(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nonlinear_stride(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %square = arith.muli %i, %i : index
      %offset = arith.muli %square, %pitch : index
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

// CHECK-LABEL: func.func @floor_stride(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @floor_stride(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %offset = affine.apply affine_map<(d0) -> ((d0 floordiv 2) * 16)>(%i)
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
// Replay the original arithmetic, including coefficient overflow, per row.

// CHECK-LABEL: func.func @coefficient_overflow(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @coefficient_overflow(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 1099511627776 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.muli %scaled, %pitch : index
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
// Replay invariant affine chains without deriving a compile-time row stride.


// CHECK-LABEL: func.func @invariant_affine_composition(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @invariant_affine_composition(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %large = affine.apply affine_map<()[s0] -> (s0 * 1099511627776)>()[%batch]
    %larger = affine.apply affine_map<()[s0] -> (s0 * 1099511627776)>()[%large]
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %offset = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%i)[%larger]
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

// Explicit AIV eligibility is required.
// CHECK-LABEL: func.func @missing_core_type(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @missing_core_type(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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

// Explicit AIV eligibility is required.
// CHECK-LABEL: func.func @mixed_core_type(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mixed_core_type(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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

// Explicit AIV eligibility is required.
// CHECK-LABEL: func.func @cube_core_type(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cube_core_type(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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

// Pair the first two executed rows and preserve the last scalar row.
// CHECK-LABEL: func.func @nonzero_partial_window(
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: sizes: [1, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nonzero_partial_window(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %lower = arith.constant 1 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %lower to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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

// The signed comparison gates an unsigned difference/parity calculation.
// CHECK-LABEL: func.func @dynamic_trip_unsigned_parity(
// CHECK: arith.cmpi slt
// CHECK: arith.subi
// CHECK: arith.remui
// CHECK-NOT: arith.remsi
// CHECK: sizes: [2, 8], strides:
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_trip_unsigned_parity(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index, %lower: index, %upper: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %lower to %upper step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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

// An empty signed range must not become an overflowing positive trip count.
// CHECK-LABEL: func.func @empty_signed_extreme_bounds(
// CHECK-NOT: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_signed_extreme_bounds(%base: memref<?xf32>, %out: memref<4x8xf32>,
      %batch: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.constant -9223372036854775808 : index
    %lower = arith.constant 9223372036854775807 : index
    %pitch = arith.constant 16 : index
    %dst = memref.alloc() : memref<4x8xf32>
    scf.for %i = %lower to %upper step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.subi %scaled, %batch : index
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
