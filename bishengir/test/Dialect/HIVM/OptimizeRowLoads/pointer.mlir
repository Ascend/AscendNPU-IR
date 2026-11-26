// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @pointer_regular(
// CHECK: scf.for {{.*}} step %c2
// CHECK: %[[BASE:.*]] = hivm.hir.pointer_cast
// CHECK: memref.reinterpret_cast %[[BASE]] {{.*}} sizes: [2, 8]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_regular(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%addr)[%size] : memref<?xf32>
      annotation.mark %ptr {address_space = #hivm.address_space<gm>, effects = []} : memref<?xf32>
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
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

// CHECK-LABEL: func.func @pointer_varying_address(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK-NOT: scf.for {{.*}} step %c2
// CHECK-NOT: sizes: [2, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_varying_address(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %i64 = arith.index_cast %i : index to i64
      %varying = arith.addi %addr, %i64 : i64
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%varying)[%size] : memref<?xf32>
      annotation.mark %ptr {address_space = #hivm.address_space<gm>, effects = []} : memref<?xf32>
      %varyingIndex = arith.index_cast %varying : i64 to index
      %scaled = arith.muli %varyingIndex, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
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

// CHECK-LABEL: func.func @pointer_effectful_mark(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK-NOT: scf.for {{.*}} step %c2
// CHECK-NOT: sizes: [2, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_effectful_mark(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%addr)[%size] : memref<?xf32>
      annotation.mark %ptr {address_space = #hivm.address_space<gm>, effects = ["read", "write"]} : memref<?xf32>
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
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

// CHECK-LABEL: func.func @pointer_no_evidence(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 8]
// CHECK-NOT: scf.for {{.*}} step %c2
// CHECK-NOT: sizes: [2, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_no_evidence(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%addr)[%size] : memref<?xf32>
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
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

// CHECK-LABEL: func.func @pointer_explicit_gm(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_explicit_gm(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%addr)[%size] : memref<?xf32, #hivm.address_space<gm>>
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32, #hivm.address_space<gm>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: func.func @pointer_conflicting_type(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: scf.for {{.*}} step %c2
// CHECK-NOT: sizes: [2, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_conflicting_type(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%addr)[%size] : memref<?xf32, #hivm.address_space<ub>>
      annotation.mark %ptr {address_space = #hivm.address_space<gm>, effects = []} : memref<?xf32, #hivm.address_space<ub>>
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
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

// CHECK-LABEL: func.func @pointer_multiple_addresses(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: scf.for {{.*}} step %c2
// CHECK-NOT: sizes: [2, 8]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_multiple_addresses(%base: memref<?xf32>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
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
      %extra = arith.addi %i, %c1 : index
      %size = arith.addi %capacity, %extra : index
      %ptr = hivm.hir.pointer_cast(%addr, %addr)[%size] : memref<?xf32>
      annotation.mark %ptr {address_space = #hivm.address_space<gm>, effects = []} : memref<?xf32>
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
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

// CHECK-LABEL: func.func @pointer_external(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 8], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pointer_external(%base: memref<?xf32, #hivm.address_space<gm>>, %ids: tensor<4xi32>,
      %out: memref<4x8xf32>, %batch: index, %position: i32, %limit: i32,
      %side: memref<1xi32>, %addr: i64, %capacity: index)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %pitch = arith.constant 16 : index
    %pitch32 = arith.constant 16 : i32
    %pitch64 = arith.constant 16 : i64
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x8xf32>
    %ptr = hivm.hir.pointer_cast(%addr)[%capacity] : memref<?xf32, #hivm.address_space<gm>>
    annotation.mark %ptr {address_space = #hivm.address_space<gm>, effects = []} : memref<?xf32, #hivm.address_space<gm>>
    scf.for %i = %c0 to %rows step %c1 {
      %scaled = arith.muli %i, %pitch : index
      %offset = arith.addi %batch, %scaled : index
      %src = memref.reinterpret_cast %ptr to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32, #hivm.address_space<gm>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
      %slice = memref.subview %dst[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%slice : memref<1x8xf32, strided<[8, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %result = bufferization.to_tensor %dst restrict writable : memref<4x8xf32>
    bufferization.materialize_in_destination %result in writable %out
      : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
    return
  }
}
