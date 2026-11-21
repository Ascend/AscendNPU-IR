// RUN: bishengir-opt %s -normalize-to-tensor -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -normalize-to-tensor -hivm-decompose-alloc-tensor-copy -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s --check-prefix=CHECK-ONE-SHOT

// CHECK-LABEL: func.func @test_to_tensorOp(
// CHECK-ONE-SHOT-LABEL: func.func @test_to_tensorOp(
func.func @test_to_tensorOp(%arg0: i32, %arg1 : memref<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK-ONE-SHOT: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %alloc = memref.alloc() : memref<256xf16>
    hivm.hir.load ins(%arg1 : memref<256xf16>) outs(%alloc : memref<256xf16>)
    %5 = bufferization.to_tensor %alloc restrict writable : memref<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK-ONE-SHOT: } else {
    // CHECK-ONE-SHOT: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK-ONE-SHOT: scf.yield %[[ALLOC_1]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
  return %3 : tensor<256xf16>
}

// -----

// Test: a writable to_tensor whose memref was DPS-written before is exposed
// as an alloc_tensor(copy...) {to_be_replaced}, making the memref-level write
// visible to One-Shot Bufferization. The copy operand provides the shape, so
// a dynamic-typed to_tensor needs no dim sizes.

// CHECK-ONE-SHOT-LABEL: func.func @dynamic_to_tensor(
// CHECK-LABEL: func.func @dynamic_to_tensor(
// CHECK: %[[T:.+]] = bufferization.to_tensor %{{.+}} restrict writable : memref<?x20x16x16xf16>
// CHECK: %[[CP:.+]] = bufferization.alloc_tensor() copy(%[[T]]) {to_be_replaced} : tensor<?x20x16x16xf16>
// CHECK: hivm.hir.vadd ins(%[[CP]], %[[CP]] :
func.func @dynamic_to_tensor(%arg0: memref<?x20x16x16xf16>, %arg1: index) -> tensor<?x20x16x16xf16> {
  %alloc = memref.alloc(%arg1) : memref<?x20x16x16xf16>
  hivm.hir.load ins(%arg0 : memref<?x20x16x16xf16>) outs(%alloc : memref<?x20x16x16xf16>)
  %t = bufferization.to_tensor %alloc restrict writable : memref<?x20x16x16xf16>
  %e = tensor.empty(%arg1) : tensor<?x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<?x20x16x16xf16>, tensor<?x20x16x16xf16>) outs(%e : tensor<?x20x16x16xf16>) -> tensor<?x20x16x16xf16>
  return %r : tensor<?x20x16x16xf16>
}

// -----

// A writable to_tensor whose memref carries an hivm address-space attribute:
// a `memref.memory_space_cast` is inserted before the to_tensor to isolate the
// memory-space info, while the DPS-write exposure still applies.

// CHECK-LABEL: func.func @memspace_to_tensor(
// CHECK: %[[CAST:.+]] = memref.memory_space_cast %{{.+}} : memref<256xf16, #hivm.address_space<ub>> to memref<256xf16>
// CHECK: %[[T:.+]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<256xf16>
func.func @memspace_to_tensor(%arg0: memref<256xf16>) -> tensor<256xf16> {
  %alloc = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>
  %t = bufferization.to_tensor %alloc restrict writable : memref<256xf16, #hivm.address_space<ub>>
  return %t : tensor<256xf16>
}

// -----

// A writable to_tensor whose operand is a pre-existing
// `memref.memory_space_cast` of a DPS-written memref (the cast may pre-date
// the pass, e.g. from a previous run of the memspace isolation or from an
// upstream lowering). The cast's only user is the to_tensor itself, so the
// write exposure must trace up through the cast's source to see the
// `hivm.hir.load` write.

// CHECK-LABEL: func.func @memspace_cast_preexisting(
// CHECK: %[[CAST:.+]] = memref.memory_space_cast %{{.+}} : memref<5x20x16x16xf16, #hivm.address_space<ub>> to memref<5x20x16x16xf16>
// CHECK: %[[T:.+]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<5x20x16x16xf16>
// CHECK: %[[CP:.+]] = bufferization.alloc_tensor() copy(%[[T]]) {to_be_replaced} : tensor<5x20x16x16xf16>
// CHECK: hivm.hir.vadd ins(%[[CP]], %[[CP]] :
func.func @memspace_cast_preexisting(%arg0: memref<5x20x16x16xf16>) -> tensor<5x20x16x16xf16> {
  %alloc = memref.alloc() : memref<5x20x16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<5x20x16x16xf16>) outs(%alloc : memref<5x20x16x16xf16, #hivm.address_space<ub>>)
  %cast = memref.memory_space_cast %alloc : memref<5x20x16x16xf16, #hivm.address_space<ub>> to memref<5x20x16x16xf16>
  %t = bufferization.to_tensor %cast restrict writable : memref<5x20x16x16xf16>
  %e = tensor.empty() : tensor<5x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<5x20x16x16xf16>, tensor<5x20x16x16xf16>) outs(%e : tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16>
  return %r : tensor<5x20x16x16xf16>
}

// -----

// A select whose other arm is not traceable to an allocation (here a
// `hivm.hir.pointer_cast`) must not be reported as an unexpected root: the
// select is reached as a *user* of the tracked memref, so the arm is only an
// alternative candidate. The write on the select result is still detected.

// CHECK-LABEL: func.func @select_user_mixed_arm(
// CHECK: %[[T:.+]] = bufferization.to_tensor {{.*}} restrict writable : memref<2x4x16x16xf32>
// CHECK: %[[CP:.+]] = bufferization.alloc_tensor() copy(%[[T]]) {to_be_replaced} : tensor<2x4x16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @select_user_mixed_arm(%arg0: tensor<64x64xf32>, %addr: i64, %cond: i1) -> tensor<2x4x16x16xf32> {
  %ping = memref.alloc() : memref<2x4x16x16xf32, #hivm.address_space<ub>>
  %other = memref.alloc() : memref<2x4x16x16xf32, #hivm.address_space<ub>>
  %sel = arith.select %cond, %ping, %other : memref<2x4x16x16xf32, #hivm.address_space<ub>>
  hivm.hir.fixpipe ins(%arg0 : tensor<64x64xf32>) outs(%sel : memref<2x4x16x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
  %cast = memref.memory_space_cast %ping : memref<2x4x16x16xf32, #hivm.address_space<ub>> to memref<2x4x16x16xf32>
  %t = bufferization.to_tensor %cast restrict writable : memref<2x4x16x16xf32>
  return %t : tensor<2x4x16x16xf32>
}
}

// -----

// Same, but here the select is only *read* (no DPS write on it or on its
// arms). The untraceable arm must not abort the pass; nothing is exposed.

// CHECK-LABEL: func.func @select_read_only_mixed_arm(
// CHECK-NOT: to_be_replaced
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @select_read_only_mixed_arm(%addr: i64, %cond: i1) -> tensor<2x4x16x16xf32> {
  %ping = memref.alloc() : memref<2x4x16x16xf32, #hivm.address_space<ub>>
  %other = hivm.hir.pointer_cast(%addr) : memref<2x4x16x16xf32, #hivm.address_space<ub>>
  %sel = arith.select %cond, %ping, %other : memref<2x4x16x16xf32, #hivm.address_space<ub>>
  %e = tensor.empty() : tensor<2x4x16x16xf32>
  %r = hivm.hir.vadd ins(%sel, %sel : memref<2x4x16x16xf32, #hivm.address_space<ub>>, memref<2x4x16x16xf32, #hivm.address_space<ub>>) outs(%e : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
  %cast = memref.memory_space_cast %ping : memref<2x4x16x16xf32, #hivm.address_space<ub>> to memref<2x4x16x16xf32>
  %t = bufferization.to_tensor %cast restrict writable : memref<2x4x16x16xf32>
  return %t : tensor<2x4x16x16xf32>
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @marked
func.func @marked(%src: memref<16xf32, #hivm.address_space<gm>>, %condition: i1) -> tensor<16xf32> {
  %buffer = memref.alloc() : memref<16xf32>
  scope.scope : () -> () {
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%buffer : memref<16xf32>)
    scope.return
  } {hivm.load_preload = 1 : i32, no_inline}
  // CHECK: bufferization.to_tensor {{.*}} restrict writable
  // CHECK: bufferization.alloc_tensor() copy({{.*}}) {to_be_replaced}
  %tensor = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
  return %tensor : tensor<16xf32>
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @ordinary_scope
func.func @ordinary_scope(%src: memref<16xf32, #hivm.address_space<gm>>, %condition: i1) -> tensor<16xf32> {
  %buffer = memref.alloc() : memref<16xf32>
  scope.scope : () -> () {
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%buffer : memref<16xf32>)
    scope.return
  } {no_inline}
  // CHECK-NOT: bufferization.alloc_tensor
  %tensor = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
  // CHECK: return
  return %tensor : tensor<16xf32>
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @conditional_write
func.func @conditional_write(%src: memref<16xf32, #hivm.address_space<gm>>, %condition: i1) -> tensor<16xf32> {
  %buffer = memref.alloc() : memref<16xf32>
  scope.scope : () -> () {
    scf.if %condition {
      hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%buffer : memref<16xf32>)
    }
    scope.return
  } {hivm.load_preload = 1 : i32, no_inline}
  // CHECK-NOT: bufferization.alloc_tensor
  %tensor = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
  // CHECK: return
  return %tensor : tensor<16xf32>
}
}

// -----

// Trace a select operand back to an allocation with a prior write.
// CHECK-LABEL: func.func @select_true_written(
// CHECK: %[[T:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<16xf32>
// CHECK-NEXT: %[[COPY:.*]] = bufferization.alloc_tensor() copy(%[[T]]) {to_be_replaced}
// CHECK-NEXT: return %[[COPY]]
func.func @select_true_written(%condition: i1, %value: f32) -> tensor<16xf32> {
  %a = memref.alloc() : memref<16xf32>
  %b = memref.alloc() : memref<16xf32>
  hivm.hir.vbrc ins(%value : f32) outs(%a : memref<16xf32>)
  %selected = arith.select %condition, %a, %b : memref<16xf32>
  %t = bufferization.to_tensor %selected restrict writable : memref<16xf32>
  return %t : tensor<16xf32>
}

// -----

// An annotation on the allocation is accepted without exposing a DPS write.
// CHECK-LABEL: func.func @annotated_alloc_without_write(
// CHECK: annotation.mark
// CHECK: %[[T:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<16xf32>
// CHECK-NEXT: return %[[T]]
func.func @annotated_alloc_without_write() -> tensor<16xf32> {
  %buffer = memref.alloc() : memref<16xf32>
  annotation.mark %buffer {test_marker} : memref<16xf32>
  %t = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
  return %t : tensor<16xf32>
}

// -----

// Inserting a copy during the walk must not skip the next to_tensor.
// Replace every tensor use with the single copy for that to_tensor.
// CHECK-LABEL: func.func @walk_multiple_to_tensor(
// CHECK: %[[T0:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<16xf32>
// CHECK-NEXT: %[[C0:.*]] = bufferization.alloc_tensor() copy(%[[T0]]) {to_be_replaced}
// CHECK-NEXT: %[[T1:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<16xf32>
// CHECK-NEXT: %[[C1:.*]] = bufferization.alloc_tensor() copy(%[[T1]]) {to_be_replaced}
// CHECK-NEXT: return %[[C0]], %[[C0]], %[[C1]]
func.func @walk_multiple_to_tensor(%value: f32) -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) {
  %a = memref.alloc() : memref<16xf32>
  %b = memref.alloc() : memref<16xf32>
  hivm.hir.vbrc ins(%value : f32) outs(%a : memref<16xf32>)
  hivm.hir.vbrc ins(%value : f32) outs(%b : memref<16xf32>)
  %t0 = bufferization.to_tensor %a restrict writable : memref<16xf32>
  %t1 = bufferization.to_tensor %b restrict writable : memref<16xf32>
  return %t0, %t0, %t1 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
}
