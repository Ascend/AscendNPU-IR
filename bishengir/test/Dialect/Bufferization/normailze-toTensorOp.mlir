// RUN: bishengir-opt %s -normalize-to-tensor -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -normalize-to-tensor -hivm-decompose-alloc-tensor-copy -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s --check-prefix=CHECK-ONE-SHOT

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

// CHECK-LABEL: func.func @dynamic_to_tensor(
// CHECK: %[[T:.+]] = bufferization.to_tensor %{{.+}} restrict writable {write_exposed} : memref<?x20x16x16xf16>
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

// Static shapes keep the same form: the copy operand provides the shape.

// CHECK-LABEL: func.func @static_to_tensor(
// CHECK: %[[T:.+]] = bufferization.to_tensor %{{.+}} restrict writable {write_exposed} : memref<5x20x16x16xf16>
// CHECK: %[[CP:.+]] = bufferization.alloc_tensor() copy(%[[T]]) {to_be_replaced} : tensor<5x20x16x16xf16>
// CHECK: hivm.hir.vadd ins(%[[CP]], %[[CP]] :
func.func @static_to_tensor(%arg0: memref<5x20x16x16xf16>) -> tensor<5x20x16x16xf16> {
  %alloc = memref.alloc() : memref<5x20x16x16xf16>
  hivm.hir.load ins(%arg0 : memref<5x20x16x16xf16>) outs(%alloc : memref<5x20x16x16xf16>)
  %t = bufferization.to_tensor %alloc restrict writable : memref<5x20x16x16xf16>
  %e = tensor.empty() : tensor<5x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<5x20x16x16xf16>, tensor<5x20x16x16xf16>) outs(%e : tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16>
  return %r : tensor<5x20x16x16xf16>
}

// -----

// Negative: no DPS write before to_tensor -> nothing is exposed.

// CHECK-LABEL: func.func @no_prior_write(
// CHECK-NOT: to_be_replaced
func.func @no_prior_write(%arg0: memref<5x20x16x16xf16>) -> tensor<5x20x16x16xf16> {
  %alloc = memref.alloc() : memref<5x20x16x16xf16>
  %t = bufferization.to_tensor %alloc restrict writable : memref<5x20x16x16xf16>
  %e = tensor.empty() : tensor<5x20x16x16xf16>
  %r = hivm.hir.vadd ins(%t, %t : tensor<5x20x16x16xf16>, tensor<5x20x16x16xf16>) outs(%e : tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16>
  return %r : tensor<5x20x16x16xf16>
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
// CHECK: %[[T:.+]] = bufferization.to_tensor %[[CAST]] restrict writable {write_exposed} : memref<5x20x16x16xf16>
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

// Negative: the DPS write on the function argument happens after the
// to_tensor and does not dominate it -> nothing is exposed.

// CHECK-LABEL: func.func @func_arg_write_after_to_tensor(
// CHECK-NOT: to_be_replaced
func.func @func_arg_write_after_to_tensor(%arg0: memref<5x20x16x16xf16>, %arg1: tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16> {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<5x20x16x16xf16>
  %e = tensor.empty() : tensor<5x20x16x16xf16>
  %r = hivm.hir.load ins(%t : tensor<5x20x16x16xf16>) outs(%e : tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16>
  return %r : tensor<5x20x16x16xf16>
}
