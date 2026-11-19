// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --verify-each 2>&1 | FileCheck %s --implicit-check-not="AutoVectorizeV2 failed"

func.func @local_multiuse(%a: f32, %b: f32, %n: index, %dst: memref<8xf32>, %scalar: memref<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %v = vector.broadcast %s : f32 to vector<8xf32>
    memref.store %s, %scalar[%iv] : memref<?xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @local_multiuse(
// CHECK: scf.for
// CHECK: %[[S:.*]] = arith.addf {{.*}} : f32
// CHECK: %[[A:.*]] = vector.broadcast {{.*}} : f32 to vector<8xf32>
// CHECK: %[[B:.*]] = vector.broadcast {{.*}} : f32 to vector<8xf32>
// CHECK: %[[V:.*]] = arith.addf %[[A]], %[[B]] : vector<8xf32>
// CHECK-NOT: vector.broadcast %[[S]]
// CHECK: memref.store %[[S]],
// CHECK: vector.transfer_write %[[V]],
// CHECK: return

func.func @cast_chain(%a: f32, %b: f32, %n: index, %dst: memref<8xi8>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %i = arith.fptosi %s {round_mode = #hfusion.round_mode<truncwithoverflow>} : f32 to i32
    %j = arith.trunci %i {round_mode = #hfusion.round_mode<trunc>} : i32 to i8
    %v = vector.broadcast %j : i8 to vector<8xi8>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi8>, memref<8xi8>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @cast_chain(
// CHECK: vector.broadcast {{.*}} : f32 to vector<8xf32>
// CHECK: vector.broadcast {{.*}} : f32 to vector<8xf32>
// CHECK: %[[SUM:.*]] = arith.addf {{.*}} : vector<8xf32>
// CHECK: %[[I:.*]] = arith.fptosi %[[SUM]] {{.*}}round_mode = #hfusion.round_mode<truncwithoverflow>{{.*}} : vector<8xf32> to vector<8xi32>
// CHECK: %[[J:.*]] = arith.trunci %[[I]] {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} : vector<8xi32> to vector<8xi8>
// CHECK: vector.transfer_write %[[J]],
// CHECK: return

func.func @repeated_operand(%a: f32, %b: f32, %n: index, %dst: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %t = arith.mulf %s, %s : f32
    %v = vector.broadcast %t : f32 to vector<8xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @repeated_operand(
// CHECK: %[[ADD:.*]] = arith.addf {{.*}} : vector<8xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[ADD]] : vector<8xf32>
// CHECK: vector.transfer_write %[[MUL]],
// CHECK: return

func.func @outer_boundary(%a: i32, %b: i64, %n: index, %dst: memref<8xi64>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ext = arith.extsi %a : i32 to i64
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addi %ext, %b : i64
    %v = vector.broadcast %s : i64 to vector<8xi64>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi64>, memref<8xi64>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @outer_boundary(
// CHECK: %[[EXT:.*]] = arith.extsi {{.*}} : i32 to i64
// CHECK: scf.for
// CHECK: %[[E:.*]] = vector.broadcast %[[EXT]] : i64 to vector<8xi64>
// CHECK: %[[B64:.*]] = vector.broadcast {{.*}} : i64 to vector<8xi64>
// CHECK: %[[SUM64:.*]] = arith.addi %[[E]], %[[B64]] : vector<8xi64>
// CHECK: vector.transfer_write %[[SUM64]],
// CHECK: return

func.func @division(%a: f32, %b: f32, %n: index, %dst: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.divf %a, %b : f32
    %v = vector.broadcast %s : f32 to vector<8xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// Division stays scalar on purpose: the vector form flushes subnormal
// operands, measured on Ascend950PR (see AutoVectorizeV2.cpp comment).
// CHECK-LABEL: func.func @division(
// CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %{{.*}} : f32
// CHECK: %[[BC:.*]] = vector.broadcast %[[DIV]] : f32 to vector<8xf32>
// CHECK: vector.transfer_write %[[BC]],
// CHECK-NOT: arith.divf {{.*}} : vector<8xf32>
// CHECK: return

// fptosi carrying an explicit truncating round mode is value preserving when
// lifted, so it is vectorized together with its producer chain.
func.func @fptosi_trunc(%a: f32, %b: f32, %n: index, %dst: memref<8xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %i = arith.fptosi %s {round_mode = #hfusion.round_mode<trunc>} : f32 to i32
    %v = vector.broadcast %i : i32 to vector<8xi32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi32>, memref<8xi32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @fptosi_trunc(
// CHECK: %[[SUM:.*]] = arith.addf {{.*}} : vector<8xf32>
// CHECK: %[[I:.*]] = arith.fptosi %[[SUM]] {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} : vector<8xf32> to vector<8xi32>
// CHECK: vector.transfer_write %[[I]],
// CHECK: return

// fptosi without a round mode must stay scalar: lifting it would let the
// ArithToHIVMAVE lowering attach its default (non-truncating) round mode and
// change the value (e.g. 1.75 -> 2 instead of 1).
func.func @fptosi_no_round(%a: f32, %b: f32, %n: index, %dst: memref<8xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %i = arith.fptosi %s : f32 to i32
    %v = vector.broadcast %i : i32 to vector<8xi32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi32>, memref<8xi32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @fptosi_no_round(
// CHECK: %[[S:.*]] = arith.fptosi %{{.*}} : f32 to i32
// CHECK: %[[BC:.*]] = vector.broadcast %[[S]] : i32 to vector<8xi32>
// CHECK: vector.transfer_write %[[BC]],
// CHECK-NOT: arith.fptosi {{.*}} : vector<8xf32> to vector<8xi32>
// CHECK: return

// fptosi with a non-truncating round mode must stay scalar: the scalar path
// truncates while the lifted vector cast would round, so the value differs.
func.func @fptosi_round(%a: f32, %b: f32, %n: index, %dst: memref<8xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %i = arith.fptosi %s {round_mode = #hfusion.round_mode<round>} : f32 to i32
    %v = vector.broadcast %i : i32 to vector<8xi32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi32>, memref<8xi32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @fptosi_round(
// CHECK: %[[S:.*]] = arith.fptosi %{{.*}}round_mode = #hfusion.round_mode<round>{{.*}} : f32 to i32
// CHECK: %[[BC:.*]] = vector.broadcast %[[S]] : i32 to vector<8xi32>
// CHECK: vector.transfer_write %[[BC]],
// CHECK-NOT: arith.fptosi {{.*}} : vector<8xf32> to vector<8xi32>
// CHECK: return

// fptoui without a round mode must stay scalar for the same reason as fptosi.
func.func @fptoui_no_round(%a: f32, %b: f32, %n: index, %dst: memref<8xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %i = arith.fptoui %s : f32 to i32
    %v = vector.broadcast %i : i32 to vector<8xi32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi32>, memref<8xi32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @fptoui_no_round(
// CHECK: %[[S:.*]] = arith.fptoui %{{.*}} : f32 to i32
// CHECK: %[[BC:.*]] = vector.broadcast %[[S]] : i32 to vector<8xi32>
// CHECK: vector.transfer_write %[[BC]],
// CHECK-NOT: arith.fptoui {{.*}} : vector<8xf32> to vector<8xi32>
// CHECK: return
