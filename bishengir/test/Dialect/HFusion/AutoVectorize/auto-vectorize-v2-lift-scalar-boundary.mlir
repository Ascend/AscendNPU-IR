// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --verify-each | FileCheck %s
// Complementary mechanism probes: select predicate dtype, boundaries, effects.
func.func @cmp_select(%a: f32, %b: f32, %n: index, %dst: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %p = arith.cmpf olt, %a, %b : f32
    %s = arith.select %p, %a, %b : f32
    %v = vector.broadcast %s : f32 to vector<8xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @cmp_select(
// CHECK: %[[P:.*]] = arith.cmpf olt, {{.*}} : vector<8xf32>
// CHECK: %[[S:.*]] = arith.select %[[P]], {{.*}} : vector<8xi1>, vector<8xf32>
// CHECK: vector.transfer_write %[[S]],

func.func @memory_leaf(%src: memref<?xf32>, %n: index, %dst: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = memref.load %src[%iv] : memref<?xf32>
    %v = vector.broadcast %s : f32 to vector<8xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @memory_leaf(
// CHECK: %[[S:.*]] = memref.load
// CHECK: %[[V:.*]] = vector.broadcast %[[S]] : f32 to vector<8xf32>
// CHECK: vector.transfer_write %[[V]],

func.func @outer_direct(%a: i32, %n: index, %dst: memref<8xi64>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s = arith.extsi %a : i32 to i64
  scf.for %iv = %c0 to %n step %c1 {
    %v = vector.broadcast %s : i64 to vector<8xi64>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xi64>, memref<8xi64>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @outer_direct(
// CHECK: %[[S:.*]] = arith.extsi {{.*}} : i32 to i64
// CHECK: scf.for
// CHECK: %[[V:.*]] = vector.broadcast %[[S]] : i64 to vector<8xi64>
// CHECK: vector.transfer_write %[[V]],
