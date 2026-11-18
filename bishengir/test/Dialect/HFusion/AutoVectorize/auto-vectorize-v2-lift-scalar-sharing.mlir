// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --verify-each 2>&1 | FileCheck %s --implicit-check-not="AutoVectorizeV2 failed"
// Distinct full vector types must not share a cached result. Two roots with
// the same full type do share, even if worklist order visits the later one first.
func.func @shared_roots(%a: f32, %b: f32, %n: index, %d8: memref<8xf32>, %d4: memref<4xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %early = vector.broadcast %s : f32 to vector<8xf32>
    vector.transfer_write %early, %d8[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
    %small = vector.broadcast %s : f32 to vector<4xf32>
    vector.transfer_write %small, %d4[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
    %late = vector.broadcast %s : f32 to vector<8xf32>
    vector.transfer_write %late, %d8[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @shared_roots(
// CHECK-DAG: %[[V8:.*]] = arith.addf {{.*}} : vector<8xf32>
// CHECK-DAG: %[[V4:.*]] = arith.addf {{.*}} : vector<4xf32>
// CHECK-NOT: arith.addf
// CHECK: vector.transfer_write %[[V8]],
// CHECK: vector.transfer_write %[[V4]],
// CHECK: vector.transfer_write %[[V8]],
// CHECK: return

// An unequal-length path reaches the same scalar producer in different rounds.
func.func @unequal_paths(%a: f32, %b: f32, %n: index, %dst: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s = arith.addf %a, %b : f32
    %l = arith.mulf %s, %a : f32
    %ll = arith.addf %l, %b : f32
    %r = arith.mulf %s, %b : f32
    %out = arith.addf %ll, %r : f32
    %v = vector.broadcast %out : f32 to vector<8xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @unequal_paths(
// CHECK: %[[SHARED:.*]] = arith.addf {{.*}} : vector<8xf32>
// CHECK: %[[L:.*]] = arith.mulf %[[SHARED]], {{.*}} : vector<8xf32>
// CHECK: %[[LL:.*]] = arith.addf %[[L]], {{.*}} : vector<8xf32>
// CHECK: %[[R:.*]] = arith.mulf %[[SHARED]], {{.*}} : vector<8xf32>
// CHECK: %[[OUT:.*]] = arith.addf %[[LL]], %[[R]] : vector<8xf32>
// CHECK: vector.transfer_write %[[OUT]],
// CHECK: return

// A deep linear chain must fully lift in a single driver invocation: the
// worklist picks up each broadcast this pattern creates for an operand, so no
// scalar arithmetic may remain even though only the outermost broadcast was
// seeded. Guards against a regression if the per-round full re-walk is removed.
func.func @deep_chain(%a: f32, %b: f32, %n: index, %dst: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %s0 = arith.addf %a, %b : f32
    %s1 = arith.mulf %s0, %a : f32
    %s2 = arith.subf %s1, %b : f32
    %s3 = arith.maxnumf %s2, %a : f32
    %v = vector.broadcast %s3 : f32 to vector<8xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  } {"outlined-loop-target-1"}
  return
}
// CHECK-LABEL: func.func @deep_chain(
// CHECK: arith.addf {{.*}} : vector<8xf32>
// CHECK: arith.mulf {{.*}} : vector<8xf32>
// CHECK: arith.subf {{.*}} : vector<8xf32>
// CHECK: arith.maxnumf {{.*}} : vector<8xf32>
// CHECK-NOT: arith.addf {{.*}} : f32
// CHECK-NOT: arith.mulf {{.*}} : f32
// CHECK-NOT: arith.subf {{.*}} : f32
// CHECK-NOT: arith.maxnumf {{.*}} : f32
// CHECK: vector.transfer_write
// CHECK: return
