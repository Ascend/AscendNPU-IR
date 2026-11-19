// RUN: bishengir-opt %s -hfusion-legalize-scalar-fp8 -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @multiple_users
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (i8) {
// CHECK: %[[BITS:.*]] = arith.bitcast %arg1 : f8E4M3FN to i8
// CHECK: scf.yield %[[BITS]] : i8
// CHECK: } else {
// CHECK: %[[FALLBACK:.*]] = arith.bitcast %arg2 : f8E4M3FN to i8
// CHECK: scf.yield %[[FALLBACK]] : i8
// CHECK: %[[VALUE:.*]] = arith.bitcast %[[IF]] : i8 to f8E4M3FN
// CHECK: %[[EXT:.*]] = arith.extf %[[VALUE]] : f8E4M3FN to f32
// CHECK: return %[[VALUE]], %[[EXT]] : f8E4M3FN, f32
func.func @multiple_users(%cond: i1, %value: f8E4M3FN, %fallback: f8E4M3FN) -> (f8E4M3FN, f32) {
  %v = scf.if %cond -> (f8E4M3FN) {
    scf.yield %value : f8E4M3FN
  } else {
    scf.yield %fallback : f8E4M3FN
  }
  %e = arith.extf %v : f8E4M3FN to f32
  return %v, %e : f8E4M3FN, f32
}

// -----

// CHECK-LABEL: func.func @mixed_results
// CHECK: %[[IF:.*]]:3 = scf.if {{.*}} -> (i8, i32, i8) {
// CHECK: %[[A:.*]] = arith.bitcast %arg1 : f8E4M3FN to i8
// CHECK: %[[B:.*]] = arith.bitcast %arg2 : f8E5M2 to i8
// CHECK: scf.yield %[[A]], %arg3, %[[B]] : i8, i32, i8
// CHECK: } else {
// CHECK: %[[FALLBACK:.*]] = arith.bitcast %arg4 : f8E4M3FN to i8
// CHECK: scf.yield {{.*}} : i8, i32, i8
// CHECK-DAG: %[[X:.*]] = arith.bitcast %[[IF]]#0 : i8 to f8E4M3FN
// CHECK-DAG: %[[Y:.*]] = arith.bitcast %[[IF]]#2 : i8 to f8E5M2
// CHECK: return %[[X]], %[[IF]]#1, %[[Y]] : f8E4M3FN, i32, f8E5M2
func.func @mixed_results(%cond: i1, %a: f8E4M3FN, %b: f8E5M2, %i: i32, %fallback_a: f8E4M3FN)
    -> (f8E4M3FN, i32, f8E5M2) {
  %fallback = arith.constant 1.5 : f8E5M2
  %neg = arith.constant -1 : i32
  %v:3 = scf.if %cond -> (f8E4M3FN, i32, f8E5M2) {
    scf.yield %a, %i, %b : f8E4M3FN, i32, f8E5M2
  } else {
    scf.yield %fallback_a, %neg, %fallback : f8E4M3FN, i32, f8E5M2
  }
  return %v#0, %v#1, %v#2 : f8E4M3FN, i32, f8E5M2
}

// -----

// CHECK-LABEL: func.func @nested_if
// CHECK: scf.if {{.*}} -> (i8) {
// CHECK: scf.if {{.*}} -> (i8) {
// CHECK: arith.bitcast {{.*}} : f8E5M2 to i8
// CHECK-NOT: scf.if
// CHECK: arith.bitcast {{.*}} : i8 to f8E5M2
// CHECK: return {{.*}} : f8E5M2
func.func @nested_if(%outer: i1, %inner: i1, %value: f8E5M2) -> f8E5M2 {
  %fallback = arith.constant 1.5 : f8E5M2
  %v = scf.if %outer -> (f8E5M2) {
    %nested = scf.if %inner -> (f8E5M2) {
      scf.yield %value : f8E5M2
    } else {
      scf.yield %fallback : f8E5M2
    }
    scf.yield %nested : f8E5M2
  } else {
    scf.yield %fallback : f8E5M2
  }
  return %v : f8E5M2
}

// -----

// The bit pattern, including NaN payloads, is observable through an i8 user.
// CHECK-LABEL: func.func @preserve_nan_bits
// CHECK: arith.constant 127 : i8
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (i8) {
// CHECK: scf.yield %c127_i8 : i8
// CHECK: %[[RESTORED:.*]] = arith.bitcast %[[IF]] : i8 to f8E4M3FN
// CHECK: %[[BITS:.*]] = arith.bitcast %[[RESTORED]] : f8E4M3FN to i8
// CHECK: return %[[BITS]] : i8
func.func @preserve_nan_bits(%cond: i1, %value: f8E4M3FN) -> i8 {
  %nan = arith.constant 0x7F : f8E4M3FN
  %v = scf.if %cond -> (f8E4M3FN) {
    scf.yield %value : f8E4M3FN
  } else {
    scf.yield %nan : f8E4M3FN
  }
  %bits = arith.bitcast %v : f8E4M3FN to i8
  return %bits : i8
}

// -----

// CHECK-LABEL: func.func @unchanged_f16
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (f16) {
// CHECK-NOT: arith.bitcast
// CHECK: return %[[IF]] : f16
func.func @unchanged_f16(%cond: i1, %a: f16, %b: f16) -> f16 {
  %v = scf.if %cond -> (f16) {
    scf.yield %a : f16
  } else {
    scf.yield %b : f16
  }
  return %v : f16
}
