// RUN: bishengir-opt -test-index-bound-analyzer %s | FileCheck %s

// CHECK: constant: [32, 32]
// CHECK: max-lower-only: [0, ?]
// CHECK: min-upper-only: [?, 32]
// CHECK: clamped: [0, 32]
// CHECK: unknown: [?, ?]
// CHECK: iv-plus-4: [4, 35]
// CHECK: iv-plus-4 <= 35: sat
// CHECK: iv-plus-4-too-large: [4, 35]
// CHECK: iv-plus-4-too-large <= 3: unsat
// CHECK: iv: [0, 31]
// CHECK: unknown-compare: [?, ?]
// CHECK: unknown-compare <= 31: unknown
func.func @index_bounds(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant {test.index_bound_label = "constant"} 32 : index

  %max = arith.maxsi %arg0, %c0 {test.index_bound_label = "max-lower-only"} : index
  %min = arith.minsi %arg0, %c32 {test.index_bound_label = "min-upper-only"} : index
  %clamped = arith.minsi %max, %c32 {test.index_bound_label = "clamped"} : index
  %unknown = arith.addi %arg0, %c32 {test.index_bound_label = "unknown"} : index

  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c32 step %c4 {
    %iv_plus_4 = arith.addi %iv, %c4 {test.index_bound_compare_le = 35 : index, test.index_bound_label = "iv-plus-4"} : index
    arith.addi %iv, %c4 {test.index_bound_compare_le = 3 : index, test.index_bound_label = "iv-plus-4-too-large"} : index
    arith.addi %iv, %c0 {test.index_bound_label = "iv"} : index
  }

  arith.addi %arg0, %c0 {test.index_bound_compare_le = 31 : index, test.index_bound_label = "unknown-compare"} : index

  return
}

// -----

// CHECK-DAG: overlap-lhs: [0, 10]
// CHECK-DAG: overlap-lhs < overlap-rhs: unknown
// CHECK-DAG: overlap-rhs: [5, 20]
func.func @overlapping_index_bound_compare(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %lhs_lower = arith.maxsi %arg0, %c0 : index
  arith.minsi %lhs_lower, %c10 {test.index_bound_compare_lt_label = "overlap-rhs", test.index_bound_label = "overlap-lhs"} : index
  %rhs_lower = arith.maxsi %arg0, %c5 : index
  arith.minsi %rhs_lower, %c20 {test.index_bound_label = "overlap-rhs"} : index

  return
}
