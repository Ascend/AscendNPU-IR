// RUN: bishengir-opt -test-index-bound-analyzer %s | FileCheck %s

// CHECK: constant: [32, 32]
// CHECK: max-lower-only: [0, ?]
// CHECK: min-upper-only: [?, 32]
// CHECK: clamped: [0, 32]
// CHECK: unknown: [?, ?]
func.func @index_bounds(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant {test.index_bound_label = "constant"} 32 : index

  %max = arith.maxsi %arg0, %c0 {test.index_bound_label = "max-lower-only"} : index
  %min = arith.minsi %arg0, %c32 {test.index_bound_label = "min-upper-only"} : index
  %clamped = arith.minsi %max, %c32 {test.index_bound_label = "clamped"} : index
  %unknown = arith.addi %arg0, %c32 {test.index_bound_label = "unknown"} : index

  return
}
