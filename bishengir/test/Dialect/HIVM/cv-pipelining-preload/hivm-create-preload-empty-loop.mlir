// RUN: bishengir-opt %s -create-preload | FileCheck %s

// Empty loops (lb >= ub) must not grow by max_preload_num * step.

// CHECK-LABEL: func.func @empty_preload_loop
// CHECK-SAME: %[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32
// CHECK: %[[EXTRA:.*]] = arith.muli %[[STEP]], %{{.+}}
// CHECK: %[[EXPANDED:.*]] = arith.addi %[[UB]], %[[EXTRA]]
// CHECK: %[[NONEMPTY:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[NEWUB:.*]] = arith.select %[[NONEMPTY]], %[[EXPANDED]], %[[LB]]
// CHECK: scf.for {{.*}} = %[[LB]] to %[[NEWUB]] step %[[STEP]]
func.func @empty_preload_loop(%lb: i32, %ub: i32, %step: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %c0) -> (i32) : i32 {
    %s0 = scope.scope : () -> i32 {
      %r0 = arith.addi %acc, %i : i32
      scope.return %r0 : i32
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    scf.yield %s0 : i32
  }
  return %0 : i32
}
