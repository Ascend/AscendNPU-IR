// RUN: bishengir-opt %s -create-preload -split-input-file -allow-unregistered-dialect | FileCheck %s

// Test 2 scopes with preload_num=1 and preload_num=0, each yielded as iter_arg.

// CHECK-LABEL: func.func @test_two_scopes
func.func @test_two_scopes() -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  %c0_init = arith.constant 0 : i32

  // CHECK: scf.for {{.*}} = %c0{{.*}} to %c18{{.*}} step {{.*}}
  %0:2 = scf.for %i = %c0 to %c16 step %c1 iter_args(%arg0 = %c0_init, %arg1 = %c0_init) -> (i32, i32) : i32 {

    // Stage 0 (preload_num=1): scf.if with cmp guards
    // CHECK: arith.cmpi sge
    // CHECK-NEXT: arith.cmpi slt
    // CHECK-NEXT: arith.andi
    // CHECK-NEXT: scf.if
    %s0 = scope.scope : () -> i32 {
      %r0 = arith.addi %arg0, %i : i32
      scope.return %r0 : i32
    } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}

    // Stage 1 (preload_num=0): scf.if with cmp guards (uses shifted iv)
    // CHECK: arith.cmpi sge
    // CHECK-NEXT: arith.cmpi slt
    // CHECK-NEXT: arith.andi
    // CHECK-NEXT: scf.if
    %s1 = scope.scope : () -> i32 {
      %r1 = arith.addi %arg1, %i : i32
      scope.return %r1 : i32
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}

    scf.yield %s0, %s1 : i32, i32
  }
  // CHECK: return
  return %0#0, %0#1 : i32, i32
}
