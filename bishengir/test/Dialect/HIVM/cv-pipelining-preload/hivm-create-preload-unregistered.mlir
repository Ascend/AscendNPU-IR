// RUN: bishengir-opt %s -create-preload -hivm-peel-preload -split-input-file | FileCheck %s

// Off-registry functions keep the unique-slot map: a later scope with the
// same preload_num overwrites the earlier one. The overwritten scope stays
// as scope.scope; only the last occupant is rewritten (then peeled). Constant
// trip >= 2 * max_preload_num folds the peel gate: prologue/epilogue are
// inlined and the middle loop drops preload_if guards.
// CHECK-LABEL: func.func @test_multiple_scopes_same_preload_num
// CHECK: "test.stage1_op"
// CHECK: scope.scope
// CHECK: "test.stage0_op_a"
// CHECK: "test.stage0_op_b"
// CHECK: scf.for
// CHECK: "test.stage1_op"
// CHECK: scope.scope
// CHECK: "test.stage0_op_a"
// CHECK: "test.stage0_op_b"
// CHECK: }
func.func @test_multiple_scopes_same_preload_num() {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32

  scf.for %i = %c0 to %c4 step %c1 : i32 {
    scope.scope : () -> () {
      "test.stage1_op"() : () -> ()
      scope.return
    } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}

    scope.scope : () -> () {
      "test.stage0_op_a"() : () -> ()
      scope.return
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}

    scope.scope : () -> () {
      "test.stage0_op_b"() : () -> ()
      scope.return
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
  }
  return
}
