// RUN: bishengir-opt -fix-call-unknown-loc --mlir-print-debuginfo %s -split-input-file | FileCheck %s

// Test 1: func.call with UnknownLoc inherits location from a result user op.
// The call and its user should share the same location alias after the pass.

func.func private @callee() -> f32

// CHECK-LABEL: func.func @test_fix_from_user
func.func @test_fix_from_user() {
  %0 = func.call @callee() : () -> f32 loc(unknown)
  // CHECK: call @callee() : () -> f32 loc(#[[FIXED_LOC:.*]])
  %1 = arith.addf %0, %0 : f32 loc("user_loc")
  // CHECK: arith.addf %0, %0 : f32 loc(#[[FIXED_LOC]])
  return
}

// -----

// Test 2: func.call with UnknownLoc and no result user inherits from a
// parent op (the enclosing func, which has a parser-assigned location).
// The call should have a non-UnknownLoc after the pass.

func.func private @void_callee() -> ()

// CHECK-LABEL: func.func @test_fix_from_parent
func.func @test_fix_from_parent() {
  func.call @void_callee() : () -> () loc(unknown)
  // CHECK: call @void_callee() : () -> () loc(#
  return
}

// -----

// Test 3: llvm.call with UnknownLoc inherits location from a result user op.
// The call and its user should share the same location alias.

llvm.func @llvm_callee() -> f32

// CHECK-LABEL: func.func @test_llvm_call_fix_from_user
func.func @test_llvm_call_fix_from_user() {
  %0 = llvm.call @llvm_callee() : () -> f32 loc(unknown)
  // CHECK: llvm.call @llvm_callee() : () -> f32 loc(#[[LLVM_FIXED_LOC:.*]])
  %1 = arith.addf %0, %0 : f32 loc("user_loc")
  // CHECK: arith.addf %0, %0 : f32 loc(#[[LLVM_FIXED_LOC]])
  return
}

// -----

// Test 4: func.call with a proper non-UnknownLoc - pass preserves it.

func.func private @callee3() -> f32

// CHECK-LABEL: func.func @test_call_with_proper_loc
func.func @test_call_with_proper_loc() {
  %0 = func.call @callee3() : () -> f32 loc("already_has_loc")
  // CHECK: call @callee3() : () -> f32 loc(#[[KEEP_LOC:.*]])
  // CHECK: #[[KEEP_LOC]] = loc("already_has_loc")
  return
}

// -----

// Test 5: llvm.call with a proper non-UnknownLoc - pass preserves it.

llvm.func @llvm_callee2() -> f32

// CHECK-LABEL: func.func @test_llvm_call_with_proper_loc
func.func @test_llvm_call_with_proper_loc() {
  %0 = llvm.call @llvm_callee2() : () -> f32 loc("already_has_loc")
  // CHECK: llvm.call @llvm_callee2() : () -> f32 loc(#[[KEEP_LLVM_LOC:.*]])
  // CHECK: #[[KEEP_LLVM_LOC]] = loc("already_has_loc")
  return
}
