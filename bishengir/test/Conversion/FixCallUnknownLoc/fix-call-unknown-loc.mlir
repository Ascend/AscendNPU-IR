// RUN: bishengir-opt -fix-call-unknown-loc --mlir-print-debuginfo %s -split-input-file | FileCheck %s

// Test 1: func.call with UnknownLoc inherits location from a result user op.
// The call and its user should share the same location alias after the pass.

func.func private @callee() -> f32

// CHECK-LABEL: func.func @test_fix_from_user
func.func @test_fix_from_user() {
  %0 = func.call @callee() : () -> f32 loc(unknown)
  // CHECK: call @callee() : () -> f32 loc(#[[FIXED_LOC:.*]])
  %1 = arith.addf %0, %0 : f32 loc("user_loc")
  // CHECK: arith.addf %0, %0 : f32 loc(#[[FIXED_LOC1:.*]])
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
  // CHECK: arith.addf %0, %0 : f32 loc(#[[LLVM_FIXED_LOC1:.*]])
  return
}

// -----

// Test 4: func.call with a proper non-UnknownLoc - pass preserves it.

func.func private @callee3() -> f32

// CHECK-LABEL: func.func @test_call_with_proper_loc
func.func @test_call_with_proper_loc() {
  %0 = func.call @callee3() : () -> f32 loc("already_has_loc":1:1)
  // CHECK: call @callee3() : () -> f32 loc(#[[KEEP_LOC:.*]])
  // CHECK: #[[KEEP_LOC]] = loc("already_has_loc":1:1)
  return
}

// -----

// Test 5: llvm.call with a proper non-UnknownLoc - pass preserves it.

llvm.func @llvm_callee2() -> f32

// CHECK-LABEL: func.func @test_llvm_call_with_proper_loc
func.func @test_llvm_call_with_proper_loc() {
  %0 = llvm.call @llvm_callee2() : () -> f32 loc("already_has_loc":1:1)
  // CHECK: llvm.call @llvm_callee2() : () -> f32 loc(#[[KEEP_LLVM_LOC:.*]])
  // CHECK: #[[KEEP_LLVM_LOC]] = loc("already_has_loc":1:1)
  return
}

// -----

// Test 6: An external func.func declaration whose own loc is unknown inherits
// a location from its call site (caller's enclosing func loc). This is the
// key case for library function declarations (func.func private @ciface_...)
// which have no body and would otherwise be skipped by the body-walk.

func.func private @extern_void_callee() -> () loc(unknown)

// CHECK: func.func private @extern_void_callee(){{.*}} loc(#[[DECL_LOC6:.*]])
// CHECK-LABEL: func.func @test_fix_func_decl_from_caller
func.func @test_fix_func_decl_from_caller() {
  func.call @extern_void_callee() : () -> () loc("caller_site_loc":1:1)
  return
}

// -----

// Test 7: A func.call whose loc is a NameLoc wrapping UnknownLoc is treated
// as effectively unknown and fixed by inheriting from the result user. This
// mirrors how LLVMDIScope::extractFileLoc unwraps NameLoc to reach the inner
// UnknownLoc; the pass must recognize such wrappers and fix them.

func.func private @callee7() -> f32

// CHECK-LABEL: func.func @test_fix_nameloc_wrapping_unknown
func.func @test_fix_nameloc_wrapping_unknown() {
  // The call's loc is NameLoc("wrapped", UnknownLoc) -> effectively unknown.
  %0 = func.call @callee7() : () -> f32 loc("wrapped")
  // CHECK: call @callee7() : () -> f32 loc(#[[FIXED_LOC7:.*]])
  %1 = arith.addf %0, %0 : f32 loc("user_loc7":1:1)
  // CHECK: arith.addf %0, %0 : f32 loc(#[[FIXED_LOC7]])
  return
}

// -----

// Test 8: A func.func declaration whose own loc is a NameLoc wrapping UnknownLoc
// is treated as effectively unknown and fixed by inheriting from its caller.

func.func private @extern_void_callee8() -> () loc("wrapped_decl8")

// CHECK: func.func private @extern_void_callee8(){{.*}} loc(#[[DECL_LOC8:.*]])
// CHECK-LABEL: func.func @test_fix_func_decl_nameloc_wrapping_unknown
func.func @test_fix_func_decl_nameloc_wrapping_unknown() {
  func.call @extern_void_callee8() : () -> () loc("caller_site_loc8":1:1)
  return
}

// -----

// Test 9: CallSiteLoc and FusedLoc wrapping unknown are treated as effectively
// unknown. Both call ops are fixed from their result users.

func.func private @callee_cs() -> f32
func.func private @callee_fl() -> f32

// CHECK-LABEL: func.func @test_wrapped_unknown
func.func @test_wrapped_unknown() {
  %0 = func.call @callee_cs() : () -> f32 loc(callsite(unknown at unknown))
  // CHECK: call @callee_cs() : () -> f32 loc(#
  %1 = arith.addf %0, %0 : f32 loc("user_cs":1:1)
  %2 = func.call @callee_fl() : () -> f32 loc(fused[unknown, unknown])
  // CHECK: call @callee_fl() : () -> f32 loc(#
  %3 = arith.addf %2, %2 : f32 loc("user_fl":1:1)
  return
}

// -----

// Test 10: findNonUnknownLocForFunc's three fallback priorities in one split.
// - @extern_walk: external decl loc(unknown), called with loc(unknown) ->
//   Priority 1 walks call site's parents.
// - @func_body: has body, loc(unknown), no callers -> Priority 2.
// - @extern_unused: external decl loc(unknown), no callers -> Priority 3.

func.func private @extern_walk() -> () loc(unknown)

// CHECK: func.func private @extern_walk(){{.*}} loc(#
// CHECK-LABEL: func.func @test_func_fallbacks
func.func @test_func_fallbacks() {
  func.call @extern_walk() : () -> () loc(unknown)
  // CHECK: call @extern_walk() : () -> () loc(#
  return
} loc("parent_func":1:1)

// CHECK: func.func @func_body
// CHECK: arith.constant
// CHECK: return
// CHECK: } loc(#
func.func @func_body() {
  %0 = arith.constant 1.0 : f32 loc("body_op":1:1)
  return
} loc(unknown)

// CHECK: func.func private @extern_unused(){{.*}} loc(#
func.func private @extern_unused() -> () loc(unknown)
