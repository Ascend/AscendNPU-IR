// RUN: bishengir-opt -inline-func-to-scope -split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: bishengir-opt -inline-func-to-scope -outline-scope -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=ROUNDTRIP

// A callee with more than one block cannot be inlined into the single-block
// scope.scope: the call is left untouched and the callee is kept.

// CHECK-LABEL: func.func @multi_block_callee
// CHECK-LABEL: func.func @multi_block_caller
// CHECK: call @multi_block_callee
// CHECK-NOT: scope.scope
// ROUNDTRIP-LABEL: func.func @multi_block_callee
// ROUNDTRIP-LABEL: func.func @multi_block_caller
// ROUNDTRIP: call @multi_block_callee
// ROUNDTRIP-NOT: scope.scope
func.func @multi_block_callee(%arg0: i1, %arg1: f32, %arg2: f32) -> f32
    attributes {hivm.vector_function} {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  return %arg1 : f32
^bb2:
  return %arg2 : f32
}

func.func @multi_block_caller(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
  // expected-warning @below {{skipping inlining: callee has multiple blocks}}
  %0 = func.call @multi_block_callee(%arg0, %arg1, %arg2) : (i1, f32, f32) -> f32
  return %0 : f32
}

// -----

// A call carrying the inherent `no_inline` marker records it in
// `call_attrs_vec`; the callee has none, so the scope itself carries no
// `no_inline`. The callee is erased after inlining and the marker is restored
// on the recreated call.

// CHECK-LABEL: func.func @call_no_inline_caller
// CHECK: scope.scope
// CHECK: scope.return
// CHECK: } {hivm.vector_function, outline, scope.call_attrs_vec = #scope.call_attrs_vec<{no_inline}>, scope.sym_func_name = #scope.sym_func_name<"call_no_inline_callee">}
// CHECK-NOT: call @
// CHECK-NOT: func.func @call_no_inline_callee
// ROUNDTRIP-LABEL: func.func @call_no_inline_callee
// ROUNDTRIP-LABEL: func.func @call_no_inline_caller
// ROUNDTRIP-NOT: scope.scope
// ROUNDTRIP: call @call_no_inline_callee({{.*}}) {hivm.vector_function, no_inline}
func.func @call_no_inline_callee(%arg0: f32) -> f32
    attributes {hivm.vector_function} {
  return %arg0 : f32
}

func.func @call_no_inline_caller(%arg0: f32) -> f32 {
  %0 = func.call @call_no_inline_callee(%arg0) {no_inline} : (f32) -> f32
  return %0 : f32
}

// -----

// Two call sites of the same callee: both are inlined and the callee is
// erased once no uses remain. On the way back, only one outlined function can
// take the original name; the other one falls back to a unique name.

// CHECK-LABEL: func.func @two_calls_caller(
// CHECK-COUNT-2: scope.sym_func_name = #scope.sym_func_name<"shared_callee">
// CHECK-NOT: call @
// CHECK-NOT: func.func @shared_callee
// ROUNDTRIP-LABEL: func.func @shared_callee
// ROUNDTRIP: func.func @two_calls_caller_scope_0
// ROUNDTRIP: func.func @two_calls_caller(
// ROUNDTRIP-NOT: scope.scope
// ROUNDTRIP: call @two_calls_caller_scope_0
// ROUNDTRIP: call @shared_callee
func.func @shared_callee(%arg0: f32) -> f32 attributes {hivm.vector_function} {
  %0 = arith.addf %arg0, %arg0 : f32
  return %0 : f32
}

func.func @two_calls_caller(%arg0: f32) -> f32 {
  %0 = func.call @shared_callee(%arg0) : (f32) -> f32
  %1 = func.call @shared_callee(%0) : (f32) -> f32
  return %1 : f32
}
