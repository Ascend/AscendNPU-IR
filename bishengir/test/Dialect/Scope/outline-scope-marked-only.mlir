// RUN: bishengir-opt -outline-scope="outline-marked-scopes-only=true" -split-input-file %s | FileCheck %s

// -----

// Only scopes carrying the `outline` attribute are outlined. The unmarked
// scope must remain untouched, while the marked one becomes a call.
// CHECK: func.func @test_scope_with_outline_attr
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: scope.scope
// CHECK: scope.return %[[CST_0]]
// CHECK: call @test_scope_with_outline_attr_scope_0()
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_scope_with_outline_attr() -> (f32, f32){
    %cst = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = scope.scope : () -> (f32) {
      scope.return %cst : f32
    }
    %1 = scope.scope : () -> (f32) {
      scope.return %cst_1 : f32
    } {outline}
    return %0, %1 : f32, f32
  }
}

// -----

// The `outline`, `sym_func_name` and `call_attrs_vec` bookkeeping attributes
// are dropped from the outlined function. `sym_func_name` names the function
// when the symbol is free, and `call_attrs_vec` is restored on the call.
// CHECK-LABEL: func.func @named_callee(
// CHECK-NOT: outline
// CHECK-NOT: scope.sym_func_name
// CHECK-NOT: scope.call_attrs_vec
// CHECK-SAME: attributes {hivm.vector_function}
// CHECK-LABEL: func.func @named_scope_caller(
// CHECK: call @named_callee({{.*}}) {hivm.vector_function, ptc_simdvf}
module {
  func.func @named_scope_caller(%arg0: f32) -> f32 {
    %0 = scope.scope : () -> (f32) {
      scope.return %arg0 : f32
    } {hivm.vector_function, outline,
       scope.call_attrs_vec = #scope.call_attrs_vec<{ptc_simdvf}>,
       scope.sym_func_name = #scope.sym_func_name<"named_callee">}
    return %0 : f32
  }
}

// -----

// When the recorded `sym_func_name` is already taken by another symbol, the
// outlined function falls back to a unique name derived from the caller.
// CHECK-LABEL: func.func @taken_name(
// CHECK-NEXT: return
// CHECK-LABEL: func.func @taken_name_caller_scope_0(
// CHECK-NOT: scope.sym_func_name
// CHECK-LABEL: func.func @taken_name_caller(
// CHECK: call @taken_name_caller_scope_0
module {
  func.func @taken_name(%arg0: f32) -> f32 {
    return %arg0 : f32
  }
  func.func @taken_name_caller(%arg0: f32) -> f32 {
    %0 = scope.scope : () -> (f32) {
      scope.return %arg0 : f32
    } {outline, scope.sym_func_name = #scope.sym_func_name<"taken_name">}
    return %0 : f32
  }
}

// -----

// A SIMT scope with sub-block tiling reverted is wrapped in an scf.if guard;
// the recorded call-site attributes must be restored on the call inside it.
// CHECK-LABEL: func.func @simt_named(
// CHECK-LABEL: func.func @simt_named_caller(
// CHECK: scf.if
// CHECK: call @simt_named({{.*}}) {ptc_simdvf}
// CHECK: limit_sub_block_id0
module attributes {"hivm.tile_and_bind_subblock_reverted"} {
  func.func @simt_named_caller(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>, outline,
       scope.call_attrs_vec = #scope.call_attrs_vec<{ptc_simdvf}>,
       scope.sym_func_name = #scope.sym_func_name<"simt_named">}
    return
  }
}
