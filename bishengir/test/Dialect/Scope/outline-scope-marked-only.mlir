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
    } {outline = true}
    return %0, %1 : f32, f32
  }
}
