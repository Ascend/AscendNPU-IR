// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect -split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_scope_scope(
func.func @test_scope_scope() {
  // CHECK: scope.scope
  scope.scope : () -> () {
    // CHECK: scope.return
    scope.return
  // CHECK: {tcore_type = #hivm.tcore_type<CUBE>}
  } {tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// CHECK-LABEL: func.func @test_scope_scope_no_inline(
func.func @test_scope_scope_no_inline() {
  // CHECK: scope.scope
  scope.scope : () -> () {
    // CHECK: scope.return
    scope.return
  // CHECK: {no_inline}
  } {no_inline}
  return
}

// -----

// CHECK-LABEL: func.func @test_scope_with_yields(
func.func @test_scope_with_yields() -> f32 {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[YIELD:.*]] = scope.scope : () -> f32 {
  %0 = scope.scope : () -> f32 {
    // CHECK: scope.return %[[CST]] : f32
    scope.return %cst : f32
  }
  // CHECK: return %[[YIELD]] : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @test_scope_with_yields(
func.func @test_scope_with_yields() {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: scope.scope : () -> f32 {
  scope.scope : () -> f32 {
    // CHECK: scope.return %[[CST]] : f32
    scope.return %cst : f32
  }
  return
}
