// RUN: bishengir-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_dead_single_return
func.func @test_dead_single_return() {
  // CHECK-NOT: scope.scope
  %cst = arith.constant 2.000000e+00 : f32
  %0 = scope.scope : () -> f32 {
    scope.return %cst : f32
  }
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: func.func @test_dead_multiple_returns
func.func @test_dead_multiple_returns() {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cst = arith.constant 3.0 : f32
  // CHECK-NOT: scope.scope
  %0:3 = scope.scope : () -> (i32, i32, f32) {
    scope.return %c1, %c2, %cst : i32, i32, f32
  }
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: func.func @test_partial_dead_returns
func.func @test_partial_dead_returns() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cst = arith.constant 3.0 : f32
  // CHECK: %[[RESULT:.*]] = scope.scope : () -> i32 {
  %0:3 = scope.scope : () -> (i32, i32, f32) {
    // CHECK: scope.return %{{.*}} : i32
    scope.return %c1, %c2, %cst : i32, i32, f32
  }
  // CHECK: return %[[RESULT]] : i32
  return %0#1 : i32
}

// -----

// CHECK-LABEL: func.func @test_all_returns_used
func.func @test_all_returns_used() -> (i32, f32) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 2.0 : f32
  // CHECK: %[[RESULT:.*]]:2 = scope.scope : () -> (i32, f32) {
  %0:2 = scope.scope : () -> (i32, f32) {
    // CHECK: scope.return %{{.*}}, %{{.*}} : i32, f32
    scope.return %c1, %cst : i32, f32
  }
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1 : i32, f32
  return %0#0, %0#1 : i32, f32
}

// -----

// CHECK-LABEL: func.func @test_nested_scopes_dead
func.func @test_nested_scopes_dead() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  // CHECK: %[[OUTER:.*]] = scope.scope : () -> i32 {
  %0:2 = scope.scope : () -> (i32, i32) {
    // CHECK: %[[INNER:.*]] = scope.scope : () -> i32 {
    %1:2 = scope.scope : () -> (i32, i32) {
      // CHECK: scope.return %{{.*}} : i32
      scope.return %c1, %c2 : i32, i32
    }
    // CHECK: scope.return %[[INNER]] : i32
    scope.return %1#0, %c2 : i32, i32
  }
  // CHECK: return %[[OUTER]] : i32
  return %0#0 : i32
}
