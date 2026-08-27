// RUN: bishengir-opt --mark-simt-scope-no-inline --split-input-file %s | FileCheck %s

// SIMT scopes (vector_mode = "simt") are marked no_inline so the generic
// inline-scope pass keeps them outlined.
// CHECK-LABEL: func.func @simt_scope_marked
// CHECK: scope.scope : () -> tensor<i32> {
// CHECK: arith.addi
// CHECK: scope.return
// CHECK: } {no_inline, vector_mode = "simt"}
module {
  func.func @simt_scope_marked(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = scope.scope : () -> tensor<i32> {
      %r = arith.addi %arg0, %arg1 : tensor<i32>
      scope.return %r : tensor<i32>
    } {vector_mode = "simt"}
    return %0 : tensor<i32>
  }
}

// -----

// SIMT scopes that already carry no_inline are left unchanged.
// CHECK-LABEL: func.func @simt_scope_already_marked
// CHECK: } {no_inline, vector_mode = "simt"}
module {
  func.func @simt_scope_already_marked(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = scope.scope : () -> tensor<i32> {
      %r = arith.addi %arg0, %arg1 : tensor<i32>
      scope.return %r : tensor<i32>
    } {no_inline, vector_mode = "simt"}
    return %0 : tensor<i32>
  }
}

// -----

// Non-SIMT scopes are left untouched.
// CHECK-LABEL: func.func @normal_scope_untouched
// CHECK: scope.scope : () -> tensor<i32> {
// CHECK: scope.return
// CHECK-NOT: no_inline
// CHECK: }
module {
  func.func @normal_scope_untouched(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = scope.scope : () -> tensor<i32> {
      %r = arith.addi %arg0, %arg1 : tensor<i32>
      scope.return %r : tensor<i32>
    }
    return %0 : tensor<i32>
  }
}
