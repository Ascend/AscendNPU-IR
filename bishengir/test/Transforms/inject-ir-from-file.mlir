// RUN: bishengir-opt --inject-ir="file-path=%S/Inputs/inject-ir-inject.mlir" %s -o - | FileCheck %s
//
// Test that inject-ir loads IR from the given file and replaces
// matching function bodies in the current module.
//
// Case 1: function with no parameters (@foo).
// Case 2: function with parameters (@bar) - injected body uses the parameters.

module {
  // Original body is just "return 0". After inject pass, body should be
  // replaced with the one from inject-ir-inject.mlir (constant 42, return).
  func.func @foo() -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
  // Original body returns first arg. After inject, body should be replaced
  // with addi of the two parameters.
  func.func @bar(%arg0: i32, %arg1: i32) -> i32 {
    return %arg0 : i32
  }
}

// CHECK: module {
// CHECK:   func.func @foo() -> i32 {
// CHECK:     %[[c42:.*]] = arith.constant 42 : i32
// CHECK:     return %[[c42]] : i32
// CHECK:   }
// CHECK:   func.func @bar(%{{.*}}: i32, %{{.*}}: i32) -> i32 {
// CHECK:     %[[sum:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:     return %[[sum]] : i32
// CHECK:   }
// CHECK: }
