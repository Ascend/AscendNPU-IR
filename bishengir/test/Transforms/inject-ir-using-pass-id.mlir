// RUN: bishengir-compile --print-pass-id --inject-ir-after=canonicalize/module/0@%S/Inputs/inject-ir-inject.mlir --mlir-print-ir-after-all %s | FileCheck %s

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // Original body is just "return 0". After inject pass, body should be
  // replaced with the one from inject-ir-inject.mlir (constant 42, return).
  func.func @foo()->i32 { 
    %c0 = arith.constant 0 : i32 
    return %c0 : i32 
    }
}

// ============================================================================
// Part 1: Verify PassID
// ============================================================================

// CHECK: [PassID] canonicalize/module/0
