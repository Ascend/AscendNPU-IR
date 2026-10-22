// RUN: bishengir-opt %s -convert-proton-to-hivm | FileCheck %s

// Bit 10 is always set; bit 11 remains 0 for start and 1 for end.
// A scope owns start = logical_id | 0x400 and end = logical_id | 0xC00.
// The outer end is emitted after the inner scope: 1024, 1025, 3073, 3072.
// CHECK-LABEL: func.func @nested
// CHECK: arith.constant 1024 : i32
// CHECK: hivm.hir.custom
// CHECK-SAME: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 1025 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 3073 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 3072 : i32
// CHECK: "__builtin_proton_scope_boundary"

// Logical ids continue across functions in the module.
// An outer scope stays open across scf.if; each branch takes the next id.
// CHECK-LABEL: func.func @across_region
// CHECK: arith.constant 1026 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 1027 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 3075 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 1028 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 3076 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK: arith.constant 3074 : i32
// CHECK: "__builtin_proton_scope_boundary"
// CHECK-NOT: proton.record

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @nested() {
    proton.record start "outer"
    proton.record start "inner"
    proton.record end "inner"
    proton.record end "outer"
    return
  }

  func.func @across_region(%cond: i1) {
    proton.record start "outer"
    scf.if %cond {
      proton.record start "then_branch"
      proton.record end "then_branch"
    } else {
      proton.record start "else_branch"
      proton.record end "else_branch"
    }
    proton.record end "outer"
    return
  }
}
