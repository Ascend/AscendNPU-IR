// RUN: bishengir-opt %s -convert-proton-to-hivm | FileCheck %s

// CHECK-LABEL: func.func @on_910b
// CHECK: proton.record start "outer"
// CHECK: proton.record end "outer"
// CHECK-NOT: hivm.hir.proton_scope_boundary
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @on_910b() {
    proton.record start "outer"
    proton.record end "outer"
    return
  }
}
