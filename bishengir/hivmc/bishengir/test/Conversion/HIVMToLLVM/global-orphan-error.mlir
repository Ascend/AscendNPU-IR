// RUN: not hivmc-a5 %s --enable-lir-compile=false --target=Ascend910B2 2>&1 | FileCheck %s
// GlobalOps with no user trigger the error in hivmc.
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK:      Unknown core type
  memref.global @public_global : memref<3x7xf32>
  func.func @foo() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    return
  }
  func.func @bar() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    return
  }
}
