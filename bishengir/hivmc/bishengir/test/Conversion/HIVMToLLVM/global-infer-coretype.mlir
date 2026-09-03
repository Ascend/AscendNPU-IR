// UNSUPPORTED: bishengir_published
// RUN: hivmc-a5 %s --enable-lir-compile=false --target=Ascend910B2 2>&1 | FileCheck %s
// GlobalOps stay in the same module with the parent function of its user.
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-NOT    Unknown core type
  // CHECK:       ; ModuleID = {{.*}}
  // CHECK-LABEL: define void @bar()
  // CHECK:       ret void
  // CHECK:       ; ModuleID = {{.*}}
  // CHECK:       @public_global = external global [3 x [7 x float]]
  // CHECK-LABEL: define float @foo()
  // CHECK:       load float, ptr @public_global
  // CHECK:       ret float
  memref.global @public_global : memref<3x7xf32>
  func.func @foo() -> f32 attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %0 = memref.get_global @public_global : memref<3x7xf32>
    %c0 = arith.constant 0 : index
    %v = memref.load %0[%c0, %c0] : memref<3x7xf32>
    return %v : f32
  }
  func.func @bar() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    return
  }
}
