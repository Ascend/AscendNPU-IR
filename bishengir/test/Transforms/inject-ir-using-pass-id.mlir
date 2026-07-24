// RUN: (bishengir-compile --print-pass-id \
// RUN:   --inject-ir-after=canonicalize-module/module/0@%S/Inputs/inject-ir-inject.mlir \
// RUN:   %s 2>&1 || true) | FileCheck %s

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @foo() -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}

// CHECK: [PassID] canonicalize-module/module/0
// CHECK: [InjectIR] replaced module at canonicalize-module/module/0
