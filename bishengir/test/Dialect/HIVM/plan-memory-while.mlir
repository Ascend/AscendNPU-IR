// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1   \
// RUN:   -hivm-plan-memory -split-input-file -verify-diagnostics       \
// RUN:   | FileCheck %s

// -----
// Verify that PlanMemory's fixMultibufferEnabledPointerCastOps hoists the
// multi-address pointer_cast to the *after region* head (not before the
// scf.while), so that the cast is evaluated only when the loop actually
// enters its body. Hoisting before the while or into the before region
// would break semantics (the before region only evaluates the condition).

// CHECK-LABEL: func.func @while_plan_memory_basic(
func.func @while_plan_memory_basic(%arg0: memref<8xf32, #hivm.address_space<gm>>,
                                   %arg1: memref<8xf32, #hivm.address_space<gm>>) {
  %true = arith.constant true
  // CHECK: scf.while
  %r = scf.while (%cond = %true) : (i1) -> i1 {
    scf.condition(%cond) %cond : i1
  } do {
  // CHECK: ^bb0
  ^bb0(%cin: i1):
    // The hoisted multi-address hivm.hir.pointer_cast must appear at the
    // head of the after region's body, immediately followed by its
    // annotation.mark.
    // CHECK-NEXT: %[[PCAST:.*]] = hivm.hir.pointer_cast({{.*}}) : memref<8xf32, #hivm.address_space<ub>>
    // CHECK-NEXT: annotation.mark %[[PCAST]] {hivm.multi_buffer = 2 : i32}
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %tmp {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%tmp : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    scf.yield %cin : i1
  }
  return
}
