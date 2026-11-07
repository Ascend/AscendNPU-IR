// RUN: bishengir-opt %s -hivm-insert-infer-task-type-func | FileCheck %s

// Outlining places VFs before their caller. Triton loads the callback using the
// entry name and must receive MIX 1:1 (31), rather than fall back to Cube-only.
// CHECK-NOT: @early_vf_infer_task_type_function
// CHECK-LABEL: func.func @kernel_infer_task_type_function() -> i8
// CHECK: hacc.function_kind = #hacc.function_kind<HOST>
// CHECK: %[[TYPE:.*]] = arith.constant 31 : i8
// CHECK: return %[[TYPE]] : i8
module {
  func.func private @early_vf() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    return
  }
  func.func @kernel() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    return
  }
}
