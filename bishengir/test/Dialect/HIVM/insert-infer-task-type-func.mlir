// RUN: bishengir-opt %s --hivm-insert-infer-task-type-func -split-input-file -verify-diagnostics | FileCheck %s

module {
  // CHECK: func.func @F1_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 10 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F1() attributes {hivm.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    return
  }
}

// -----

module {
  // CHECK: func.func @F2_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 20 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F2() attributes {hivm.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    return
  }
}

// -----

module {
  // CHECK: func.func @F3_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 32 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F3() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
}