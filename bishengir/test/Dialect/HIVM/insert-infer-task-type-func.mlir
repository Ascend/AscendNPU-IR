// RUN: bishengir-opt %s --hivm-insert-infer-task-type-func -split-input-file -verify-diagnostics | FileCheck %s

module {
  // CHECK: func.func @F1_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 10 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F1() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    return
  }
}

// -----

module {
  // CHECK: func.func @F2_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 20 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F2() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
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

// -----

module {
  // CHECK: func.func @F4_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 41 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F4() attributes {hacc.entry,
                              hivm.func_core_type = #hivm.func_core_type<MIX>,
                              hivm.core_ratio = #hivm.core_ratio<0, 1>} {
    return
  }
}

// -----

module {
  // CHECK: func.func @F5_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 30 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F5() attributes {hacc.entry,
                              hivm.func_core_type = #hivm.func_core_type<MIX>,
                              hivm.core_ratio = #hivm.core_ratio<1, 0>} {
    return
  }
}

// -----

module {
  // CHECK: func.func @F6_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 31 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F6() attributes {hacc.entry,
                              hivm.func_core_type = #hivm.func_core_type<MIX>,
                              hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    return
  }
}

// -----

module {
  // CHECK: func.func @F7_infer_task_type_function() -> i8
  // CHECK: %[[TASK_TYPE:.*]] = arith.constant 32 : i8
  // CHECK: return %[[TASK_TYPE]]
  func.func @F7() attributes {hacc.entry,
                              hivm.func_core_type = #hivm.func_core_type<MIX>,
                              hivm.core_ratio = #hivm.core_ratio<1, 2>} {
    return
  }
}
