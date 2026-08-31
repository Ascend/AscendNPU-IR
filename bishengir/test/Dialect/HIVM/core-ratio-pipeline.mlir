// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-insert-infer-task-type-func \
// RUN:   -split-input-file | FileCheck %s

// CHECK: hivm.module_core_type = #hivm.module_core_type<MIX>
// CHECK: func.func @vector_body_0_1_infer_task_type_function
// CHECK: arith.constant 41 : i8
// CHECK: func.func @vector_body_0_1(
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<0, 1>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
module {
  func.func @vector_body_0_1(%base : memref<?xf32>,
                             %indices : tensor<8xi64>,
                             %dst : tensor<8xf32>) -> tensor<8xf32>
      attributes {hacc.entry, hivm.core_ratio = #hivm.core_ratio<0, 1>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%dst : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// CHECK: hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK: func.func @vector_body_no_ratio_infer_task_type_function
// CHECK: arith.constant 10 : i8
// CHECK: func.func @vector_body_no_ratio(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
module {
  func.func @vector_body_no_ratio(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %dst : tensor<8xf32>) -> tensor<8xf32>
      attributes {hacc.entry} {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%dst : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
