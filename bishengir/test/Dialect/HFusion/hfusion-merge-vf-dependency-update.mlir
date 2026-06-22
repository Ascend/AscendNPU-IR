// RUN: bishengir-opt --hfusion-merge-vf="merge-level=1" %s | FileCheck %s

// The initial dependency graph is:
//
//   zero_vf -> reduce_a_vf
//           -> reduce_b_vf
//
// producer_vf can merge with zero_vf, and reduce_a_vf can merge with
// reduce_b_vf. After those merges, the dependency graph must be updated to:
//
//   producer_merged_vf -> reduce_a_merged_vf
//
// Otherwise the two merged VFs are incorrectly merged once more.

// CHECK-DAG: func.func @dependency_update_producer_merged_vf_0
// CHECK-DAG: func.func @dependency_update_reduce_a_merged_vf_0
// CHECK-NOT: func.func @dependency_update_producer_merged_merged_vf

func.func @dependency_update_producer_vf_0(%arg0: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<4xf32>
}

func.func @dependency_update_zero_vf_0(%arg0: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  return %cst : tensor<4xf32>
}

func.func @dependency_update_reduce_a_vf_0(%arg0: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<4xf32>
}

func.func @dependency_update_reduce_b_vf_0(%arg0: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @dependency_update_main
// CHECK-NOT: call @dependency_update_producer_merged_merged_vf
// CHECK: %[[PRODUCER_ZERO:.*]]:2 = {{.*}}call @dependency_update_producer_merged_vf_0
// CHECK: %[[REDUCTIONS:.*]]:2 = {{.*}}call @dependency_update_reduce_a_merged_vf_0
// CHECK-SAME: %[[PRODUCER_ZERO]]#1
// CHECK-SAME: %[[PRODUCER_ZERO]]#1
// CHECK: return %[[PRODUCER_ZERO]]#0, %[[REDUCTIONS]]#0, %[[REDUCTIONS]]#1
func.func @dependency_update_main(
    %producer_input: tensor<4xf32>,
    %zero_input: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %producer = func.call @dependency_update_producer_vf_0(%producer_input)
      {hivm.vector_function, no_inline}
      : (tensor<4xf32>) -> tensor<4xf32>
  %zero = func.call @dependency_update_zero_vf_0(%zero_input)
      {hivm.vector_function, no_inline}
      : (tensor<4xf32>) -> tensor<4xf32>
  %reduce_a = func.call @dependency_update_reduce_a_vf_0(%zero)
      {hivm.vector_function, no_inline}
      : (tensor<4xf32>) -> tensor<4xf32>
  %reduce_b = func.call @dependency_update_reduce_b_vf_0(%zero)
      {hivm.vector_function, no_inline}
      : (tensor<4xf32>) -> tensor<4xf32>
  return %producer, %reduce_a, %reduce_b
      : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}
