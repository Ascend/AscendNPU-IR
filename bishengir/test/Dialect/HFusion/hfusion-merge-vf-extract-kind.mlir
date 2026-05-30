// RUN: bishengir-opt -hfusion-merge-vf %s -split-input-file | FileCheck %s

// Extracted-result VF + non-extracted-result VF must NOT merge.
// Without the new guard, these two VFs are mergeable because vf_1 does not
// depend on the tensor.extract result of vf_0.

// CHECK-NOT: extract_mismatch_merged_vf
// CHECK-LABEL: func.func @extract_mismatch_vf_0
func.func @extract_mismatch_vf_0(%arg0: tensor<1xf32>) -> tensor<1xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<1xf32>
}

// CHECK-LABEL: func.func @extract_mismatch_vf_1
func.func @extract_mismatch_vf_1(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @extract_mismatch_main
func.func @extract_mismatch_main(%arg0: tensor<1xf32>,
                                 %arg1: tensor<32x128xf32>)
    -> (f32, tensor<32x128xf32>) {
  // CHECK: %[[MISMATCH_CALL0:.*]] = {{.*}}call @extract_mismatch_vf_0
  // CHECK-NEXT: tensor.extract %[[MISMATCH_CALL0]]
  // CHECK-NEXT: {{.*}}call @extract_mismatch_vf_1
  // CHECK-NOT: extract_mismatch_merged_vf

  %c0 = arith.constant 0 : index
  %0 = func.call @extract_mismatch_vf_0(%arg0) {hivm.vector_function, no_inline}
      : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %0[%c0] : tensor<1xf32>

  %1 = func.call @extract_mismatch_vf_1(%arg1) {hivm.vector_function, no_inline}
      : (tensor<32x128xf32>) -> tensor<32x128xf32>

  return %extracted, %1 : f32, tensor<32x128xf32>
}

// -----

// Extracted-result VF + extracted-result VF should still merge.

// CHECK-LABEL: func.func @same_extract_merged_vf_0
// CHECK-SAME: -> (tensor<1xf32>, tensor<1xf32>)
// CHECK-NOT: func.func @same_extract_vf_0
// CHECK-NOT: func.func @same_extract_vf_1
func.func @same_extract_vf_0(%arg0: tensor<1xf32>) -> tensor<1xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<1xf32>
}

func.func @same_extract_vf_1(%arg0: tensor<1xf32>) -> tensor<1xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<1xf32>
}

// CHECK-LABEL: func.func @same_extract_main
func.func @same_extract_main(%arg0: tensor<1xf32>,
                             %arg1: tensor<1xf32>) -> (f32, f32) {
  // CHECK: %[[SAME_EXTRACT_CALL:[A-Za-z0-9_]+]]:2 = {{.*}}call @same_extract_merged_vf_0
  // CHECK: tensor.extract %[[SAME_EXTRACT_CALL]]#0
  // CHECK: tensor.extract %[[SAME_EXTRACT_CALL]]#1

  %c0 = arith.constant 0 : index
  %0 = func.call @same_extract_vf_0(%arg0) {hivm.vector_function, no_inline}
      : (tensor<1xf32>) -> tensor<1xf32>
  %extracted0 = tensor.extract %0[%c0] : tensor<1xf32>

  %1 = func.call @same_extract_vf_1(%arg1) {hivm.vector_function, no_inline}
      : (tensor<1xf32>) -> tensor<1xf32>
  %extracted1 = tensor.extract %1[%c0] : tensor<1xf32>

  return %extracted0, %extracted1 : f32, f32
}

// -----

// Non-extracted-result VF + non-extracted-result VF should still merge.

// CHECK-LABEL: func.func @same_no_extract_merged_vf_0
// CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: func.func @same_no_extract_vf_0
// CHECK-NOT: func.func @same_no_extract_vf_1
func.func @same_no_extract_vf_0(%arg0: tensor<4xf32>) -> tensor<4xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<4xf32>
}

func.func @same_no_extract_vf_1(%arg0: tensor<4xf32>) -> tensor<4xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %arg0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @same_no_extract_main
func.func @same_no_extract_main(%arg0: tensor<4xf32>,
                                %arg1: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK: %[[SAME_NO_EXTRACT_CALL:[A-Za-z0-9_]+]]:2 = {{.*}}call @same_no_extract_merged_vf_0
  // CHECK: return %[[SAME_NO_EXTRACT_CALL]]#0, %[[SAME_NO_EXTRACT_CALL]]#1

  %0 = func.call @same_no_extract_vf_0(%arg0) {hivm.vector_function, no_inline}
      : (tensor<4xf32>) -> tensor<4xf32>
  %1 = func.call @same_no_extract_vf_1(%arg1) {hivm.vector_function, no_inline}
      : (tensor<4xf32>) -> tensor<4xf32>

  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}
