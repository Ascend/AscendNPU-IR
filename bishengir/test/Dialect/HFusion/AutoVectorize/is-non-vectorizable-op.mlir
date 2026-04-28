// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -outline-vector-function -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @scalar_pred_select_blocks_vectorization(
// CHECK: func.call {{.*}}vf_0
// CHECK: arith.select %arg3
// CHECK: func.call {{.*}}vf_1
module {
  func.func @scalar_pred_select_blocks_vectorization(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: i1) -> tensor<64xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %3 = arith.select %arg3, %2, %arg2 : tensor<64xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
}
