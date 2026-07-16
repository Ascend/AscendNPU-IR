// RUN: bishengir-opt %s  -hfusion-fold-extract-insert-pair | FileCheck %s

// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert

func.func @test_fold_extract_insert_pair(%arg0: i32, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %tmp = tensor.empty() : tensor<1xf32>
  %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg1, %arg2 : tensor<1xf32>, tensor<1xf32>) outs(%tmp : tensor<1xf32>) -> tensor<1xf32>
  %extracted_14 = tensor.extract %sub[%c0] : tensor<1xf32>
  %inserted_15 = tensor.insert %extracted_14 into %tmp[%c0] : tensor<1xf32>
  %exp = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%inserted_15 : tensor<1xf32>) outs(%tmp : tensor<1xf32>) -> tensor<1xf32>
  return %exp : tensor<1xf32>
}
