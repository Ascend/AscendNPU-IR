// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops --split-input-file %s | FileCheck %s

// CHECK: func.func @test_load_store_0
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.store
// CHECK: func.func @test_load_store
// CHECK: call @test_load_store_0
module {
  func.func @test_load_store(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %cst = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %cst : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%1 : tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = hfusion.store ins(%1 : tensor<1xf32>) outs(%arg0 : tensor<1xf32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
  }
}

// -----

// CHECK: func.func @test_load_store_A_0
// CHECK: hfusion.load
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.store
// CHECK: func.func @test_load_store_A
// CHECK: call @test_load_store_A_0
module {
  func.func @test_load_store_A(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %0 = tensor.empty() : tensor<1xf32>
    %1 = hfusion.load ins(%arg1 : tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%2 : tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %3 = hfusion.store ins(%2 : tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    return %3 : tensor<1xf32>
  }
}