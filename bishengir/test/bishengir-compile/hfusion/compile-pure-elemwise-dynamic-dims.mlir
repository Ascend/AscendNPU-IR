// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s

// CHECK: LLVMDialectModule
module {
  func.func @test_reshape_multi_dynamic_dims(%arg0: tensor<1x?x?xf32>, %arg1: tensor<1x?x?xf32>) -> tensor<1x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 45.254833995939045 : f64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %dim = tensor.dim %arg0, %c1 : tensor<1x?x?xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x?x?xf32>
    %0 = arith.truncf %cst : f64 to f32
    %1 = tensor.empty(%dim, %dim_1) : tensor<1x?x?xf32>
    %2 = linalg.fill ins(%0 : f32) outs(%1 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %2 : tensor<1x?x?xf32>, tensor<1x?x?xf32>) outs(%1 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %4 : tensor<1x?x?xf32>, tensor<1x?x?xf32>) outs(%1 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<1x?x?xf32>, tensor<1x?x?xf32>) outs(%1 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    return %6 : tensor<1x?x?xf32>
  }
}