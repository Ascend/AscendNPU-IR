// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s

func.func @add_mul_reduce(%arg0: tensor<8xf32>, %arg1 : tensor<8xf32>, %arg2 : tensor<8xf32>, %arg3 : tensor<8xf32>) -> tensor<8xf32>
attributes {hacc.function_kind = #hacc.function_kind<HOST>}  {
  %1 = tensor.empty() : tensor<8xf32>
  %2 = linalg.elemwise_binary { mul, fun = #linalg.binary_fn<mul> } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>


  %4 = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%2, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%arg3 : tensor<8xf32>) -> tensor<8xf32>
  return %4 : tensor<8xf32>
}
