// RUN: bishengir-compile --enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20  %s | FileCheck %s

// CHECK: LLVMDialectModule
module {
  func.func @fused_mul_pow_mul_split(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32>
  attributes {OperatorType = "Default", compute_capability = "", hacc.function_kind = #hacc.function_kind<DEVICE>,
  mindspore_kernel, process = "aicore"} {
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %cst = arith.constant dense<0.51> : tensor<1xf32>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%1, %cst : tensor<1xf32>, tensor<1xf32>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    %cst_0 = arith.constant dense<0.001953125> : tensor<1xf32>
    %4 = tensor.empty() : tensor<1xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %cst_0 : tensor<1xf32>, tensor<1xf32>) outs(%arg2 : tensor<1xf32>) -> tensor<1xf32>
    return %5 : tensor<1xf32>
  }
}