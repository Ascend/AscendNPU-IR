// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s
 
// CHECK: LLVMDialectModule
module {
  func.func @matmul_add_mul(%arg0: tensor<1024x1024xf16>, %arg1: tensor<1024x1024xf16>, %arg2: tensor<1024x1024xf16>, %arg3: tensor<1024x1024xf16>, %arg4: tensor<1024x1024xf16>) -> tensor<1024x1024xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
    %0 = tensor.empty() : tensor<1024x1024xf16>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %2 = tensor.empty() : tensor<1024x1024xf16>
    %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%2 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, sub} ins(%3, %arg3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%arg4 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    return %4 : tensor<1024x1024xf16>
  }
}