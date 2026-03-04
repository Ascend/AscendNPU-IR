// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s

// CHECK-LABEL: ; ModuleID = 'LLVMDialectModule'
// CHECK: ptr @Fused_Mul_Add_fusion_3414230412939713453_0_0, !"kernel"
module {
  func.func @Fused_Mul_Add_fusion_3414230412939713453(%arg0: tensor<8x?xf16>, %arg1: tensor<8x?xf16>, %arg2: tensor<8x?xf16>) -> tensor<8x?xf16> attributes {OperatorType = "Elementwise", compute_capability = "", frontend_symbol = {input_0 = ["8", "s13"], input_1 = ["8", "s14"], input_2 = ["8", "s16"], output_0 = ["8", "s17"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<8x?xf16>
    %0 = tensor.empty(%dim) : tensor<8x?xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8x?xf16>, tensor<8x?xf16>) outs(%0 : tensor<8x?xf16>) -> tensor<8x?xf16>
    %dim_0 = tensor.dim %1, %c1 : tensor<8x?xf16>
    %2 = tensor.empty(%dim_0) : tensor<8x?xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<8x?xf16>, tensor<8x?xf16>) outs(%2 : tensor<8x?xf16>) -> tensor<8x?xf16>
    return %3 : tensor<8x?xf16>
  }
}