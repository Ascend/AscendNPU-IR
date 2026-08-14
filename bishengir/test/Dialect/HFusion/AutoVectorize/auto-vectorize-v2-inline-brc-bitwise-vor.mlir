// RUN: bishengir-opt %s --hfusion-inline-brc --hfusion-auto-vectorize-v2 --outline-vector-function | FileCheck %s

// CHECK-LABEL: func.func @inline_fill_to_bitwise_vor_outlined_vf_0(
// CHECK-NOT: linalg.fill
// CHECK: arith.ori
// CHECK-NOT: linalg.fill
// CHECK-LABEL: func.func @inline_fill_to_bitwise_vor(
// CHECK-NOT: linalg.fill
// CHECK: func.call @inline_fill_to_bitwise_vor_outlined_vf_0
// CHECK-NOT: linalg.fill
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fill_to_bitwise_vor(%arg0: tensor<64x128xi8>) -> tensor<64x128xi8>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %mask = arith.constant 127 : i8
    %fill_empty = tensor.empty() : tensor<64x128xi8>
    %filled = linalg.fill ins(%mask : i8) outs(%fill_empty : tensor<64x128xi8>) -> tensor<64x128xi8>
    %result_empty = tensor.empty() : tensor<64x128xi8>
    %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%arg0, %filled : tensor<64x128xi8>, tensor<64x128xi8>) outs(%result_empty : tensor<64x128xi8>) -> tensor<64x128xi8>
    return %result : tensor<64x128xi8>
  }
}
