// RUN: bishengir-opt %s --convert-linalg-to-hfusion --hfusion-inline-brc | FileCheck %s --check-prefix=INLINE
// RUN: bishengir-opt %s --convert-linalg-to-hfusion --hfusion-inline-brc --hfusion-auto-vectorize-v2 --outline-vector-function | FileCheck %s --check-prefix=VF

// INLINE-LABEL: func.func @inline_fill_to_bitwise_map(
// INLINE: %[[MASK:.*]] = arith.constant 127 : i8
// INLINE-NOT: linalg.fill
// INLINE-NOT: linalg.map
// INLINE: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, %[[MASK]] : tensor<64x128xi8>, i8)
// INLINE-NOT: linalg.map

// VF-LABEL: func.func @inline_fill_to_bitwise_map_outlined_vf_0(
// VF-NOT: linalg.fill
// VF: arith.andi
// VF-NOT: linalg.fill
// VF-NOT: func.func @inline_fill_to_bitwise_map_outlined_vf_
// VF-LABEL: func.func @inline_fill_to_bitwise_map(
// VF-NOT: linalg.fill
// VF: func.call @inline_fill_to_bitwise_map_outlined_vf_0
// VF-NOT: linalg.fill
// VF-NOT: func.func @inline_fill_to_bitwise_map_outlined_vf_
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fill_to_bitwise_map(%arg0: tensor<64x128xi8>) -> tensor<64x128xi8>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %mask = arith.constant 127 : i8
    %fill_empty = tensor.empty() : tensor<64x128xi8>
    %filled = linalg.fill ins(%mask : i8) outs(%fill_empty : tensor<64x128xi8>) -> tensor<64x128xi8>
    %result_empty = tensor.empty() : tensor<64x128xi8>
    %mapped = linalg.map { arith.andi } ins(%arg0, %filled : tensor<64x128xi8>, tensor<64x128xi8>) outs(%result_empty : tensor<64x128xi8>)
    return %mapped : tensor<64x128xi8>
  }
}
