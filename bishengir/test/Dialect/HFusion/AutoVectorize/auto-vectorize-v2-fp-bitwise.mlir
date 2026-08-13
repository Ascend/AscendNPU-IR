// RUN: bishengir-opt %s --execution-engine-convert-hivm-to-upstream="convert-to-named-op=true" --hfusion-generalize --hfusion-auto-vectorize-v2 --outline-vector-function | FileCheck %s

// CHECK-LABEL: func.func @fp_vand_to_vf_outlined_vf_0(
// CHECK-NOT: linalg.map
// CHECK: arith.bitcast
// CHECK: arith.bitcast
// CHECK: arith.andi
// CHECK: arith.bitcast
// CHECK-NOT: linalg.map
// CHECK-LABEL: func.func @fp_vand_to_vf(
// CHECK-NOT: linalg.map
// CHECK: func.call @fp_vand_to_vf_outlined_vf_0
// CHECK-NOT: linalg.map
// CHECK-LABEL: func.func @fp_vor_to_vf_outlined_vf_0(
// CHECK-NOT: linalg.map
// CHECK: arith.bitcast
// CHECK: arith.bitcast
// CHECK: arith.ori
// CHECK: arith.bitcast
// CHECK-NOT: linalg.map
// CHECK-LABEL: func.func @fp_vor_to_vf(
// CHECK-NOT: linalg.map
// CHECK: func.call @fp_vor_to_vf_outlined_vf_0
// CHECK-NOT: linalg.map
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fp_vand_to_vf(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>) -> tensor<64xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<64xf32>
    %result = hivm.hir.vand ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
        outs(%dst : tensor<64xf32>) -> tensor<64xf32>
    return %result : tensor<64xf32>
  }

  func.func @fp_vor_to_vf(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>) -> tensor<64xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<64xf32>
    %result = hivm.hir.vor ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
        outs(%dst : tensor<64xf32>) -> tensor<64xf32>
    return %result : tensor<64xf32>
  }
}
