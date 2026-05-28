// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -outline-vector-function -split-input-file 2>&1 | FileCheck %s
//
//==============================================================================
// TEST SPECIFICATION: DeinterleaveOp Shape Alignment for VF Fusion
//==============================================================================
//
// PURPOSE: Verify that when DeinterleaveOp is fused with downstream consumers,
// the loop shape is computed from result types (output shape [N]) rather than
// operand types (input shape [2*N]), enabling hasCommonAxis fusion.
//
// BEFORE the fix (6096d101d): DeinterleaveOp shape reflected input [2*N],
// preventing fusion with downstream [N] consumers.
// AFTER the fix: DeinterleaveOp shape [N] aligns with consumers [N].
//
// COVERAGE:
//   - deinterleave_consumer_fusion: Deinterleave [128]→[64] + elemwise on [64]
//   - deinterleave_elemwise_chain: Deinterleave + chain of elemwise ops
//   - deinterleave_bf16: bf16 type variant
//==============================================================================

// TEST_SPEC: deinterleave_consumer_fusion | Shape: 128xf16 → 64xf16
//   DeinterleaveOp input [128] → output [64], consumer on [64].
//   EXPECTED: Both fused into one _outlined_vf_ function.
// CHECK-LABEL: func @deinterleave_consumer_fusion(
// CHECK: _outlined_vf_
module {
  func.func @deinterleave_consumer_fusion(%arg0: tensor<128xf16>) -> tensor<64xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<64xf16>
    %cst = arith.constant 1.000000e+00 : f16
    %1 = hfusion.deinterleave %arg0 channel<0> : tensor<128xf16> -> tensor<64xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %cst : tensor<64xf16>, f16) outs(%0 : tensor<64xf16>) -> tensor<64xf16>
    return %2 : tensor<64xf16>
  }
}

// -----

// TEST_SPEC: deinterleave_elemwise_chain | Shape: 32xf32 → 16xf32
//   Deinterleave [32]→[16] + 3 elemwise ops on [16].
//   EXPECTED: All fused into one _outlined_vf_ function.
// CHECK-LABEL: func @deinterleave_elemwise_chain(
// CHECK: _outlined_vf_
module {
  func.func @deinterleave_elemwise_chain(%arg0: tensor<32xf32>) -> tensor<16xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<16xf32>
    %cst = arith.constant -1.000000e+00 : f32
    %1 = hfusion.deinterleave %arg0 channel<1> : tensor<32xf32> -> tensor<16xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %cst : tensor<16xf32>, f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%2 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
}

// -----

// TEST_SPEC: deinterleave_bf16 | Shape: 256xbf16 → 128xbf16
//   bf16 type variant — verify shape fix works across element types.
//   EXPECTED: Fused into one _outlined_vf_ function.
// CHECK-LABEL: func @deinterleave_bf16(
// CHECK: _outlined_vf_
module {
  func.func @deinterleave_bf16(%arg0: tensor<256xbf16>) -> tensor<128xbf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<128xbf16>
    %cst = arith.constant 1.000000e+00 : bf16
    %1 = hfusion.deinterleave %arg0 channel<1> : tensor<256xbf16> -> tensor<128xbf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %cst : tensor<128xbf16>, bf16) outs(%0 : tensor<128xbf16>) -> tensor<128xbf16>
    return %2 : tensor<128xbf16>
  }
}
