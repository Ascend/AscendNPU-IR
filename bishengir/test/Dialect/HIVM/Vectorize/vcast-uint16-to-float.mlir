// RUN: bishengir-opt %s --hivm-vectorize-ops --cse --canonicalize | FileCheck %s --implicit-check-not=hivm.hir.vcast

// The regbase verifier must accept unsigned i16 -> f32 casts with rint
// rounding, and vectorization must preserve unsigned conversion semantics.
// CHECK-LABEL: func.func @cast_u16_to_f32(
// CHECK: %[[INPUT:.*]] = vector.transfer_read
// CHECK: %[[CAST:.*]] = arith.uitofp %[[INPUT]] : vector<64xi16> to vector<64xf32>
// CHECK: %[[OUTPUT:.*]] = vector.transfer_write %[[CAST]],
// CHECK: return %[[OUTPUT]] : tensor<64xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_u16_to_f32(%input: tensor<64xi16>) -> tensor<64xf32> attributes {hivm.vector_function} {
    %init = tensor.empty() : tensor<64xf32>
    %out = hivm.hir.vcast ins(%input : tensor<64xi16>) outs(%init : tensor<64xf32>) round_mode = <rint> cast = <cast_unsigned> -> tensor<64xf32>
    return %out : tensor<64xf32>
  }
}
