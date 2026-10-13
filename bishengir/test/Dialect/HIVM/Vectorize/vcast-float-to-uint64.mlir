// RUN: bishengir-opt %s --hivm-vectorize-ops --cse --canonicalize | FileCheck %s --implicit-check-not=hivm.hir.vcast

// Accept the regbase f32 -> u64 truncating cast and preserve its unsigned
// semantics during vectorization, rather than lowering it as fptosi.
// CHECK-LABEL: func.func @cast_f32_to_u64(
// CHECK: %[[INPUT:.*]] = vector.transfer_read
// CHECK: %[[CAST:.*]] = arith.fptoui %[[INPUT]] : vector<32xf32> to vector<32xi64>
// CHECK: %[[OUTPUT:.*]] = vector.transfer_write %[[CAST]],
// CHECK: return %[[OUTPUT]] : tensor<32xi64>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_f32_to_u64(%input: tensor<32xf32>) -> tensor<32xi64> attributes {hivm.vector_function} {
    %init = tensor.empty() : tensor<32xi64>
    %out = hivm.hir.vcast ins(%input : tensor<32xf32>) outs(%init : tensor<32xi64>) round_mode = <trunc> cast = <cast_unsigned> -> tensor<32xi64>
    return %out : tensor<32xi64>
  }
}
