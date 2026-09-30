// RUN: bishengir-opt %s --hivm-vectorize-ops --cse --canonicalize | FileCheck %s

// The regbase verifier must accept unsigned i16 -> i32 casts, and
// vectorization must preserve zero-extension semantics.
// CHECK-LABEL: func.func @cast_u16_to_u32(
// CHECK: %[[INPUT:.*]] = vector.transfer_read
// CHECK: %[[EXT:.*]] = arith.extui %[[INPUT]] : vector<64xi16> to vector<64xi32>
// CHECK: vector.transfer_write %[[EXT]],
// CHECK-NOT: hivm.hir.vcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cast_u16_to_u32(%input: tensor<64xi16>) -> tensor<64xi32> attributes {hivm.vector_function} {
    %init = tensor.empty() : tensor<64xi32>
    %out = hivm.hir.vcast ins(%input : tensor<64xi16>) outs(%init : tensor<64xi32>) cast = <cast_unsigned> -> tensor<64xi32>
    return %out : tensor<64xi32>
  }
}
