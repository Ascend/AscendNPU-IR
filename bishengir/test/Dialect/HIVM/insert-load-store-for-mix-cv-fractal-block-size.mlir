// RUN: bishengir-opt %s \
// RUN:   -hivm-insert-load-store-for-mix-cv="target=Ascend950PR_9599" \
// RUN:   | FileCheck %s

// Verify that for Ascend950 (A5) with f8E4M3FN (1-byte) element type,
// the B operand (isA=false, no b_transpose → isA==isTranspose) uses
// alignM=32 (fractal block size for A5 B8), not the hardcoded alignM=16.
//
// Without the fix (alignM=16): final expand_shape produces tensor<1x2x16x32xf8E4M3FN>
// With the fix (alignM=32):    final expand_shape produces tensor<1x1x32x32xf8E4M3FN>

// CHECK-LABEL: @test_ub2l1_fractal_block_size_a5_b8
// CHECK: tensor.expand_shape {{.*}} : tensor<1x32x32xf8E4M3FN> into tensor<1x1x32x32xf8E4M3FN>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">} {
  func.func @test_ub2l1_fractal_block_size_a5_b8(%arg0: tensor<32x32xf8E4M3FN>, %arg1: tensor<32x32xf8E4M3FN>)
      -> tensor<32x32xf32>
      attributes {hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %init = tensor.empty() : tensor<32x32xf32>
    %scope_result = scope.scope : () -> tensor<32x32xf8E4M3FN> {
      scope.return %arg1 : tensor<32x32xf8E4M3FN>
    }
    %res = hivm.hir.mmadL1 {already_set_real_mkn}
        ins(%arg0, %scope_result, %true, %c32, %c32, %c32
            : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>, i1, index, index, index)
        outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %res : tensor<32x32xf32>
  }
}
