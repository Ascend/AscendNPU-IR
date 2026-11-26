// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s

// The outline stage extracts a selected group into a private vector function.
// Escaping results flow through the function result, the call site keeps the
// group order, and inherited core metadata is corrected to AIV.

module {
  func.func @mix(%a: tensor<64xf32>, %b: tensor<64xf32>,
                 %init: tensor<64xf32>) -> tensor<64xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>,
                  mix_mode = "mix"} {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %product = hivm.hir.vmul ins(%sum, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %product : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @mix_fused_0(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.vector_function
// CHECK-SAME: mix_mode = "aiv"
// CHECK-SAME: no_inline
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: return
// CHECK-LABEL: func.func @mix(
// CHECK: call @mix_fused_0
// CHECK-SAME: {hivm.vector_function, no_inline}
// CHECK-NOT: hivm.hir.vadd
