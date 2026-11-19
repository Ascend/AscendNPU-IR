// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s

// Mixed-kernel matrix. A MIX kernel produces one vector function per
// contiguous group, both corrected to AIV. A function without HIVM compute is
// left untouched.

module {
  func.func private @opaque()

  func.func @mix(%a: tensor<64xf32>, %b: tensor<64xf32>,
                 %init: tensor<64xf32>) -> tensor<64xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>,
                  mix_mode = "mix"} {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    func.call @opaque() : () -> ()
    %1 = hivm.hir.vmul ins(%0, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }

  func.func @cpu(%a: tensor<64xf32>) -> tensor<64xf32> {
    return %a : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @mix_fused_0(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.vector_function
// CHECK-SAME: mix_mode = "aiv"
// CHECK-SAME: no_inline
// CHECK-LABEL: func.func private @mix_fused_1(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.vector_function
// CHECK-SAME: mix_mode = "aiv"
// CHECK-SAME: no_inline
// CHECK-LABEL: func.func @mix(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
// CHECK: call @mix_fused_1
// CHECK-SAME: {hivm.vector_function, no_inline}
// CHECK-NEXT: call @opaque() : () -> ()
// CHECK-NEXT: call @mix_fused_0
// CHECK-SAME: {hivm.vector_function, no_inline}
// CHECK-LABEL: func.func @cpu(
// CHECK-NOT: hivm.vector_function
// CHECK: return
