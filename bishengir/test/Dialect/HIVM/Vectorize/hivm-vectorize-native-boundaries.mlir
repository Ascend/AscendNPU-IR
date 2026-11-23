// RUN: bishengir-opt %s --hivm-vectorize --hivm-vectorize-buffers --verify-each | FileCheck %s

module {
  // CHECK-LABEL: func.func private @mixed_compute_fused_0
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
  // CHECK-SAME: hivm.vector_function, mix_mode = "aiv", no_inline
  // CHECK: arith.addf
  func.func @mixed_compute(%a: tensor<64xf32>, %b: tensor<64xf32>,
                           %init: tensor<64xf32>) -> tensor<64xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %result = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %result : tensor<64xf32>
  }

  // CHECK-LABEL: func.func @cbuf_fill
  // CHECK: hivm.hir.vbrc
  // CHECK-NOT: call
  // CHECK: return
  // CHECK-NOT: @cbuf_fill_fused
  func.func @cbuf_fill(%value: f16, %dst: memref<2048xf16, #hivm.address_space<cbuf>>) {
    hivm.hir.vbrc ins(%value : f16)
        outs(%dst : memref<2048xf16, #hivm.address_space<cbuf>>)
    return
  }
}
