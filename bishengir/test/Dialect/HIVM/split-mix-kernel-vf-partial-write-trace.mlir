// RUN: bishengir-opt %s -hivm-split-mix-kernel -split-input-file | FileCheck %s
//
// Root cause: VF out-operand recovery returned transfer_write arg indices even
// when their count did not match the call's result count, then asserted in
// replaceResultWithInitOperand. Partial traces must fall back to stubs/attrs.

// CHECK-LABEL: @vf_partial_write_trace_mix_aic
// CHECK:       hivm.hir.matmul
// CHECK-NOT:   call @vf_partial_write
// CHECK-LABEL: @vf_partial_write_trace_mix_aiv
// CHECK:       call @vf_partial_write
// CHECK:       hivm.hir.vadd
module {
  // Two results, but only the first return traces to transfer_write → size 1
  // vs getNumResults() 2.
  func.func @vf_partial_write(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>)
      -> (tensor<16xf32>, tensor<16xf32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f32
    %v = vector.transfer_read %arg0[%c0], %pad
        : tensor<16xf32>, vector<16xf32>
    %w = vector.transfer_write %v, %arg1[%c0]
        : vector<16xf32>, tensor<16xf32>
    return %w, %arg0 : tensor<16xf32>, tensor<16xf32>
  }

  func.func @vf_partial_write_trace(%lhs: tensor<16x16xf16>,
                                    %rhs: tensor<16x16xf16>,
                                    %dst: tensor<16x16xf16>,
                                    %t0: tensor<16xf32>,
                                    %t1: tensor<16xf32>,
                                    %vout: tensor<16xf32>)
      -> (tensor<16x16xf16>, tensor<16xf32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %vf:2 = func.call @vf_partial_write(%t0, %t1) {hivm.vector_function}
        : (tensor<16xf32>, tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>)
    %mm = hivm.hir.matmul
        ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
        outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>
    %va = hivm.hir.vadd
        ins(%vf#0, %vout : tensor<16xf32>, tensor<16xf32>)
        outs(%vout : tensor<16xf32>) -> tensor<16xf32>
    return %mm, %va : tensor<16x16xf16>, tensor<16xf32>
  }
}
