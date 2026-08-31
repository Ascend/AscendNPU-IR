// RUN: bishengir-opt %s -hivm-split-mix-kernel | FileCheck %s

// A vector region can contain nested loops with tensor results. When the
// region is filtered from the AIC side, each loop result must be forwarded to
// its matching iter_args init instead of reaching getOutOperands'
// unsupported-op failure.

// CHECK-LABEL: func.func @nested_scf_for_result_mix_aic(
// CHECK-NOT:     scope.scope
// CHECK-NOT:     scf.for
// CHECK-NOT:     memref.load
// CHECK:         return %arg1 : tensor<4x4xbf16>

// CHECK-LABEL: func.func @nested_scf_for_result_mix_aiv(
// CHECK:         scope.scope
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               memref.load
// CHECK:               tensor.insert

module {
  func.func @nested_scf_for_result(%src: memref<16xbf16>,
                                   %init: tensor<4x4xbf16>)
      -> tensor<4x4xbf16>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>,
                  mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %result = scope.scope : () -> tensor<4x4xbf16> {
      %outer = scf.for %i = %c0 to %c4 step %c1
          iter_args(%outer_iter = %init) -> tensor<4x4xbf16> {
        %inner = scf.for %j = %c0 to %c4 step %c1
            iter_args(%inner_iter = %outer_iter) -> tensor<4x4xbf16> {
          %offset = arith.muli %i, %c4 : index
          %linear = arith.addi %offset, %j : index
          %value = memref.load %src[%linear] : memref<16xbf16>
          %inserted = tensor.insert %value into %inner_iter[%i, %j]
              : tensor<4x4xbf16>
          scf.yield %inserted : tensor<4x4xbf16>
        } {ExtractedLoadOrStore, pipeline.veconly}
        scf.yield %inner : tensor<4x4xbf16>
      } {ExtractedLoadOrStore, pipeline.veconly}
      scope.return %outer : tensor<4x4xbf16>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return %result : tensor<4x4xbf16>
  }
}
