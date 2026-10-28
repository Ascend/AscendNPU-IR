// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -hivm-split-mix-kernel -split-input-file | FileCheck %s
//
// Root cause: annotateOpOperand walked elseYield() on scf.if with an empty
// else region (legal for result-less ifs). After SplitMixedIf tags a no-else
// cube if as cube_only, SplitMix erases that if from the AIV kernel and must
// not crash when annotating through it.

// CHECK-LABEL: @annotate_no_else_if_mix_aic
// CHECK:       scf.if
// CHECK:         hivm.hir.mmadL1
// CHECK:       } {{{.*}}hivm.cube_only}
// CHECK-NOT:   hivm.hir.vadd
// CHECK-LABEL: @annotate_no_else_if_mix_aiv
// CHECK-NOT:   hivm.hir.mmadL1
// CHECK:       hivm.hir.vadd
module {
  func.func @annotate_no_else_if(%cond: i1,
                                 %a: tensor<32x32xf16>,
                                 %bias: tensor<32x32xf32>)
      -> tensor<32x32xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c32 = arith.constant 32 : index
    %ce = tensor.empty() : tensor<32x32xf32>
    // Result-less, no-else if: Stage C tags cube_only; AIV erase path
    // annotateOpOperand(if) must guard elseYield().
    scf.if %cond {
      %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
          ins(%a, %a, %cond, %c32, %c32, %c32
              : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index)
          outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
      "test.consume"(%mm) : (tensor<32x32xf32>) -> ()
    }
    %1 = hivm.hir.vadd
        ins(%bias, %bias : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
