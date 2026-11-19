// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv %s | FileCheck %s

// Mem-based targets have no direct L0C->UB move. The mix-CV pass stages the
// L0C accumulator through GM by inserting a fixpipe, and that insertion is
// kept only when the Fixpipe catalogue says the candidate is Native.
// CHECK-LABEL: func.func @l0c_to_ub_inserts_native_fixpipe
// CHECK: %[[GM:.*]] = tensor.empty() {hivm.address_space = #hivm.address_space<gm>, "hivm.inserted-tensor"} : tensor<16x32xf32>
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, "hivm.inserted-fixpipe"}
// CHECK-SAME: ins(%[[ACC:.*]] : tensor<16x32xf32>) outs(%[[GM]] : tensor<16x32xf32>)
// CHECK: hivm.hir.load ins(%[[FIX]] : tensor<16x32xf32>)
// CHECK-SAME: {"hivm.inserted-load"} core_type = <VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @l0c_to_ub_inserts_native_fixpipe(
      %lhs: tensor<16x32xf32>, %rhs: tensor<32x32xf32>, %count: i32,
      %out: memref<16x32xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  mix_mode = "mix"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x32xf32>
    %while:2 = scf.while (%acc = %init, %n = %count)
        : (tensor<16x32xf32>, i32) -> (tensor<16x32xf32>, i32) {
      %cond = arith.cmpi sgt, %n, %c0_i32 : i32
      scf.condition(%cond) %acc, %n : tensor<16x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<16x32xf32>, %n: i32):
      %next = arith.subi %n, %c1_i32 : i32
      %mm = hivm.hir.mmadL1
          {already_set_real_mkn, fixpipe_for_result_already_inserted = true,
           hivm.remain_in_l0c, normalized_in_L0C}
          ins(%lhs, %rhs, %true, %c16, %c32, %c32
              : tensor<16x32xf32>, tensor<32x32xf32>, i1, index, index, index)
          outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
      scf.yield %mm, %next : tensor<16x32xf32>, i32
    } attributes {may_not_exec, normalized_in_L0C = [0 : i32]}
    %no_mm = arith.cmpi eq, %count, %c0_i32 : i32
    %merged = scf.if %no_mm -> (tensor<16x32xf32>) {
      %empty = tensor.empty() : tensor<16x32xf32>
      %zero = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x32xf32>)
          -> tensor<16x32xf32>
      scf.yield %zero : tensor<16x32xf32>
    } else {
      scf.yield %while#0 : tensor<16x32xf32>
    } {fallback_not_exec}
    hivm.hir.store ins(%merged : tensor<16x32xf32>) outs(%out : memref<16x32xf32>)
    return
  }
}
