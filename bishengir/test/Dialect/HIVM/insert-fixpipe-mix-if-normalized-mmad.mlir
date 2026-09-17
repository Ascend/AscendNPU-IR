// RUN: bishengir-opt -hivm-insert-fixpipe %s | FileCheck %s

// Distilled from chunk_oja_bwd_kernel_dA: Cube mmad with normalized_in_L0C
// (no remain_in_l0c) is yielded through a mix scf.if whose else path is
// Vector (vsel / vbrc). getInsertPointOutOfIf must not hoist past the if
// just because isResultInL0C is true from normalized_in_L0C; fixpipe
// belongs after the mmad so the merged if-result is never treated as L0C.
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
// CHECK-LABEL: func.func @mmad_normalized_mix_if_fixpipe_inside
// CHECK: %[[IF:.*]] = scf.if
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {{.*}}fixpipe_for_result_already_inserted = true, normalized_in_L0C
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MMAD]]
// CHECK: scf.yield %[[FIX]]
// CHECK: } else {
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: tensor.extract_slice %[[IF]]
// CHECK: hivm.hir.store
func.func @mmad_normalized_mix_if_fixpipe_inside(
    %cond: i1, %eq: i1,
    %a: tensor<16x64xf32>,
    %b: tensor<64x16xf32>,
    %mask: tensor<16x16xi1>,
    %out: memref<?x?xf32, strided<[256, 1], offset: ?>>) {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %zero = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>)
      -> tensor<16x16xf32>
  %res = scf.if %cond -> (tensor<16x16xf32>) {
    %acc = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%a, %b, %true, %c16, %c64, %c16
            : tensor<16x64xf32>, tensor<64x16xf32>, i1, index, index, index)
        outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  } else {
    %inner = scf.if %eq -> (tensor<16x16xf32>) {
      %sel = hivm.hir.vsel ins(%mask, %zero, %cst
          : tensor<16x16xi1>, tensor<16x16xf32>, f32)
          outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %sel : tensor<16x16xf32>
    } else {
      scf.yield %zero : tensor<16x16xf32>
    }
    scf.yield %inner : tensor<16x16xf32>
  }
  %slice = tensor.extract_slice %res[0, 0] [%c16, %c16] [1, 1]
      : tensor<16x16xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%slice : tensor<?x?xf32>)
      outs(%out : memref<?x?xf32, strided<[256, 1], offset: ?>>)
  return
}
}
