// RUN: bishengir-opt -fold-slice-into-scf-for --split-input-file %s | FileCheck %s

// Negative: the write-back insert_slice sits inside an scf.if, so it does not
// belong to the loop's block. Folding it into the loop body would make the
// write execute unconditionally on every iteration. The offsets match and all
// other conditions hold, so only the block check keeps the fold from firing.
// CHECK-LABEL: func.func @no_fold_insert_in_scf_if(
// CHECK: %[[EXTRACT:.*]] = tensor.extract_slice %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1]
// CHECK: scf.for {{.*}} iter_args({{%.*}} = %[[EXTRACT]]) -> (tensor<16x16xf32>) {
// CHECK: scf.if
// CHECK: %[[INS:.*]] = tensor.insert_slice %{{.*}} into %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1]
// CHECK: scf.yield %[[INS]]
// CHECK: return
func.func @no_fold_insert_in_scf_if(%arg0: tensor<16x16x16xf32>, %arg1: index, %ub: index, %step: index, %cond: i1) -> tensor<16x16x16xf32> {
  %extracted = tensor.extract_slice %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16x16xf32>
  %r = scf.for %i = %arg1 to %ub step %step iter_args(%arg = %extracted) -> (tensor<16x16xf32>) {
    scf.yield %arg : tensor<16x16xf32>
  }
  %inserted = scf.if %cond -> (tensor<16x16x16xf32>) {
    %ins = tensor.insert_slice %r into %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16xf32> into tensor<16x16x16xf32>
    scf.yield %ins : tensor<16x16x16xf32>
  } else {
    scf.yield %arg0 : tensor<16x16x16xf32>
  }
  return %inserted : tensor<16x16x16xf32>
}
