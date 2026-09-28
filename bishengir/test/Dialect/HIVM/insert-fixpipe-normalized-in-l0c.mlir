// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// Test 1: mmadL1 with `init = false` and `normalized_in_L0C` attribute.
// Previously, `skipFixpipeForBiasDecompose` would return failure because init is constant false.
// With the fix, since `normalized_in_L0C` is present, `fixpipe` should be correctly inserted.

// CHECK-LABEL: func.func @mmad_normalized_in_l0c_insert_fixpipe
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32x64xf32>, %[[ARG1:.*]]: tensor<64x64xf32>, %[[OUT:.*]]: memref<32x64xf32>)
// CHECK:        %[[C32:.*]] = arith.constant 32 : index
// CHECK:        %[[C64:.*]] = arith.constant 64 : index
// CHECK:        %[[INIT:.*]] = tensor.empty() : tensor<32x64xf32>
// CHECK:        %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %false, %[[C32]], %[[C64]], %[[C64]] : tensor<32x64xf32>, tensor<64x64xf32>, i1, index, index, index)
// CHECK-SAME:       outs(%[[INIT]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:        %[[DST:.*]] = tensor.empty() {hivm.address_space = #hivm.address_space<ub>, "hivm.inserted-tensor"} : tensor<32x64xf32>
// CHECK:        %[[FIXPIPE:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MMAD]] : tensor<32x64xf32>) outs(%[[DST]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:        hivm.hir.store ins(%[[FIXPIPE]] : tensor<32x64xf32>) outs(%[[OUT]] : memref<32x64xf32>)
func.func @mmad_normalized_in_l0c_insert_fixpipe(
    %arg0: tensor<32x64xf32>,
    %arg1: tensor<64x64xf32>,
    %out: memref<32x64xf32>) {
  %false = arith.constant false
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %init = tensor.empty() : tensor<32x64xf32>

  %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
      ins(%arg0, %arg1, %false, %c32, %c64, %c64 : tensor<32x64xf32>, tensor<64x64xf32>, i1, index, index, index)
      outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>

  hivm.hir.store ins(%mmad : tensor<32x64xf32>) outs(%out : memref<32x64xf32>)
  return
}

}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// Test 2: mmadL1 with `normalized_in_L0C` where result is consumed via `tensor.extract_slice` before store.
// Verify that fixpipe is inserted after mmadL1 and users (extract_slice) are redirected to the fixpipe result.

// CHECK-LABEL: func.func @mmad_normalized_in_l0c_slice_and_store
// CHECK:        %[[MMAD:.*]] = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
// CHECK:        %[[DST:.*]] = tensor.empty() {hivm.address_space = #hivm.address_space<ub>, "hivm.inserted-tensor"} : tensor<32x64xf32>
// CHECK:        %[[FIXPIPE:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MMAD]] : tensor<32x64xf32>) outs(%[[DST]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:        %[[SLICE:.*]] = tensor.extract_slice %[[FIXPIPE]][0, 0] [%c32, %c64] [1, 1] : tensor<32x64xf32> to tensor<?x?xf32>
// CHECK:        hivm.hir.store ins(%[[SLICE]] : tensor<?x?xf32>) outs(%{{.*}} : memref<?x?xf32, strided<[512, 1], offset: ?>>)
func.func @mmad_normalized_in_l0c_slice_and_store(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x64xf32>,
    %acc_in: tensor<32x64xf32>,
    %out_mem: memref<?x?xf32, strided<[512, 1], offset: ?>>) {
  %false = arith.constant false
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  %mmad = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, normalized_in_L0C}
      ins(%arg0, %arg1, %false, %c32, %c32, %c64 : tensor<32x32xf32>, tensor<32x64xf32>, i1, index, index, index)
      outs(%acc_in : tensor<32x64xf32>) -> tensor<32x64xf32>

  %extracted_slice = tensor.extract_slice %mmad[0, 0] [%c32, %c64] [1, 1] : tensor<32x64xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%out_mem : memref<?x?xf32, strided<[512, 1], offset: ?>>)
  return
}

}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// Test 3: nested loops. The innermost mmad accumulates in L0C. The middle
// loop only forwards that result. The outer loop both feeds it to a later
// mmad and yields it under normalized_in_L0C. Fixpipe must move a copy out
// for the later mmad, while every yield of the accumulator stays on the raw
// L0C value.

// CHECK-LABEL: func.func @nested_loop_normalized_in_l0c
// CHECK: scf.for
// CHECK:   %[[MID:.*]] = scf.for
// CHECK:     %[[INNER:.*]] = scf.for
// CHECK:       %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield %[[MMAD]]
// CHECK:     } {fixpipe_for_mmad_result_already_inserted = true}
// CHECK:     scf.yield %[[INNER]]
// CHECK:   %[[FP_ACC:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MID]] : tensor<32x32xf32>)
// CHECK:   %[[MMAD2:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%[[FP_ACC]]
// CHECK:   %[[FP_OUT:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MMAD2]] : tensor<32x1xf32>)
// CHECK:   hivm.hir.store ins(%[[FP_OUT]]
// CHECK:   scf.yield %[[MID]]
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true, may_not_exec, normalized_in_L0C = [0 : i32]}
func.func @nested_loop_normalized_in_l0c(
    %a: tensor<32x1xf16>,
    %b: tensor<1x32xf16>,
    %vec: tensor<32x1xf32>,
    %out: memref<32x1xf32>) {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %init = tensor.empty() : tensor<32x32xf32>

  %outer = scf.for %i = %c0 to %c2 step %c1
      iter_args(%acc_outer = %init) -> (tensor<32x32xf32>) {
    %mid = scf.for %j = %c0 to %c2 step %c1
        iter_args(%acc_mid = %acc_outer) -> (tensor<32x32xf32>) {
      %inner = scf.for %k = %c0 to %c2 step %c1
          iter_args(%acc_inner = %acc_mid) -> (tensor<32x32xf32>) {
        %mmad = hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
            ins(%a, %b, %false, %c32, %c1, %c32 : tensor<32x1xf16>, tensor<1x32xf16>, i1, index, index, index)
            outs(%acc_inner : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %mmad : tensor<32x32xf32>
      }
      scf.yield %inner : tensor<32x32xf32>
    }
    %empty = tensor.empty() : tensor<32x1xf32>
    %mmad2 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%mid, %vec, %true, %c32, %c32, %c1 : tensor<32x32xf32>, tensor<32x1xf32>, i1, index, index, index)
        outs(%empty : tensor<32x1xf32>) -> tensor<32x1xf32>
    hivm.hir.store ins(%mmad2 : tensor<32x1xf32>) outs(%out : memref<32x1xf32>)
    scf.yield %mid : tensor<32x32xf32>
  } {may_not_exec, normalized_in_L0C = [0 : i32]}
  return
}

}
