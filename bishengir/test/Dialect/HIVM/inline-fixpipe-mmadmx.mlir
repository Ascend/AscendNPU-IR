// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// Elementwise bias (full C) with init=false: InsertFixpipe should refuse to
// insert because the op will later decompose to mmad+vadd.
// CHECK-LABEL: func.func @test_mmadmx_fixpipe_skip_elemwise_bias
func.func @test_mmadmx_fixpipe_skip_elemwise_bias(%bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %a = tensor.empty() : tensor<16x16xf8E5M2>
  %b = tensor.empty() : tensor<16x16xf8E5M2>
  %sa = tensor.empty() : tensor<1xui8>
  %sb = tensor.empty() : tensor<1xui8>
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-NOT: hivm.hir.fixpipe
  %mad = hivm.hir.mmadmxL1
    ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
        tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
        tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
    outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mad : tensor<16x16xf32>
}

// -----
// Clean empty init: fixpipe should be inserted.
// CHECK-LABEL: func.func @test_mmadmx_fixpipe_insert_on_empty
func.func @test_mmadmx_fixpipe_insert_on_empty() -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %a = tensor.empty() : tensor<16x16xf8E5M2>
  %b = tensor.empty() : tensor<16x16xf8E5M2>
  %sa = tensor.empty() : tensor<1xui8>
  %sb = tensor.empty() : tensor<1xui8>
  %c = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.mmadmxL1
  // CHECK: hivm.hir.fixpipe
  %mad = hivm.hir.mmadmxL1
    ins(%a, %b, %sa, %sb, %true, %c16, %c16, %c16 :
        tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
        tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
    outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mad : tensor<16x16xf32>
}

// -----
// K-accumulation stays in L0C inside the loop. The store consumes a slice of
// the loop result, so the fixpipe belongs after the scf.for, on the static
// accumulator, and the in-loop yield stays on the raw mmadmxL1.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @mmadmx_k_loop_fixpipe_before_slice
// CHECK: %[[ACC:.*]] = scf.for
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadmxL1
// CHECK-SAME: fixpipe_for_result_already_inserted = true
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.yield %[[MMAD]]
// CHECK: %[[FP:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[ACC]] : tensor<16x16xf32>)
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[FP]]
// CHECK: hivm.hir.store ins(%[[SLICE]]
func.func @mmadmx_k_loop_fixpipe_before_slice(
    %a: tensor<16x16xf8E5M2>, %b: tensor<16x16xf8E5M2>,
    %sa: tensor<1xui8>, %sb: tensor<1xui8>,
    %rows: index, %cols: index,
    %out: memref<?x?xf32, strided<[16, 1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %init = tensor.empty() : tensor<16x16xf32>
  %acc = scf.for %k = %c0 to %c2 step %c1
      iter_args(%c = %init) -> (tensor<16x16xf32>) {
    %mmad = hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
        ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
            tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
            tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
        outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  } {normalized_in_L0C = [0 : i32]}
  %slice = tensor.extract_slice %acc[0, 0] [%rows, %cols] [1, 1]
      : tensor<16x16xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%slice : tensor<?x?xf32>)
      outs(%out : memref<?x?xf32, strided<[16, 1], offset: ?>>)
  return
}
}

// -----
// The first loop's result stays in L0C for the next mmadmxL1. Do not insert
// an outer fixpipe between the loops. The value that leaves the chain still
// gets one.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @mmadmx_accumulation_no_outer_fixpipe
// CHECK: %[[ACC:.*]] = scf.for
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadmxL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.yield %[[MMAD]]
// CHECK: } {hivm.remain_in_l0c, normalized_in_L0C = [0 : i32]}
// CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[ACC]]
// CHECK: %[[NEXT:.*]] = scf.for
// CHECK: %[[OUT:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[NEXT]]
// CHECK: return %[[OUT]]
func.func @mmadmx_accumulation_no_outer_fixpipe(
    %a: tensor<16x16xf8E5M2>, %b: tensor<16x16xf8E5M2>,
    %sa: tensor<1xui8>, %sb: tensor<1xui8>,
    %cInit: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %acc = scf.for %i = %c0 to %c2 step %c1
      iter_args(%cCurr = %cInit) -> (tensor<16x16xf32>) {
    %mmad = hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
        ins(%a, %b, %sa, %sb, %true, %c16, %c16, %c16 :
            tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
            tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
        outs(%cCurr : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  } {hivm.remain_in_l0c, normalized_in_L0C = [0 : i32]}
  %next = scf.for %j = %c0 to %c2 step %c1
      iter_args(%cNext = %acc) -> (tensor<16x16xf32>) {
    %mmad2 = hivm.hir.mmadmxL1 {already_set_real_mkn}
        ins(%a, %b, %sa, %sb, %true, %c16, %c16, %c16 :
            tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
            tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
        outs(%cNext : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad2 : tensor<16x16xf32>
  }
  return %next : tensor<16x16xf32>
}
}
