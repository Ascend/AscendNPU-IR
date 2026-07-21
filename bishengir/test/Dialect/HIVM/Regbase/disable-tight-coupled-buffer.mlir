// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv="disable-tight-coupled-buffer=true" -split-input-file %s | FileCheck %s

// CHECK: func.func @test_fixpipe_load_to_vector
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK: %[[FIXPIPE:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x16xf16>
// CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%[[FIXPIPE]] : tensor<16x16xf16>) outs(%[[EMPTY]] : tensor<16x16xf16>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR> -> tensor<16x16xf16>
// CHECK: hivm.hir.vmul ins(%[[LOAD]], 
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_fixpipe_load_to_vector(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>) -> tensor<16x16xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %cst = arith.constant 2.000000e+00 : f16
    %c16 = arith.constant 16 : index
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %2 = tensor.empty() : tensor<16x16xf16>
    %3 = tensor.empty() : tensor<16x16xf16>
    %4 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%1 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %5 = hivm.hir.vmul ins(%4, %cst : tensor<16x16xf16>, f16) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %5 : tensor<16x16xf16>
  }
}
