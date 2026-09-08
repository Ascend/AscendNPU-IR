// RUN: bishengir-opt %s -hivm-insert-load-store-for-mix-cv | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CV11 keeps a workspace producer and loads its result on the vector core.
  // CHECK-LABEL: func.func @cv11
  // CHECK: %[[FIXED:.*]] = hivm.hir.fixpipe
  // CHECK: %[[LOADED:.*]] = hivm.hir.load ins(%[[FIXED]]
  // CHECK: hivm.hir.vrelu ins(%[[LOADED]]
  func.func @cv11(%acc: tensor<16x16xf32>, %out: memref<16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    %dst = tensor.empty() : tensor<16x16xf32>
    %fixed = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%acc : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    %empty = tensor.empty() : tensor<16x16xf32>
    %relu = hivm.hir.vrelu ins(%fixed : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    bufferization.materialize_in_destination %relu in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
    return
  }
  // CHECK-LABEL: func.func @cv12
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.fixpipe {{.*}}outs(%[[UB]] : memref<16x16xf32, #hivm.address_space<ub>>)
  // CHECK-NOT: hivm.hir.load
  // CHECK: hivm.hir.vrelu
  func.func @cv12(%acc: tensor<16x16xf32>, %out: memref<16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.core_ratio = #hivm.core_ratio<1, 2>} {
    %dst = tensor.empty() : tensor<16x16xf32>
    %fixed = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%acc : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    %empty = tensor.empty() : tensor<16x16xf32>
    %relu = hivm.hir.vrelu ins(%fixed : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    bufferization.materialize_in_destination %relu in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
    return
  }
}
