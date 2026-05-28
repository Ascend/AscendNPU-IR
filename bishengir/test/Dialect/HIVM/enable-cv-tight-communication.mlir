// Test 1: Tight-coupled CV communication pass.
// This pass is used by lower-hivm-pipeline when
// enable-cv-tight-communication=true on Ascend950.
// RUN: bishengir-opt -hivm-insert-cv-tight-coupled-buffer \
// RUN:   %s | FileCheck %s -check-prefix=TIGHT

// Test 2: MixCV communication pass.
// This pass is used by lower-hivm-pipeline when
// enable-cv-tight-communication=false or tight path is not selected.
// RUN: bishengir-opt \
// RUN: -hivm-insert-load-store-for-mix-cv="disable-tight-coupled-buffer=true" \
// RUN:   %s | FileCheck %s -check-prefix=MIXCV

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // TIGHT-LABEL: func.func @test_fixpipe_to_vector(
  // TIGHT: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // TIGHT: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]]
  // TIGHT: hivm.hir.fixpipe {{.*}} outs(%[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>>)
  // TIGHT: %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]]
  // TIGHT-NOT: hivm.hir.load
  // TIGHT: hivm.hir.vmul ins(%[[TENSOR]]

  // MIXCV-LABEL: func.func @test_fixpipe_to_vector(
  // MIXCV-NOT: memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // MIXCV-NOT: memref.memory_space_cast
  // MIXCV: hivm.hir.fixpipe
  // MIXCV: hivm.hir.vmul

  func.func @test_fixpipe_to_vector(%src : tensor<16x16xf32>)
      -> tensor<16x16xf16> {
    %dst = tensor.empty() : tensor<16x16xf16>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
           ins(%src : tensor<16x16xf32>)
           outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>

    %one = arith.constant 1.000000e+00 : f16
    %vdst = tensor.empty() : tensor<16x16xf16>
    %res = hivm.hir.vmul ins(%fix, %one : tensor<16x16xf16>, f16)
           outs(%vdst : tensor<16x16xf16>) -> tensor<16x16xf16>

    return %res : tensor<16x16xf16>
  }
}