// RUN: bishengir-opt -hivm-inline-fixpipe %s -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// TODO: Check the explicit `dma_mode = normal` spelling after A3 and A5
// FixpipeOp printers are unified. A3 currently elides this default attribute.
// CHECK-LABEL: func.func @inline_fixpipe_vcast_extract_slice_4d
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %arg0[0, 0, 0, 0] [4, %arg4, 16, 16] [1, 1, 1, 1] : tensor<4x1x16x16xf32> to tensor<4x?x16x16xf32>
// CHECK: hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%[[SLICE]] : tensor<4x?x16x16xf32>) outs(%arg3 : memref<4x?x16x16xf16>)
func.func @inline_fixpipe_vcast_extract_slice_4d(
    %arg0: tensor<4x1x16x16xf32>,
    %arg1: tensor<4x1x16x16xf32>,
    %arg2: tensor<4x1x16x16xf16>,
    %arg3: memref<4x?x16x16xf16>,
    %arg4: index) {
  %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<normal>} ins(%arg0 : tensor<4x1x16x16xf32>) outs(%arg1 : tensor<4x1x16x16xf32>) -> tensor<4x1x16x16xf32>
  %1 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false} ins(%0 : tensor<4x1x16x16xf32>) outs(%arg2 : tensor<4x1x16x16xf16>) -> tensor<4x1x16x16xf16>
  %2 = tensor.extract_slice %1[0, 0, 0, 0] [4, %arg4, 16, 16] [1, 1, 1, 1] : tensor<4x1x16x16xf16> to tensor<4x?x16x16xf16>
  hivm.hir.store ins(%2 : tensor<4x?x16x16xf16>) outs(%arg3 : memref<4x?x16x16xf16>)
  return
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @inline_fixpipe_no_swap_rank_mismatch
// CHECK: %[[FP:.*]] = hivm.hir.fixpipe
// CHECK-SAME: pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>
// CHECK-SAME: ins(%arg0 : tensor<4x1x16x16xf32>)
// CHECK-SAME: outs({{.*}} : tensor<16x64xf16>) -> tensor<16x64xf16>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[FP]][0, 0] [%arg4, 64] [1, 1] : tensor<16x64xf16> to tensor<?x64xf16>
// CHECK-NOT: tensor.extract_slice %arg0[0, 0]
func.func @inline_fixpipe_no_swap_rank_mismatch(
    %arg0: tensor<4x1x16x16xf32>,
    %arg1: tensor<16x64xf32>,
    %arg2: tensor<16x64xf16>,
    %arg3: memref<?x64xf16>,
    %arg4: index) {
  %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg0 : tensor<4x1x16x16xf32>) outs(%arg1 : tensor<16x64xf32>) -> tensor<16x64xf32>
  %1 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false} ins(%0 : tensor<16x64xf32>) outs(%arg2 : tensor<16x64xf16>) -> tensor<16x64xf16>
  %2 = tensor.extract_slice %1[0, 0] [%arg4, 64] [1, 1] : tensor<16x64xf16> to tensor<?x64xf16>
  hivm.hir.store ins(%2 : tensor<?x64xf16>) outs(%arg3 : memref<?x64xf16>)
  return
}
}
