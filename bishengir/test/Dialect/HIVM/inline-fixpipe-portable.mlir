// RUN: bishengir-opt -hivm-inline-fixpipe %s -split-input-file | FileCheck %s
// RUN: bishengir-opt -hivm-inline-fixpipe='inline-quant-scale=true' %s -split-input-file | FileCheck %s --check-prefix=QUANT

// Small, self-contained coverage for the core InlineFixpipe rewrites shared
// by memory-based and register-based targets.

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @fuse_vcast_and_store(
  // CHECK-NOT: hivm.hir.vcast
  // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
  // CHECK-SAME: ins(%arg0 : tensor<16x16xf32>) outs(%arg3 : memref<16x16xf16>)
  // CHECK-NEXT: return
  func.func @fuse_vcast_and_store(
      %arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>,
      %arg2: tensor<16x16xf16>, %arg3: memref<16x16xf16>) {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<16x16xf32>) outs(%arg1 : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %1 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false}
        ins(%0 : tensor<16x16xf32>) outs(%arg2 : tensor<16x16xf16>)
        -> tensor<16x16xf16>
    hivm.hir.store ins(%1 : tensor<16x16xf16>)
        outs(%arg3 : memref<16x16xf16>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @inline_nz2dn_fixpipe_vtranspose_return
  // CHECK-NOT: nz2nd
  // CHECK: %[[RESULT:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>}
  // CHECK: return %[[RESULT]]
  func.func @inline_nz2dn_fixpipe_vtranspose_return(
      %arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>,
      %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<128x64xf32>) outs(%arg1 : tensor<128x64xf32>)
        -> tensor<128x64xf32>
    %1 = hivm.hir.vtranspose ins(%0 : tensor<128x64xf32>)
        outs(%arg2 : tensor<64x128xf32>) permutation = [1, 0]
        -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @inline_quant_scale_fixpipe_vmul_store
  // CHECK: hivm.hir.fixpipe
  // CHECK-NOT: quant_scale
  // CHECK: hivm.hir.vmul
  // QUANT-LABEL: func.func @inline_quant_scale_fixpipe_vmul_store
  // QUANT-NOT: hivm.hir.vmul
  // QUANT: hivm.hir.fixpipe
  // QUANT-SAME: pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>
  // QUANT-SAME: quant_scale = %arg2 : f32
  func.func @inline_quant_scale_fixpipe_vmul_store(
      %arg0: tensor<?x64xf32>, %arg1: tensor<?x64xf32>,
      %arg2: f32, %arg3: tensor<?x64xf32>,
      %arg4: memref<?x64xf32, strided<[256, 1], offset: ?>>) {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<?x64xf32>) outs(%arg1 : tensor<?x64xf32>)
        -> tensor<?x64xf32>
    %1 = hivm.hir.vmul ins(%0, %arg2 : tensor<?x64xf32>, f32)
        outs(%arg3 : tensor<?x64xf32>) -> tensor<?x64xf32>
    hivm.hir.store ins(%1 : tensor<?x64xf32>)
        outs(%arg4 : memref<?x64xf32, strided<[256, 1], offset: ?>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @no_fuse_i32_to_i8_without_saturate
  // CHECK: hivm.hir.fixpipe
  // CHECK-NOT: pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>
  // CHECK: hivm.hir.vcast {enable_overflow = true, enable_saturate = false
  func.func @no_fuse_i32_to_i8_without_saturate(
      %arg0: tensor<16x16xi32>, %arg1: tensor<16x16xi32>,
      %arg2: tensor<16x16xi8>, %arg3: memref<16x16xi8>) {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<16x16xi32>) outs(%arg1 : tensor<16x16xi32>)
        -> tensor<16x16xi32>
    %1 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = false}
        ins(%0 : tensor<16x16xi32>) outs(%arg2 : tensor<16x16xi8>)
        round_mode = <truncwithoverflow> -> tensor<16x16xi8>
    hivm.hir.store ins(%1 : tensor<16x16xi8>)
        outs(%arg3 : memref<16x16xi8>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @fuse_relu_and_store(
  // CHECK-NOT: hivm.hir.vrelu
  // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_relu = #hivm.fixpipe_pre_relu_mode<NORMAL_RELU>}
  // CHECK-SAME: ins(%arg0 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
  // CHECK-NEXT: return
  func.func @fuse_relu_and_store(
      %arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>,
      %arg2: memref<16x16xf32>) {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<16x16xf32>) outs(%arg1 : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %1 = hivm.hir.vrelu ins(%0 : tensor<16x16xf32>)
        outs(%arg1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%1 : tensor<16x16xf32>)
        outs(%arg2 : memref<16x16xf32>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @fuse_atomic_add_store(
  // CHECK: hivm.hir.set_atomic kind = <add>[type = f32]
  // CHECK-NEXT: hivm.hir.fixpipe
  // CHECK-SAME: ins(%arg0 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
  // CHECK-NEXT: hivm.hir.set_atomic kind = <none>[type = f32]
  // CHECK-NEXT: return
  func.func @fuse_atomic_add_store(
      %arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>,
      %arg2: memref<16x16xf32>) {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<16x16xf32>) outs(%arg1 : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    hivm.hir.store ins(%0 : tensor<16x16xf32>)
        outs(%arg2 : memref<16x16xf32>) atomic = <add>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @swap_extract_slice_before_fixpipe(
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %arg0[%arg3, 0] [%arg4, 16] [1, 1]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty(%arg4)
  // CHECK-NEXT: %[[FIXPIPE:.*]] = hivm.hir.fixpipe
  // CHECK-SAME: ins(%[[SLICE]] : tensor<?x16xf32>) outs(%[[EMPTY]] : tensor<?x16xf32>)
  // CHECK-NEXT: return %[[FIXPIPE]]
  func.func @swap_extract_slice_before_fixpipe(
      %arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>,
      %arg2: tensor<?x16xf32>, %arg3: index,
      %arg4: index) -> tensor<?x16xf32> {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<16x16xf32>) outs(%arg1 : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %1 = tensor.extract_slice %0[%arg3, 0] [%arg4, 16] [1, 1]
        : tensor<16x16xf32> to tensor<?x16xf32>
    return %1 : tensor<?x16xf32>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @swap_insert_slice_before_fixpipe(
  // CHECK: %[[INSERT:.*]] = tensor.insert_slice %arg0 into %arg2[0, 0] [8, 16] [1, 1]
  // CHECK: hivm.hir.fixpipe
  // CHECK-SAME: ins(%[[INSERT]] : tensor<16x16xf32>) outs(%arg3 : memref<16x16xf32>)
  // CHECK-NEXT: return
  func.func @swap_insert_slice_before_fixpipe(
      %arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>,
      %arg2: tensor<16x16xf32>, %arg3: memref<16x16xf32>) {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<8x16xf32>) outs(%arg1 : tensor<8x16xf32>)
        -> tensor<8x16xf32>
    %1 = tensor.insert_slice %0 into %arg2[0, 0] [8, 16] [1, 1]
        : tensor<8x16xf32> into tensor<16x16xf32>
    hivm.hir.store ins(%1 : tensor<16x16xf32>)
        outs(%arg3 : memref<16x16xf32>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Multiple non-debug users must prevent all inlining.
  // CHECK-LABEL: func.func @keep_fixpipe_with_multiple_users(
  // CHECK: %[[FIXPIPE:.*]] = hivm.hir.fixpipe
  // CHECK: hivm.hir.store ins(%[[FIXPIPE]]
  // CHECK: return %[[FIXPIPE]]
  func.func @keep_fixpipe_with_multiple_users(
      %arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>,
      %arg2: memref<16x16xf32>) -> tensor<16x16xf32> {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%arg0 : tensor<16x16xf32>) outs(%arg1 : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    hivm.hir.store ins(%0 : tensor<16x16xf32>)
        outs(%arg2 : memref<16x16xf32>)
    return %0 : tensor<16x16xf32>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // TODO: Use the original register-based negative case after the A3 and A5
  // VTransposeOp verifier/assembly behavior is unified. A3 rejects its
  // identity permutation, so this uses an A3-valid non-foldable equivalent.
  // A transpose after a non-NZ2ND fixpipe cannot be folded.
  // CHECK-LABEL: func.func @keep_unsupported_transpose(
  // CHECK: %[[FIXPIPE:.*]] = hivm.hir.fixpipe ins(%arg0
  // CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[FIXPIPE]]
  // CHECK-SAME: permutation = [1, 0]
  // CHECK: return %[[TRANSPOSE]]
  func.func @keep_unsupported_transpose(
      %arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>,
      %arg2: tensor<32x16xf32>) -> tensor<32x16xf32> {
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<normal>}
        ins(%arg0 : tensor<16x32xf32>) outs(%arg1 : tensor<16x32xf32>)
        -> tensor<16x32xf32>
    %1 = hivm.hir.vtranspose ins(%0 : tensor<16x32xf32>)
        outs(%arg2 : tensor<32x16xf32>) permutation = [1, 0]
        -> tensor<32x16xf32>
    return %1 : tensor<32x16xf32>
  }
}
