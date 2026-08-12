// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// CHECK-LABEL: @fixpipe_fractal_rhs_no_convert_layout(
// CHECK-NOT: {"hivm.inserted-copy"}
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: %[[LHS_LOAD:.*]] = hivm.hir.load ins(%arg0 : tensor<2x1x16x8xf32>) outs(%{{.*}} : tensor<2x1x16x8xf32>) {"hivm.inserted-load"} -> tensor<2x1x16x8xf32>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<2x1x16x8xf32>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg1 : tensor<2x1x16x8xf32>) outs(%[[ALLOC]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%[[LHS_LOAD]], %[[TENSOR]], %true, %c16, %c16, %c16 : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index) outs(%{{.*}} : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fixpipe_fractal_rhs_no_convert_layout(
      %lhs: tensor<2x1x16x8xf32>, %rhs: tensor<2x1x16x8xf32>) -> tensor<1x1x16x16xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %fix_out = tensor.empty() : tensor<2x1x16x8xf32>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%rhs : tensor<2x1x16x8xf32>) outs(%fix_out : tensor<2x1x16x8xf32>)
        -> tensor<2x1x16x8xf32>
    %out = tensor.empty() : tensor<1x1x16x16xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%lhs, %fix, %true, %c16, %c16, %c16
            : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index)
        outs(%out : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    return %mmad : tensor<1x1x16x16xf32>
  }
}

// -----

// Contrast: rank-2/ND fixpipe result still needs UB->L1 layout conversion.
// CHECK-LABEL: @fixpipe_nd_rhs_gets_convert_layout(
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK: hivm.hir.vtranspose
// CHECK: %[[TENSOR:.*]] = tensor.empty() {hivm.address_space = #hivm.address_space<cbuf>, "hivm.inserted-tensor"} : tensor<2x1x16x8xf32>
// CHECK: %[[COPY:.*]] = hivm.hir.copy ins(%{{.*}} : tensor<2x1x16x8xf32>) outs(%[[TENSOR]] : tensor<2x1x16x8xf32>) {"hivm.inserted-copy"}
// CHECK: hivm.hir.mmadL1 {{.*}} ins(%{{.*}}, %[[COPY]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fixpipe_nd_rhs_gets_convert_layout(
      %lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf32>) -> tensor<16x16xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %fix_out = tensor.empty() : tensor<16x16xf32>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%rhs : tensor<16x16xf32>) outs(%fix_out : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %out = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%lhs, %fix, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf32>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mmad : tensor<16x16xf32>
  }
}
