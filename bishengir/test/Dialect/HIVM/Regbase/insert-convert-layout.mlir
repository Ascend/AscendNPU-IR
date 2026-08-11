// RUN: bishengir-opt %s --hivm-insert-convert-layout --split-input-file | FileCheck %s

// When mmadL1 result is only consumed by fixpipe writing to L1 (cbuf), skip
// Fractal->ND convert_layout on the mmad result so fixpipe can stay NZ2NZ.
// CHECK-LABEL: func.func @mmad_result_to_fixpipe_cbuf_no_nd_convert(
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.convert_layout %[[MMAD]]
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MMAD]] : tensor<1x1x16x16xf32>) outs(%{{.*}} : memref<16x16xf32, #hivm.address_space<cbuf>>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mmad_result_to_fixpipe_cbuf_no_nd_convert(
      %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %out = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1
        ins(%a, %b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad : tensor<16x16xf32>)
        outs(%alloc : memref<16x16xf32, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

// Contrast: fixpipe writing to UB still needs Fractal->ND convert_layout.
// CHECK-LABEL: func.func @mmad_result_to_fixpipe_ub_keeps_nd_convert(
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK: %[[ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[ND]] : tensor<16x16xf32>) outs(%{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mmad_result_to_fixpipe_ub_keeps_nd_convert(
      %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %out = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1
        ins(%a, %b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad : tensor<16x16xf32>)
        outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
    return
  }
}
