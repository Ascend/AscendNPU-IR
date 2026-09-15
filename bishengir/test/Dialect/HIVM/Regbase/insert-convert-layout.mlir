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

// CHECK-LABEL: func.func @insert_conv2d_input_convert_layout(
// CHECK: %[[CONVERTED:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [1, 2, 8, 8, 16]
// CHECK-SAME: dstLayout = #hivm.data_layout<NC1HWC0>
// CHECK-SAME: groups = 2 : i64
// CHECK-SAME: srcLayout = #hivm.data_layout<NCHW>
// CHECK: hivm.hir.Conv2dL1
// CHECK-SAME: ins(%[[CONVERTED]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @insert_conv2d_input_convert_layout(
      %input: tensor<1x30x8x8xf16>,
      %weight: tensor<1x3x3x32x16xf16>) -> tensor<64x32xf32> {
    %true = arith.constant true
    %init = tensor.empty() : tensor<64x32xf32>
    %conv = hivm.hir.Conv2dL1
        {dilation = [1, 1], groups = 2 : i32, padding = [1, 1],
         stride = [1, 1]}
        ins(%input, %weight, %true
            : tensor<1x30x8x8xf16>, tensor<1x3x3x32x16xf16>, i1)
        outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %conv : tensor<64x32xf32>
  }
}

// -----

// CHECK-LABEL: func.func @insert_conv2d_weight_convert_layout(
// CHECK: %[[CONVERTED:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [1, 3, 3, 32, 16]
// CHECK-SAME: dstLayout = #hivm.data_layout<C1HWNC0>
// CHECK-SAME: groups = 2 : i64
// CHECK-SAME: srcLayout = #hivm.data_layout<NCHW>
// CHECK: hivm.hir.Conv2dL1
// CHECK-SAME: ins(%{{.*}}, %[[CONVERTED]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @insert_conv2d_weight_convert_layout(
      %input: tensor<1x2x8x8x16xf16>,
      %weight: tensor<30x15x3x3xf16>) -> tensor<64x32xf32> {
    %true = arith.constant true
    %init = tensor.empty() : tensor<64x32xf32>
    %conv = hivm.hir.Conv2dL1
        {dilation = [1, 1], groups = 2 : i32, padding = [1, 1],
         stride = [1, 1]}
        ins(%input, %weight, %true
            : tensor<1x2x8x8x16xf16>, tensor<30x15x3x3xf16>, i1)
        outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %conv : tensor<64x32xf32>
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
