// RUN: bishengir-opt %s --hivm-insert-convert-layout --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @insert_for_mmad_basic(
// CHECK-SAME: %[[A:.*]]: tensor<64x16xf16>, %[[B:.*]]: tensor<16x32xf16>)
// CHECK: %[[OUT_INIT:.*]] = tensor.empty() : tensor<64x32xf32>
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[A]] output_shape [1, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B]] output_shape [2, 1, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %[[OUT_INIT]] output_shape [2, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 ins(%[[A_FR]], %[[B_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x16x16xf16>, tensor<2x1x16x16xf16>, i1, index, index, index) outs(%[[C_FR]] : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[RES_ND]] : tensor<64x32xf32>
func.func @insert_for_mmad_basic(
    %arg0: tensor<64x16xf16>, %arg1: tensor<16x32xf16>) -> tensor<64x32xf32> {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %out = tensor.empty() : tensor<64x32xf32>
  %res = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<64x16xf16>, tensor<16x32xf16>, i1, index, index, index)
                        outs(%out : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %res : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: func.func @insert_for_mmad_transpose_b(
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B:.+]] output_shape [1, 2, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {b_transpose} ins(%{{.*}}, %[[B_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x16x16xf16>, tensor<1x2x16x16xf16>, i1, index, index, index) outs(%{{.*}} : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 32]
// CHECK: return %[[RES_ND]] : tensor<64x32xf32>
func.func @insert_for_mmad_transpose_b(
    %arg0: tensor<64x16xf16>, %arg1: tensor<32x16xf16>) -> tensor<64x32xf32> {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %out = tensor.empty() : tensor<64x32xf32>
  %res = hivm.hir.mmadL1 {b_transpose} ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<64x16xf16>, tensor<32x16xf16>, i1, index, index, index)
                                      outs(%out : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %res : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: func.func @insert_for_mmad_same_source_bf16(
// CHECK-SAME: %[[SRC:.*]]: tensor<64x32xbf16>, %[[OUT:.*]]: tensor<64x64xf32>, %[[INIT:.*]]: i1)
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape [2, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape [2, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %[[OUT]] output_shape [4, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%[[A_FR]], %[[B_FR]], %[[INIT]], %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x16x16xbf16>, tensor<2x4x16x16xbf16>, i1, index, index, index) outs(%[[C_FR]] : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
// CHECK: %[[RES:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 64]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[RES]] : tensor<64x64xf32>
func.func @insert_for_mmad_same_source_bf16(
    %arg0: tensor<64x32xbf16>, %arg1: tensor<64x64xf32>, %arg2: i1) -> tensor<64x64xf32> {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %res = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true}
      ins(%arg0, %arg0, %arg2, %c64, %c32, %c64 : tensor<64x32xbf16>, tensor<64x32xbf16>, i1, index, index, index)
      outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %res : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @insert_for_mmad_same_source_f32(
// CHECK-SAME: %[[SRC:.*]]: tensor<64x32xf32>, %[[OUT:.*]]: tensor<64x64xf32>, %[[INIT:.*]]: i1)
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape [4, 4, 16, 8]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape [4, 4, 16, 8]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %[[OUT]] output_shape [4, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%[[A_FR]], %[[B_FR]], %[[INIT]], %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x4x16x8xf32>, tensor<4x4x16x8xf32>, i1, index, index, index) outs(%[[C_FR]] : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
// CHECK: %[[RES:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 64]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[RES]] : tensor<64x64xf32>
func.func @insert_for_mmad_same_source_f32(
    %arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>, %arg2: i1) -> tensor<64x64xf32> {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %res = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true}
      ins(%arg0, %arg0, %arg2, %c64, %c32, %c64 : tensor<64x32xf32>, tensor<64x32xf32>, i1, index, index, index)
      outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %res : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @insert_for_mmad_diff_blocksizes_i8(
// CHECK-SAME: %[[SRC:.*]]: tensor<64x64xi8>, %[[OUT:.*]]: tensor<64x64xi32>, %[[INIT:.*]]: i1)
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %[[OUT]] output_shape
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%[[A_FR]], %[[B_FR]], %[[INIT]], %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: %[[RES:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 64]
// CHECK: return %[[RES]] : tensor<64x64xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @insert_for_mmad_diff_blocksizes_i8(
    %arg0: tensor<64x64xi8>, %arg1: tensor<64x64xi32>, %arg2: i1) -> tensor<64x64xi32> {
  %c64 = arith.constant 64 : index
  %res = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
      ins(%arg0, %arg0, %arg2, %c64, %c64, %c64 : tensor<64x64xi8>, tensor<64x64xi8>, i1, index, index, index)
      outs(%arg1 : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %res : tensor<64x64xi32>
}
}
