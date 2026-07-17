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

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @insert_for_mmad_transpose_a(
// For i8 type with a_transpose: A uses block size 32x32 (nZ layout)
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[A:.+]] output_shape [2, 1, 32, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B:.+]] output_shape [1, 1, 32, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {a_transpose} ins(%[[A_FR]], %[[B_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x1x32x32xi8>, tensor<1x1x32x32xi8>, i1, index, index, index) outs(%{{.*}} : tensor<2x4x16x16xi32>) -> tensor<2x4x16x16xi32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 32]
// CHECK: return %[[RES_ND]] : tensor<64x32xi32>
  func.func @insert_for_mmad_transpose_a(
      %arg0: tensor<16x64xi8>, %arg1: tensor<16x32xi8>) -> tensor<64x32xi32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %out = tensor.empty() : tensor<64x32xi32>
    %res = hivm.hir.mmadL1 {a_transpose} ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<16x64xi8>, tensor<16x32xi8>, i1, index, index, index)
                                        outs(%out : tensor<64x32xi32>) -> tensor<64x32xi32>
    return %res : tensor<64x32xi32>
  }
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

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @insert_for_mmadmx_A5(
// CHECK-SAME: %[[A:.*]]: tensor<208x64xf8E4M3FN>, %[[B:.*]]: tensor<64x224xf8E4M3FN>, %[[SA:.*]]: tensor<208x2xi8>, %[[SB:.*]]: tensor<224x2xi8>)
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[A]] output_shape [2, 13, 16, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B]] output_shape [7, 2, 32, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[SA_FR:.*]] = hivm.hir.convert_layout %[[SA]] output_shape [13, 1, 16, 2]
// CHECK-SAME: {dstLayout = #hivm.data_layout<SCALEA_zZ, fractalSizes = [16, 2]>, srcLayout = #hivm.data_layout<SCALEA_ND>}
// CHECK: %[[SB_FR:.*]] = hivm.hir.convert_layout %[[SB]] output_shape [14, 1, 16, 2]
// CHECK-SAME: {dstLayout = #hivm.data_layout<SCALEB_nN, fractalSizes = [16, 2]>, srcLayout = #hivm.data_layout<SCALEB_DN>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [14, 13, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadmxL1 {already_set_real_mkn} ins(%[[A_FR]], %[[B_FR]], %[[SA_FR]], %[[SB_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x13x16x32xf8E4M3FN>, tensor<7x2x32x32xf8E4M3FN>, tensor<13x1x16x2xi8>, tensor<14x1x16x2xi8>, i1, index, index, index) outs(%[[C_FR]] : tensor<14x13x16x16xf32>) -> tensor<14x13x16x16xf32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [208, 224]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[RES_ND]] : tensor<208x224xf32>
  func.func @insert_for_mmadmx_A5(
      %arg0: tensor<208x64xf8E4M3FN>, %arg1: tensor<64x224xf8E4M3FN>,
      %arg2: tensor<208x2xi8>, %arg3: tensor<224x2xi8>) -> tensor<208x224xf32> {
    %true = arith.constant true
    %c208 = arith.constant 208 : index
    %c64 = arith.constant 64 : index
    %c224 = arith.constant 224 : index
    %out = tensor.empty() : tensor<208x224xf32>
    %res = hivm.hir.mmadmxL1 {already_set_real_mkn}
        ins(%arg0, %arg1, %arg2, %arg3, %true, %c208, %c64, %c224
            : tensor<208x64xf8E4M3FN>, tensor<64x224xf8E4M3FN>,
              tensor<208x2xi8>, tensor<224x2xi8>, i1, index, index, index)
        outs(%out : tensor<208x224xf32>) -> tensor<208x224xf32>
    return %res : tensor<208x224xf32>
  }
}
