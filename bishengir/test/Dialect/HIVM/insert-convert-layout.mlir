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

// CHECK-LABEL: func.func @skip_when_mmad_is_already_fractal(
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x1x16x16xf16>, tensor<1x2x16x16xf16>, i1, index, index, index) outs(%{{.*}} : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
// CHECK: return %[[MMAD]] : tensor<4x2x16x16xf32>
func.func @skip_when_mmad_is_already_fractal(
    %arg0: tensor<4x1x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<4x2x16x16xf32> {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %out = tensor.empty() : tensor<4x2x16x16xf32>
  %res = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<4x1x16x16xf16>, tensor<1x2x16x16xf16>, i1, index, index, index)
                        outs(%out : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
  return %res : tensor<4x2x16x16xf32>
}
