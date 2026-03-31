// RUN: bishengir-opt -hivm-normalize-convops %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_nobias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf16> into tensor<1x32x128xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xf16> into tensor<1x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xf16>) outs(%{{.*}} : tensor<1x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xf16> into tensor<1x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xf16> into tensor<32x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xf16>) outs(%{{.*}} : tensor<1x32x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xf16>) outs(%{{.*}} : tensor<1x32x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xf16>) outs(%{{.*}} : tensor<1x5x32x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xf16> into tensor<1x1x5x32x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xf16>, tensor<1x1x5x32x16xf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf16> to tensor<126x32xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf16>) outs(%{{.*}} : tensor<32x126xf16>) permutation = [1, 0] -> tensor<32x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_nobias_ocaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<32x126xf16>) -> tensor<32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf16>, tensor<32x16x5xf16>, i1) outs(%arg2 : tensor<32x126xf16>) -> tensor<32x126xf16>
  return %0 : tensor<32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_nobias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xf16> into tensor<2x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xf16>) outs(%{{.*}} : tensor<2x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xf16> into tensor<2x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xf16> into tensor<32x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xf16>) outs(%{{.*}} : tensor<1x32x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xf16>) outs(%{{.*}} : tensor<1x32x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xf16>) outs(%{{.*}} : tensor<1x5x32x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xf16> into tensor<1x1x5x32x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xf16>, tensor<1x1x5x32x16xf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf16> to tensor<126x64xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf16>) outs(%{{.*}} : tensor<64x126xf16>) permutation = [1, 0] -> tensor<64x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf16> into tensor<2x32x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_nobias_ocaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf16>, tensor<32x16x5xf16>, i1) outs(%arg2 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  return %0 : tensor<2x32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_nobias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xbf16> into tensor<1x32x128xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xbf16> into tensor<1x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xbf16>) outs(%{{.*}} : tensor<1x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xbf16> into tensor<1x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xbf16> into tensor<32x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xbf16>) outs(%{{.*}} : tensor<1x32x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xbf16>) outs(%{{.*}} : tensor<1x32x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xbf16>) outs(%{{.*}} : tensor<1x5x32x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xbf16> into tensor<1x1x5x32x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xbf16>, tensor<1x1x5x32x16xbf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<128x32xbf16>) -> tensor<128x32xbf16>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xbf16> to tensor<126x32xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xbf16>) outs(%{{.*}} : tensor<32x126xbf16>) permutation = [1, 0] -> tensor<32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_nobias_ocaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<32x126xbf16>) -> tensor<32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%arg2 : tensor<32x126xbf16>) -> tensor<32x126xbf16>
  return %0 : tensor<32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_nobias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xbf16> into tensor<2x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xbf16>) outs(%{{.*}} : tensor<2x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xbf16> into tensor<2x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xbf16> into tensor<32x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xbf16>) outs(%{{.*}} : tensor<1x32x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xbf16>) outs(%{{.*}} : tensor<1x32x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xbf16>) outs(%{{.*}} : tensor<1x5x32x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xbf16> into tensor<1x1x5x32x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xbf16>, tensor<1x1x5x32x16xbf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xbf16> to tensor<126x64xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xbf16>) outs(%{{.*}} : tensor<64x126xbf16>) permutation = [1, 0] -> tensor<64x126xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xbf16> into tensor<2x32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_nobias_ocaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%arg2 : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
  return %0 : tensor<2x32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_nobias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf32> into tensor<1x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 4, 8, 128] : tensor<1x32x128xf32> into tensor<1x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x4x8x128xf32>) outs(%{{.*}} : tensor<1x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<1x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 4, 1, 128, 8] : tensor<1x4x128x8xf32> into tensor<1x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 2, 8, 5] : tensor<32x16x5xf32> into tensor<32x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x2x8x5xf32>) outs(%{{.*}} : tensor<2x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x8x5xf32>) outs(%{{.*}} : tensor<2x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x5x8xf32>) outs(%{{.*}} : tensor<2x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 32, 8] : tensor<2x5x32x8xf32> into tensor<2x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x1x128x8xf32>, tensor<2x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf32>) outs(%{{.*}} : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_nobias_ocaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<32x126xf32>) -> tensor<32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf32>, tensor<32x16x5xf32>, i1) outs(%arg2 : tensor<32x126xf32>) -> tensor<32x126xf32>
  return %0 : tensor<32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_nobias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 8, 128] : tensor<2x32x128xf32> into tensor<2x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x4x8x128xf32>) outs(%{{.*}} : tensor<2x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 4, 1, 128, 8] : tensor<2x4x128x8xf32> into tensor<2x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 2, 8, 5] : tensor<32x16x5xf32> into tensor<32x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x2x8x5xf32>) outs(%{{.*}} : tensor<2x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x8x5xf32>) outs(%{{.*}} : tensor<2x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x5x8xf32>) outs(%{{.*}} : tensor<2x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 32, 8] : tensor<2x5x32x8xf32> into tensor<2x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x1x128x8xf32>, tensor<2x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf32>) outs(%{{.*}} : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_nobias_ocaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf32>, tensor<32x16x5xf32>, i1) outs(%arg2 : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
  return %0 : tensor<2x32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_bias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf16> into tensor<1x32x128xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xf16> into tensor<1x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xf16>) outs(%{{.*}} : tensor<1x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xf16> into tensor<1x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xf16> into tensor<32x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xf16>) outs(%{{.*}} : tensor<1x32x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xf16>) outs(%{{.*}} : tensor<1x32x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xf16>) outs(%{{.*}} : tensor<1x5x32x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xf16> into tensor<1x1x5x32x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xf16>, tensor<1x1x5x32x16xf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf16> to tensor<126x32xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf16>) outs(%{{.*}} : tensor<32x126xf16>) permutation = [1, 0] -> tensor<32x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf16> into tensor<32x1xf16>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<32x126xf16>, tensor<32x1xf16>) outs(%{{.*}} : tensor<32x126xf16>) broadcast = [1] -> tensor<32x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_bias_ocaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32x126xf16>) -> tensor<32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf16>, tensor<32x16x5xf16>, i1, tensor<32xf16>) outs(%arg3 : tensor<32x126xf16>) -> tensor<32x126xf16>
  return %0 : tensor<32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_bias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xf16> into tensor<2x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xf16>) outs(%{{.*}} : tensor<2x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xf16> into tensor<2x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xf16> into tensor<32x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xf16>) outs(%{{.*}} : tensor<1x32x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xf16>) outs(%{{.*}} : tensor<1x32x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xf16>) outs(%{{.*}} : tensor<1x5x32x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xf16> into tensor<1x1x5x32x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xf16>, tensor<1x1x5x32x16xf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf16> to tensor<126x64xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf16>) outs(%{{.*}} : tensor<64x126xf16>) permutation = [1, 0] -> tensor<64x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf16> into tensor<2x32x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf16> into tensor<1x32x1xf16>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x32x126xf16>, tensor<1x32x1xf16>) outs(%{{.*}} : tensor<2x32x126xf16>) broadcast = [0, 2] -> tensor<2x32x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_bias_ocaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf16>, tensor<32x16x5xf16>, i1, tensor<32xf16>) outs(%arg3 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  return %0 : tensor<2x32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_bias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<32xbf16>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xbf16> into tensor<1x32x128xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xbf16> into tensor<1x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xbf16>) outs(%{{.*}} : tensor<1x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xbf16> into tensor<1x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xbf16> into tensor<32x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xbf16>) outs(%{{.*}} : tensor<1x32x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xbf16>) outs(%{{.*}} : tensor<1x32x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xbf16>) outs(%{{.*}} : tensor<1x5x32x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xbf16> into tensor<1x1x5x32x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xbf16>, tensor<1x1x5x32x16xbf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf32>) outs(%{{.*}} : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<32x126xf32>, tensor<32x1xf32>) outs(%{{.*}} : tensor<32x126xf32>) broadcast = [1] -> tensor<32x126xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<32x126xf32>) outs(%{{.*}} : tensor<32x126xbf16>) -> tensor<32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_bias_ocaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<32xbf16>, %arg3: tensor<32x126xbf16>) -> tensor<32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xbf16>, tensor<32x16x5xbf16>, i1, tensor<32xbf16>) outs(%arg3 : tensor<32x126xbf16>) -> tensor<32x126xbf16>
  return %0 : tensor<32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_bias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<32xbf16>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xbf16> into tensor<2x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xbf16>) outs(%{{.*}} : tensor<2x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xbf16> into tensor<2x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 1, 16, 5] : tensor<32x16x5xbf16> into tensor<32x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x1x16x5xbf16>) outs(%{{.*}} : tensor<1x32x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x32x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x16x5xbf16>) outs(%{{.*}} : tensor<1x32x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x32x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x32x5x16xbf16>) outs(%{{.*}} : tensor<1x5x32x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x32x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 32, 16] : tensor<1x5x32x16xbf16> into tensor<1x1x5x32x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xbf16>, tensor<1x1x5x32x16xbf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf32>) outs(%{{.*}} : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf32> into tensor<1x32x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x32x126xf32>, tensor<1x32x1xf32>) outs(%{{.*}} : tensor<2x32x126xf32>) broadcast = [0, 2] -> tensor<2x32x126xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<2x32x126xf32>) outs(%{{.*}} : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_bias_ocaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<32xbf16>, %arg3: tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xbf16>, tensor<32x16x5xbf16>, i1, tensor<32xbf16>) outs(%arg3 : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
  return %0 : tensor<2x32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_bias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf32> into tensor<1x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 4, 8, 128] : tensor<1x32x128xf32> into tensor<1x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x4x8x128xf32>) outs(%{{.*}} : tensor<1x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<1x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 4, 1, 128, 8] : tensor<1x4x128x8xf32> into tensor<1x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 2, 8, 5] : tensor<32x16x5xf32> into tensor<32x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x2x8x5xf32>) outs(%{{.*}} : tensor<2x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x8x5xf32>) outs(%{{.*}} : tensor<2x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x5x8xf32>) outs(%{{.*}} : tensor<2x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 32, 8] : tensor<2x5x32x8xf32> into tensor<2x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x1x128x8xf32>, tensor<2x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf32>) outs(%{{.*}} : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<32x126xf32>, tensor<32x1xf32>) outs(%{{.*}} : tensor<32x126xf32>) broadcast = [1] -> tensor<32x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_bias_ocaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x126xf32>) -> tensor<32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf32>, tensor<32x16x5xf32>, i1, tensor<32xf32>) outs(%arg3 : tensor<32x126xf32>) -> tensor<32x126xf32>
  return %0 : tensor<32x126xf32>
}


// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_bias_ocaligned(
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 8, 128] : tensor<2x32x128xf32> into tensor<2x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x4x8x128xf32>) outs(%{{.*}} : tensor<2x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 4, 1, 128, 8] : tensor<2x4x128x8xf32> into tensor<2x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 2, 8, 5] : tensor<32x16x5xf32> into tensor<32x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x2x8x5xf32>) outs(%{{.*}} : tensor<2x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x8x5xf32>) outs(%{{.*}} : tensor<2x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x5x8xf32>) outs(%{{.*}} : tensor<2x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 32, 8] : tensor<2x5x32x8xf32> into tensor<2x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x1x128x8xf32>, tensor<2x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf32>) outs(%{{.*}} : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf32> into tensor<1x32x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x32x126xf32>, tensor<1x32x1xf32>) outs(%{{.*}} : tensor<2x32x126xf32>) broadcast = [0, 2] -> tensor<2x32x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_bias_ocaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<32xf32>, %arg3: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf32>, tensor<32x16x5xf32>, i1, tensor<32xf32>) outs(%arg3 : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
  return %0 : tensor<2x32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_nobias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf16> into tensor<1x32x128xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xf16> into tensor<1x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xf16>) outs(%{{.*}} : tensor<1x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xf16> into tensor<1x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xf16> into tensor<30x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xf16>) outs(%{{.*}} : tensor<1x30x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xf16>) outs(%{{.*}} : tensor<1x30x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xf16>) outs(%{{.*}} : tensor<1x5x30x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xf16> into tensor<1x1x5x30x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xf16>, tensor<1x1x5x30x16xf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x32xf16>) outs(%{{.*}} : tensor<32x128xf16>) permutation = [1, 0] -> tensor<32x128xf16>
// CHECK:           %{{.*}} = tensor.empty() : tensor<30x128xf16>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<30x128xf16>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<32x128xf16> to tensor<15x128xf16>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<30x128xf16>
// CHECK:             scf.yield %{{.*}} : tensor<30x128xf16>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [30, 126] [1, 1] : tensor<30x128xf16> to tensor<30x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_nobias_ocunaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<30x126xf16>) -> tensor<30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf16>, tensor<30x16x5xf16>, i1) outs(%arg2 : tensor<30x126xf16>) -> tensor<30x126xf16>
  return %0 : tensor<30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_nobias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xf16> into tensor<2x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xf16>) outs(%{{.*}} : tensor<2x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xf16> into tensor<2x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xf16> into tensor<30x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xf16>) outs(%{{.*}} : tensor<1x30x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xf16>) outs(%{{.*}} : tensor<1x30x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xf16>) outs(%{{.*}} : tensor<1x5x30x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xf16> into tensor<1x1x5x30x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xf16>, tensor<1x1x5x30x16xf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x64xf16>) outs(%{{.*}} : tensor<64x128xf16>) permutation = [1, 0] -> tensor<64x128xf16>
// CHECK:           %{{.*}} = tensor.empty() : tensor<60x128xf16>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<60x128xf16>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<64x128xf16> to tensor<15x128xf16>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<60x128xf16>
// CHECK:             scf.yield %{{.*}} : tensor<60x128xf16>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [60, 126] [1, 1] : tensor<60x128xf16> to tensor<60x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf16> into tensor<2x30x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_nobias_ocunaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<2x30x126xf16>) -> tensor<2x30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf16>, tensor<30x16x5xf16>, i1) outs(%arg2 : tensor<2x30x126xf16>) -> tensor<2x30x126xf16>
  return %0 : tensor<2x30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_nobias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xbf16> into tensor<1x32x128xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xbf16> into tensor<1x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xbf16>) outs(%{{.*}} : tensor<1x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xbf16> into tensor<1x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xbf16> into tensor<30x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xbf16>) outs(%{{.*}} : tensor<1x30x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xbf16>) outs(%{{.*}} : tensor<1x30x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xbf16>) outs(%{{.*}} : tensor<1x5x30x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xbf16> into tensor<1x1x5x30x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xbf16>, tensor<1x1x5x30x16xbf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<128x32xbf16>) -> tensor<128x32xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x32xbf16>) outs(%{{.*}} : tensor<32x128xbf16>) permutation = [1, 0] -> tensor<32x128xbf16>
// CHECK:           %{{.*}} = tensor.empty() : tensor<30x128xbf16>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<30x128xbf16>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<32x128xbf16> to tensor<15x128xbf16>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xbf16> into tensor<30x128xbf16>
// CHECK:             scf.yield %{{.*}} : tensor<30x128xbf16>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [30, 126] [1, 1] : tensor<30x128xbf16> to tensor<30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_nobias_ocunaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<30x126xbf16>) -> tensor<30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%arg2 : tensor<30x126xbf16>) -> tensor<30x126xbf16>
  return %0 : tensor<30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_nobias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xbf16> into tensor<2x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xbf16>) outs(%{{.*}} : tensor<2x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xbf16> into tensor<2x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xbf16> into tensor<30x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xbf16>) outs(%{{.*}} : tensor<1x30x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xbf16>) outs(%{{.*}} : tensor<1x30x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xbf16>) outs(%{{.*}} : tensor<1x5x30x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xbf16> into tensor<1x1x5x30x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xbf16>, tensor<1x1x5x30x16xbf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x64xbf16>) outs(%{{.*}} : tensor<64x128xbf16>) permutation = [1, 0] -> tensor<64x128xbf16>
// CHECK:           %{{.*}} = tensor.empty() : tensor<60x128xbf16>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<60x128xbf16>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<64x128xbf16> to tensor<15x128xbf16>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xbf16> into tensor<60x128xbf16>
// CHECK:             scf.yield %{{.*}} : tensor<60x128xbf16>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [60, 126] [1, 1] : tensor<60x128xbf16> to tensor<60x126xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xbf16> into tensor<2x30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_nobias_ocunaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%arg2 : tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16>
  return %0 : tensor<2x30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_nobias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf32> into tensor<1x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 4, 8, 128] : tensor<1x32x128xf32> into tensor<1x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x4x8x128xf32>) outs(%{{.*}} : tensor<1x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<1x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 4, 1, 128, 8] : tensor<1x4x128x8xf32> into tensor<1x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 2, 8, 5] : tensor<30x16x5xf32> into tensor<30x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x2x8x5xf32>) outs(%{{.*}} : tensor<2x30x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x30x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x8x5xf32>) outs(%{{.*}} : tensor<2x30x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x30x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x5x8xf32>) outs(%{{.*}} : tensor<2x5x30x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x30x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 30, 8] : tensor<2x5x30x8xf32> into tensor<2x1x5x30x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x1x128x8xf32>, tensor<2x1x5x30x8xf32>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<32x128xf32>) permutation = [1, 0] -> tensor<32x128xf32>
// CHECK:           %{{.*}} = tensor.empty() : tensor<30x128xf32>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<30x128xf32>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<32x128xf32> to tensor<15x128xf32>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<30x128xf32>
// CHECK:             scf.yield %{{.*}} : tensor<30x128xf32>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [30, 126] [1, 1] : tensor<30x128xf32> to tensor<30x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_nobias_ocunaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<30x126xf32>) -> tensor<30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf32>, tensor<30x16x5xf32>, i1) outs(%arg2 : tensor<30x126xf32>) -> tensor<30x126xf32>
  return %0 : tensor<30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_nobias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 8, 128] : tensor<2x32x128xf32> into tensor<2x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x4x8x128xf32>) outs(%{{.*}} : tensor<2x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 4, 1, 128, 8] : tensor<2x4x128x8xf32> into tensor<2x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 2, 8, 5] : tensor<30x16x5xf32> into tensor<30x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x2x8x5xf32>) outs(%{{.*}} : tensor<2x30x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x30x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x8x5xf32>) outs(%{{.*}} : tensor<2x30x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x30x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x5x8xf32>) outs(%{{.*}} : tensor<2x5x30x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x30x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 30, 8] : tensor<2x5x30x8xf32> into tensor<2x1x5x30x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x1x128x8xf32>, tensor<2x1x5x30x8xf32>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<64x128xf32>) permutation = [1, 0] -> tensor<64x128xf32>
// CHECK:           %{{.*}} = tensor.empty() : tensor<60x128xf32>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<60x128xf32>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<64x128xf32> to tensor<15x128xf32>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<60x128xf32>
// CHECK:             scf.yield %{{.*}} : tensor<60x128xf32>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [60, 126] [1, 1] : tensor<60x128xf32> to tensor<60x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf32> into tensor<2x30x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_nobias_ocunaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<2x30x126xf32>) -> tensor<2x30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf32>, tensor<30x16x5xf32>, i1) outs(%arg2 : tensor<2x30x126xf32>) -> tensor<2x30x126xf32>
  return %0 : tensor<2x30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_bias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf16> into tensor<1x32x128xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xf16> into tensor<1x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xf16>) outs(%{{.*}} : tensor<1x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xf16> into tensor<1x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xf16> into tensor<30x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xf16>) outs(%{{.*}} : tensor<1x30x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xf16>) outs(%{{.*}} : tensor<1x30x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xf16>) outs(%{{.*}} : tensor<1x5x30x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xf16> into tensor<1x1x5x30x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xf16>, tensor<1x1x5x30x16xf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x32xf16>) outs(%{{.*}} : tensor<32x128xf16>) permutation = [1, 0] -> tensor<32x128xf16>
// CHECK:           %{{.*}} = tensor.empty() : tensor<30x128xf16>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<30x128xf16>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<32x128xf16> to tensor<15x128xf16>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<30x128xf16>
// CHECK:             scf.yield %{{.*}} : tensor<30x128xf16>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [30, 126] [1, 1] : tensor<30x128xf16> to tensor<30x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf16> into tensor<30x1xf16>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<30x126xf16>, tensor<30x1xf16>) outs(%{{.*}} : tensor<30x126xf16>) broadcast = [1] -> tensor<30x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_bias_ocunaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<30xf16>, %arg3: tensor<30x126xf16>) -> tensor<30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf16>, tensor<30x16x5xf16>, i1, tensor<30xf16>) outs(%arg3 : tensor<30x126xf16>) -> tensor<30x126xf16>
  return %0 : tensor<30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_bias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xf16> into tensor<2x2x16x128xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xf16>) outs(%{{.*}} : tensor<2x2x128x16xf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xf16> into tensor<2x2x1x128x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xf16> into tensor<30x1x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xf16>) outs(%{{.*}} : tensor<1x30x16x5xf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xf16>) outs(%{{.*}} : tensor<1x30x5x16xf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xf16>) outs(%{{.*}} : tensor<1x5x30x16xf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xf16> into tensor<1x1x5x30x16xf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xf16>, tensor<1x1x5x30x16xf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x64xf16>) outs(%{{.*}} : tensor<64x128xf16>) permutation = [1, 0] -> tensor<64x128xf16>
// CHECK:           %{{.*}} = tensor.empty() : tensor<60x128xf16>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<60x128xf16>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<64x128xf16> to tensor<15x128xf16>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<60x128xf16>
// CHECK:             scf.yield %{{.*}} : tensor<60x128xf16>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [60, 126] [1, 1] : tensor<60x128xf16> to tensor<60x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf16> into tensor<2x30x126xf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 30, 1] : tensor<30xf16> into tensor<1x30x1xf16>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x30x126xf16>, tensor<1x30x1xf16>) outs(%{{.*}} : tensor<2x30x126xf16>) broadcast = [0, 2] -> tensor<2x30x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_bias_ocunaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<30xf16>, %arg3: tensor<2x30x126xf16>) -> tensor<2x30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf16>, tensor<30x16x5xf16>, i1, tensor<30xf16>) outs(%arg3 : tensor<2x30x126xf16>) -> tensor<2x30x126xf16>
  return %0 : tensor<2x30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_bias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30xbf16>) outs(%{{.*}} : tensor<30xf32>) -> tensor<30xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xbf16> into tensor<1x32x128xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 16, 128] : tensor<1x32x128xbf16> into tensor<1x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x2x16x128xbf16>) outs(%{{.*}} : tensor<1x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 1, 128, 16] : tensor<1x2x128x16xbf16> into tensor<1x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xbf16> into tensor<30x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xbf16>) outs(%{{.*}} : tensor<1x30x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xbf16>) outs(%{{.*}} : tensor<1x30x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xbf16>) outs(%{{.*}} : tensor<1x5x30x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xbf16> into tensor<1x1x5x30x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x1x128x16xbf16>, tensor<1x1x5x30x16xbf16>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<32x128xf32>) permutation = [1, 0] -> tensor<32x128xf32>
// CHECK:           %{{.*}} = tensor.empty() : tensor<30x128xf32>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<30x128xf32>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<32x128xf32> to tensor<15x128xf32>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<30x128xf32>
// CHECK:             scf.yield %{{.*}} : tensor<30x128xf32>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [30, 126] [1, 1] : tensor<30x128xf32> to tensor<30x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<30x126xf32>, tensor<30x1xf32>) outs(%{{.*}} : tensor<30x126xf32>) broadcast = [1] -> tensor<30x126xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30x126xf32>) outs(%{{.*}} : tensor<30x126xbf16>) -> tensor<30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_bias_ocunaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<30xbf16>, %arg3: tensor<30x126xbf16>) -> tensor<30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xbf16>, tensor<30x16x5xbf16>, i1, tensor<30xbf16>) outs(%arg3 : tensor<30x126xbf16>) -> tensor<30x126xbf16>
  return %0 : tensor<30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_bias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30xbf16>) outs(%{{.*}} : tensor<30xf32>) -> tensor<30xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 16, 128] : tensor<2x32x128xbf16> into tensor<2x2x16x128xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x2x16x128xbf16>) outs(%{{.*}} : tensor<2x2x128x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<2x2x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 2, 1, 128, 16] : tensor<2x2x128x16xbf16> into tensor<2x2x1x128x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 1, 16, 5] : tensor<30x16x5xbf16> into tensor<30x1x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x1x16x5xbf16>) outs(%{{.*}} : tensor<1x30x16x5xbf16>) permutation = [1, 0, 2, 3] -> tensor<1x30x16x5xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x16x5xbf16>) outs(%{{.*}} : tensor<1x30x5x16xbf16>) permutation = [0, 1, 3, 2] -> tensor<1x30x5x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x30x5x16xbf16>) outs(%{{.*}} : tensor<1x5x30x16xbf16>) permutation = [0, 2, 1, 3] -> tensor<1x5x30x16xbf16>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 1, 5, 30, 16] : tensor<1x5x30x16xbf16> into tensor<1x1x5x30x16xbf16>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x1x128x16xbf16>, tensor<1x1x5x30x16xbf16>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<64x128xf32>) permutation = [1, 0] -> tensor<64x128xf32>
// CHECK:           %{{.*}} = tensor.empty() : tensor<60x128xf32>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<60x128xf32>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<64x128xf32> to tensor<15x128xf32>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<60x128xf32>
// CHECK:             scf.yield %{{.*}} : tensor<60x128xf32>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [60, 126] [1, 1] : tensor<60x128xf32> to tensor<60x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf32> into tensor<2x30x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 30, 1] : tensor<30xf32> into tensor<1x30x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x30x126xf32>, tensor<1x30x1xf32>) outs(%{{.*}} : tensor<2x30x126xf32>) broadcast = [0, 2] -> tensor<2x30x126xf32>
// CHECK:           %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<2x30x126xf32>) outs(%{{.*}} : tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_bias_ocunaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<30xbf16>, %arg3: tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xbf16>, tensor<30x16x5xbf16>, i1, tensor<30xbf16>) outs(%arg3 : tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16>
  return %0 : tensor<2x30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_bias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 32, 128] : tensor<32x128xf32> into tensor<1x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 4, 8, 128] : tensor<1x32x128xf32> into tensor<1x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x4x8x128xf32>) outs(%{{.*}} : tensor<1x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<1x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 4, 1, 128, 8] : tensor<1x4x128x8xf32> into tensor<1x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 2, 8, 5] : tensor<30x16x5xf32> into tensor<30x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x2x8x5xf32>) outs(%{{.*}} : tensor<2x30x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x30x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x8x5xf32>) outs(%{{.*}} : tensor<2x30x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x30x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x5x8xf32>) outs(%{{.*}} : tensor<2x5x30x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x30x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 30, 8] : tensor<2x5x30x8xf32> into tensor<2x1x5x30x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x1x128x8xf32>, tensor<2x1x5x30x8xf32>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x32xf32>) outs(%{{.*}} : tensor<32x128xf32>) permutation = [1, 0] -> tensor<32x128xf32>
// CHECK:           %{{.*}} = tensor.empty() : tensor<30x128xf32>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<30x128xf32>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<32x128xf32> to tensor<15x128xf32>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<30x128xf32>
// CHECK:             scf.yield %{{.*}} : tensor<30x128xf32>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [30, 126] [1, 1] : tensor<30x128xf32> to tensor<30x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<30x126xf32>, tensor<30x1xf32>) outs(%{{.*}} : tensor<30x126xf32>) broadcast = [1] -> tensor<30x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_bias_ocunaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<30xf32>, %arg3: tensor<30x126xf32>) -> tensor<30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf32>, tensor<30x16x5xf32>, i1, tensor<30xf32>) outs(%arg3 : tensor<30x126xf32>) -> tensor<30x126xf32>
  return %0 : tensor<30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_bias_ocunaligned(
// CHECK:           %{{.*}} = arith.constant 15 : index
// CHECK:           %{{.*}} = arith.constant 16 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 8, 128] : tensor<2x32x128xf32> into tensor<2x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x4x8x128xf32>) outs(%{{.*}} : tensor<2x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 4, 1, 128, 8] : tensor<2x4x128x8xf32> into tensor<2x4x1x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [30, 2, 8, 5] : tensor<30x16x5xf32> into tensor<30x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<30x2x8x5xf32>) outs(%{{.*}} : tensor<2x30x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x30x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x8x5xf32>) outs(%{{.*}} : tensor<2x30x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x30x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x30x5x8xf32>) outs(%{{.*}} : tensor<2x5x30x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x30x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 30, 8] : tensor<2x5x30x8xf32> into tensor<2x1x5x30x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x1x128x8xf32>, tensor<2x1x5x30x8xf32>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<64x128xf32>) permutation = [1, 0] -> tensor<64x128xf32>
// CHECK:           %{{.*}} = tensor.empty() : tensor<60x128xf32>
// CHECK:           %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<60x128xf32>) {
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<64x128xf32> to tensor<15x128xf32>
// CHECK:             %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<60x128xf32>
// CHECK:             scf.yield %{{.*}} : tensor<60x128xf32>
// CHECK:           }
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [60, 126] [1, 1] : tensor<60x128xf32> to tensor<60x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf32> into tensor<2x30x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 30, 1] : tensor<30xf32> into tensor<1x30x1xf32>
// CHECK:           %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x30x126xf32>, tensor<1x30x1xf32>) outs(%{{.*}} : tensor<2x30x126xf32>) broadcast = [0, 2] -> tensor<2x30x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_bias_ocunaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<30xf32>, %arg3: tensor<2x30x126xf32>) -> tensor<2x30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf32>, tensor<30x16x5xf32>, i1, tensor<30xf32>) outs(%arg3 : tensor<2x30x126xf32>) -> tensor<2x30x126xf32>
  return %0 : tensor<2x30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_icunaligned_1(
// CHECK:           %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 15, 128] : tensor<2x30x128xf32> into tensor<2x2x15x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<2x2x1x128xf32>) -> tensor<2x2x1x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(2) ins(%{{.*}}, %{{.*}} : tensor<2x2x15x128xf32>, tensor<2x2x1x128xf32>) outs(%{{.*}} : tensor<2x2x16x128xf32>) -> tensor<2x2x16x128xf32>
// CHECK:           %{{.*}} = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] : tensor<2x2x16x128xf32> into tensor<2x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 8, 128] : tensor<2x32x128xf32> into tensor<2x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x4x8x128xf32>) outs(%{{.*}} : tensor<2x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 4, 1, 128, 8] : tensor<2x4x128x8xf32> into tensor<2x4x1x128x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x1x5xf32>) -> tensor<32x1x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(1) ins(%{{.*}}, %{{.*}} : tensor<32x15x5xf32>, tensor<32x1x5xf32>) outs(%{{.*}} : tensor<32x16x5xf32>) -> tensor<32x16x5xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 2, 8, 5] : tensor<32x16x5xf32> into tensor<32x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x2x8x5xf32>) outs(%{{.*}} : tensor<2x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x8x5xf32>) outs(%{{.*}} : tensor<2x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x5x8xf32>) outs(%{{.*}} : tensor<2x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 32, 8] : tensor<2x5x32x8xf32> into tensor<2x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x1x128x8xf32>, tensor<2x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf32>) outs(%{{.*}} : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_icunaligned_1(%arg0: tensor<2x30x128xf32>, %arg1: tensor<32x15x5xf32>, %arg2: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x30x128xf32>, tensor<32x15x5xf32>, i1) outs(%arg2 : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
  return %0 : tensor<2x32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_icunaligned_2(
// CHECK:           %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 1, 30, 128] : tensor<2x30x128xf32> into tensor<2x1x30x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<2x1x2x128xf32>) -> tensor<2x1x2x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(2) ins(%{{.*}}, %{{.*}} : tensor<2x1x30x128xf32>, tensor<2x1x2x128xf32>) outs(%{{.*}} : tensor<2x1x32x128xf32>) -> tensor<2x1x32x128xf32>
// CHECK:           %{{.*}} = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] : tensor<2x1x32x128xf32> into tensor<2x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 8, 128] : tensor<2x32x128xf32> into tensor<2x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x4x8x128xf32>) outs(%{{.*}} : tensor<2x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [2, 4, 1, 128, 8] : tensor<2x4x128x8xf32> into tensor<2x4x1x128x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x2x5xf32>) -> tensor<32x2x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(1) ins(%{{.*}}, %{{.*}} : tensor<32x30x5xf32>, tensor<32x2x5xf32>) outs(%{{.*}} : tensor<32x32x5xf32>) -> tensor<32x32x5xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 4, 8, 5] : tensor<32x32x5xf32> into tensor<32x4x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x4x8x5xf32>) outs(%{{.*}} : tensor<4x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<4x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<4x32x8x5xf32>) outs(%{{.*}} : tensor<4x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<4x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<4x32x5x8xf32>) outs(%{{.*}} : tensor<4x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<4x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [4, 1, 5, 32, 8] : tensor<4x5x32x8xf32> into tensor<4x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x1x128x8xf32>, tensor<4x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x64xf32>) outs(%{{.*}} : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_icunaligned_2(%arg0: tensor<2x30x128xf32>, %arg1: tensor<32x30x5xf32>, %arg2: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x30x128xf32>, tensor<32x30x5xf32>, i1) outs(%arg2 : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
  return %0 : tensor<2x32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_icunaligned_1(
// CHECK:           %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 30, 128] : tensor<30x128xf32> into tensor<1x30x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 2, 15, 128] : tensor<1x30x128xf32> into tensor<1x2x15x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<1x2x1x128xf32>) -> tensor<1x2x1x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(2) ins(%{{.*}}, %{{.*}} : tensor<1x2x15x128xf32>, tensor<1x2x1x128xf32>) outs(%{{.*}} : tensor<1x2x16x128xf32>) -> tensor<1x2x16x128xf32>
// CHECK:           %{{.*}} = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] : tensor<1x2x16x128xf32> into tensor<1x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 4, 8, 128] : tensor<1x32x128xf32> into tensor<1x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x4x8x128xf32>) outs(%{{.*}} : tensor<1x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<1x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 4, 1, 128, 8] : tensor<1x4x128x8xf32> into tensor<1x4x1x128x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x1x5xf32>) -> tensor<32x1x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(1) ins(%{{.*}}, %{{.*}} : tensor<32x15x5xf32>, tensor<32x1x5xf32>) outs(%{{.*}} : tensor<32x16x5xf32>) -> tensor<32x16x5xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 2, 8, 5] : tensor<32x16x5xf32> into tensor<32x2x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x2x8x5xf32>) outs(%{{.*}} : tensor<2x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<2x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x8x5xf32>) outs(%{{.*}} : tensor<2x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<2x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<2x32x5x8xf32>) outs(%{{.*}} : tensor<2x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<2x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [2, 1, 5, 32, 8] : tensor<2x5x32x8xf32> into tensor<2x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x1x128x8xf32>, tensor<2x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf32>) outs(%{{.*}} : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_icunaligned_1(%arg0: tensor<30x128xf32>, %arg1: tensor<32x15x5xf32>, %arg2: tensor<32x126xf32>) -> tensor<32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<30x128xf32>, tensor<32x15x5xf32>, i1) outs(%arg2 : tensor<32x126xf32>) -> tensor<32x126xf32>
  return %0 : tensor<32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_icunaligned_2(
// CHECK:           %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 30, 128] : tensor<30x128xf32> into tensor<1x30x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 1, 30, 128] : tensor<1x30x128xf32> into tensor<1x1x30x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<1x1x2x128xf32>) -> tensor<1x1x2x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(2) ins(%{{.*}}, %{{.*}} : tensor<1x1x30x128xf32>, tensor<1x1x2x128xf32>) outs(%{{.*}} : tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
// CHECK:           %{{.*}} = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] : tensor<1x1x32x128xf32> into tensor<1x32x128xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [1, 4, 8, 128] : tensor<1x32x128xf32> into tensor<1x4x8x128xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<1x4x8x128xf32>) outs(%{{.*}} : tensor<1x4x128x8xf32>) permutation = [0, 1, 3, 2] -> tensor<1x4x128x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 4, 1, 128, 8] : tensor<1x4x128x8xf32> into tensor<1x4x1x128x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x2x5xf32>) -> tensor<32x2x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vconcat dim(1) ins(%{{.*}}, %{{.*}} : tensor<32x30x5xf32>, tensor<32x2x5xf32>) outs(%{{.*}} : tensor<32x32x5xf32>) -> tensor<32x32x5xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] output_shape [32, 4, 8, 5] : tensor<32x32x5xf32> into tensor<32x4x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<32x4x8x5xf32>) outs(%{{.*}} : tensor<4x32x8x5xf32>) permutation = [1, 0, 2, 3] -> tensor<4x32x8x5xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<4x32x8x5xf32>) outs(%{{.*}} : tensor<4x32x5x8xf32>) permutation = [0, 1, 3, 2] -> tensor<4x32x5x8xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<4x32x5x8xf32>) outs(%{{.*}} : tensor<4x5x32x8xf32>) permutation = [0, 2, 1, 3] -> tensor<4x5x32x8xf32>
// CHECK:           %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] output_shape [4, 1, 5, 32, 8] : tensor<4x5x32x8xf32> into tensor<4x1x5x32x8xf32>
// CHECK:           %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x1x128x8xf32>, tensor<4x1x5x32x8xf32>, i1) outs(%{{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<126x32xf32>) outs(%{{.*}} : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_icunaligned_2(%arg0: tensor<30x128xf32>, %arg1: tensor<32x30x5xf32>, %arg2: tensor<32x126xf32>) -> tensor<32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<30x128xf32>, tensor<32x30x5xf32>, i1) outs(%arg2 : tensor<32x126xf32>) -> tensor<32x126xf32>
  return %0 : tensor<32x126xf32>
}