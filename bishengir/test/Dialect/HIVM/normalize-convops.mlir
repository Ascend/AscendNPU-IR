// RUN: bishengir-opt -hivm-normalize-convops %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_nobias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32x126xf16>) -> tensor<32x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<32x128xf16>, tensor<32x16x5xf16>, i1) outs(%[[VAL_1]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x32xf16>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vcast ins(%[[VAL_2]] : tensor<128x32xf32>) outs(%[[VAL_3]] : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 32] [1, 1] : tensor<128x32xf16> to tensor<126x32xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<32x126xf16>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x32xf16>) outs(%[[VAL_6]] : tensor<32x126xf16>) permutation = [1, 0] -> tensor<32x126xf16>
// CHECK:           return %[[VAL_7]] : tensor<32x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_nobias_ocaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<32x126xf16>) -> tensor<32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf16>, tensor<32x16x5xf16>, i1) outs(%arg2 : tensor<32x126xf16>) -> tensor<32x126xf16>
  return %0 : tensor<32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_nobias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<2x32x128xf16>, tensor<32x16x5xf16>, i1) outs(%[[VAL_1]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x64xf16>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vcast ins(%[[VAL_2]] : tensor<128x64xf32>) outs(%[[VAL_3]] : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 64] [1, 1] : tensor<128x64xf16> to tensor<126x64xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64x126xf16>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x64xf16>) outs(%[[VAL_6]] : tensor<64x126xf16>) permutation = [1, 0] -> tensor<64x126xf16>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf16> into tensor<2x32x126xf16>
// CHECK:           return %[[VAL_8]] : tensor<2x32x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_nobias_ocaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf16>, tensor<32x16x5xf16>, i1) outs(%arg2 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  return %0 : tensor<2x32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_nobias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32x126xbf16>) -> tensor<32x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%[[VAL_1]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x32xbf16>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vcast ins(%[[VAL_2]] : tensor<128x32xf32>) outs(%[[VAL_3]] : tensor<128x32xbf16>) -> tensor<128x32xbf16>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 32] [1, 1] : tensor<128x32xbf16> to tensor<126x32xbf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<32x126xbf16>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x32xbf16>) outs(%[[VAL_6]] : tensor<32x126xbf16>) permutation = [1, 0] -> tensor<32x126xbf16>
// CHECK:           return %[[VAL_7]] : tensor<32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_nobias_ocaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<32x126xbf16>) -> tensor<32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%arg2 : tensor<32x126xbf16>) -> tensor<32x126xbf16>
  return %0 : tensor<32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_nobias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<2x32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%[[VAL_1]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x64xbf16>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vcast ins(%[[VAL_2]] : tensor<128x64xf32>) outs(%[[VAL_3]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 64] [1, 1] : tensor<128x64xbf16> to tensor<126x64xbf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64x126xbf16>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x64xbf16>) outs(%[[VAL_6]] : tensor<64x126xbf16>) permutation = [1, 0] -> tensor<64x126xbf16>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xbf16> into tensor<2x32x126xbf16>
// CHECK:           return %[[VAL_8]] : tensor<2x32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_nobias_ocaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%arg2 : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
  return %0 : tensor<2x32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_nobias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32x126xf32>) -> tensor<32x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<32x128xf32>, tensor<32x16x5xf32>, i1) outs(%[[VAL_1]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_2]][0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<32x126xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vtranspose ins(%[[VAL_3]] : tensor<126x32xf32>) outs(%[[VAL_4]] : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:           return %[[VAL_5]] : tensor<32x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_nobias_ocaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<32x126xf32>) -> tensor<32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf32>, tensor<32x16x5xf32>, i1) outs(%arg2 : tensor<32x126xf32>) -> tensor<32x126xf32>
  return %0 : tensor<32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_nobias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<2x32x128xf32>, tensor<32x16x5xf32>, i1) outs(%[[VAL_1]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_2]][0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64x126xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vtranspose ins(%[[VAL_3]] : tensor<126x64xf32>) outs(%[[VAL_4]] : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.expand_shape %[[VAL_5]] {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:           return %[[VAL_6]] : tensor<2x32x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_nobias_ocaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf32>, tensor<32x16x5xf32>, i1) outs(%arg2 : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
  return %0 : tensor<2x32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_bias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32xf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<32x126xf16>) -> tensor<32x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<32x128xf16>, tensor<32x16x5xf16>, i1) outs(%[[VAL_1]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x32xf16>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vcast ins(%[[VAL_2]] : tensor<128x32xf32>) outs(%[[VAL_3]] : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 32] [1, 1] : tensor<128x32xf16> to tensor<126x32xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<32x126xf16>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x32xf16>) outs(%[[VAL_6]] : tensor<32x126xf16>) permutation = [1, 0] -> tensor<32x126xf16>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf16> into tensor<32x1xf16>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<32x126xf16>
// CHECK:           %[[VAL_10:.*]] = hivm.hir.vadd ins(%[[VAL_7]], %[[VAL_8]] : tensor<32x126xf16>, tensor<32x1xf16>) outs(%[[VAL_9]] : tensor<32x126xf16>) broadcast = [1] -> tensor<32x126xf16>
// CHECK:           return %[[VAL_10]] : tensor<32x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_bias_ocaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32x126xf16>) -> tensor<32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf16>, tensor<32x16x5xf16>, i1, tensor<32xf16>) outs(%arg3 : tensor<32x126xf16>) -> tensor<32x126xf16>
  return %0 : tensor<32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_bias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32xf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<2x32x128xf16>, tensor<32x16x5xf16>, i1) outs(%[[VAL_1]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x64xf16>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vcast ins(%[[VAL_2]] : tensor<128x64xf32>) outs(%[[VAL_3]] : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 64] [1, 1] : tensor<128x64xf16> to tensor<126x64xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64x126xf16>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x64xf16>) outs(%[[VAL_6]] : tensor<64x126xf16>) permutation = [1, 0] -> tensor<64x126xf16>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf16> into tensor<2x32x126xf16>
// CHECK:           %[[VAL_9:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf16> into tensor<1x32x1xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<2x32x126xf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vadd ins(%[[VAL_8]], %[[VAL_9]] : tensor<2x32x126xf16>, tensor<1x32x1xf16>) outs(%[[VAL_10]] : tensor<2x32x126xf16>) broadcast = [0, 2] -> tensor<2x32x126xf16>
// CHECK:           return %[[VAL_11]] : tensor<2x32x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_bias_ocaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<32x16x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf16>, tensor<32x16x5xf16>, i1, tensor<32xf16>) outs(%arg3 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  return %0 : tensor<2x32x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_bias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32xbf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<32x126xbf16>) -> tensor<32x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.vcast ins(%[[ARG2]] : tensor<32xbf16>) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%[[VAL_3]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<32x126xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x32xf32>) outs(%[[VAL_6]] : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_2]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<32x126xf32>
// CHECK:           %[[VAL_10:.*]] = hivm.hir.vadd ins(%[[VAL_7]], %[[VAL_8]] : tensor<32x126xf32>, tensor<32x1xf32>) outs(%[[VAL_9]] : tensor<32x126xf32>) broadcast = [1] -> tensor<32x126xf32>
// CHECK:           %[[VAL_11:.*]] = tensor.empty() : tensor<32x126xbf16>
// CHECK:           %[[VAL_12:.*]] = hivm.hir.vcast ins(%[[VAL_10]] : tensor<32x126xf32>) outs(%[[VAL_11]] : tensor<32x126xbf16>) -> tensor<32x126xbf16>
// CHECK:           return %[[VAL_12]] : tensor<32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_bias_ocaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<32xbf16>, %arg3: tensor<32x126xbf16>) -> tensor<32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xbf16>, tensor<32x16x5xbf16>, i1, tensor<32xbf16>) outs(%arg3 : tensor<32x126xbf16>) -> tensor<32x126xbf16>
  return %0 : tensor<32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_bias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32xbf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.vcast ins(%[[ARG2]] : tensor<32xbf16>) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<2x32x128xbf16>, tensor<32x16x5xbf16>, i1) outs(%[[VAL_3]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64x126xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vtranspose ins(%[[VAL_5]] : tensor<126x64xf32>) outs(%[[VAL_6]] : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.expand_shape %[[VAL_2]] {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf32> into tensor<1x32x1xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<2x32x126xf32>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vadd ins(%[[VAL_8]], %[[VAL_9]] : tensor<2x32x126xf32>, tensor<1x32x1xf32>) outs(%[[VAL_10]] : tensor<2x32x126xf32>) broadcast = [0, 2] -> tensor<2x32x126xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<2x32x126xbf16>
// CHECK:           %[[VAL_13:.*]] = hivm.hir.vcast ins(%[[VAL_11]] : tensor<2x32x126xf32>) outs(%[[VAL_12]] : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
// CHECK:           return %[[VAL_13]] : tensor<2x32x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_bias_ocaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<32x16x5xbf16>, %arg2: tensor<32xbf16>, %arg3: tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xbf16>, tensor<32x16x5xbf16>, i1, tensor<32xbf16>) outs(%arg3 : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
  return %0 : tensor<2x32x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_bias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32xf32>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<32x126xf32>) -> tensor<32x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<32x128xf32>, tensor<32x16x5xf32>, i1) outs(%[[VAL_1]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_2]][0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<32x126xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vtranspose ins(%[[VAL_3]] : tensor<126x32xf32>) outs(%[[VAL_4]] : tensor<32x126xf32>) permutation = [1, 0] -> tensor<32x126xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<32x126xf32>
// CHECK:           %[[VAL_8:.*]] = hivm.hir.vadd ins(%[[VAL_5]], %[[VAL_6]] : tensor<32x126xf32>, tensor<32x1xf32>) outs(%[[VAL_7]] : tensor<32x126xf32>) broadcast = [1] -> tensor<32x126xf32>
// CHECK:           return %[[VAL_8]] : tensor<32x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_bias_ocaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x126xf32>) -> tensor<32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf32>, tensor<32x16x5xf32>, i1, tensor<32xf32>) outs(%arg3 : tensor<32x126xf32>) -> tensor<32x126xf32>
  return %0 : tensor<32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_bias_ocaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<32x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<32xf32>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_0]] : tensor<2x32x128xf32>, tensor<32x16x5xf32>, i1) outs(%[[VAL_1]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_2]][0, 0] [126, 64] [1, 1] : tensor<128x64xf32> to tensor<126x64xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64x126xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vtranspose ins(%[[VAL_3]] : tensor<126x64xf32>) outs(%[[VAL_4]] : tensor<64x126xf32>) permutation = [1, 0] -> tensor<64x126xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.expand_shape %[[VAL_5]] {{\[\[}}0, 1], [2]] output_shape [2, 32, 126] : tensor<64x126xf32> into tensor<2x32x126xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf32> into tensor<1x32x1xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<2x32x126xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vadd ins(%[[VAL_6]], %[[VAL_7]] : tensor<2x32x126xf32>, tensor<1x32x1xf32>) outs(%[[VAL_8]] : tensor<2x32x126xf32>) broadcast = [0, 2] -> tensor<2x32x126xf32>
// CHECK:           return %[[VAL_9]] : tensor<2x32x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_bias_ocaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<32x16x5xf32>, %arg2: tensor<32xf32>, %arg3: tensor<2x32x126xf32>) -> tensor<2x32x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf32>, tensor<32x16x5xf32>, i1, tensor<32xf32>) outs(%arg3 : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
  return %0 : tensor<2x32x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_nobias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30x126xf16>) -> tensor<30x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<32x128xf16>, tensor<30x16x5xf16>, i1) outs(%[[VAL_6]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x32xf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vcast ins(%[[VAL_7]] : tensor<128x32xf32>) outs(%[[VAL_8]] : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<32x128xf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x32xf16>) outs(%[[VAL_10]] : tensor<32x128xf16>) permutation = [1, 0] -> tensor<32x128xf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<30x128xf16>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<30x128xf16>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<32x128xf16> to tensor<15x128xf16>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<30x128xf16>
// CHECK:             scf.yield %[[VAL_19]] : tensor<30x128xf16>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [30, 126] [1, 1] : tensor<30x128xf16> to tensor<30x126xf16>
// CHECK:           return %[[VAL_20]] : tensor<30x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_nobias_ocunaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<30x126xf16>) -> tensor<30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf16>, tensor<30x16x5xf16>, i1) outs(%arg2 : tensor<30x126xf16>) -> tensor<30x126xf16>
  return %0 : tensor<30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_nobias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x30x126xf16>) -> tensor<2x30x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<2x32x128xf16>, tensor<30x16x5xf16>, i1) outs(%[[VAL_6]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x64xf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vcast ins(%[[VAL_7]] : tensor<128x64xf32>) outs(%[[VAL_8]] : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x64xf16>) outs(%[[VAL_10]] : tensor<64x128xf16>) permutation = [1, 0] -> tensor<64x128xf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<60x128xf16>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<60x128xf16>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<64x128xf16> to tensor<15x128xf16>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<60x128xf16>
// CHECK:             scf.yield %[[VAL_19]] : tensor<60x128xf16>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [60, 126] [1, 1] : tensor<60x128xf16> to tensor<60x126xf16>
// CHECK:           %[[VAL_21:.*]] = tensor.expand_shape %[[VAL_20]] {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf16> into tensor<2x30x126xf16>
// CHECK:           return %[[VAL_21]] : tensor<2x30x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_nobias_ocunaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<2x30x126xf16>) -> tensor<2x30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf16>, tensor<30x16x5xf16>, i1) outs(%arg2 : tensor<2x30x126xf16>) -> tensor<2x30x126xf16>
  return %0 : tensor<2x30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_nobias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30x126xbf16>) -> tensor<30x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%[[VAL_6]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x32xbf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vcast ins(%[[VAL_7]] : tensor<128x32xf32>) outs(%[[VAL_8]] : tensor<128x32xbf16>) -> tensor<128x32xbf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<32x128xbf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x32xbf16>) outs(%[[VAL_10]] : tensor<32x128xbf16>) permutation = [1, 0] -> tensor<32x128xbf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<30x128xbf16>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<30x128xbf16>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<32x128xbf16> to tensor<15x128xbf16>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xbf16> into tensor<30x128xbf16>
// CHECK:             scf.yield %[[VAL_19]] : tensor<30x128xbf16>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [30, 126] [1, 1] : tensor<30x128xbf16> to tensor<30x126xbf16>
// CHECK:           return %[[VAL_20]] : tensor<30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_nobias_ocunaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<30x126xbf16>) -> tensor<30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%arg2 : tensor<30x126xbf16>) -> tensor<30x126xbf16>
  return %0 : tensor<30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_nobias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<2x32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%[[VAL_6]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x64xbf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vcast ins(%[[VAL_7]] : tensor<128x64xf32>) outs(%[[VAL_8]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64x128xbf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x64xbf16>) outs(%[[VAL_10]] : tensor<64x128xbf16>) permutation = [1, 0] -> tensor<64x128xbf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<60x128xbf16>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<60x128xbf16>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<64x128xbf16> to tensor<15x128xbf16>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xbf16> into tensor<60x128xbf16>
// CHECK:             scf.yield %[[VAL_19]] : tensor<60x128xbf16>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [60, 126] [1, 1] : tensor<60x128xbf16> to tensor<60x126xbf16>
// CHECK:           %[[VAL_21:.*]] = tensor.expand_shape %[[VAL_20]] {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xbf16> into tensor<2x30x126xbf16>
// CHECK:           return %[[VAL_21]] : tensor<2x30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_nobias_ocunaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%arg2 : tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16>
  return %0 : tensor<2x30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_nobias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30x126xf32>) -> tensor<30x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<32x128xf32>, tensor<30x16x5xf32>, i1) outs(%[[VAL_6]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vtranspose ins(%[[VAL_7]] : tensor<128x32xf32>) outs(%[[VAL_8]] : tensor<32x128xf32>) permutation = [1, 0] -> tensor<32x128xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<30x128xf32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<30x128xf32>) {
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_1]] : index
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_12]], %[[VAL_0]] : index
// CHECK:             %[[VAL_16:.*]] = tensor.extract_slice %[[VAL_9]]{{\[}}%[[VAL_14]], 0] [15, 128] [1, 1] : tensor<32x128xf32> to tensor<15x128xf32>
// CHECK:             %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_13]]{{\[}}%[[VAL_15]], 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<30x128xf32>
// CHECK:             scf.yield %[[VAL_17]] : tensor<30x128xf32>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]][0, 0] [30, 126] [1, 1] : tensor<30x128xf32> to tensor<30x126xf32>
// CHECK:           return %[[VAL_18]] : tensor<30x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_nobias_ocunaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<30x126xf32>) -> tensor<30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<32x128xf32>, tensor<30x16x5xf32>, i1) outs(%arg2 : tensor<30x126xf32>) -> tensor<30x126xf32>
  return %0 : tensor<30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_nobias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x30x126xf32>) -> tensor<2x30x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<2x32x128xf32>, tensor<30x16x5xf32>, i1) outs(%[[VAL_6]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vtranspose ins(%[[VAL_7]] : tensor<128x64xf32>) outs(%[[VAL_8]] : tensor<64x128xf32>) permutation = [1, 0] -> tensor<64x128xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<60x128xf32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<60x128xf32>) {
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_1]] : index
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_12]], %[[VAL_0]] : index
// CHECK:             %[[VAL_16:.*]] = tensor.extract_slice %[[VAL_9]]{{\[}}%[[VAL_14]], 0] [15, 128] [1, 1] : tensor<64x128xf32> to tensor<15x128xf32>
// CHECK:             %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_13]]{{\[}}%[[VAL_15]], 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<60x128xf32>
// CHECK:             scf.yield %[[VAL_17]] : tensor<60x128xf32>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]][0, 0] [60, 126] [1, 1] : tensor<60x128xf32> to tensor<60x126xf32>
// CHECK:           %[[VAL_19:.*]] = tensor.expand_shape %[[VAL_18]] {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf32> into tensor<2x30x126xf32>
// CHECK:           return %[[VAL_19]] : tensor<2x30x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_nobias_ocunaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<2x30x126xf32>) -> tensor<2x30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true : tensor<2x32x128xf32>, tensor<30x16x5xf32>, i1) outs(%arg2 : tensor<2x30x126xf32>) -> tensor<2x30x126xf32>
  return %0 : tensor<2x30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp16_bias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30xf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<30x126xf16>) -> tensor<30x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<32x128xf16>, tensor<30x16x5xf16>, i1) outs(%[[VAL_6]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x32xf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vcast ins(%[[VAL_7]] : tensor<128x32xf32>) outs(%[[VAL_8]] : tensor<128x32xf16>) -> tensor<128x32xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<32x128xf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x32xf16>) outs(%[[VAL_10]] : tensor<32x128xf16>) permutation = [1, 0] -> tensor<32x128xf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<30x128xf16>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<30x128xf16>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<32x128xf16> to tensor<15x128xf16>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<30x128xf16>
// CHECK:             scf.yield %[[VAL_19]] : tensor<30x128xf16>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [30, 126] [1, 1] : tensor<30x128xf16> to tensor<30x126xf16>
// CHECK:           %[[VAL_21:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf16> into tensor<30x1xf16>
// CHECK:           %[[VAL_22:.*]] = tensor.empty() : tensor<30x126xf16>
// CHECK:           %[[VAL_23:.*]] = hivm.hir.vadd ins(%[[VAL_20]], %[[VAL_21]] : tensor<30x126xf16>, tensor<30x1xf16>) outs(%[[VAL_22]] : tensor<30x126xf16>) broadcast = [1] -> tensor<30x126xf16>
// CHECK:           return %[[VAL_23]] : tensor<30x126xf16>
// CHECK:         }
func.func @triton_conv1d_2d_fp16_bias_ocunaligned(%arg0: tensor<32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<30xf16>, %arg3: tensor<30x126xf16>) -> tensor<30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf16>, tensor<30x16x5xf16>, i1, tensor<30xf16>) outs(%arg3 : tensor<30x126xf16>) -> tensor<30x126xf16>
  return %0 : tensor<30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_bias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30xf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x30x126xf16>) -> tensor<2x30x126xf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<2x32x128xf16>, tensor<30x16x5xf16>, i1) outs(%[[VAL_6]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x64xf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vcast ins(%[[VAL_7]] : tensor<128x64xf32>) outs(%[[VAL_8]] : tensor<128x64xf16>) -> tensor<128x64xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x64xf16>) outs(%[[VAL_10]] : tensor<64x128xf16>) permutation = [1, 0] -> tensor<64x128xf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<60x128xf16>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<60x128xf16>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<64x128xf16> to tensor<15x128xf16>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xf16> into tensor<60x128xf16>
// CHECK:             scf.yield %[[VAL_19]] : tensor<60x128xf16>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [60, 126] [1, 1] : tensor<60x128xf16> to tensor<60x126xf16>
// CHECK:           %[[VAL_21:.*]] = tensor.expand_shape %[[VAL_20]] {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf16> into tensor<2x30x126xf16>
// CHECK:           %[[VAL_22:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1, 2]] output_shape [1, 30, 1] : tensor<30xf16> into tensor<1x30x1xf16>
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<2x30x126xf16>
// CHECK:           %[[VAL_24:.*]] = hivm.hir.vadd ins(%[[VAL_21]], %[[VAL_22]] : tensor<2x30x126xf16>, tensor<1x30x1xf16>) outs(%[[VAL_23]] : tensor<2x30x126xf16>) broadcast = [0, 2] -> tensor<2x30x126xf16>
// CHECK:           return %[[VAL_24]] : tensor<2x30x126xf16>
// CHECK:         }
func.func @triton_conv1d_3d_fp16_bias_ocunaligned(%arg0: tensor<2x32x128xf16>, %arg1: tensor<30x16x5xf16>, %arg2: tensor<30xf16>, %arg3: tensor<2x30x126xf16>) -> tensor<2x30x126xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf16>, tensor<30x16x5xf16>, i1, tensor<30xf16>) outs(%arg3 : tensor<2x30x126xf16>) -> tensor<2x30x126xf16>
  return %0 : tensor<2x30x126xf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_bf16_bias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30xbf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<30x126xbf16>) -> tensor<30x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<30xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vcast ins(%[[ARG2]] : tensor<30xbf16>) outs(%[[VAL_6]] : tensor<30xf32>) -> tensor<30xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%[[VAL_8]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x32xf32>) outs(%[[VAL_10]] : tensor<32x128xf32>) permutation = [1, 0] -> tensor<32x128xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<30x128xf32>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<30x128xf32>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<32x128xf32> to tensor<15x128xf32>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<30x128xf32>
// CHECK:             scf.yield %[[VAL_19]] : tensor<30x128xf32>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [30, 126] [1, 1] : tensor<30x128xf32> to tensor<30x126xf32>
// CHECK:           %[[VAL_21:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
// CHECK:           %[[VAL_22:.*]] = tensor.empty() : tensor<30x126xf32>
// CHECK:           %[[VAL_23:.*]] = hivm.hir.vadd ins(%[[VAL_20]], %[[VAL_21]] : tensor<30x126xf32>, tensor<30x1xf32>) outs(%[[VAL_22]] : tensor<30x126xf32>) broadcast = [1] -> tensor<30x126xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.empty() : tensor<30x126xbf16>
// CHECK:           %[[VAL_25:.*]] = hivm.hir.vcast ins(%[[VAL_23]] : tensor<30x126xf32>) outs(%[[VAL_24]] : tensor<30x126xbf16>) -> tensor<30x126xbf16>
// CHECK:           return %[[VAL_25]] : tensor<30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_2d_bf16_bias_ocunaligned(%arg0: tensor<32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<30xbf16>, %arg3: tensor<30x126xbf16>) -> tensor<30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xbf16>, tensor<30x16x5xbf16>, i1, tensor<30xbf16>) outs(%arg3 : tensor<30x126xbf16>) -> tensor<30x126xbf16>
  return %0 : tensor<30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_bf16_bias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30xbf16>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<30xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vcast ins(%[[ARG2]] : tensor<30xbf16>) outs(%[[VAL_6]] : tensor<30xf32>) -> tensor<30xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<2x32x128xbf16>, tensor<30x16x5xbf16>, i1) outs(%[[VAL_8]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.vtranspose ins(%[[VAL_9]] : tensor<128x64xf32>) outs(%[[VAL_10]] : tensor<64x128xf32>) permutation = [1, 0] -> tensor<64x128xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<60x128xf32>
// CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<60x128xf32>) {
// CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_17:.*]] = arith.muli %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_16]], 0] [15, 128] [1, 1] : tensor<64x128xf32> to tensor<15x128xf32>
// CHECK:             %[[VAL_19:.*]] = tensor.insert_slice %[[VAL_18]] into %[[VAL_15]]{{\[}}%[[VAL_17]], 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<60x128xf32>
// CHECK:             scf.yield %[[VAL_19]] : tensor<60x128xf32>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_13]][0, 0] [60, 126] [1, 1] : tensor<60x128xf32> to tensor<60x126xf32>
// CHECK:           %[[VAL_21:.*]] = tensor.expand_shape %[[VAL_20]] {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf32> into tensor<2x30x126xf32>
// CHECK:           %[[VAL_22:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0, 1, 2]] output_shape [1, 30, 1] : tensor<30xf32> into tensor<1x30x1xf32>
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<2x30x126xf32>
// CHECK:           %[[VAL_24:.*]] = hivm.hir.vadd ins(%[[VAL_21]], %[[VAL_22]] : tensor<2x30x126xf32>, tensor<1x30x1xf32>) outs(%[[VAL_23]] : tensor<2x30x126xf32>) broadcast = [0, 2] -> tensor<2x30x126xf32>
// CHECK:           %[[VAL_25:.*]] = tensor.empty() : tensor<2x30x126xbf16>
// CHECK:           %[[VAL_26:.*]] = hivm.hir.vcast ins(%[[VAL_24]] : tensor<2x30x126xf32>) outs(%[[VAL_25]] : tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16>
// CHECK:           return %[[VAL_26]] : tensor<2x30x126xbf16>
// CHECK:         }
func.func @triton_conv1d_3d_bf16_bias_ocunaligned(%arg0: tensor<2x32x128xbf16>, %arg1: tensor<30x16x5xbf16>, %arg2: tensor<30xbf16>, %arg3: tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xbf16>, tensor<30x16x5xbf16>, i1, tensor<30xbf16>) outs(%arg3 : tensor<2x30x126xbf16>) -> tensor<2x30x126xbf16>
  return %0 : tensor<2x30x126xbf16>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_2d_fp32_bias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30xf32>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<30x126xf32>) -> tensor<30x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x32xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<32x128xf32>, tensor<30x16x5xf32>, i1) outs(%[[VAL_6]] : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vtranspose ins(%[[VAL_7]] : tensor<128x32xf32>) outs(%[[VAL_8]] : tensor<32x128xf32>) permutation = [1, 0] -> tensor<32x128xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<30x128xf32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<30x128xf32>) {
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_1]] : index
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_12]], %[[VAL_0]] : index
// CHECK:             %[[VAL_16:.*]] = tensor.extract_slice %[[VAL_9]]{{\[}}%[[VAL_14]], 0] [15, 128] [1, 1] : tensor<32x128xf32> to tensor<15x128xf32>
// CHECK:             %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_13]]{{\[}}%[[VAL_15]], 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<30x128xf32>
// CHECK:             scf.yield %[[VAL_17]] : tensor<30x128xf32>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]][0, 0] [30, 126] [1, 1] : tensor<30x128xf32> to tensor<30x126xf32>
// CHECK:           %[[VAL_19:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
// CHECK:           %[[VAL_20:.*]] = tensor.empty() : tensor<30x126xf32>
// CHECK:           %[[VAL_21:.*]] = hivm.hir.vadd ins(%[[VAL_18]], %[[VAL_19]] : tensor<30x126xf32>, tensor<30x1xf32>) outs(%[[VAL_20]] : tensor<30x126xf32>) broadcast = [1] -> tensor<30x126xf32>
// CHECK:           return %[[VAL_21]] : tensor<30x126xf32>
// CHECK:         }
func.func @triton_conv1d_2d_fp32_bias_ocunaligned(%arg0: tensor<32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<30xf32>, %arg3: tensor<30x126xf32>) -> tensor<30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<32x128xf32>, tensor<30x16x5xf32>, i1, tensor<30xf32>) outs(%arg3 : tensor<30x126xf32>) -> tensor<30x126xf32>
  return %0 : tensor<30x126xf32>
}

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp32_bias_ocunaligned(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x32x128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<30x16x5xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<30xf32>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x30x126xf32>) -> tensor<2x30x126xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.Conv1dL1 {groups = 2 : i32, outputAlreadyNormalized, padding = 1 : i32} ins(%[[ARG0]], %[[ARG1]], %[[VAL_5]] : tensor<2x32x128xf32>, tensor<30x16x5xf32>, i1) outs(%[[VAL_6]] : tensor<128x64xf32>) -> tensor<128x64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.vtranspose ins(%[[VAL_7]] : tensor<128x64xf32>) outs(%[[VAL_8]] : tensor<64x128xf32>) permutation = [1, 0] -> tensor<64x128xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<60x128xf32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<60x128xf32>) {
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_1]] : index
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_12]], %[[VAL_0]] : index
// CHECK:             %[[VAL_16:.*]] = tensor.extract_slice %[[VAL_9]]{{\[}}%[[VAL_14]], 0] [15, 128] [1, 1] : tensor<64x128xf32> to tensor<15x128xf32>
// CHECK:             %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_13]]{{\[}}%[[VAL_15]], 0] [15, 128] [1, 1] : tensor<15x128xf32> into tensor<60x128xf32>
// CHECK:             scf.yield %[[VAL_17]] : tensor<60x128xf32>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]][0, 0] [60, 126] [1, 1] : tensor<60x128xf32> to tensor<60x126xf32>
// CHECK:           %[[VAL_19:.*]] = tensor.expand_shape %[[VAL_18]] {{\[\[}}0, 1], [2]] output_shape [2, 30, 126] : tensor<60x126xf32> into tensor<2x30x126xf32>
// CHECK:           %[[VAL_20:.*]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1, 2]] output_shape [1, 30, 1] : tensor<30xf32> into tensor<1x30x1xf32>
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<2x30x126xf32>
// CHECK:           %[[VAL_22:.*]] = hivm.hir.vadd ins(%[[VAL_19]], %[[VAL_20]] : tensor<2x30x126xf32>, tensor<1x30x1xf32>) outs(%[[VAL_21]] : tensor<2x30x126xf32>) broadcast = [0, 2] -> tensor<2x30x126xf32>
// CHECK:           return %[[VAL_22]] : tensor<2x30x126xf32>
// CHECK:         }
func.func @triton_conv1d_3d_fp32_bias_ocunaligned(%arg0: tensor<2x32x128xf32>, %arg1: tensor<30x16x5xf32>, %arg2: tensor<30xf32>, %arg3: tensor<2x30x126xf32>) -> tensor<2x30x126xf32> {
  %true = arith.constant true
  %0 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%arg0, %arg1, %true, %arg2 : tensor<2x32x128xf32>, tensor<30x16x5xf32>, i1, tensor<30xf32>) outs(%arg3 : tensor<2x30x126xf32>) -> tensor<2x30x126xf32>
  return %0 : tensor<2x30x126xf32>
}