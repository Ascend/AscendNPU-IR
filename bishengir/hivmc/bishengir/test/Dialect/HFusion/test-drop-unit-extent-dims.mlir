// RUN: bishengir-opt %s -linalg-fold-unit-extent-dims -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -linalg-fold-unit-extent-dims="use-rank-reducing-slices" -cse -split-input-file | FileCheck %s --check-prefix=CHECK-SLICES

func.func @drop_broadcast(%arg0 : tensor<1x2xf32>) -> tensor<1x2x3xf32> {
  %0 = tensor.empty() : tensor<1x2x3xf32>
  %1 = linalg.broadcast ins(%arg0: tensor<1x2xf32>) outs(%0 : tensor<1x2x3xf32>) dimensions = [2]
  return %1 : tensor<1x2x3xf32>
}

// CHECK-LABEL:   func.func @drop_broadcast(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<1x2xf32>) -> tensor<1x2x3xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<1x2xf32> into tensor<2xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x3xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.broadcast ins(%[[VAL_1]] : tensor<2xf32>) outs(%[[VAL_2]] : tensor<2x3xf32>) dimensions = [1]
// CHECK:           %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_3]] {{\[\[}}0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x2x3xf32>
// CHECK:         }

// CHECK-SLICES-LABEL:   func.func @drop_broadcast(
// CHECK-SLICES-SAME:                              %[[VAL_0:.*]]: tensor<1x2xf32>) -> tensor<1x2x3xf32> {
// CHECK-SLICES:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x2x3xf32>
// CHECK-SLICES:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [1, 2] [1, 1] : tensor<1x2xf32> to tensor<2xf32>
// CHECK-SLICES:           %[[VAL_3:.*]] = tensor.empty() : tensor<2x3xf32>
// CHECK-SLICES:           %[[VAL_4:.*]] = linalg.broadcast ins(%[[VAL_2]] : tensor<2xf32>) outs(%[[VAL_3]] : tensor<2x3xf32>) dimensions = [1]
// CHECK-SLICES:           %[[VAL_5:.*]] = tensor.insert_slice %[[VAL_4]] into %[[VAL_1]][0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<2x3xf32> into tensor<1x2x3xf32>
// CHECK-SLICES:           return %[[VAL_5]] : tensor<1x2x3xf32>
// CHECK-SLICES:         }

// -----

func.func @drop_reduce(%arg0 : tensor<1x2x3xf32>) -> tensor<1x2xf32> {
  %0 = tensor.empty() : tensor<1x2xf32>
  %1 = linalg.reduce ins(%arg0: tensor<1x2x3xf32>) outs(%0 : tensor<1x2xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %2 = arith.addf %in, %init : f32
      linalg.yield %2 : f32
    }
  return %1 : tensor<1x2xf32>
}

// CHECK-LABEL:   func.func @drop_reduce(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x2x3xf32>) -> tensor<1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.reduce ins(%[[VAL_1]] : tensor<2x3xf32>) outs(%[[VAL_2]] : tensor<2xf32>) dimensions = [1]
// CHECK:             (%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32) {
// CHECK:               %[[VAL_6:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:               linalg.yield %[[VAL_6]] : f32
// CHECK:             }
// CHECK:           %[[VAL_7:.*]] = tensor.expand_shape %[[VAL_3]] {{\[\[}}0, 1]] output_shape [1, 2] : tensor<2xf32> into tensor<1x2xf32>
// CHECK:           return %[[VAL_7]] : tensor<1x2xf32>
// CHECK:         }

// CHECK-SLICES-LABEL:   func.func @drop_reduce(
// CHECK-SLICES-SAME:                           %[[VAL_0:.*]]: tensor<1x2x3xf32>) -> tensor<1x2xf32> {
// CHECK-SLICES:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x2xf32>
// CHECK-SLICES:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<1x2x3xf32> to tensor<2x3xf32>
// CHECK-SLICES:           %[[VAL_3:.*]] = tensor.empty() : tensor<2xf32>
// CHECK-SLICES:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_2]] : tensor<2x3xf32>) outs(%[[VAL_3]] : tensor<2xf32>) dimensions = [1]
// CHECK-SLICES:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
// CHECK-SLICES:               %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK-SLICES:               linalg.yield %[[VAL_7]] : f32
// CHECK-SLICES:             }
// CHECK-SLICES:           %[[VAL_8:.*]] = tensor.insert_slice %[[VAL_4]] into %[[VAL_1]][0, 0] [1, 2] [1, 1] : tensor<2xf32> into tensor<1x2xf32>
// CHECK-SLICES:           return %[[VAL_8]] : tensor<1x2xf32>
// CHECK-SLICES:         }

// -----

func.func @drop_transpose(%arg0 : tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = tensor.empty() : tensor<3x2x1xf32>
  %1 = linalg.transpose ins(%arg0: tensor<1x2x3xf32>) outs(%0 : tensor<3x2x1xf32>) permutation = [2, 1, 0]
  return %1 : tensor<3x2x1xf32>
}

// CHECK-LABEL:   func.func @drop_transpose(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<3x2xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.transpose ins(%[[VAL_1]] : tensor<2x3xf32>) outs(%[[VAL_2]] : tensor<3x2xf32>) permutation = [1, 0]
// CHECK:           %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_3]] {{\[\[}}0], [1, 2]] output_shape [3, 2, 1] : tensor<3x2xf32> into tensor<3x2x1xf32>
// CHECK:           return %[[VAL_4]] : tensor<3x2x1xf32>
// CHECK:         }

// CHECK-SLICES-LABEL:   func.func @drop_transpose(
// CHECK-SLICES-SAME:                              %[[VAL_0:.*]]: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK-SLICES:           %[[VAL_1:.*]] = tensor.empty() : tensor<3x2x1xf32>
// CHECK-SLICES:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<1x2x3xf32> to tensor<2x3xf32>
// CHECK-SLICES:           %[[VAL_3:.*]] = tensor.empty() : tensor<3x2xf32>
// CHECK-SLICES:           %[[VAL_4:.*]] = linalg.transpose ins(%[[VAL_2]] : tensor<2x3xf32>) outs(%[[VAL_3]] : tensor<3x2xf32>) permutation = [1, 0]
// CHECK-SLICES:           %[[VAL_5:.*]] = tensor.insert_slice %[[VAL_4]] into %[[VAL_1]][0, 0, 0] [3, 2, 1] [1, 1, 1] : tensor<3x2xf32> into tensor<3x2x1xf32>
// CHECK-SLICES:           return %[[VAL_5]] : tensor<3x2x1xf32>
// CHECK-SLICES:         }
