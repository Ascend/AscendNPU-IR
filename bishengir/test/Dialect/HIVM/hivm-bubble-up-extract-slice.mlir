// RUN: bishengir-opt %s  --hivm-bubble-up-extract-slice --split-input-file -verify-diagnostics  | FileCheck %s

// CHECK-LABEL:   func.func @bubble_up_hivm_unary_1d_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vln ins(%[[VAL_1]] : tensor<2xf32>) outs(%[[VAL_2]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return %[[VAL_3]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm_unary_1d_static(%arg0: tensor<4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() :  tensor<4xf32>
    %29 = hivm.hir.vln ins(%arg0: tensor<4xf32>) outs(%0  : tensor<4xf32>) -> tensor<4xf32>
    %extracted_slice = tensor.extract_slice %29[1] [2] [1] {to_be_bubbled_slice} : tensor<4xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_unary_1d_dyn(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty(%[[SIZE]]) : tensor<?xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vln ins(%[[VAL_1]] : tensor<?xf32>) outs(%[[VAL_2]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return %[[VAL_3]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_unary_1d_dyn(%arg0: tensor<5xf32>, %offset: index, %size: index) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() :  tensor<5xf32>
  %1 = hivm.hir.vln ins(%arg0: tensor<5xf32>) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] {to_be_bubbled_slice}: tensor<5xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_unary_1d_dyn2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?xf32>, %[[VAL_1:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vln ins(%[[VAL_2]] : tensor<?xf32>) outs(%[[VAL_3]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return %[[VAL_4]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_unary_1d_dyn2(%arg0: tensor<?xf32>, %0: tensor<?xf32>, %offset: index, %size: index) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vln ins(%arg0: tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] {to_be_bubbled_slice}: tensor<?xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_unary_2d_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<2x8xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x8xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vln ins(%[[VAL_1]] : tensor<2x8xf32>) outs(%[[VAL_2]] : tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK:           return %[[VAL_3]] : tensor<2x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_unary_2d_static(%arg0: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() :  tensor<4x8xf32>
    %29 = hivm.hir.vln ins(%arg0: tensor<4x8xf32>) outs(%0  : tensor<4x8xf32>) -> tensor<4x8xf32>
    %extracted_slice = tensor.extract_slice %29[1,0] [2,8] [1,1] {to_be_bubbled_slice} : tensor<4x8xf32> to tensor<2x8xf32>
    return %extracted_slice : tensor<2x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_unary_2d_dyn(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5x8xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty(%[[SIZE]]) : tensor<?x8xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vln ins(%[[VAL_1]] : tensor<?x8xf32>) outs(%[[VAL_2]] : tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           return %[[VAL_3]] : tensor<?x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_unary_2d_dyn(%arg0: tensor<5x8xf32>, %offset: index, %size: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() :  tensor<5x8xf32>
  %1 = hivm.hir.vln ins(%arg0: tensor<5x8xf32>) outs(%0 : tensor<5x8xf32>) -> tensor<5x8xf32>
  %extracted_slice = tensor.extract_slice %1[%offset,0] [%size,8] [1,1] {to_be_bubbled_slice}: tensor<5x8xf32> to tensor<?x8xf32>
  return %extracted_slice : tensor<?x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_unary_2d_dyn2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x8xf32>, %[[VAL_1:.*]]: tensor<?x8xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vln ins(%[[VAL_2]] : tensor<?x8xf32>) outs(%[[VAL_3]] : tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           return %[[VAL_4]] : tensor<?x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_unary_2d_dyn2(%arg0: tensor<?x8xf32>, %0: tensor<?x8xf32>, %offset: index, %size: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vln ins(%arg0: tensor<?x8xf32>) outs(%0 : tensor<?x8xf32>) -> tensor<?x8xf32>
  %extracted_slice = tensor.extract_slice %1[%offset,0] [%size,8] [1,1] {to_be_bubbled_slice}: tensor<?x8xf32> to tensor<?x8xf32>
  return %extracted_slice : tensor<?x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_binary_1d_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4xf32>, %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vadd ins(%[[VAL_2]], %[[VAL_3]] : tensor<2xf32>, tensor<2xf32>) outs(%[[VAL_4]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return %[[VAL_5]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm_binary_1d_static(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4xf32>
    %29 = hivm.hir.vadd ins(%arg0, %arg1: tensor<4xf32>, tensor<4xf32>) outs(%0: tensor<4xf32>) -> tensor<4xf32>
    %extracted_slice = tensor.extract_slice %29[1] [2] [1] {to_be_bubbled_slice} : tensor<4xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_binary_1d_dyn(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5xf32>, %[[VAL_1:.*]]: tensor<5xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_4:.*]] = tensor.empty(%[[SIZE]]) : tensor<?xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vadd ins(%[[VAL_2]], %[[VAL_3]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_4]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return %[[VAL_5]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_binary_1d_dyn(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %offset: index, %size: index) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1: tensor<5xf32>, tensor<5xf32>) outs(%0: tensor<5xf32>) -> tensor<5xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] {to_be_bubbled_slice}: tensor<5xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_binary_1d_dyn2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?xf32>, %[[VAL_1:.*]]: tensor<?xf32>, %[[VAL_2:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_6:.*]] = hivm.hir.vadd ins(%[[VAL_3]], %[[VAL_4]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_5]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return %[[VAL_6]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_binary_1d_dyn2(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %0: tensor<?xf32>, %offset: index, %size: index) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vadd ins(%arg0, %arg1: tensor<?xf32>, tensor<?xf32>) outs(%0: tensor<?xf32>) -> tensor<?xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] {to_be_bubbled_slice}: tensor<?xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_binary_2d_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4x8xf32>, %[[VAL_1:.*]]: tensor<4x8xf32>) -> tensor<2x8xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<2x8xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vadd ins(%[[VAL_2]], %[[VAL_3]] : tensor<2x8xf32>, tensor<2x8xf32>) outs(%[[VAL_4]] : tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_binary_2d_static(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x8xf32>
    %29 = hivm.hir.vadd ins(%arg0, %arg1: tensor<4x8xf32>, tensor<4x8xf32>) outs(%0: tensor<4x8xf32>) -> tensor<4x8xf32>
    %extracted_slice = tensor.extract_slice %29[1,0] [2,8] [1,1] {to_be_bubbled_slice} : tensor<4x8xf32> to tensor<2x8xf32>
    return %extracted_slice : tensor<2x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_binary_2d_dyn(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5x8xf32>, %[[VAL_1:.*]]: tensor<5x8xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_4:.*]] = tensor.empty(%[[SIZE]]) : tensor<?x8xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vadd ins(%[[VAL_2]], %[[VAL_3]] : tensor<?x8xf32>, tensor<?x8xf32>) outs(%[[VAL_4]] : tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           return %[[VAL_5]] : tensor<?x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_binary_2d_dyn(%arg0: tensor<5x8xf32>, %arg1: tensor<5x8xf32>, %offset: index, %size: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x8xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1: tensor<5x8xf32>, tensor<5x8xf32>) outs(%0: tensor<5x8xf32>) -> tensor<5x8xf32>
  %extracted_slice = tensor.extract_slice %1[%offset,0] [%size,8] [1,1] {to_be_bubbled_slice}: tensor<5x8xf32> to tensor<?x8xf32>
  return %extracted_slice : tensor<?x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_binary_2d_dyn2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x8xf32>, %[[VAL_1:.*]]: tensor<?x8xf32>, %[[VAL_2:.*]]: tensor<?x8xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_6:.*]] = hivm.hir.vadd ins(%[[VAL_3]], %[[VAL_4]] : tensor<?x8xf32>, tensor<?x8xf32>) outs(%[[VAL_5]] : tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           return %[[VAL_6]] : tensor<?x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_binary_2d_dyn2(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %0: tensor<?x8xf32>, %offset: index, %size: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vadd ins(%arg0, %arg1: tensor<?x8xf32>, tensor<?x8xf32>) outs(%0: tensor<?x8xf32>) -> tensor<?x8xf32>
  %extracted_slice = tensor.extract_slice %1[%offset,0] [%size,8] [1,1] {to_be_bubbled_slice}: tensor<?x8xf32> to tensor<?x8xf32>
  return %extracted_slice : tensor<?x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_ternary_1d_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4xi1>, %[[VAL_1:.*]]: tensor<4xf32>, %[[VAL_2:.*]]: tensor<4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vsel ins(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) outs(%[[VAL_6]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return %[[VAL_7]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm_ternary_1d_static(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4xf32>
    %29 = hivm.hir.vsel ins(%arg0, %arg1, %arg2: tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%0: tensor<4xf32>) -> tensor<4xf32>
    %extracted_slice = tensor.extract_slice %29[1] [2] [1] {to_be_bubbled_slice} : tensor<4xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_ternary_1d_dyn(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5xi1>, %[[VAL_1:.*]]: tensor<5xf32>, %[[VAL_2:.*]]: tensor<5xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_6:.*]] = tensor.empty(%[[SIZE]]) : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vsel ins(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_6]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return %[[VAL_7]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_ternary_1d_dyn(%arg0: tensor<5xi1>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>, %offset: index, %size: index) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5xf32>
  %1 = hivm.hir.vsel ins(%arg0, %arg1, %arg2: tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) outs(%0: tensor<5xf32>) -> tensor<5xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] {to_be_bubbled_slice}: tensor<5xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_ternary_1d_dyn2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?xi1>, %[[VAL_1:.*]]: tensor<?xf32>, %[[VAL_2:.*]]: tensor<?xf32>, %[[VAL_3:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_3]]
// CHECK:           %[[VAL_8:.*]] = hivm.hir.vsel ins(%[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_7]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return %[[VAL_8]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_ternary_1d_dyn2(%arg0: tensor<?xi1>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %0: tensor<?xf32>, %offset: index, %size: index) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vsel ins(%arg0, %arg1, %arg2: tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) outs(%0: tensor<?xf32>) -> tensor<?xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] {to_be_bubbled_slice}: tensor<?xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_ternary_2d_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4x8xi1>, %[[VAL_1:.*]]: tensor<4x8xf32>, %[[VAL_2:.*]]: tensor<4x8xf32>) -> tensor<2x8xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<2x8xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vsel ins(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) outs(%[[VAL_6]] : tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK:           return %[[VAL_7]] : tensor<2x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_ternary_2d_static(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x8xf32>
    %29 = hivm.hir.vsel ins(%arg0, %arg1, %arg2: tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) outs(%0: tensor<4x8xf32>) -> tensor<4x8xf32>
    %extracted_slice = tensor.extract_slice %29[1,0] [2,8] [1,1] {to_be_bubbled_slice} : tensor<4x8xf32> to tensor<2x8xf32>
    return %extracted_slice : tensor<2x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_ternary_2d_dyn(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5x8xi1>, %[[VAL_1:.*]]: tensor<5x8xf32>, %[[VAL_2:.*]]: tensor<5x8xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_6:.*]] = tensor.empty(%[[SIZE]]) : tensor<?x8xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vsel ins(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : tensor<?x8xi1>, tensor<?x8xf32>, tensor<?x8xf32>) outs(%[[VAL_6]] : tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           return %[[VAL_7]] : tensor<?x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_ternary_2d_dyn(%arg0: tensor<5x8xi1>, %arg1: tensor<5x8xf32>, %arg2: tensor<5x8xf32>, %offset: index, %size: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x8xf32>
  %1 = hivm.hir.vsel ins(%arg0, %arg1, %arg2: tensor<5x8xi1>, tensor<5x8xf32>, tensor<5x8xf32>) outs(%0: tensor<5x8xf32>) -> tensor<5x8xf32>
  %extracted_slice = tensor.extract_slice %1[%offset,0] [%size,8] [1,1] {to_be_bubbled_slice}: tensor<5x8xf32> to tensor<?x8xf32>
  return %extracted_slice : tensor<?x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_ternary_2d_dyn2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x8xi1>, %[[VAL_1:.*]]: tensor<?x8xf32>, %[[VAL_2:.*]]: tensor<?x8xf32>, %[[VAL_3:.*]]: tensor<?x8xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index)
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]]
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]]
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_2]]
// CHECK:           %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_3]]
// CHECK:           %[[VAL_8:.*]] = hivm.hir.vsel ins(%[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : tensor<?x8xi1>, tensor<?x8xf32>, tensor<?x8xf32>) outs(%[[VAL_7]] : tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           return %[[VAL_8]] : tensor<?x8xf32>
// CHECK:         }
func.func @bubble_up_hivm_ternary_2d_dyn2(%arg0: tensor<?x8xi1>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %0: tensor<?x8xf32>, %offset: index, %size: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vsel ins(%arg0, %arg1, %arg2: tensor<?x8xi1>, tensor<?x8xf32>, tensor<?x8xf32>) outs(%0: tensor<?x8xf32>) -> tensor<?x8xf32>
  %extracted_slice = tensor.extract_slice %1[%offset,0] [%size,8] [1,1] {to_be_bubbled_slice}: tensor<?x8xf32> to tensor<?x8xf32>
  return %extracted_slice : tensor<?x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_1d_static(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<5xf32>) -> tensor<1xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_0]] : tensor<5xf32>) outs(%[[VAL_1]] : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
// CHECK:           return %[[VAL_2]] : tensor<1xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_1d_static(%arg0: tensor<5xf32>) -> tensor<1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5xf32>) outs(%0 : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %51 : tensor<1xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_static_dim1_1(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<5x4xf32>) -> tensor<2x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [2, 4] [1, 1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<2x4xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x1xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_1]] : tensor<2x4xf32>) outs(%[[VAL_2]] : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
// CHECK:           return %[[VAL_3]] : tensor<2x1xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_static_dim1_1(%arg0: tensor<5x4xf32>) -> tensor<2x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5x1xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%0 : tensor<5x1xf32>) reduce_dims = [1] -> tensor<5x1xf32>
    %extracted_slice = tensor.extract_slice %51[0, 0] [2,1] [1,1] {to_be_bubbled_slice} : tensor<5x1xf32> to tensor<2x1xf32>
    return %extracted_slice : tensor<2x1xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_static_dim1_2(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<5x4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x1xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_1]] : tensor<2x4xf32>) outs(%[[VAL_2]] : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_3]] {{\[\[}}0, 1]] : tensor<2x1xf32> into tensor<2xf32>
// CHECK:           return %[[VAL_4]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_static_dim1_2(%arg0: tensor<5x4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5xf32>
    %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [5, 1] : tensor<5xf32> into tensor<5x1xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%expanded : tensor<5x1xf32>) reduce_dims = [1] -> tensor<5x1xf32>
    %collapsed = tensor.collapse_shape %51 [[0, 1]] : tensor<5x1xf32> into tensor<5xf32>

    %extracted_slice = tensor.extract_slice %collapsed[0] [2] [1] {to_be_bubbled_slice} : tensor<5xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_static_dim0(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<5x4xf32>) -> tensor<1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [5, 2] [1, 1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<5x2xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1x2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_1]] : tensor<5x2xf32>) outs(%[[VAL_2]] : tensor<1x2xf32>) reduce_dims = [0] -> tensor<1x2xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x2xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_static_dim0(%arg0: tensor<5x4xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%0 : tensor<1x4xf32>) reduce_dims = [0] -> tensor<1x4xf32>
    %extracted_slice = tensor.extract_slice %51[0, 0] [1,2] [1,1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x2xf32>
    return %extracted_slice : tensor<1x2xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_static_1(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1xf32>) -> tensor<5xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<5xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.vbrc ins(%[[VAL_0]] : tensor<1xf32>) outs(%[[VAL_1]] : tensor<5xf32>) broadcast_dims = [0] -> tensor<5xf32>
// CHECK:           return %[[VAL_2]] : tensor<5xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_static_1(%arg0: tensor<1xf32>) -> tensor<5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<5xf32>) broadcast_dims = [0] -> tensor<5xf32>
    return %35 : tensor<5xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_static_2(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x4xf32>) -> tensor<1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [1, 2] [1, 1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x2xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x2xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_static_2(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<1x4xf32>) broadcast_dims = [0] -> tensor<1x4xf32>
    %extracted_slice = tensor.extract_slice %35[0, 0] [1, 2] [1,1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x2xf32>
    return %extracted_slice : tensor<1x2xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_static_3(
// CHECK-SAME:                                    %arg0: tensor<1x1x1xf32>) -> tensor<1x2x3xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x2x3xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%arg0 : tensor<1x1x1xf32>) outs(%[[EMPTY]] : tensor<1x2x3xf32>) broadcast_dims = [1, 2] -> tensor<1x2x3xf32>
// CHECK:           return %[[VBRC]] : tensor<1x2x3xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_static_3(%arg0: tensor<1x1x1xf32>) -> tensor<1x2x3xf32> {
    %0 = tensor.empty() : tensor<2x3x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x1x1xf32>) outs(%0 : tensor<2x3x4xf32>) broadcast_dims = [0, 1, 2] -> tensor<2x3x4xf32>
    %extracted_slice = tensor.extract_slice %35[0,0,0] [1,2,3] [1,1,1] {to_be_bubbled_slice} : tensor<2x3x4xf32> to tensor<1x2x3xf32>
    return %extracted_slice : tensor<1x2x3xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_static_4(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x0x3xf32>) -> tensor<4x0x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0] [1, 0, 2] [1, 1, 1] {to_be_bubbled_slice} : tensor<1x0x3xf32> to tensor<1x0x2xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4x0x2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vbrc ins(%[[VAL_1]] : tensor<1x0x2xf32>) outs(%[[VAL_2]] : tensor<4x0x2xf32>) broadcast_dims = [0] -> tensor<4x0x2xf32>
// CHECK:           return %[[VAL_3]] : tensor<4x0x2xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_static_4(%arg0: tensor<1x0x3xf32>) -> tensor<4x0x2xf32> {
    %0 = tensor.empty() : tensor<4x0x3xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x0x3xf32>) outs(%0 : tensor<4x0x3xf32>) broadcast_dims = [0] -> tensor<4x0x3xf32>
    %extracted_slice = tensor.extract_slice %35[0,0,0] [4,0,2] [1,1,1] {to_be_bubbled_slice} : tensor<4x0x3xf32> to tensor<4x0x2xf32>
    return %extracted_slice : tensor<4x0x2xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_static_5(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<1x4xf32>) -> tensor<5x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vbrc ins(%[[VAL_1]] : tensor<1x2xf32>) outs(%[[VAL_2]] : tensor<5x2xf32>) broadcast_dims = [0] -> tensor<5x2xf32>
// CHECK:           return %[[VAL_3]] : tensor<5x2xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_static_5(%arg0: tensor<1x4xf32>) -> tensor<5x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<5x4xf32>) broadcast_dims = [0] -> tensor<5x4xf32>
    %extracted_slice = tensor.extract_slice %35[0, 0] [5, 2] [1,1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<5x2xf32>
    return %extracted_slice : tensor<5x2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_brcOTF_op_static(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<1x16x1x16xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<16x16x1x1xf32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: index) -> tensor<16x8x16x16xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, %[[VAL_2]], 0, 0] [1, 8, 1, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<1x16x1x16xf32> to tensor<1x8x1x16xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]][0, %[[VAL_2]], 0, 0] [16, 8, 1, 1] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x16x1x1xf32> to tensor<16x8x1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<16x8x16x16xf32>
// CHECK:           %[[VAL_6:.*]] = hivm.hir.vmul ins(%[[VAL_3]], %[[VAL_4]] : tensor<1x8x1x16xf32>, tensor<16x8x1x1xf32>) outs(%[[VAL_5]] : tensor<16x8x16x16xf32>) broadcast = [0, 2, 3] -> tensor<16x8x16x16xf32>
// CHECK:           return %[[VAL_6]] : tensor<16x8x16x16xf32>
// CHECK:         }
func.func @bubble_up_brcOTF_op_static(%arg0: tensor<1x16x1x16xf32>, %arg1: tensor<16x16x1x1xf32>, %57: index) ->tensor<16x8x16x16xf32> {
  %0 = tensor.empty() : tensor<16x16x16x16xf32>
  %1 = hivm.hir.vmul ins(%arg0, %arg1 : tensor<1x16x1x16xf32>, tensor<16x16x1x1xf32>) outs(%0 : tensor<16x16x16x16xf32>) broadcast = [0, 2, 3] -> tensor<16x16x16x16xf32>
  %extracted_slice_12 = tensor.extract_slice %1[0, %57, 0, 0] [16, 8, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x16x16x16xf32> to tensor<16x8x16x16xf32>
  return %extracted_slice_12 : tensor<16x8x16x16xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_dim0_dyn_1(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<5x5xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: index,
// CHECK-SAME:                                                %[[VAL_2:.*]]: index) -> tensor<1x?xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, %[[VAL_1]]] [5, %[[VAL_2]]] [1, 1] {to_be_bubbled_slice} : tensor<5x5xf32> to tensor<5x?xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty(%[[VAL_2]]) : tensor<1x?xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_3]] : tensor<5x?xf32>) outs(%[[VAL_4]] : tensor<1x?xf32>) reduce_dims = [0] -> tensor<1x?xf32>
// CHECK:           return %[[VAL_5]] : tensor<1x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_dim0_dyn_1(%arg0: tensor<5x5xf32>, %offset: index, %size: index) -> tensor<1x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x5xf32>
  %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x5xf32>) outs(%0 : tensor<1x5xf32>) reduce_dims = [0] -> tensor<1x5xf32>
  %extracted_slice = tensor.extract_slice %1[0, %offset] [1, %size] [1, 1] {to_be_bubbled_slice} : tensor<1x5xf32> to tensor<1x?xf32>
  return %extracted_slice : tensor<1x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_dim0_dyn_2(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<5x?xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<1x?xf32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: index,
// CHECK-SAME:                                                %[[VAL_3:.*]]: index) -> tensor<1x?xf32> {
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0, %[[VAL_2]]] [5, %[[VAL_3]]] [1, 1] {to_be_bubbled_slice} : tensor<5x?xf32> to tensor<5x?xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]][0, %[[VAL_2]]] [1, %[[VAL_3]]] [1, 1] {to_be_bubbled_slice} : tensor<1x?xf32> to tensor<1x?xf32>
// CHECK:           %[[VAL_6:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_4]] : tensor<5x?xf32>) outs(%[[VAL_5]] : tensor<1x?xf32>) reduce_dims = [0] -> tensor<1x?xf32>
// CHECK:           return %[[VAL_6]] : tensor<1x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_dim0_dyn_2(%arg0: tensor<5x?xf32>, %0 : tensor<1x?xf32>, %offset: index, %size: index) -> tensor<1x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x?xf32>) outs(%0 : tensor<1x?xf32>) reduce_dims = [0] -> tensor<1x?xf32>
  %extracted_slice = tensor.extract_slice %1[0, %offset] [1, %size] [1, 1] {to_be_bubbled_slice} : tensor<1x?xf32> to tensor<1x?xf32>
  return %extracted_slice : tensor<1x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_dim1_dyn_1(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<5x4xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: index,
// CHECK-SAME:                                                %[[VAL_2:.*]]: index) -> tensor<?x1xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][%[[VAL_1]], 0] [%[[VAL_2]], 4] [1, 1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<?x4xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty(%[[VAL_2]]) : tensor<?x1xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_3]] : tensor<?x4xf32>) outs(%[[VAL_4]] : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
// CHECK:           return %[[VAL_5]] : tensor<?x1xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_dim1_dyn_1(%arg0: tensor<5x4xf32>, %offset: index, %size: index) -> tensor<?x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%0 : tensor<5x1xf32>) reduce_dims = [1] -> tensor<5x1xf32>
  %extracted_slice = tensor.extract_slice %1[%offset, 0] [%size, 1] [1, 1] {to_be_bubbled_slice} : tensor<5x1xf32> to tensor<?x1xf32>
  return %extracted_slice : tensor<?x1xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_reduce_2d_dim1_dyn_2(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<?x4xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<?x1xf32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: index,
// CHECK-SAME:                                                %[[VAL_3:.*]]: index) -> tensor<?x1xf32> {
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][%[[VAL_2]], 0] [%[[VAL_3]], 4] [1, 1] {to_be_bubbled_slice} : tensor<?x4xf32> to tensor<?x4xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]][%[[VAL_2]], 0] [%[[VAL_3]], 1] [1, 1] {to_be_bubbled_slice} : tensor<?x1xf32> to tensor<?x1xf32>
// CHECK:           %[[VAL_6:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_4]] : tensor<?x4xf32>) outs(%[[VAL_5]] : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
// CHECK:           return %[[VAL_6]] : tensor<?x1xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce_2d_dim1_dyn_2(%arg0: tensor<?x4xf32>, %0 : tensor<?x1xf32>, %offset: index, %size: index) -> tensor<?x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<?x4xf32>) outs(%0 : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
  %extracted_slice = tensor.extract_slice %1[%offset, 0] [%size, 1] [1, 1] {to_be_bubbled_slice} : tensor<?x1xf32> to tensor<?x1xf32>
  return %extracted_slice : tensor<?x1xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_dyn_0(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x4xf32>,
// CHECK-SAME:                                       %[[OFFSET0:.*]]: index, %[[OFFSET1:.*]]: index, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index)
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty(%[[SIZE0]], %[[SIZE1]]) : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vbrc ins(%[[VAL_1]] : tensor<1x?xf32>) outs(%[[VAL_2]] : tensor<?x?xf32>) broadcast_dims = [0] -> tensor<?x?xf32>
// CHECK:           return %[[VAL_3]] : tensor<?x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_dyn_0(%arg0: tensor<1x4xf32>, %offset0: index, %offset1: index, %size0: index, %size1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x4xf32>
  %1 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<5x4xf32>) broadcast_dims = [0] -> tensor<5x4xf32>
  %extracted_slice = tensor.extract_slice %1[%offset0, %offset1] [%size0, %size1] [1,1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<?x?xf32>
  return %extracted_slice : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_dyn_1(
// CHECK-SAME:                                          %arg0: tensor<1x4xf32>, %arg1: tensor<?x4xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
// CHECK:           %[[SLICE_IN:.*]] = tensor.extract_slice %arg0[0, %arg3] [1, %arg5] [1, 1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x?xf32>
// CHECK:           %[[SLICE_OUT:.*]] = tensor.extract_slice %arg1[%arg2, %arg3] [%arg4, %arg5] [1, 1] {to_be_bubbled_slice} : tensor<?x4xf32> to tensor<?x?xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[SLICE_IN]] : tensor<1x?xf32>) outs(%[[SLICE_OUT]] : tensor<?x?xf32>) broadcast_dims = [0] -> tensor<?x?xf32>
// CHECK:           return %[[VBRC]] : tensor<?x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_dyn_1(%arg0: tensor<1x4xf32>,%0 : tensor<?x4xf32>, %offset0: index, %offset1: index, %size0: index, %size1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  
  %1 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<?x4xf32>) broadcast_dims = [0] -> tensor<?x4xf32>
  %extracted_slice = tensor.extract_slice %1[%offset0, %offset1] [%size0, %size1] [1,1] {to_be_bubbled_slice} : tensor<?x4xf32> to tensor<?x?xf32>
  return %extracted_slice : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_dyn_2(
// CHECK-SAME:                                          %arg0: tensor<1xf32>, %arg1: index, %arg2: index) -> tensor<?xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%arg2) : tensor<?xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%arg0 : tensor<1xf32>) outs(%[[EMPTY]] : tensor<?xf32>) broadcast_dims = [0] -> tensor<?xf32>
// CHECK:           return %[[VBRC]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_dyn_2(%arg0: tensor<1xf32>,%offset0: index, %size0: index) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<5xf32>) broadcast_dims = [0] -> tensor<5xf32>
    %extracted_slice = tensor.extract_slice %35[%offset0] [%size0] [1] {to_be_bubbled_slice} : tensor<5xf32> to tensor<?xf32>
    return %extracted_slice : tensor<?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_dyn_3(
// CHECK-SAME:                                          %arg0: tensor<1xf32>, %arg1: tensor<?xf32>, %arg2: index, %arg3: index) -> tensor<?xf32> {
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %arg1[%arg2] [%arg3] [1] {to_be_bubbled_slice} : tensor<?xf32> to tensor<?xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%arg0 : tensor<1xf32>) outs(%[[SLICE]] : tensor<?xf32>) broadcast_dims = [0] -> tensor<?xf32>
// CHECK:           return %[[VBRC]] : tensor<?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_dyn_3(%arg0: tensor<1xf32>,%0 : tensor<?xf32>, %offset0: index, %size0: index) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<?xf32>) broadcast_dims = [0] -> tensor<?xf32>
    %extracted_slice = tensor.extract_slice %35[%offset0] [%size0] [1] {to_be_bubbled_slice} : tensor<?xf32> to tensor<?xf32>
    return %extracted_slice : tensor<?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_dyn_4(
// CHECK-SAME:                                    %arg0: tensor<1x4xf32>, %arg1: index, %arg2: index) -> tensor<1x?xf32> {
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %arg0[0, %arg1] [1, %arg2] [1, 1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x?xf32>
// CHECK:           return %[[SLICE]] : tensor<1x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_dyn_4(%arg0: tensor<1x4xf32>, %offset0: index, %size0: index) -> tensor<1x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<1x4xf32>) broadcast_dims = [0] -> tensor<1x4xf32>
    %extracted_slice = tensor.extract_slice %35[0, %offset0] [1, %size0] [1,1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x?xf32>
    return %extracted_slice : tensor<1x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_dyn_5(
// CHECK-SAME:                                    %arg0: tensor<1x1x1xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<?x?x?xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%arg4, %arg5, %arg6) : tensor<?x?x?xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%arg0 : tensor<1x1x1xf32>) outs(%[[EMPTY]] : tensor<?x?x?xf32>) broadcast_dims = [0, 1, 2] -> tensor<?x?x?xf32>
// CHECK:           return %[[VBRC]] : tensor<?x?x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_dyn_5(%arg0: tensor<1x1x1xf32>,%offset0: index, %offset1: index,%offset2: index, %size0: index, %size1: index, %size2: index) -> tensor<?x?x?xf32> {
    %0 = tensor.empty() : tensor<2x3x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x1x1xf32>) outs(%0 : tensor<2x3x4xf32>) broadcast_dims = [0, 1, 2] -> tensor<2x3x4xf32>
    %extracted_slice = tensor.extract_slice %35[%offset0,%offset1,%offset2] [%size0,%size1,%size2] [1,1,1] {to_be_bubbled_slice} : tensor<2x3x4xf32> to tensor<?x?x?xf32>
    return %extracted_slice : tensor<?x?x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_6(
// CHECK-SAME:                                    %arg0: tensor<1x0x3xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x0x?xf32> {
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %arg0[0, 0, %arg2] [1, 0, %arg4] [1, 1, 1] {to_be_bubbled_slice} : tensor<1x0x3xf32> to tensor<1x0x?xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%arg3, %arg4) : tensor<?x0x?xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[SLICE]] : tensor<1x0x?xf32>) outs(%[[EMPTY]] : tensor<?x0x?xf32>) broadcast_dims = [0] -> tensor<?x0x?xf32>
// CHECK:           return %[[VBRC]] : tensor<?x0x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_6(%arg0: tensor<1x0x3xf32>, %offset0: index,%offset1: index, %size0: index, %size1: index) -> tensor<?x0x?xf32> {
    %0 = tensor.empty() : tensor<4x0x3xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x0x3xf32>) outs(%0 : tensor<4x0x3xf32>) broadcast_dims = [0] -> tensor<4x0x3xf32>
    %extracted_slice = tensor.extract_slice %35[%offset0,0, %offset1][%size0,0,%size1] [1,1,1] {to_be_bubbled_slice} : tensor<4x0x3xf32> to tensor<?x0x?xf32>
    return %extracted_slice : tensor<?x0x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc_7(
// CHECK-SAME:                                    %arg0: tensor<1x4xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?xf32> {
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %arg0[0, %arg2] [1, %arg4] [1, 1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x?xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%arg3, %arg4) : tensor<?x?xf32>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[SLICE]] : tensor<1x?xf32>) outs(%[[EMPTY]] : tensor<?x?xf32>) broadcast_dims = [0] -> tensor<?x?xf32>
// CHECK:           return %[[VBRC]] : tensor<?x?xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc_7(%arg0: tensor<1x4xf32>, %offset0: index,%offset1: index, %size0: index, %size1: index) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<5x4xf32>) broadcast_dims = [0] -> tensor<5x4xf32>
    %extracted_slice = tensor.extract_slice %35[%offset0, %offset1] [%size0,%size1] [1,1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<?x?xf32>
    return %extracted_slice : tensor<?x?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_brcOTF_op(
// CHECK-SAME:                              %arg0: tensor<1x16x1x16xf32>, %arg1: tensor<16x16x1x1xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[SLICE0:.*]] = tensor.extract_slice %arg0[0, %arg3, 0, %arg5] [1, %arg7, 1, %arg9] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<1x16x1x16xf32> to tensor<1x?x1x?xf32>
// CHECK:           %[[SLICE1:.*]] = tensor.extract_slice %arg1[%arg2, %arg3, 0, 0] [%arg6, %arg7, 1, 1] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x16x1x1xf32> to tensor<?x?x1x1xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%arg6, %arg7, %arg8, %arg9) : tensor<?x?x?x?xf32>
// CHECK:           %[[VMUL:.*]] = hivm.hir.vmul ins(%[[SLICE0]], %[[SLICE1]] : tensor<1x?x1x?xf32>, tensor<?x?x1x1xf32>) outs(%[[EMPTY]] : tensor<?x?x?x?xf32>) broadcast = [0, 2, 3] -> tensor<?x?x?x?xf32>
// CHECK:           return %[[VMUL]] : tensor<?x?x?x?xf32>
// CHECK:         }
func.func @bubble_up_brcOTF_op(
     %arg0: tensor<1x16x1x16xf32>,%arg1: tensor<16x16x1x1xf32>, %offset0: index,%offset1: index,%offset2: index,%offset3: index, %size0: index, %size1: index, %size2: index, %size3: index) ->tensor<?x?x?x?xf32> {
  %0 = tensor.empty() : tensor<16x16x16x16xf32>
  %1 = hivm.hir.vmul ins(%arg0, %arg1 : tensor<1x16x1x16xf32>, tensor<16x16x1x1xf32>) outs(%0 : tensor<16x16x16x16xf32>) broadcast = [0, 2, 3] -> tensor<16x16x16x16xf32>
  %extracted_slice_12 = tensor.extract_slice %1[%offset0, %offset1,%offset2, %offset3] [%size0,%size1, %size2,%size3] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x16x16x16xf32> to tensor<?x?x?x?xf32>
  return %extracted_slice_12 : tensor<?x?x?x?xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_collapse_shape(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<64x1xf32>) -> tensor<32xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1]] : tensor<32x1xf32> into tensor<32xf32>
// CHECK:           return %[[VAL_2]] : tensor<32xf32>
// CHECK:         }
func.func @bubble_up_collapse_shape(%arg0: tensor<64x1xf32>) -> tensor<32xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
  %extracted_slice_10 = tensor.extract_slice %collapsed[0] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
  return %extracted_slice_10 : tensor<32xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_expand_shape(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<64xf32>) -> tensor<32x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x1xf32>
// CHECK:         }
func.func @bubble_up_expand_shape(%arg0: tensor<64xf32>) -> tensor<32x1xf32> {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %extracted_slice_10 = tensor.extract_slice %expanded[0, 0] [32, 1] [1, 1] {to_be_bubbled_slice} : tensor<64x1xf32> to tensor<32x1xf32>
    return %extracted_slice_10 : tensor<32x1xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_for_loop3(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<64x32xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<32x16xf32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: tensor<64x16xf32>) -> (tensor<32x32xf32>, index) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_7:.*]]:2 = scf.for %[[VAL_8:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]], %[[VAL_10:.*]] = %[[VAL_3]]) -> (tensor<32x32xf32>, index) {
// CHECK-DAG:             %[[VAL_12:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK-DAG:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_4]] : index
// CHECK-DAG:             %[[VAL_13:.*]] = hivm.hir.vln ins(%[[VAL_9]] : tensor<32x32xf32>) outs(%[[VAL_12]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:             scf.yield %[[VAL_13]], %[[VAL_11]] : tensor<32x32xf32>, index
// CHECK:           } {to_be_tiled_op}
// CHECK:           return %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<32x32xf32>, index
// CHECK:         }
func.func @bubble_up_for_loop3(%arg0: tensor<64x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<64x16xf32>) -> (tensor<32x32xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_counter = arith.constant 0 : index
  %result:2 = scf.for %i = %c0 to %c10 step %c1
    iter_args(%temp = %arg0, %counter = %init_counter) -> (tensor<64x32xf32>, index) {
        %expanded = tensor.empty() : tensor<64x32xf32>
    %29 = hivm.hir.vln ins(%temp: tensor<64x32xf32>) outs(%expanded : tensor<64x32xf32>) -> tensor<64x32xf32>
    %new_counter = arith.addi %counter, %c1 : index
    scf.yield %29, %new_counter : tensor<64x32xf32>, index
  } {to_be_tiled_op}
  %extracted_slice_10 = tensor.extract_slice %result#0[0, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xf32> to tensor<32x32xf32>
  return  %extracted_slice_10  , %result#1 : tensor<32x32xf32>, index
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @bubble_up_tiling_loop(
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]] {
// CHECK:             %[[VAL_7:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]]]
// CHECK:             %[[VAL_8:.*]] = tensor.extract_slice %{{.*}}{{\[}}%[[VAL_7]]] [256] [1] {to_be_bubbled_slice} : tensor<512xf32> to tensor<256xf32>
// CHECK:             %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (tensor<256xf32>) {
// CHECK:               %[[VAL_12:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_13:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_12]]] [32] [1] : tensor<256xf32> to tensor<32xf32>
// CHECK:               %[[VAL_14:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:               %[[VAL_15:.*]] = hivm.hir.vln ins(%[[VAL_13]] : tensor<32xf32>) outs(%[[VAL_14]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:               %[[VAL_16:.*]] = tensor.insert_slice %[[VAL_15]] into %[[VAL_11]]{{\[}}%[[VAL_12]]] [32] [1] : tensor<32xf32> into tensor<256xf32>
// CHECK:               scf.yield %[[VAL_16]] : tensor<256xf32>
// CHECK:             }
// CHECK:             hivm.hir.store ins(%[[VAL_9]] : tensor<256xf32>) outs(%{{.*}} : memref<256xf32>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @bubble_up_tiling_loop(%arg0: tensor<512xf32>, %arg1: memref<256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %0 = tensor.empty() : tensor<64xf32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %1 = affine.apply #map()[%arg2]
      %2 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %arg0) -> (tensor<512xf32>) {
        %3 = affine.apply #map1()[%arg3]
        %extracted_slice_0 = tensor.extract_slice %arg4[%3] [64] [1] : tensor<512xf32> to tensor<64xf32>
        %4 = hivm.hir.vln ins(%extracted_slice_0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%3] [64] [1] : tensor<64xf32> into tensor<512xf32>
        scf.yield %inserted_slice : tensor<512xf32>
      }
      %extracted_slice = tensor.extract_slice %2[%1] [256] [1] {to_be_bubbled_slice} : tensor<512xf32> to tensor<256xf32>
      hivm.hir.store ins(%extracted_slice : tensor<256xf32>) outs(%arg1 : memref<256xf32>) {tiled_op}
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}

// -----
// CHECK-NOT: bubble_up_slice_non_hivm
#map1 = affine_map<()[s0] -> (s0 * 2)>
func.func @bubble_up_slice_non_hivm(%arg0: tensor<4xf32>, %arg3: memref<2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() :  tensor<4xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %10 = affine.apply #map1()[%arg2]
      %29 = math.sqrt %arg0: tensor<4xf32>
      %extracted_slice = tensor.extract_slice %29[%10] [2] [1] {to_be_bubbled_slice} : tensor<4xf32> to   tensor<2xf32>
      hivm.hir.store ins(%extracted_slice : tensor<2xf32>) outs(%arg3 : memref<2xf32>)
      scf.yield
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
}
