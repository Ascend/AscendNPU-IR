// RUN: bishengir-opt %s -propagate-reshape="for-regbased=false" -split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: bishengir-opt %s -propagate-reshape="for-regbased=true" -split-input-file | FileCheck %s --check-prefix=REGBASE

// DEFAULT-LABEL: func.func @scalar_reduce_unit_expand(
// DEFAULT: %[[EXPANDED_INIT:.*]] = tensor.expand_shape %arg1 [] output_shape [1]
// DEFAULT: %[[EXPANDED_SRC:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 4]
// DEFAULT: linalg.reduce ins(%[[EXPANDED_SRC]] : tensor<1x4xi32>) outs(%[[EXPANDED_INIT]] : tensor<1xi32>) dimensions = [1]
// DEFAULT-NOT: tensor.expand_shape
// REGBASE-LABEL: func.func @scalar_reduce_unit_expand(
// REGBASE: %[[REDUCED:.*]] = linalg.reduce ins(%arg0 : tensor<4xi32>) outs(%arg1 : tensor<i32>) dimensions = [0]
// REGBASE: tensor.expand_shape %[[REDUCED]] [] output_shape [1]
func.func @scalar_reduce_unit_expand(
    %src: tensor<4xi32>, %init: tensor<i32>) -> tensor<1xi32> {
  %reduced = linalg.reduce
      ins(%src : tensor<4xi32>) outs(%init : tensor<i32>) dimensions = [0]
      (%in: i32, %acc: i32) {
        %sum = arith.addi %in, %acc : i32
        linalg.yield %sum : i32
      }
  %expanded = tensor.expand_shape %reduced [] output_shape [1] :
      tensor<i32> into tensor<1xi32>
  %out = tensor.empty() : tensor<1xi32>
  %copied = linalg.copy ins(%expanded : tensor<1xi32>)
      outs(%out : tensor<1xi32>) -> tensor<1xi32>
  return %copied : tensor<1xi32>
}

// -----

// DEFAULT-LABEL: func.func @extract_slice_unit_expand(
// DEFAULT: %[[EXPANDED_SRC:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [2, 1, 4]
// DEFAULT: tensor.extract_slice %[[EXPANDED_SRC]][0, 0, 0] [1, 1, 4] [1, 1, 1]
// REGBASE-LABEL: func.func @extract_slice_unit_expand(
// REGBASE: %[[SLICE:.*]] = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1]
// REGBASE: tensor.expand_shape %[[SLICE]] {{\[\[}}0], [1, 2]] output_shape [1, 1, 4]
func.func @extract_slice_unit_expand(
    %src: tensor<2x4xi32>) -> tensor<1x1x4xi32> {
  %slice = tensor.extract_slice %src[0, 0] [1, 4] [1, 1] :
      tensor<2x4xi32> to tensor<1x4xi32>
  %expanded = tensor.expand_shape %slice [[0], [1, 2]]
      output_shape [1, 1, 4] :
      tensor<1x4xi32> into tensor<1x1x4xi32>
  %out = tensor.empty() : tensor<1x1x4xi32>
  %copied = linalg.copy ins(%expanded : tensor<1x1x4xi32>)
      outs(%out : tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
  return %copied : tensor<1x1x4xi32>
}

// -----

// DEFAULT-LABEL: func.func @collapse_unit_insert_slice(
// DEFAULT: %[[EXPANDED_DEST:.*]] = tensor.expand_shape %arg2 {{\[\[}}0, 1]] output_shape [4, 1]
// DEFAULT: %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into %[[EXPANDED_DEST]][1, 0] [2, 1] [1, 1]
// DEFAULT: tensor.collapse_shape %[[INSERTED]] {{\[\[}}0, 1]]
// REGBASE-LABEL: func.func @collapse_unit_insert_slice(
// REGBASE: %[[COLLAPSED:.*]] = tensor.collapse_shape {{.*}} {{\[\[}}0, 1]]
// REGBASE: tensor.insert_slice %[[COLLAPSED]] into %arg2[1] [2] [1]
func.func @collapse_unit_insert_slice(
    %src: tensor<2x1xi32>, %init: tensor<2x1xi32>,
    %dest: tensor<4xi32>) -> tensor<4xi32> {
  %copied = linalg.copy ins(%src : tensor<2x1xi32>)
      outs(%init : tensor<2x1xi32>) -> tensor<2x1xi32>
  %collapsed = tensor.collapse_shape %copied [[0, 1]] :
      tensor<2x1xi32> into tensor<2xi32>
  %inserted = tensor.insert_slice %collapsed into %dest[1] [2] [1] :
      tensor<2xi32> into tensor<4xi32>
  return %inserted : tensor<4xi32>
}
