// RUN: bishengir-opt %s -transform-interpreter | FileCheck %s

// Copied from upstream LLVM:
// mlir/test/Dialect/Tensor/bubble-up-extract-slice-op.mlir (expand_shape cases).
// Licensed under Apache-2.0 WITH LLVM-exception.
// The Transform op and offset checks are adapted to this MLIR revision,
// which uses affine.apply instead of affine.linearize_index.
// TODO: Move these tests to our LLVM fork together with the pattern.

// CHECK-DAG: #[[$OFFSET_MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 30 + s1 * 10 + 5)>
// CHECK-DAG: #[[$SIZE_MAP:.*]] = affine_map<()[s0] -> (s0 * 10)>

module attributes {transform.with_named_sequence} {

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_expand_shape(
// CHECK-SAME:                %[[SRC:.*]]: tensor<60xf32>) -> tensor<1x1x5xf32> {
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][5] [5] [1] : tensor<60xf32> to tensor<5xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 5] : tensor<5xf32> into tensor<1x1x5xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x1x5xf32>

func.func @bubble_up_extract_slice_through_expand_shape(%src: tensor<60xf32>) -> tensor<1x1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5][1, 1, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
  return %extract : tensor<1x1x5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_on_non_contiguous(
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:           return %[[EXTRACT]]

func.func @no_bubble_up_extract_slice_on_non_contiguous(%src: tensor<60xf32>) -> tensor<1x2x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5][1, 2, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x2x5xf32>
  return %extract : tensor<1x2x5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_on_stride(
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:           return %[[EXTRACT]]

func.func @no_bubble_up_extract_slice_on_stride(%src: tensor<60xf32>) -> tensor<1x1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0][1, 1, 5][1, 1, 2] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
  return %extract : tensor<1x1x5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_on_rank_reducing(
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:           return %[[EXTRACT]]

func.func @no_bubble_up_extract_slice_on_rank_reducing(%src: tensor<60xf32>) -> tensor<1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5][1, 1, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x5xf32>
  return %extract : tensor<1x5xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_expand_shape_multiple_expanded_dims(
// CHECK-SAME:                  %[[SRC:.*]]: tensor<120x56xf32>) -> tensor<1x2x10x1x4xf32> {
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][0, 0] [20, 4] [1, 1] : tensor<120x56xf32> to tensor<20x4xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2], [3, 4]] output_shape [1, 2, 10, 1, 4] : tensor<20x4xf32> into tensor<1x2x10x1x4xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x2x10x1x4xf32>

func.func @bubble_up_extract_slice_through_expand_shape_multiple_expanded_dims(%src: tensor<120x56xf32>) -> tensor<1x2x10x1x4xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [3, 4, 10, 7, 8] : tensor<120x56xf32> into tensor<3x4x10x7x8xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0, 0, 0][1, 2, 10, 1, 4][1, 1, 1, 1, 1] : tensor<3x4x10x7x8xf32> to tensor<1x2x10x1x4xf32>
  return %extract : tensor<1x2x10x1x4xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_with_trailing_full_dims(
// CHECK-SAME:                %[[SRC:.*]]: tensor<60xf32>) -> tensor<2x5x2xf32> {
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][0] [20] [1] : tensor<60xf32> to tensor<20xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape [2, 5, 2] : tensor<20xf32> into tensor<2x5x2xf32>
// CHECK:           return %[[EXPAND]] : tensor<2x5x2xf32>
func.func @bubble_up_extract_slice_with_trailing_full_dims(%src: tensor<60xf32>) -> tensor<2x5x2xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [6, 5, 2] : tensor<60xf32> into tensor<6x5x2xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0][2, 5, 2][1, 1, 1] : tensor<6x5x2xf32> to tensor<2x5x2xf32>
  return %extract : tensor<2x5x2xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_dont_fold_linearize_index(
// CHECK-SAME:                 %[[SRC:.*]]: tensor<60xf32>,
// CHECK-SAME:                 %[[OFFSET_0:.*]]: index,
// CHECK-SAME:                 %[[OFFSET_1:.*]]: index) -> tensor<1x1x5xf32> {
// CHECK:           %[[LINEARIZE:.*]] = affine.apply #[[$OFFSET_MAP]]()[%[[OFFSET_0]], %[[OFFSET_1]]]
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[LINEARIZE]]] [5] [1] : tensor<60xf32> to tensor<5xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 5] : tensor<5xf32> into tensor<1x1x5xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x1x5xf32>
func.func @bubble_up_extract_slice_dont_fold_linearize_index(%src: tensor<60xf32>, %offset_0 : index, %offset_1 : index) -> tensor<1x1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[%offset_0, %offset_1, 5][1, 1, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
  return %extract : tensor<1x1x5xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_not_all_dims_expanded(
// CHECK-SAME:                %[[SRC:.*]]: tensor<60x12xf32>) -> tensor<1x1x5x12xf32> {
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][5, 0] [5, 12] [1, 1] : tensor<60x12xf32> to tensor<5x12xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2], [3]] output_shape [1, 1, 5, 12] : tensor<5x12xf32> into tensor<1x1x5x12xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x1x5x12xf32>
func.func @bubble_up_extract_slice_not_all_dims_expanded(%src: tensor<60x12xf32>) -> tensor<1x1x5x12xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2], [3]] output_shape [2, 3, 10, 12] : tensor<60x12xf32> into tensor<2x3x10x12xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5, 0][1, 1, 5, 12][1, 1, 1, 1] : tensor<2x3x10x12xf32> to tensor<1x1x5x12xf32>
  return %extract : tensor<1x1x5x12xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_affine_apply_not_folded(
// CHECK-SAME:                   %[[SRC:.*]]: tensor<60xf32>,
// CHECK-SAME:                   %[[SLICE_SIZE:.*]]: index) -> tensor<?x5x2xf32> {
// CHECK:           %[[AFFINE_APPLY:.*]] = affine.apply #[[$SIZE_MAP]](){{\[}}%[[SLICE_SIZE]]]
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][0] {{\[}}%[[AFFINE_APPLY]]] [1] : tensor<60xf32> to tensor<?xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape {{\[}}%[[SLICE_SIZE]], 5, 2] : tensor<?xf32> into tensor<?x5x2xf32>
// CHECK:           return %[[EXPAND]] : tensor<?x5x2xf32>
func.func @bubble_up_extract_slice_affine_apply_not_folded(%src: tensor<60xf32>, %slice_size : index) -> tensor<?x5x2xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [6, 5, 2] : tensor<60xf32> into tensor<6x5x2xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0][%slice_size, 5, 2][1, 1, 1] : tensor<6x5x2xf32> to tensor<?x5x2xf32>
  return %extract : tensor<?x5x2xf32>
}

// Upstream LLVM bug: affine folding makes the collapsed size static while the
// expanded result remains dynamic. This test checks the compatibility fix.
// CHECK-LABEL:   func.func @bubble_up_extract_slice_ssa_constant_dynamic_result(
// CHECK:           %[[EXTRACT_DYNAMIC:.*]] = tensor.extract_slice %{{.*}}[0] [20] [1] : tensor<60xf32> to tensor<20xf32>
// CHECK:           %[[EXPAND_DYNAMIC:.*]] = tensor.expand_shape %[[EXTRACT_DYNAMIC]] {{\[\[}}0, 1]] output_shape [2, 10] : tensor<20xf32> into tensor<2x10xf32>
// CHECK:           %[[CAST_DYNAMIC:.*]] = tensor.cast %[[EXPAND_DYNAMIC]] : tensor<2x10xf32> to tensor<?x10xf32>
// CHECK:           return %[[CAST_DYNAMIC]] : tensor<?x10xf32>
func.func @bubble_up_extract_slice_ssa_constant_dynamic_result(%src: tensor<60xf32>) -> tensor<?x10xf32> {
  %c2 = arith.constant 2 : index
  %expand = tensor.expand_shape %src [[0, 1]] output_shape [6, 10] : tensor<60xf32> into tensor<6x10xf32>
  %extract = tensor.extract_slice %expand[0, 0] [%c2, 10] [1, 1] : tensor<6x10xf32> to tensor<?x10xf32>
  return %extract : tensor<?x10xf32>
}

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.hfusion.forward_extract_slice
    } : !transform.any_op
    transform.yield
  }
}
