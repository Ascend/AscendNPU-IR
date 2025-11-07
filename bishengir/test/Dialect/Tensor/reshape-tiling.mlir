// RUN: bishengir-opt %s -transform-interpreter -canonicalize-ext -cse -split-input-file | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:3 = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [1, 1, 256]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @expand_shape_elementwise_tile_1x1xN(%arg0: tensor<131072xf32>) -> tensor<8x16x1024xf32> {
  %empty= tensor.empty() : tensor<8x16x1024xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [8, 16, 1024] : tensor<131072xf32> into tensor<8x16x1024xf32>
  %0 = linalg.elemwise_unary ins(%expanded : tensor<8x16x1024xf32>)
                             outs(%empty: tensor<8x16x1024xf32>) -> tensor<8x16x1024xf32>
  return %0 : tensor<8x16x1024xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 1024 + d2)>
// CHECK: func.func @expand_shape_elementwise_tile_1x1xN
//   CHECK-SAME: (%[[ARG0:.+]]: tensor<131072xf32>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: scf.for %[[ARG1:.+]] = %[[C0]]
// CHECK:   scf.for %[[ARG2:.+]] = %[[C0]]
// CHECK:     scf.for %[[ARG3:.+]] = %[[C0]]
// CHECK:       %[[OFFSET:.+]] = affine.apply #[[MAP]](%[[ARG1]], %[[ARG2]], %[[ARG3]])
// CHECK:       tensor.extract_slice %[[ARG0]][%[[OFFSET]]] [256] [1]
// CHECK:       tensor.expand_shape {{.*}} output_shape [1, 1, 256]
// CHECK:       linalg.elemwise_unary
// CHECK:       tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield
// CHECK: scf.yield

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__root__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [8, 4]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @elementwise_collapse_shape(%arg0: tensor<128x1x8x2xf32>) ->tensor<128x16xf32> {
  %empty= tensor.empty() : tensor<128x1x8x2xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<128x1x8x2xf32>)
                             outs(%empty: tensor<128x1x8x2xf32>) -> tensor<128x1x8x2xf32>
  %collapsed = tensor.collapse_shape %0 [[0], [1, 2, 3]] {__root__} : tensor<128x1x8x2xf32> into tensor<128x16xf32>
  return %collapsed : tensor<128x16xf32>
}

// CHECK-LABEL: func.func @elementwise_collapse_shape
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.collapse_shape
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [8, 4]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @collapse_shape_elementwise(%arg0: tensor<128x1x8x2xf32>) ->tensor<128x16xf32> {
  %empty= tensor.empty() : tensor<128x16xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<128x1x8x2xf32> into tensor<128x16xf32>
  %0 = linalg.elemwise_unary ins(%collapsed : tensor<128x16xf32>)
                             outs(%empty: tensor<128x16xf32>) -> tensor<128x16xf32>
  return %0 : tensor<128x16xf32>
}

// CHECK-LABEL: func.func @collapse_shape_elementwise
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.collapse_shape
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [256]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @collapse_shape_elementwise_slice_1x1xN(%arg0: tensor<8x16x1024xf32>) -> tensor<131072xf32> {
  %empty= tensor.empty() : tensor<131072xf32>
  %expanded = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<8x16x1024xf32> into tensor<131072xf32>
  %0 = linalg.elemwise_unary ins(%expanded : tensor<131072xf32>)
                             outs(%empty: tensor<131072xf32>) -> tensor<131072xf32>
  return %0 : tensor<131072xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> (((d0 floordiv 1024) floordiv 16) mod 8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> ((d0 floordiv 1024) mod 16)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (d0 mod 1024)>
// CHECK: func.func @collapse_shape_elementwise_slice_1x1xN
//   CHECK-SAME: (%[[ARG0:.+]]: tensor<8x16x1024xf32>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: scf.for %[[ARG1:.+]] = %[[C0]]
// CHECK-DAG: %[[OFFSET0:.+]] = affine.apply #[[MAP0]](%[[ARG1]])
// CHECK-DAG: %[[OFFSET1:.+]] = affine.apply #[[MAP1]](%[[ARG1]])
// CHECK-DAG: %[[OFFSET2:.+]] = affine.apply #[[MAP2]](%[[ARG1]])
// CHECK:   tensor.extract_slice %[[ARG0]][%[[OFFSET0]], %[[OFFSET1]], %[[OFFSET2]]] [1, 1, 256] [1, 1, 1]
// CHECK:   linalg.elemwise_unary
// CHECK:   tensor.insert_slice
// CHECK: scf.yield


// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__root__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [64, 8]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @elementwise_expand_shape(%arg0: tensor<128x16xf32>) -> tensor<128x16x1xf32> {
  %empty= tensor.empty() : tensor<128x16xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<128x16xf32>)
                             outs(%empty: tensor<128x16xf32>) -> tensor<128x16xf32>
  %expanded = tensor.expand_shape %0 [[0], [1, 2]]  output_shape [128, 16, 1] {__root__} : tensor<128x16xf32> into tensor<128x16x1xf32>
  return %expanded : tensor<128x16x1xf32>
}

// CHECK-LABEL:   func.func @elementwise_expand_shape(
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               tensor.extract_slice
// CHECK:               tensor.extract_slice
// CHECK:               linalg.elemwise_unary
// CHECK:               tensor.expand_shape
// CHECK:               %[[VAL_18:.*]] = tensor.insert_slice
// CHECK:               scf.yield %[[VAL_18]] : tensor<128x16x1xf32>
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__root__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [128, 16]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @elementwise_expand_shape_full_slice(%arg0: tensor<128x16xf32>) -> tensor<128x16x1xf32> {
  %empty= tensor.empty() : tensor<128x16xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<128x16xf32>)
                             outs(%empty: tensor<128x16xf32>) -> tensor<128x16xf32>
  %expanded = tensor.expand_shape %0 [[0], [1, 2]]  output_shape [128, 16, 1] {__root__} : tensor<128x16xf32> into tensor<128x16x1xf32>
  return %expanded : tensor<128x16x1xf32>
}

// CHECK-LABEL:   func.func @elementwise_expand_shape_full_slice(
// CHECK:           tensor.empty() : tensor<128x16xf32>
// CHECK:           linalg.elemwise_unary
// CHECK:           tensor.expand_shape 
// CHECK:           return 

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:1 = transform.structured.fuse %0 {tile_interchange = [], tile_sizes = [1024]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @collapse_shape_full_slice(%arg0: tensor<1x2x1024xf32>) -> tensor<2x1024xf32> {
  %empty= tensor.empty() : tensor<2x1024xf32>
  %expanded = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x2x1024xf32> into tensor<2x1024xf32>
  %0 = linalg.elemwise_unary ins(%expanded : tensor<2x1024xf32>)
                             outs(%empty: tensor<2x1024xf32>) -> tensor<2x1024xf32>
  return %0 : tensor<2x1024xf32>
}

// CHECK-LABEL:   func.func @collapse_shape_full_slice(
// CHECK:           tensor.empty() : tensor<2x1024xf32>
// CHECK:           tensor.collapse_shape
// CHECK:           linalg.elemwise_unary
// CHECK:           return
// CHECK:         }

