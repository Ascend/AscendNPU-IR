// RUN: bishengir-opt %s -transform-interpreter -canonicalize --split-input-file | FileCheck %s

// Check that the unified row-major tiling algorithm for expand_shape
// accepts patterns following [unit|shape==1]* [mainDim] [NoTile]*
// with dynamic tile sizes and dynamic shapes.

// Case 1: shape==1 prefix before the main dim.

// CHECK: func.func @shape1_prefix
// CHECK: scf.for
// CHECK: %[[slice:.*]] = tensor.extract_slice
// CHECK: %[[exp:.*]] = tensor.expand_shape %[[slice]]

module attributes { transform.with_named_sequence } {
  func.func @shape1_prefix(%arg0: tensor<32xf16>) -> tensor<1x8x1x2x2xf16> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %init = tensor.empty() : tensor<1x8x1x2x2xf16>
    %exp = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 8, 1, 2, 2] : tensor<32xf16> into tensor<1x8x1x2x2xf16>
    %r = scf.for %i = %c0 to %c8 step %c4 iter_args(%a = %init) -> tensor<1x8x1x2x2xf16> {
      %s = tensor.extract_slice %exp[0, %i, 0, 0, 0] [1, 4, 1, 2, 2] [1, 1, 1, 1, 1] : tensor<1x8x1x2x2xf16> to tensor<1x4x1x2x2xf16>
      %u = tensor.insert_slice %s into %a[0, %i, 0, 0, 0] [1, 4, 1, 2, 2] [1, 1, 1, 1, 1] : tensor<1x4x1x2x2xf16> into tensor<1x8x1x2x2xf16>
      scf.yield %u : tensor<1x8x1x2x2xf16>
    }
    return %r : tensor<1x8x1x2x2xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %exp_h = transform.structured.match ops{["tensor.expand_shape"]} in %f : (!transform.any_op) -> !transform.any_op
    %loop_h = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.structured.fuse_into_containing_op %exp_h into %loop_h : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Case 2: dynamic tile size on the main dim.

// CHECK: func.func @dynamic_tile
// CHECK: scf.for
// CHECK: %[[slice:.*]] = tensor.extract_slice
// CHECK: %[[exp:.*]] = tensor.expand_shape %[[slice]]

module attributes { transform.with_named_sequence } {
  func.func @dynamic_tile(%arg0: tensor<1x128xf16>, %step: index) -> tensor<1x8x16xf16> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %init = tensor.empty() : tensor<1x8x16xf16>
    %exp = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [1, 8, 16] : tensor<1x128xf16> into tensor<1x8x16xf16>
    %r = scf.for %i = %c0 to %c8 step %step iter_args(%a = %init) -> tensor<1x8x16xf16> {
      %s = tensor.extract_slice %exp[0, %i, 0] [1, %step, 16] [1, 1, 1] : tensor<1x8x16xf16> to tensor<1x?x16xf16>
      %u = tensor.insert_slice %s into %a[0, %i, 0] [1, %step, 16] [1, 1, 1] : tensor<1x?x16xf16> into tensor<1x8x16xf16>
      scf.yield %u : tensor<1x8x16xf16>
    }
    return %r : tensor<1x8x16xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %exp_h = transform.structured.match ops{["tensor.expand_shape"]} in %f : (!transform.any_op) -> !transform.any_op
    %loop_h = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.structured.fuse_into_containing_op %exp_h into %loop_h : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Case 3: dynamic expanded shape on the main dim.

// CHECK: func.func @dynamic_shape
// CHECK: scf.for
// CHECK: %[[slice:.*]] = tensor.extract_slice
// CHECK: %[[exp:.*]] = tensor.expand_shape %[[slice]]

module attributes {transform.with_named_sequence} {
  func.func @dynamic_shape(%arg0: tensor<1x128xf16>, %arg1: index, %step: index) -> tensor<1x?x16xf16> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %size = arith.muli %arg1, %c16 : index
    %slice = tensor.extract_slice %arg0[0, 0] [1, %size] [1, 1] : tensor<1x128xf16> to tensor<1x?xf16>

    %init = tensor.empty(%arg1) : tensor<1x?x16xf16>
    %exp = tensor.expand_shape %slice [[0], [1, 2]] output_shape [1, %arg1, 16] : tensor<1x?xf16> into tensor<1x?x16xf16>
    %2 = scf.for %i = %c0 to %arg1 step %step iter_args(%a = %init) -> (tensor<1x?x16xf16>) {
      %s = tensor.extract_slice %exp[0, %i, 0] [1, %step, 16] [1, 1, 1] : tensor<1x?x16xf16> to tensor<1x?x16xf16>
      %u = tensor.insert_slice %s into %a[0, %i, 0] [1, %step, 16] [1, 1, 1] : tensor<1x?x16xf16> into tensor<1x?x16xf16>
      scf.yield %u : tensor<1x?x16xf16>
    }
    return %2 : tensor<1x?x16xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.expand_shape"]} in %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["scf.for"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Case 4: invalid.

// CHECK: func.func @invalid
// CHECK: %[[exp:.*]] = tensor.expand_shape
// CHECK: scf.for
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[exp]]

module attributes {transform.with_named_sequence} {
  func.func @invalid(%arg0: tensor<1x128xf16>, %arg1: index, %step: index) -> tensor<1x?x?x16xf16> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %size = arith.muli %arg1, %c16 : index
    %slice = tensor.extract_slice %arg0[0, 0] [1, %size] [1, 1] : tensor<1x128xf16> to tensor<1x?xf16>

    %init = tensor.empty(%arg1, %arg1) : tensor<1x?x?x16xf16>
    %exp = tensor.expand_shape %slice [[0], [1, 2, 3]] output_shape [1, %arg1, %arg1, 16] : tensor<1x?xf16> into tensor<1x?x?x16xf16>
    %2 = scf.for %i = %c0 to %arg1 step %step iter_args(%a = %init) -> (tensor<1x?x?x16xf16>) {
      %s = tensor.extract_slice %exp[0, %i, 0, 0] [1, %step, %step, 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16> to tensor<1x?x?x16xf16>
      %u = tensor.insert_slice %s into %a[0, %i, 0, 0] [1, %step, %step, 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16> into tensor<1x?x?x16xf16>
      scf.yield %u : tensor<1x?x?x16xf16>
    }
    return %2 : tensor<1x?x?x16xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.expand_shape"]} in %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["scf.for"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
