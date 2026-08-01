// RUN: bishengir-opt -transform-interpreter -canonicalize --split-input-file %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: func.func @tile_expand_shape_static
  // CHECK: scf.for
  // CHECK: tensor.expand_shape {{.*}} output_shape [1, 16, 8, 16] : tensor<1x16x128xf16> into tensor<1x16x8x16xf16>
  func.func @tile_expand_shape_static(%arg0: tensor<1x128x128xf16>) -> tensor<1x128x8x16xf16> {
    %c0 = arith.constant 0 : index
    %step = arith.constant 16 : index
    %c128 = arith.constant 128 : index

    %init = tensor.empty() : tensor<1x128x8x16xf16>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 128, 8, 16] : tensor<1x128x128xf16> into tensor<1x128x8x16xf16>
    %r = scf.for %i = %c0 to %c128 step %step iter_args(%a = %init) -> tensor<1x128x8x16xf16> {
      %slice = tensor.extract_slice %expanded[0, %i, 0, 0] [1, 16, 8, 16] [1, 1, 1, 1] : tensor<1x128x8x16xf16> to tensor<1x16x8x16xf16>
      %upd = tensor.insert_slice %slice into %a[0, %i, 0, 0] [1, 16, 8, 16] [1, 1, 1, 1] : tensor<1x16x8x16xf16> into tensor<1x128x8x16xf16>
      scf.yield %upd : tensor<1x128x8x16xf16>
    }
    return %r : tensor<1x128x8x16xf16>
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

module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: func.func @tile_expand_shape_dynamic
  // CHECK: scf.for
  // CHECK: tensor.expand_shape {{.*}} output_shape [1, %arg1, 8, 16] : tensor<1x?x128xf16> into tensor<1x?x8x16xf16>
  func.func @tile_expand_shape_dynamic(%arg0: tensor<1x128x128xf16>, %step: index) -> tensor<1x128x8x16xf16> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index

    %init = tensor.empty() : tensor<1x128x8x16xf16>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 128, 8, 16] : tensor<1x128x128xf16> into tensor<1x128x8x16xf16>
    %r = scf.for %i = %c0 to %c128 step %step iter_args(%a = %init) -> tensor<1x128x8x16xf16> {
      %slice = tensor.extract_slice %expanded[0, %i, 0, 0] [1, %step, 8, 16] [1, 1, 1, 1] : tensor<1x128x8x16xf16> to tensor<1x?x8x16xf16>
      %upd = tensor.insert_slice %slice into %a[0, %i, 0, 0] [1, %step, 8, 16] [1, 1, 1, 1] : tensor<1x?x8x16xf16> into tensor<1x128x8x16xf16>
      scf.yield %upd : tensor<1x128x8x16xf16>
    }
    return %r : tensor<1x128x8x16xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %exp_h = transform.structured.match ops{["tensor.expand_shape"]} in %f : (!transform.any_op) -> !transform.any_op
    %loop_h = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.structured.fuse_into_containing_op %exp_h into %loop_h : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
