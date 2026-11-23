// RUN: bishengir-opt -transform-interpreter -canonicalize-ext %s | FileCheck %s

// Two extract_slice users of one producer differ only by `iv` versus
// `iv mod 4` inside `scf.for %iv = 0 to 4`. The windows are equivalent, so
// fusion must reuse the first slice instead of building a dynamic union.

// CHECK-LABEL: func.func @equivalent_mod_offset_slices
// CHECK: scf.for
// CHECK-NOT: affine.min
// CHECK-NOT: affine.max
// CHECK-NOT: tensor<?
// CHECK: %[[FILL:.*]] = linalg.fill {{.*}} -> tensor<4x1x16xf32>
// CHECK: linalg.elemwise_binary {{.*}} ins(%[[FILL]], %[[FILL]]
// CHECK-NOT: tensor<?
module attributes {transform.with_named_sequence} {
  func.func @equivalent_mod_offset_slices(%src: tensor<4x4x16xf32>, %dst: tensor<4x4x16xf32>) -> tensor<4x4x16xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %filled = linalg.fill ins(%cst : f32) outs(%src : tensor<4x4x16xf32>) -> tensor<4x4x16xf32>
    %out = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %dst) -> (tensor<4x4x16xf32>) {
      %mod = affine.apply affine_map<(d0) -> (d0 mod 4)>(%iv)
      %slice_iv = tensor.extract_slice %filled[0, %iv, 0] [4, 1, 16] [1, 1, 1] : tensor<4x4x16xf32> to tensor<4x1x16xf32>
      %slice_mod = tensor.extract_slice %filled[0, %mod, 0] [4, 1, 16] [1, 1, 1] : tensor<4x4x16xf32> to tensor<4x1x16xf32>
      %empty = tensor.extract_slice %acc[0, %iv, 0] [4, 1, 16] [1, 1, 1] : tensor<4x4x16xf32> to tensor<4x1x16xf32>
      %added = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%slice_iv, %slice_mod : tensor<4x1x16xf32>, tensor<4x1x16xf32>) outs(%empty : tensor<4x1x16xf32>) -> tensor<4x1x16xf32>
      %inserted = tensor.insert_slice %added into %acc[0, %iv, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf32> into tensor<4x4x16xf32>
      scf.yield %inserted : tensor<4x4x16xf32>
    }
    return %out : tensor<4x4x16xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 {duplicate_producer = true} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
