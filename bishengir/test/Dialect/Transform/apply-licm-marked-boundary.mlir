// RUN: bishengir-opt %s --split-input-file --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @marked_boundary
  func.func @marked_boundary(%src: tensor<8xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: arith.addi
    // CHECK: scf.for
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract
    // CHECK: scf.for
    // CHECK-NOT: tensor.extract_slice
    // CHECK: memref.store
    scf.for %i = %c0 to %n step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %sum = arith.addi %n, %n : index
        %s = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
        %v = tensor.extract %s[%c0] : tensor<4xf32>
        memref.store %v, %dst[%j] : memref<?xf32>
        %f = arith.index_cast %sum : index to i32
        %fv = arith.sitofp %f : i32 to f32
        memref.store %fv, %dst[%i] : memref<?xf32>
      }
    } {"outlined-loop-target-1"}
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}

// -----

// A pure inner loop must not carry an IV-dependent slice across the boundary.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @region_slice
  func.func @region_slice(%src: tensor<8xf32>, %init: tensor<4xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: tensor.extract_slice
    // CHECK: memref.store
    scf.for %i = %c0 to %n step %c1 {
      %r = scf.for %j = %c0 to %c4 step %c1 iter_args(%t = %init) -> tensor<4xf32> {
        %s = tensor.extract_slice %src[%j] [4] [1] : tensor<8xf32> to tensor<4xf32>
        scf.yield %s : tensor<4xf32>
      }
      %v = tensor.extract %r[%c0] : tensor<4xf32>
      memref.store %v, %dst[%i] : memref<?xf32>
    } {"outlined-loop-target-2"}
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}
