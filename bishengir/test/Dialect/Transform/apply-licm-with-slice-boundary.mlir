// RUN: bishengir-opt %s --split-input-file --transform-interpreter | FileCheck %s
// RUN: bishengir-opt %s --split-input-file --transform-interpreter -test-transform-dialect-erase-schedule > %t.custom
// RUN: sed 's/apply_licm_with_slice_boundary/apply_licm/g' %s > %t.mlir
// RUN: bishengir-opt %t.mlir --split-input-file --transform-interpreter -test-transform-dialect-erase-schedule > %t.upstream
// RUN: diff %t.custom %t.upstream

// The custom operation has exactly the upstream target-local behavior,
// including overlapping targets, non-for loops, and ordinary slice hoisting.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @nested_loops
  func.func @nested_loops(%src: tensor<8xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: memref.store
    scf.for %i = %c0 to %n step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %s = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
        %v = tensor.extract %s[%c0] : tensor<4xf32>
        memref.store %v, %dst[%j] : memref<?xf32>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @parallel_loop
  func.func @parallel_loop(%dst: memref<?xindex>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: arith.addi
    // CHECK: scf.parallel
    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      %sum = arith.addi %n, %n : index
      memref.store %sum, %dst[%i] : memref<?xindex>
      scf.reduce
    }
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.parallel"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}
