// RUN: bishengir-opt %s --split-input-file --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @pending_op
  func.func @pending_op(%x: f32, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: arith.negf %{{.*}} : f32
    // CHECK: scf.for
    // CHECK: arith.negf {{.*}} {"hfusion-auto-vectorize-target-1"}
    // CHECK: arith.addf
    // CHECK: memref.store
    scf.for %i = %c0 to %n step %c1 {
      %free = arith.negf %x : f32
      %pending = arith.negf %x {"hfusion-auto-vectorize-target-1"} : f32
      %sum = arith.addf %free, %pending : f32
      memref.store %sum, %dst[%i] : memref<?xf32>
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

// A pure region must not carry a tagged operation out of its enclosing loop.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @pending_in_region
  func.func @pending_in_region(%x: f32, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: arith.negf {{.*}} {"hfusion-auto-vectorize-target-2"}
    // CHECK: memref.store
    scf.for %i = %c0 to %n step %c1 {
      %r = scf.for %j = %c0 to %n step %c1 iter_args(%v = %x) -> f32 {
        %pending = arith.negf %x {"hfusion-auto-vectorize-target-2"} : f32
        scf.yield %pending : f32
      }
      memref.store %r, %dst[%i] : memref<?xf32>
    }
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}
