// RUN: bishengir-opt %s --split-input-file --transform-interpreter --verify-diagnostics | FileCheck %s

// Invariant slices and their consumers may leave an inner loop, but must
// remain inside the outermost for. Other invariant ops still leave both loops.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @slice_boundary(
  // CHECK-SAME: %[[SRC:.*]]: tensor<8xf32>, %[[DST:.*]]: memref<?xf32>, %[[N:.*]]: index
  func.func @slice_boundary(%src: tensor<8xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[SUM:.*]] = arith.addi %[[N]], %[[N]] : index
    // CHECK-NOT: tensor.extract_slice
    // CHECK: scf.for
    // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]][0] [4] [1]
    // CHECK: %[[NEG:.*]] = arith.negf %[[SLICE]] : tensor<4xf32>
    // CHECK: %[[VALUE:.*]] = tensor.extract %[[NEG]]
    // CHECK: scf.for %[[J:arg[0-9]+]]
    // CHECK-NOT: tensor.extract_slice
    // CHECK-NOT: arith.negf
    // CHECK: memref.store %[[VALUE]], %[[DST]][%[[J]]]
    scf.for %i = %c0 to %n step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %sum = arith.addi %n, %n : index
        %slice = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
        %neg = arith.negf %slice : tensor<4xf32>
        %value = tensor.extract %neg[%c0] : tensor<4xf32>
        memref.store %value, %dst[%j] : memref<?xf32>
        // Keep the scalar computation live without constraining its LICM.
        %as_float = arith.index_cast %sum : index to i32
        %float = arith.sitofp %as_float : i32 to f32
        memref.store %float, %dst[%i] : memref<?xf32>
      }
    } {slice_boundary}
    return
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} attributes{slice_boundary} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}

// -----

// A is outside the explicit boundary B. LICM visits C before B, allowing
// slices to leave C but not B, regardless of B's enclosing loops.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @nested_boundary(
  // CHECK-SAME: %[[SRC:.*]]: tensor<8xf32>, %[[DST:.*]]: memref<?xf32>, %[[N:.*]]: index
  func.func @nested_boundary(%src: tensor<8xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK-NOT: arith.addi
    // CHECK-NOT: tensor.extract_slice
    // CHECK: scf.for %[[A:arg[0-9]+]]
    // CHECK: %[[SUM:.*]] = arith.addi %[[N]], %[[N]] : index
    // CHECK-NOT: tensor.extract_slice
    // CHECK: scf.for %[[B:arg[0-9]+]]
    // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]][0] [4] [1]
    // CHECK: %[[NEG:.*]] = arith.negf %[[SLICE]] : tensor<4xf32>
    // CHECK: %[[VALUE:.*]] = tensor.extract %[[NEG]]
    // CHECK-NOT: arith.addi
    // CHECK: scf.for %[[C:arg[0-9]+]]
    // CHECK-NOT: tensor.extract_slice
    // CHECK-NOT: arith.negf
    // CHECK: memref.store %[[VALUE]], %[[DST]][%[[C]]]
    // CHECK: memref.store {{.*}}, %[[DST]][%[[B]]]
    // CHECK: memref.store {{.*}}, %[[DST]][%[[A]]]
    scf.for %a = %c0 to %n step %c1 {
      scf.for %b = %c0 to %n step %c1 {
        scf.for %c = %c0 to %n step %c1 {
          %sum = arith.addi %n, %n : index
          %slice = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
          %neg = arith.negf %slice : tensor<4xf32>
          %value = tensor.extract %neg[%c0] : tensor<4xf32>
          memref.store %value, %dst[%c] : memref<?xf32>
          %as_int = arith.index_cast %sum : index to i32
          %as_float = arith.sitofp %as_int : i32 to f32
          memref.store %as_float, %dst[%b] : memref<?xf32>
          memref.store %as_float, %dst[%a] : memref<?xf32>
        }
      } {slice_boundary}
    }
    return
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %boundary = transform.structured.match ops{["scf.for"]} attributes{slice_boundary} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %boundary : !transform.any_op
    transform.yield
  }
}

// -----

// Three nested loops need a post-order walk to move the slice
// all the way into the containing loop in one application.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @deep_boundary(
  // CHECK-NOT: tensor.extract_slice
  // CHECK: scf.for
  // CHECK: tensor.extract_slice
  // CHECK: tensor.extract
  // CHECK: scf.for
  // CHECK-NOT: tensor.extract_slice
  // CHECK: scf.for
  // CHECK-NOT: tensor.extract_slice
  // CHECK: memref.store
  func.func @deep_boundary(%src: tensor<8xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %a = %c0 to %n step %c1 {
      scf.for %b = %c0 to %n step %c1 {
        scf.for %c = %c0 to %n step %c1 {
          %slice = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
          %value = tensor.extract %slice[%c0] : tensor<4xf32>
          memref.store %value, %dst[%c] : memref<?xf32>
        }
      }
    } {slice_boundary}
    return
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %boundary = transform.structured.match ops{["scf.for"]} attributes{slice_boundary} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %boundary : !transform.any_op
    transform.yield
  }
}

// -----

// Disjoint boundaries can share a handle. Both keep their slices inside.
module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @sibling_boundaries(
  // CHECK-NOT: tensor.extract_slice
  // CHECK: scf.for
  // CHECK: tensor.extract_slice
  // CHECK: memref.store
  // CHECK: scf.for
  // CHECK: tensor.extract_slice
  // CHECK: memref.store
  func.func @sibling_boundaries(%src: tensor<8xf32>, %dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %slice = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
      %value = tensor.extract %slice[%c0] : tensor<4xf32>
      memref.store %value, %dst[%i] : memref<?xf32>
    }
    scf.for %i = %c0 to %n step %c1 {
      %slice = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
      %value = tensor.extract %slice[%c0] : tensor<4xf32>
      memref.store %value, %dst[%i] : memref<?xf32>
    }
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %boundaries = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm_with_slice_boundary to %boundaries : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @invalid_boundary() {
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected an scf.for slice boundary}}
    transform.apply_licm_with_slice_boundary to %func : !transform.any_op
    transform.yield
  }
}

// -----

// Reject overlapping targets before running LICM at either boundary.
module attributes {transform.with_named_sequence} {
  func.func @overlapping_boundaries(%dst: memref<?xf32>, %n: index, %value: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        memref.store %value, %dst[%j] : memref<?xf32>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{slice boundaries must not contain one another}}
    transform.apply_licm_with_slice_boundary to %loops : !transform.any_op
    transform.yield
  }
}
