// Canonicalization must leave control flow marked with
// `hivm.keep_control_flow` alone, and must still fold the same control flow
// when the marker is absent. The negative checks are what give the positive
// ones their meaning: without them this file would also pass against a build
// where the two folds never fire at all.
// RUN: bishengir-opt %s -canonicalize -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -canonicalize-ext -split-input-file | FileCheck %s
// Bufferizing an `scf.if` recreates the op, which must not lose the marker:
// that happens in the middle of the window which needs it, and nothing later
// would report it.
// RUN: bishengir-opt %s -split-input-file \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   | FileCheck %s --check-prefix=BUFFERIZE

// A constant condition is normally inlined away. Marked, the region stays.
// CHECK-LABEL: func.func @marked_if_survives
// CHECK:         scf.if %true
// CHECK:           memref.store
func.func @marked_if_survives(%m: memref<4xf32>, %v: f32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  scf.if %true {
    memref.store %v, %m[%c0] : memref<4xf32>
  } {hivm.keep_control_flow}
  return
}

// -----

// Negative control for the case above.
// CHECK-LABEL: func.func @unmarked_if_folds
// CHECK-NOT:     scf.if
// CHECK:         memref.store
func.func @unmarked_if_folds(%m: memref<4xf32>, %v: f32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  scf.if %true {
    memref.store %v, %m[%c0] : memref<4xf32>
  }
  return
}

// -----

// A single-iteration loop is normally replaced by its body. Marked, it stays a
// loop.
// CHECK-LABEL: func.func @marked_single_trip_for_survives
// CHECK:         scf.for
// CHECK:           memref.store
func.func @marked_single_trip_for_survives(%m: memref<4xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    memref.store %v, %m[%i] : memref<4xf32>
  } {hivm.keep_control_flow}
  return
}

// -----

// Negative control for the case above.
// CHECK-LABEL: func.func @unmarked_single_trip_for_folds
// CHECK-NOT:     scf.for
// CHECK:         memref.store
func.func @unmarked_single_trip_for_folds(%m: memref<4xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    memref.store %v, %m[%i] : memref<4xf32>
  }
  return
}

// -----

// A loop that provably never runs is normally erased along with its body.
// Marked, both stay: the delayed cross-core solver still has to see the region
// the anchors inside it belong to.
// CHECK-LABEL: func.func @marked_zero_trip_for_survives
// CHECK:         scf.for
// CHECK:           memref.store
func.func @marked_zero_trip_for_survives(%m: memref<4xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
    memref.store %v, %m[%i] : memref<4xf32>
  } {hivm.keep_control_flow}
  return
}

// -----

// Negative control for the case above.
// CHECK-LABEL: func.func @unmarked_zero_trip_for_folds
// CHECK-NOT:     scf.for
// CHECK-NOT:     memref.store
func.func @unmarked_zero_trip_for_folds(%m: memref<4xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
    memref.store %v, %m[%i] : memref<4xf32>
  }
  return
}

// -----

// A constant-false condition without an else region normally erases the whole
// op. Marked, it stays.
// CHECK-LABEL: func.func @marked_false_if_survives
// CHECK:         scf.if %false
func.func @marked_false_if_survives(%m: memref<4xf32>, %v: f32) {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  scf.if %false {
    memref.store %v, %m[%c0] : memref<4xf32>
  } {hivm.keep_control_flow}
  return
}

// -----

// Negative control for the case above.
// CHECK-LABEL: func.func @unmarked_false_if_folds
// CHECK-NOT:     scf.if
func.func @unmarked_false_if_folds(%m: memref<4xf32>, %v: f32) {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  scf.if %false {
    memref.store %v, %m[%c0] : memref<4xf32>
  }
  return
}

// -----

// The marker must not keep the rest of canonicalization from running: the
// store of a value that is never read is still removed from the marked region,
// and the marked op itself is untouched.
// CHECK-LABEL: func.func @marker_does_not_block_other_folds
// CHECK:         scf.if %true
// CHECK-NOT:       memref.cast
func.func @marker_does_not_block_other_folds(%m: memref<4xf32>, %v: f32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  scf.if %true {
    %cast = memref.cast %m : memref<4xf32> to memref<4xf32>
    memref.store %v, %cast[%c0] : memref<4xf32>
  } {hivm.keep_control_flow}
  return
}

// -----

// Bufferization builds a new `scf.if` for the memref types and moves the
// regions across; the marker has to come along.
// BUFFERIZE-LABEL: func.func @marked_if_survives_bufferization
// BUFFERIZE:         scf.if
// BUFFERIZE:         hivm.keep_control_flow
func.func @marked_if_survives_bufferization(%c: i1, %t: tensor<4xf32>,
                                            %v: f32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %r = scf.if %c -> tensor<4xf32> {
    %i = tensor.insert %v into %t[%c0] : tensor<4xf32>
    scf.yield %i : tensor<4xf32>
  } else {
    scf.yield %t : tensor<4xf32>
  } {hivm.keep_control_flow}
  return %r : tensor<4xf32>
}
