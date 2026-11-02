// RUN: bishengir-opt -hivm-verify-vector-function-top-level-copy -split-input-file -verify-diagnostics %s | FileCheck %s

// A copy nested in a loop of a vector function is accepted.

// CHECK-LABEL: func.func @vf_nested_copy
func.func @vf_nested_copy(%arg0: memref<4xf32>, %arg1: memref<4xf32>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    memref.copy %arg0, %arg1 : memref<4xf32> to memref<4xf32>
  }
  return
}

// A top-level copy in a non-vector function is accepted.

// CHECK-LABEL: func.func @plain_copy
func.func @plain_copy(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
  memref.copy %arg0, %arg1 : memref<4xf32> to memref<4xf32>
  return
}

// -----

func.func @vf_top_level_copy(%arg0: memref<4xf32>, %arg1: memref<4xf32>) attributes {hivm.vector_function} {
  // expected-error @below {{is not supported in the top-level block of a vector function}}
  memref.copy %arg0, %arg1 : memref<4xf32> to memref<4xf32>
  return
}
