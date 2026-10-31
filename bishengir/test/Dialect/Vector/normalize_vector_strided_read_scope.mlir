// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Scope checks only: do not assert correctness of the existing 2-D backend.
// CHECK-LABEL: func.func @keep_2d_identity
// CHECK-NOT: vector.gather
// CHECK: vector.transfer_read {{.*}}, vector<2x2xf32>
// CHECK: return
func.func @keep_2d_identity(%src: memref<2x2xf32, strided<[4, 1]>>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x2xf32, strided<[4, 1]>>, vector<2x2xf32>
  return %v : vector<2x2xf32>
}

// -----

// Keep a one-element source on its existing padding path.
// CHECK-LABEL: func.func @single_element
// CHECK-NOT: vector.gather
// CHECK: vector.transfer_read
// CHECK: return
func.func @single_element(%src: memref<1xf32, strided<[2]>>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0], %zero : memref<1xf32, strided<[2]>>, vector<4xf32>
  return %v : vector<4xf32>
}
