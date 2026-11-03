// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// A 2-D identity read over a non-packed layout used to stay on the
// transfer_read path; since Issue #588 it is converted to a gather so the
// row gap (stride 4 vs row size 2) is not read as if it were contiguous.
// CHECK-LABEL: func.func @gather_2d_identity
// CHECK: arith.constant dense<[0, 1, 4, 5]> : vector<4xi32>
// CHECK-NOT: vector.transfer_read
// CHECK: vector.gather
// CHECK: return
func.func @gather_2d_identity(%src: memref<2x2xf32, strided<[4, 1]>>) -> vector<2x2xf32> {
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
