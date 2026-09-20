// RUN: bishengir-opt %s -normalize-vector | FileCheck %s

// Do not treat an unknown stride as a compile-time gather offset.
// CHECK-LABEL: func.func @keep_dynamic_stride
// CHECK-NOT: vector.gather
// CHECK: vector.transfer_read
// CHECK-NOT: vector.gather
// CHECK: return
func.func @keep_dynamic_stride(%src: memref<?xf32, strided<[?]>>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<?xf32, strided<[?]>>, vector<64xf32>
  return %v : vector<64xf32>
}

// A gather cannot replace the implicit bounds checks of transfer_read.
// CHECK-LABEL: func.func @keep_implicit_bounds
// CHECK-NOT: vector.gather
// CHECK: vector.transfer_read
// CHECK-NOT: vector.gather
// CHECK: return
func.func @keep_implicit_bounds(%src: memref<?xf32, strided<[2]>>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0], %zero : memref<?xf32, strided<[2]>>, vector<64xf32>
  return %v : vector<64xf32>
}
