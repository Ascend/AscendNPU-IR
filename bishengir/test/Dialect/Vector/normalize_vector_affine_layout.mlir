// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Both dimension strides come from the affine layout, not the shape.
// CHECK-LABEL: func.func @affine_transpose
// CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 8, 2, 10]> : vector<4xi32>
// CHECK-NOT: vector.transfer_read
// CHECK: %[[GATHER:.*]] = vector.gather {{.*}}[%[[INDEX]]],
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[GATHER]] : vector<4xf32> to vector<2x2xf32>
// CHECK: return %[[RESULT]]
func.func @affine_transpose(%src: memref<2x2xf32, affine_map<(d0, d1)[s0] -> (d0 * 8 + d1 * 2 + s0)>>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<2x2xf32, affine_map<(d0, d1)[s0] -> (d0 * 8 + d1 * 2 + s0)>>, vector<2x2xf32>
  return %v : vector<2x2xf32>
}

// -----

// Preserve the 1-D identity fix for affine layouts with a dynamic offset.
// CHECK-LABEL: func.func @affine_identity
// CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 2, 4, 6]> : vector<4xi32>
// CHECK-NOT: vector.transfer_read
// CHECK: %[[GATHER:.*]] = vector.gather {{.*}}[%[[INDEX]]],
// CHECK: return %[[GATHER]]
func.func @affine_identity(%src: memref<4xf32, affine_map<(d0)[s0] -> (d0 * 2 + s0)>>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<4xf32, affine_map<(d0)[s0] -> (d0 * 2 + s0)>>, vector<4xf32>
  return %v : vector<4xf32>
}

// -----

// A dynamic affine stride cannot be used to build constant gather indices.
// CHECK-LABEL: func.func @keep_dynamic_affine_stride
// CHECK-NOT: vector.gather
// CHECK: %[[READ:.*]] = vector.transfer_read
// CHECK-NOT: vector.gather
// CHECK: return %[[READ]]
func.func @keep_dynamic_affine_stride(%src: memref<2x2xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<2x2xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>, vector<2x2xf32>
  return %v : vector<2x2xf32>
}

// -----

// A non-strided layout must not be interpreted as a contiguous layout.
// CHECK-LABEL: func.func @keep_non_strided_layout
// CHECK-NOT: vector.gather
// CHECK: %[[READ:.*]] = vector.transfer_read
// CHECK-NOT: vector.gather
// CHECK: return %[[READ]]
func.func @keep_non_strided_layout(%src: memref<2x4xf32, affine_map<(d0, d1) -> (d0, d1 floordiv 2, d1 mod 2)>>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %src[%c0, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<2x4xf32, affine_map<(d0, d1) -> (d0, d1 floordiv 2, d1 mod 2)>>, vector<2x2xf32>
  return %v : vector<2x2xf32>
}
