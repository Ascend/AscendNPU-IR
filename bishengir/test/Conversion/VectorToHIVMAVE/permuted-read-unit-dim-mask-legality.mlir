// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics | FileCheck %s

// Exercise the legacy mask-bound exemption before producer conversion and
// the subsequent load-distribution checks. Unit expansion may pass the
// pre-walk while its result layout is still rejected by the classifier.
// These checks cover legalization, not execution of masked loads.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @constant_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [1, 16] : vector<4x16xi1>
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @shape_cast_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %flat = vector.constant_mask [16] : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<4x16xi1>
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// Regression: the 8-lane prefix reshapes to bounds [1, 1, 8], so the
// expanded unit source dimension retains the constant_mask exemption.
// CHECK-LABEL: func.func @shape_cast_expanded_unit_bound
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x1x8xf32, strided<[8, 8, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @shape_cast_expanded_unit_bound(%src: memref<1x1x8xf32, strided<[8, 8, 1], offset: ?>, #hivm.address_space<ub>>) -> vector<8x1x8xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %flat = vector.constant_mask [8] : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<1x8x8xi1>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : memref<1x1x8xf32, strided<[8, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x1x8xf32>
    return %read : vector<8x1x8xf32>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_bound_two(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [2, 16] : vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>, %extent: index) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0 : i8
    %mask = vector.create_mask %extent, %c16 : vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @strided_real_dimension(%src: memref<1x16xi8, strided<[32, 2]>, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [1, 16] : vector<4x16xi1>
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, strided<[32, 2]>, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// The projected unit source dimension has stride 1 and mask bound 1.
// CHECK-LABEL: func.func @projected_last_unit_dimension
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<23x1xf32, #hivm.address_space<ub>> into vector<64xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_last_unit_dimension(%src: memref<23x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1] : vector<64xi1>
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// The mask passes the unit-dimension check, but source d1 has stride 7,
// so the existing layout classification still rejects this projected read.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_middle_unit_dimension(%src: memref<23x1x7xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1] : vector<64xi1>
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : memref<23x1x7xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// The mask omits source dimensions fixed by scalar indices.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_permuted_unit_dimensions(%src: memref<23x1x16xf32, #hivm.address_space<ub>>) -> vector<16x4xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1, 16] : vector<4x16xi1>
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<23x1x16xf32, #hivm.address_space<ub>>, vector<16x4xf32>
    return %read : vector<16x4xf32>
  }
}

// -----

// A bound of two does not qualify for the constant_mask unit-dimension
// exemption, which requires a bound of exactly one.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_last_unit_dimension_bound_two(%src: memref<23x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [2] : vector<64xi1>
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}
