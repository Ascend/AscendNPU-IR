// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics | FileCheck %s

// Check layout classification independently of mask bounds. A unit result
// dimension can be reordered; expanding it does not gain a layout exemption.
// These checks cover legalization, not execution of masked loads.

// CHECK-LABEL: func.func @constant_unit_bound
// CHECK-NOT: vector.transfer_read
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x16xi8, #hivm.address_space<ub>> into vector<16xi8>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @constant_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [1, 16] : vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

// CHECK-LABEL: func.func @shape_cast_unit_bound
// CHECK-NOT: vector.transfer_read
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x16xi8, #hivm.address_space<ub>> into vector<16xi8>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @shape_cast_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %flat = vector.constant_mask [16] : vector<16xi1>
    %mask = vector.shape_cast %flat : vector<16xi1> to vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_bound_two(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %flat = vector.constant_mask [32] : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<4x16xi1>
    // expected-error@+1 {{transfer_read expands a unit source dimension; cannot lower to an unmasked load}}
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
    %flat = vector.create_mask %extent : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<4x16xi1>
    // expected-error@+1 {{transfer_read expands a unit source dimension; cannot lower to an unmasked load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @strided_real_dimension(%src: memref<1x16xi8, strided<[32, 2]>, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [1, 16] : vector<1x16xi1>
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, strided<[32, 2]>, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

// A mask cannot make an unmasked NORM load safe when the read expands
// a unit source dimension, regardless of whether one or two lanes are active.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_last_unit_dimension(%src: memref<23x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1] : vector<64xi1>
    // expected-error@+1 {{transfer_read expands a unit source dimension; cannot lower to an unmasked load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// The mask omits source dimensions fixed by scalar indices.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_middle_unit_dimension(%src: memref<23x1x7xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1] : vector<64xi1>
    // expected-error@+1 {{transfer_read expands a unit source dimension; cannot lower to an unmasked load}}
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : memref<23x1x7xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// The mask omits source dimensions fixed by scalar indices.
// CHECK-LABEL: func.func @projected_permuted_unit_dimensions
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<23x1x16xf32, #hivm.address_space<ub>> into vector<16xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_permuted_unit_dimensions(%src: memref<23x1x16xf32, #hivm.address_space<ub>>) -> vector<16x1xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1, 16] : vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<23x1x16xf32, #hivm.address_space<ub>>, vector<16x1xf32>
    return %read : vector<16x1xf32>
  }
}

// -----

// A mask cannot make an unmasked NORM load safe when the read expands
// a unit source dimension, regardless of whether one or two lanes are active.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_last_unit_dimension_bound_two(%src: memref<23x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [2] : vector<64xi1>
    // expected-error@+1 {{transfer_read expands a unit source dimension; cannot lower to an unmasked load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}
