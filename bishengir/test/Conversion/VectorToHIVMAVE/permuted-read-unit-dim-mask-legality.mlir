// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics | FileCheck %s

// Exercise the legality check before conversion rewrites the mask producers.
// Positive cases check the intended exemption for a masked unit source
// dimension. They only check conversion legality, not masked-load semantics.

// CHECK-LABEL: func.func @constant_unit_bound
// CHECK-NOT: vector.transfer_read
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x16xi8, #hivm.address_space<ub>> into vector<64xi8>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @constant_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [1, 16] : vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// CHECK-LABEL: func.func @shape_cast_unit_bound
// CHECK-NOT: vector.transfer_read
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x16xi8, #hivm.address_space<ub>> into vector<64xi8>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @shape_cast_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %flat = vector.constant_mask [16] : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
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
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, strided<[32, 2]>, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// The mask omits source dimensions fixed by scalar indices.
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

// The mask omits source dimensions fixed by scalar indices.
// CHECK-LABEL: func.func @projected_middle_unit_dimension
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<23x1x7xf32, #hivm.address_space<ub>> into vector<64xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_middle_unit_dimension(%src: memref<23x1x7xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1] : vector<64xi1>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : memref<23x1x7xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// The mask omits source dimensions fixed by scalar indices.
// CHECK-LABEL: func.func @projected_permuted_unit_dimensions
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<23x1x16xf32, #hivm.address_space<ub>> into vector<64xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_permuted_unit_dimensions(%src: memref<23x1x16xf32, #hivm.address_space<ub>>) -> vector<16x4xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1, 16] : vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<23x1x16xf32, #hivm.address_space<ub>>, vector<16x4xf32>
    return %read : vector<16x4xf32>
  }
}

// -----

// The mask omits source dimensions fixed by scalar indices.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @projected_last_unit_dimension_bound_two(%src: memref<23x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [2] : vector<64xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}
