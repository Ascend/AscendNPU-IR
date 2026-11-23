// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics | FileCheck %s

// Preserve mask positions through shape casts. Reject masks that do not
// prove the unit-dimension bound, before converting their producers.
// Check constant_mask and create_mask with statically known bounds.
// create_mask must not authorize masked expansion of a unit source dimension;
// keep the legacy constant_mask exception covered by the existing tests.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @rectangle_with_holes(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %mask = vector.constant_mask [2, 8] : vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// Counting the 16 active bits as a prefix would yield [1, 16] and falsely
// satisfy the unit-dimension exemption. Preserve the holes instead.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @reshape_nonprefix_rectangle(%src: memref<1x32xi8, #hivm.address_space<ub>>) -> vector<32x2xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %rectangle = vector.constant_mask [2, 8] : vector<4x16xi1>
    %mask = vector.shape_cast %rectangle : vector<4x16xi1> to vector<2x32xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x32xi8, #hivm.address_space<ub>>, vector<32x2xi8>
    return %read : vector<32x2xi8>
  }
}

// -----

// Rounding 17 active lanes to [1, 2, 16] would incorrectly satisfy the
// unit-dimension exemption. Reject the non-rectangular prefix itself.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @prefix_not_rectangular_in_target(%src: memref<1x4x16xi8, #hivm.address_space<ub>>) -> vector<4x16x2xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %flat = vector.constant_mask [17] : vector<128xi1>
    %mask = vector.shape_cast %flat : vector<128xi1> to vector<2x4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0, %c0], %padding, %mask {in_bounds = [true, true, false], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, #hivm.address_space<ub>>, vector<4x16x2xi8>
    return %read : vector<4x16x2xi8>
  }
}

// -----

// Rounding 17 active lanes to [1, 2, 16] would incorrectly satisfy the
// unit-dimension exemption. Reject the non-rectangular prefix itself.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @create_prefix_not_rectangular_in_target(%src: memref<1x4x16xi8, #hivm.address_space<ub>>) -> vector<4x16x2xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c17 = arith.constant 17 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %c17 : vector<128xi1>
    %mask = vector.shape_cast %flat : vector<128xi1> to vector<2x4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0, %c0], %padding, %mask {in_bounds = [true, true, false], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, #hivm.address_space<ub>>, vector<4x16x2xi8>
    return %read : vector<4x16x2xi8>
  }
}

// -----

// Although the recovered bounds are [1, 1], a 64-element plain load would
// lose padding and overread this 23-element source. Reject this exemption.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @create_projected_unit_bound(%src: memref<23x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %padding = arith.constant 0.0 : f32
    %mask = vector.create_mask %c1 : vector<64xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// Shape casts must not hide create_mask from the unit-dimension expansion
// restriction, even when the recovered prefix bounds are exactly [1, 16].
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @create_shape_cast_unit_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %c16 : vector<64xi1>
    %intermediate = vector.shape_cast %flat : vector<64xi1> to vector<8x8xi1>
    %mask = vector.shape_cast %intermediate : vector<8x8xi1> to vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// A dynamic bound remains unknown even through a shape cast.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @create_dynamic_shape_cast_bound(%src: memref<1x16xi8, #hivm.address_space<ub>>, %extent: index) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %extent : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// A larger allocation does not make the masked expansion legal: the read
// must return padding, not the values from subsequent source rows.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @create_projected_unit_bound_large_source(%src: memref<128x1xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %padding = arith.constant 7.0 : f32
    %mask = vector.create_mask %c1 : vector<64xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<128x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// create_mask remains supported when no masked unit-dimension expansion is
// needed. All 64 values are active, contiguous and in bounds; the permutation
// only moves a unit dimension, so flattening preserves their order.
// CHECK-LABEL: func.func @create_full_unit_reorder
// CHECK-NOT: vector.transfer_read
// CHECK: %[[LOAD:.*]] = ave.hir.vload <NORM> {{.*}} into vector<64xf32>
// CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[LOAD]] : vector<64xf32> to vector<64x1xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[RESULT]] : vector<64x1xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @create_full_unit_reorder(%src: memref<1x64xf32, #hivm.address_space<ub>>) -> vector<64x1xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %padding = arith.constant 7.0 : f32
    %flat = vector.create_mask %c64 : vector<64xi1>
    %mask = vector.shape_cast %flat : vector<64xi1> to vector<1x64xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x64xf32, #hivm.address_space<ub>>, vector<64x1xf32>
    return %read : vector<64x1xf32>
  }
}
