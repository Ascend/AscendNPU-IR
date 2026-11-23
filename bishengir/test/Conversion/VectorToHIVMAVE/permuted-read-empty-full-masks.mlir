// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics | FileCheck %s

// Empty masked reads return padding without loading memory, including masks
// behind shape_cast and runtime padding. Full masks allow ordinary loads only
// when the permuted read is already legal, including higher-rank masks.
// A full mask must not make expansion of a unit source dimension legal.
// Full-mask cases also check that unused masks leave no AVE predicates behind.

// CHECK-LABEL: func.func @empty_mask
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<16x4xi8>
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[PAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_mask(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %mask = vector.constant_mask [0, 0] : vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----
// CHECK-LABEL: func.func @full_mask_contiguous_unit_reorder
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<16xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_mask_contiguous_unit_reorder(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %mask = vector.constant_mask [1, 16] : vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----
// CHECK-LABEL: func.func @full_shape_cast
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<16xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_shape_cast(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %flat = vector.constant_mask [16] : vector<16xi1>
    %mask = vector.shape_cast %flat : vector<16xi1> to vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_mask_cannot_expand_unit_dim(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %mask = vector.constant_mask [4, 16] : vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// A full higher-rank mask may be removed only after proving that the read is
// contiguous. Its producer must not fail the 1-D predicate conversion later.
// CHECK-LABEL: func.func @full_mask_dense_block
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<64xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_mask_dense_block(%src: memref<1x4x16xi8, #hivm.address_space<ub>>) -> vector<4x16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %mask = vector.constant_mask [1, 4, 16] : vector<1x4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %zero, %mask {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, #hivm.address_space<ub>>, vector<4x16x1xi8>
    return %read : vector<4x16x1xi8>
  }
}

// -----

// Empty reads must preserve a runtime padding value, too.
// CHECK-LABEL: func.func @empty_dynamic_padding
// CHECK-NOT: ave.hir.vload
// CHECK: %[[BROADCAST:.*]] = ave.hir.scalar_broadcast %arg1 : f32 -> vector<64xf32>
// CHECK-NOT: ave.hir.vload
// CHECK: return %[[BROADCAST]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_dynamic_padding(%src: memref<23x1xf32, #hivm.address_space<ub>>, %padding: f32) -> vector<64xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %mask = vector.constant_mask [0] : vector<64xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<23x1xf32, #hivm.address_space<ub>>, vector<64xf32>
    return %read : vector<64xf32>
  }
}

// -----

// A constant zero bound makes create_mask empty without prior canonicalization.
// CHECK-LABEL: func.func @empty_create_mask
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<16x4xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[PAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_create_mask(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %padding = arith.constant 7 : i8
    %mask = vector.create_mask %c0, %c16 : vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// Negative bounds clamp to zero; cleanup follows the entire shape-cast chain.
// CHECK-LABEL: func.func @empty_create_mask_shape_cast
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<16x4xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[PAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_create_mask_shape_cast(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %negative = arith.constant -1 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %negative : vector<64xi1>
    %intermediate = vector.shape_cast %flat : vector<64xi1> to vector<8x8xi1>
    %mask = vector.shape_cast %intermediate : vector<8x8xi1> to vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// Oversized bounds clamp to the mask shape and enable every lane.
// CHECK-LABEL: func.func @full_create_mask_clamped
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<16xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_create_mask_clamped(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %padding = arith.constant 7 : i8
    %mask = vector.create_mask %c2, %c32 : vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

// A full create_mask remains full through shape casts; erase its unused chain.
// CHECK-LABEL: func.func @full_create_mask_shape_cast
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<16xi8>
// CHECK-NOT: vector.constant_mask
// CHECK-NOT: vector.create_mask
// CHECK-NOT: ave.hir.pge
// CHECK-NOT: ave.hir.plt
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_create_mask_shape_cast(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %c16 : vector<16xi1>
    %intermediate = vector.shape_cast %flat : vector<16xi1> to vector<4x4xi1>
    %mask = vector.shape_cast %intermediate : vector<4x4xi1> to vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

// A proven full create_mask must still pass the plain-load legality check.
// In that case the plain-load legality check fails
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_create_mask_cannot_expand_unit_dim(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %padding = arith.constant 7 : i8
    %mask = vector.create_mask %c4, %c16 : vector<4x16xi1>
    // expected-error @+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// A constant_mask uses canonical all-zero bounds for an empty rectangle.
// CHECK-LABEL: func.func @empty_rectangle_constant_shape_cast
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<16x4xi8>
// CHECK-NOT: ave.hir.vload
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[PAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_rectangle_constant_shape_cast(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %rectangle = vector.constant_mask [0, 0] : vector<2x32xi1>
    %mask = vector.shape_cast %rectangle : vector<2x32xi1> to vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}

// -----

// A zero inner bound makes the mask empty before checking prefix continuity.
// CHECK-LABEL: func.func @empty_inner_bound_create
// CHECK-NOT: ave.hir.vload
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<16x4xi8>
// CHECK-NOT: ave.hir.vload
// CHECK: return %[[PAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_inner_bound_create(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %c2 = arith.constant 2 : index
    %rectangle = vector.create_mask %c2, %c0 : vector<2x32xi1>
    %mask = vector.shape_cast %rectangle : vector<2x32xi1> to vector<4x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x4xi8>
    return %read : vector<16x4xi8>
  }
}
// -----

// A partial create_mask must not be cleared as if it enabled every lane.
// This checks predicate preservation, not the runtime semantics of plain loads.
// CHECK-LABEL: func.func @partial_create_mask_preserved
// CHECK: ave.hir.{{pge|plt}}
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<16xi8>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @partial_create_mask_preserved(%src: memref<1x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %c8 : vector<16xi1>
    %mask = vector.shape_cast %flat : vector<16xi1> to vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}

// -----

// An unknown bound must not be classified as either empty or full.
// CHECK-LABEL: func.func @dynamic_create_mask_preserved
// CHECK: ave.hir.plt %arg1
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<16xi8>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_create_mask_preserved(%src: memref<1x16xi8, #hivm.address_space<ub>>, %extent: index) -> vector<16x1xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %padding = arith.constant 7 : i8
    %flat = vector.create_mask %extent : vector<16xi1>
    %mask = vector.shape_cast %flat : vector<16xi1> to vector<1x16xi1>
    %read = vector.transfer_read %src[%c0, %c0], %padding, %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, #hivm.address_space<ub>>, vector<16x1xi8>
    return %read : vector<16x1xi8>
  }
}
