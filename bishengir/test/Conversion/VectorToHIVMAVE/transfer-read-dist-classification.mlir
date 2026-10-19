// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics | FileCheck %s

// Permuting a single-element vector is a no-op; classify the read as BRC.
// CHECK-LABEL: func.func @single_element_permutation_is_exempt
// CHECK-NOT: failed to legalize
// CHECK: ave.hir.vload <BRC_B8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @single_element_permutation_is_exempt(%arg0: memref<16x16xi8, #hivm.address_space<ub>>) -> vector<1x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg0[0, 0] [1, 1] [1, 1] : memref<16x16xi8, #hivm.address_space<ub>> to memref<1x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>, vector<1x1xi8>
    return %0 : vector<1x1xi8>
  }
}

// -----

// Moving unit dimensions preserves a contiguous load when the remaining
// dimensions are contiguous in memory. Here a 1x16 row is read
// as a 16x1 column: vector[j][0] = src[0][j] lies at base + j, exactly the
// bytes a plain load reads, in the same order.
// CHECK-LABEL: func.func @unit_dim_reorder_of_contiguous_row_is_exempt
// CHECK-NOT: failed to legalize
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x16xi8, strided<[16, 1]>, #hivm.address_space<ub>> into vector<16xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_of_contiguous_row_is_exempt(%arg0: memref<16x16xi8, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg0[0, 0] [1, 16] [1, 1] : memref<16x16xi8, #hivm.address_space<ub>> to memref<1x16xi8, strided<[16, 1]>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, strided<[16, 1]>, #hivm.address_space<ub>>, vector<16x1xi8>
    return %0 : vector<16x1xi8>
  }
}

// -----

// Moving only unit dimensions is not enough when the real dimension is not
// contiguous: a 16x1 column with row stride 16 read as vector<1x16xi8> needs
// vector[0][j] = src[j][0] at base + 16 * j, while a plain load would read
// base + j. It is a real transpose that must be gathered, so it is rejected.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_of_strided_column_is_rejected(%arg0: memref<16x16xi8, #hivm.address_space<ub>>) -> vector<1x16xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg0[0, 0] [16, 1] [1, 1] : memref<16x16xi8, #hivm.address_space<ub>> to memref<16x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>, vector<1x16xi8>
    return %0 : vector<1x16xi8>
  }
}

// -----

// A non-identity-permutation vector.transfer_read must already have been
// lowered to a vector.gather before it reaches this pass. If one somehow arrives
// here unconverted, TransferReadOpPattern must reject it outright instead
// of silently emitting a plain load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unconverted_transpose(%arg0: memref<16x16xi8, #hivm.address_space<ub>>) -> vector<16x16xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x16xi8, #hivm.address_space<ub>>, vector<16x16xi8>
    return %0 : vector<16x16xi8>
  }
}

// -----

// Two real dimensions moved together past a unit one: 4x16 is one dense
// row-major block (strides 16 and 1), so vector[i][j][0] = src[0][i][j] lies at
// base + 16 * i + j, which is exactly what a plain load of 64 elements reads.
// CHECK-LABEL: func.func @unit_dim_reorder_of_dense_block_is_exempt
// CHECK-NOT: failed to legalize
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x4x16xi8, #hivm.address_space<ub>> into vector<64xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_of_dense_block_is_exempt(%arg0: memref<1x4x16xi8, #hivm.address_space<ub>>) -> vector<4x16x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, #hivm.address_space<ub>>, vector<4x16x1xi8>
    return %0 : vector<4x16x1xi8>
  }
}

// -----

// Same map, but the rows are padded (row stride 32 instead of 16): the inner
// dimension is contiguous, the outer one is not, so the 64 elements are not
// one consecutive run and a plain load would read the padding. Rejected.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_of_padded_rows_is_rejected(%arg0: memref<1x4x16xi8, strided<[128, 32, 1]>, #hivm.address_space<ub>>) -> vector<4x16x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, strided<[128, 32, 1]>, #hivm.address_space<ub>>, vector<4x16x1xi8>
    return %0 : vector<4x16x1xi8>
  }
}

// -----

// A constant result of extent 1 in the map only adds a unit vector
// dimension (no real broadcast): vector[0][j] = src[j], read by a plain load.
// CHECK-LABEL: func.func @unit_dim_broadcast_is_exempt
// CHECK-NOT: failed to legalize
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<16xi8, #hivm.address_space<ub>> into vector<16xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_broadcast_is_exempt(%arg0: memref<16xi8, #hivm.address_space<ub>>) -> vector<1x16xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (0, d0)>} : memref<16xi8, #hivm.address_space<ub>>, vector<1x16xi8>
    return %0 : vector<1x16xi8>
  }
}

// -----

// The same constant result with extent 4 is a real broadcast: every row of
// the vector repeats src, which a plain load of 64 consecutive elements does
// not produce. The repeated row is only 16 bytes, below the 32-byte BLK
// granularity, so BLK cannot implement it either.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @real_broadcast_is_rejected(%arg0: memref<16xi8, #hivm.address_space<ub>>) -> vector<4x16xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (0, d0)>} : memref<16xi8, #hivm.address_space<ub>>, vector<4x16xi8>
    return %0 : vector<4x16xi8>
  }
}

// -----

// With dynamic strides the contiguity of the real dimension cannot be proven,
// so moving a unit dimension is conservatively not exempt. Rejected.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_with_dynamic_strides_is_rejected(%arg0: memref<1x16xi8, strided<[?, ?]>, #hivm.address_space<ub>>) -> vector<16x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, strided<[?, ?]>, #hivm.address_space<ub>>, vector<16x1xi8>
    return %0 : vector<16x1xi8>
  }
}

// -----

// A dynamic extent is no obstacle for the innermost real dimension: only its
// stride decides whether a plain load reads the right elements. Here d0 has
// stride 1, so vector[0][j] = src[j] is contiguous no matter how long src is.
// CHECK-LABEL: func.func @unit_dim_broadcast_of_dynamic_contiguous_is_exempt
// CHECK-NOT: failed to legalize
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_broadcast_of_dynamic_contiguous_is_exempt(%arg0: memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> vector<1x64xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (0, d0)>} : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xbf16>
    return %0 : vector<1x64xbf16>
  }
}

// -----

// The source strides alone do not make the read contiguous: the vector must
// also cover every inner real dimension in full. Here only 8 of the 16
// elements of each row are read, so vector[i][j][0] = src[0][i][j] lies at
// base + 16 * i + j for j < 8, i.e. {0-7, 16-23, 32-39, 48-55}, while a plain
// load of 32 elements reads 0-31. Rejected.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_of_partial_inner_dim_is_rejected(%arg0: memref<1x4x16xi8, #hivm.address_space<ub>>) -> vector<4x8x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, #hivm.address_space<ub>>, vector<4x8x1xi8>
    return %0 : vector<4x8x1xi8>
  }
}

// -----

// Only the outermost real dimension may be read partially: two full rows of
// the dense 4x16 block are still one consecutive run of 32 elements.
// CHECK-LABEL: func.func @unit_dim_reorder_of_partial_outer_dim_is_exempt
// CHECK-NOT: failed to legalize
// CHECK: ave.hir.vload <NORM> %{{.*}} : memref<1x4x16xi8, #hivm.address_space<ub>> into vector<32xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_reorder_of_partial_outer_dim_is_exempt(%arg0: memref<1x4x16xi8, #hivm.address_space<ub>>) -> vector<2x16x1xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<1x4x16xi8, #hivm.address_space<ub>>, vector<2x16x1xi8>
    return %0 : vector<2x16x1xi8>
  }
}

// -----

// A unit source dimension is only skippable when the vector does not step
// over it either. Here the vector takes 4 elements along d0 (extent 1):
// vector[i][j] = src[j][i], and j >= 1 is out of bounds and must be padding,
// while a plain load of 64 consecutive elements reads real memory. Rejected.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unit_dim_stepped_over_by_vector_is_rejected(%arg0: memref<16x32xi8, #hivm.address_space<ub>>) -> vector<16x4xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg0[0, 0] [1, 16] [1, 1] : memref<16x32xi8, #hivm.address_space<ub>> to memref<1x16xi8, strided<[32, 1]>, #hivm.address_space<ub>>
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<1x16xi8, strided<[32, 1]>, #hivm.address_space<ub>>, vector<16x4xi8>
    return %0 : vector<16x4xi8>
  }
}

// -----

// Each f16 element is repeated across a 32-byte block. Dynamic row offsets
// preserve 16-byte E2B alignment, but not necessarily 32-byte alignment.
// CHECK-LABEL: func.func @e2b_b16(
// CHECK: ave.hir.vload <E2B_B16> %{{.*}}[%{{.*}}, %{{.*}}] : memref<1x4xf16, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @e2b_b16(%src: memref<8x8xf16, #hivm.address_space<ub>>, %row: index) -> vector<1x4x16xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f16
    %subview = memref.subview %src[%row, 0] [1, 4] [1, 1] : memref<8x8xf16, #hivm.address_space<ub>> to memref<1x4xf16, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %read = vector.transfer_read %subview[%c0, %c0], %pad {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1, 0)>} : memref<1x4xf16, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x4x16xf16>
    return %read : vector<1x4x16xf16>
  }
}

// -----

// An unknown stride on a unit vector dimension does not affect E2B.
// CHECK-LABEL: func.func @e2b_b32_dynamic_unit_stride(
// CHECK: ave.hir.vload <E2B_B32> %{{.*}}[%{{.*}}, %{{.*}}] : memref<1x4xf32, strided<[?, 1]>, #hivm.address_space<ub>> into vector<32xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @e2b_b32_dynamic_unit_stride(%src: memref<1x4xf32, strided<[?, 1]>, #hivm.address_space<ub>>) -> vector<1x4x8xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f32
    %read = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1, 0)>} : memref<1x4xf32, strided<[?, 1]>, #hivm.address_space<ub>>, vector<1x4x8xf32>
    return %read : vector<1x4x8xf32>
  }
}

// -----

// Broadcast the middle dimension by repeating a contiguous 32-byte block.
// CHECK-LABEL: func.func @blk_middle_dimension(
// CHECK: ave.hir.vload <BLK>
// CHECK-SAME: into vector<64xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @blk_middle_dimension(%src: memref<1x16xf16, #hivm.address_space<ub>>) -> vector<1x4x16xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f16
    %read = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (d0, 0, d1)>} : memref<1x16xf16, #hivm.address_space<ub>>, vector<1x4x16xf16>
    return %read : vector<1x4x16xf16>
  }
}

// -----

// Neither source nor vector is a singleton: classify the broadcast range.
// CHECK-LABEL: func.func @broadcast_range_b16(
// CHECK: ave.hir.vload <BRC_B16>
// CHECK-SAME: into vector<16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @broadcast_range_b16(%src: memref<16xf16, #hivm.address_space<ub>>) -> vector<16xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f16
    %read = vector.transfer_read %src[%c0], %pad {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>} : memref<16xf16, #hivm.address_space<ub>>, vector<16xf16>
    return %read : vector<16xf16>
  }
}

// -----

// Retain the singleton NORM fallback for types without a BRC dist.
// CHECK-LABEL: func.func @singleton_i1_norm(
// CHECK: ave.hir.vload <NORM>
// CHECK-SAME: into vector<1xi1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @singleton_i1_norm(%src: memref<16xi1, #hivm.address_space<ub>>) -> vector<1xi1> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant false
    %read = vector.transfer_read %src[%c0], %pad {in_bounds = [true], permutation_map = affine_map<(d0) -> (d0)>} : memref<16xi1, #hivm.address_space<ub>>, vector<1xi1>
    return %read : vector<1xi1>
  }
}

// -----

// The total size is block-aligned, but each element repeats only 16 bytes.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @e2b_short_block_rejected(%src: memref<4xf16, #hivm.address_space<ub>>) -> vector<4x8xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f16
    // expected-error@+2 {{unsupported transfer_read: expected BRC/NORM/E2B/BLK layout}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0], %pad {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<4xf16, #hivm.address_space<ub>>, vector<4x8xf16>
    return %read : vector<4x8xf16>
  }
}

// -----

// An f16 offset of one element does not meet the 16-byte E2B requirement.
// Reject it before the generic unaligned lowering can drop the broadcast.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @e2b_b16_unaligned_rejected(%src: memref<16xf16, #hivm.address_space<ub>>) -> vector<8x16xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0.0 : f16
    // expected-error@+2 {{E2B_B16 load source address is not aligned to 16 bytes}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c1], %pad {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<16xf16, #hivm.address_space<ub>>, vector<8x16xf16>
    return %read : vector<8x16xf16>
  }
}

// -----

// Non-unit source dimensions must remain in source order, even when a
// transposed strided layout would make the result physically contiguous.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unconverted_transpose_contiguous_layout(%arg0: memref<16x16xi8, strided<[1, 16]>, #hivm.address_space<ub>>) -> vector<16x16xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{non-identity permutation transfer_read must be lowered to a gather, not a plain load}}
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x16xi8, strided<[1, 16]>, #hivm.address_space<ub>>, vector<16x16xi8>
    return %0 : vector<16x16xi8>
  }
}
