// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector -convert-vector-to-hivmave -convert-arith-to-hivmave -split-input-file | FileCheck %s --check-prefix=AVE

// Review comment #14: the gather index vector of a masked read must stay
// lowerable by `ConstantOpToHivmSegmentedVCILowering`.
//
// `computeGatherIndicesAndMasks` zeroes the index of every masked-off lane --
// it has to, because a masked-off lane often maps to a coordinate outside the
// source, so its "real" index would address past the end of the memref.  For a
// prefix mask that leaves a staircase with a flat zero tail, e.g. vector<3x4>
// on strided<[7, 1]> with constant_mask [2, 4]:
//
//   real indices : [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]
//   flat mask    : [T, T, T, T, T, T, T,  T,  F,  F,  F,  F]
//   zeroed tail  : [0, 1, 2, 3, 7, 8, 9, 10,  0,  0,  0,  0]
//
// Matching the whole vector then fails (peeling the gap-3 staircase leaves a
// residual that steps at lane 8, and 12 % 8 != 0) and the index constant used
// to fail to legalize.  The matcher now fits the closed form to the active
// prefix and writes the tail back with a prefix predicate and a select, so the
// rewritten value still equals the original on every lane.

// -----

// The reproducer from the review.  Lane k reads (k / 4) * 7 + (k % 4), i.e.
// `k + 3 * (k / 4)`: one staircase term over the first 8 lanes.  8 is a `pge`
// lane count, so the prefix predicate is `pge <VL8>`.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @prefix_mask_segmented_index
  // NORMALIZE-DAG: %[[IDX:.*]] = arith.constant dense<[0, 1, 2, 3, 7, 8, 9, 10, 0, 0, 0, 0]> : vector<12xi32>
  // NORMALIZE: vector.constant_mask [8] : vector<12xi1>
  // NORMALIZE: vector.gather {{.*}}[%[[IDX]]]

  // The dense index constant must be gone, replaced by vci + vdiv arithmetic
  // and a select that restores the zero tail.
  // AVE-LABEL: func.func @prefix_mask_segmented_index
  // AVE-NOT: arith.constant dense<[0, 1, 2, 3, 7, 8, 9, 10, 0, 0, 0, 0]>
  // AVE: %[[VCI:.*]] = ave.hir.vci
  // AVE: ave.hir.vdiv
  // AVE: %[[SEQ:.*]] = ave.hir.vadd
  // AVE: %[[ZERO:.*]] = ave.hir.broadcast %{{.*}} : i32, vector<12xi1> -> vector<12xi32>
  // AVE: %[[PRED:.*]] = ave.hir.pge <VL8> : vector<12xi1>
  // AVE: ave.hir.vsel %[[PRED]], %[[SEQ]], %[[ZERO]] : vector<12xi1>, vector<12xi32>
  // AVE: ave.hir.vgather
  func.func @prefix_mask_segmented_index(
      %src: memref<3x4xf32, strided<[7, 1]>, #hivm.address_space<ub>>,
      %dst: memref<12xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [2, 4] : vector<3x4xi1>
    %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<3x4xf32, strided<[7, 1]>, #hivm.address_space<ub>>,
          vector<3x4xf32>
    %flat = vector.shape_cast %v : vector<3x4xf32> to vector<12xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<12xf32>, memref<12xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Review comment #8, the same root cause: vector<4x16> on strided<[32, 1]>
// with constant_mask [3, 16] yields [0..15, 32..47, 64..79, 0...0].  48 active
// lanes is *not* one of the `pge` lane counts, so this one exercises the `plt`
// fallback that builds a prefix predicate for an arbitrary active length.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @prefix_mask_segmented_index_4x16
  // NORMALIZE: vector.constant_mask [48] : vector<64xi1>
  // NORMALIZE: vector.gather

  // AVE-LABEL: func.func @prefix_mask_segmented_index_4x16
  // AVE-NOT: arith.constant dense<[0, 1
  // AVE: ave.hir.vci
  // AVE: ave.hir.vdiv
  // AVE: %[[SEQ:.*]] = ave.hir.vadd
  // AVE: %[[ZERO:.*]] = ave.hir.broadcast %{{.*}} : i32, vector<64xi1> -> vector<64xi32>
  // AVE: %[[PRED:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<64xi1>, index
  // AVE: ave.hir.vsel %[[PRED]], %[[SEQ]], %[[ZERO]] : vector<64xi1>, vector<64xi32>
  // AVE: ave.hir.vgather
  func.func @prefix_mask_segmented_index_4x16(
      %src: memref<4x16xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [3, 16] : vector<4x16xi1>
    %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<4x16xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
          vector<4x16xf32>
    %flat = vector.shape_cast %v : vector<4x16xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// An unmasked staircase keeps the cheaper form: matching the whole vector
// succeeds, so no padding splat and no select are emitted.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // AVE-LABEL: func.func @full_mask_segmented_index_no_select
  // AVE: ave.hir.vci
  // AVE: ave.hir.vdiv
  // AVE-NOT: ave.hir.vsel
  // AVE: ave.hir.vgather
  func.func @full_mask_segmented_index_no_select(
      %src: memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>,
          vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// A plain arithmetic prefix with a zero tail must stay with the cheaper
// VCI/VCP pattern: the segmented matcher declines it (no staircase term), so
// no vdiv appears.  This guards the hand-off between the two sibling patterns,
// which is what review comment #14 found inconsistent.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // AVE-LABEL: func.func @plain_arithmetic_prefix_stays_vcivcp
  // AVE-NOT: ave.hir.vdiv
  // AVE: ave.hir.vmuls
  // AVE: ave.hir.vsel
  // AVE: ave.hir.vgather
  func.func @plain_arithmetic_prefix_stays_vcivcp(
      %src: memref<64xf32, strided<[2]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %m = vector.constant_mask [16] : vector<64xi1>
    %v = vector.transfer_read %src[%c0], %zero, %m {in_bounds = [true]}
        : memref<64xf32, strided<[2]>, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}
