// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector -convert-vector-to-hivmave -convert-arith-to-hivmave -split-input-file | FileCheck %s --check-prefix=AVE

// Issue #588 use-case 1: rows are contiguous but separated by a gap.
// Source layout strided<[64, 1]> on a 2x32 view: row 0 lives at element
// offsets 0..31, row 1 at 64..95.  Lane k must read (k / 32) * 64 + (k % 32),
// i.e. [0..31, 64..95] - a contiguous load of 64 elements would wrongly read
// 0..63 and pull in the 32..63 gap.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank2_row_gap
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE-DAG: %[[IDX:.*]] = arith.constant dense<[0, 1, 2,{{.*}}30, 31, 64, 65,{{.*}}94, 95]> : vector<64xi32>
  // NORMALIZE: vector.gather {{.*}}[%[[IDX]]]
  // The index vector is piecewise-arithmetic, so it must lower through the
  // segmented VCI form rather than being rejected by the plain VCI pattern.
  // AVE-LABEL: func.func @rank2_row_gap
  // AVE-NOT: arith.constant dense
  // AVE: ave.hir.vci
  // AVE: ave.hir.vdiv
  // AVE: ave.hir.vgather
  func.func @rank2_row_gap(
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

// Inner stride > 1 on top of a row gap: lane k reads
// (k / 16) * 64 + (k % 16) * 2 => [0,2,..,30, 64,66,..,94, 128,...].
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank2_inner_stride
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // AVE-LABEL: func.func @rank2_inner_stride
  // AVE: ave.hir.vgather
  func.func @rank2_inner_stride(
      %src: memref<4x16xf32, strided<[64, 2]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<4x16xf32, strided<[64, 2]>, #hivm.address_space<ub>>,
          vector<4x16xf32>
    %flat = vector.shape_cast %v : vector<4x16xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// 3-D strided read: only the outermost dim introduces a gap
// (strided<[128, 32, 1]> on 2x2x16 => inner 2x16 block is packed).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank3_row_gap
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  func.func @rank3_row_gap(
      %src: memref<2x2x16xf32, strided<[128, 16, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true],
         permutation_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}
        : memref<2x2x16xf32, strided<[128, 16, 1]>, #hivm.address_space<ub>>,
          vector<2x2x16xf32>
    %flat = vector.shape_cast %v : vector<2x2x16xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// 3-D read where *two* dims introduce a gap: strided<[128, 32, 1]> on
// 2x2x16 has a 16-element gap between the middle-dim rows (32 != 16 * 1) and
// a further gap between the outer slabs (128 != 2 * 32).  The index vector
// then has two distinct jumps, so the closed form needs two staircase terms:
//   offset(k) = k + 16 * (k / 16) + 64 * (k / 32)
// A single-term form cannot express this, so it must be lowered with two
// vdiv/vadd pairs rather than being rejected.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank3_multi_gap
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // AVE-LABEL: func.func @rank3_multi_gap
  // AVE-NOT: arith.constant dense
  // AVE: ave.hir.vci
  // AVE: ave.hir.vdiv
  // AVE: ave.hir.vdiv
  // AVE: ave.hir.vgather
  func.func @rank3_multi_gap(
      %src: memref<2x2x16xf32, strided<[128, 32, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true],
         permutation_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}
        : memref<2x2x16xf32, strided<[128, 32, 1]>, #hivm.address_space<ub>>,
          vector<2x2x16xf32>
    %flat = vector.shape_cast %v : vector<2x2x16xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Packed multi-dim source must stay on the contiguous NORM load path.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank2_packed
  // NORMALIZE-NOT: vector.gather
  // NORMALIZE: vector.transfer_read
  // AVE-LABEL: func.func @rank2_packed
  // AVE-NOT: ave.hir.vgather
  // AVE: ave.hir.vload
  func.func @rank2_packed(
      %src: memref<2x32xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, #hivm.address_space<ub>>, vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// A mask whose active region *is* a row-major prefix may still be flattened:
// `[1, 16]` on <2x32xi1> covers lanes 0..15 only.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank2_prefix_mask
  // NORMALIZE: vector.constant_mask [16] : vector<64xi1>
  func.func @rank2_prefix_mask(
      %src: memref<2x32xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [1, 16] : vector<2x32xi1>
    %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, #hivm.address_space<ub>>, vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// A fully-active multi-dim mask flattens to an all-true 1-D mask, which the
// transfer_read folder then drops entirely (an all-true mask is a no-op).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank2_full_mask
  // NORMALIZE-NOT: vector.constant_mask
  // NORMALIZE: vector.transfer_read
  func.func @rank2_full_mask(
      %src: memref<2x32xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [2, 32] : vector<2x32xi1>
    %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, #hivm.address_space<ub>>, vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Regression: the rank-1 strided identity path from PR #3600 is unchanged.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank1_stride2_regression
  // NORMALIZE-DAG: %[[INDEX:.*]] = arith.constant dense<[0, 2, 4, {{.*}}, 124, 126]> : vector<64xi32>
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather {{.*}}[%[[INDEX]]]
  func.func @rank1_stride2_regression(
      %src: memref<72xf32, strided<[2], offset: 32>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c8], %zero {in_bounds = [true]}
        : memref<72xf32, strided<[2], offset: 32>, #hivm.address_space<ub>>,
          vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Regression: a rank-1 contiguous read still uses the NORM load path.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @rank1_contiguous_regression
  // NORMALIZE-NOT: vector.gather
  // NORMALIZE: vector.transfer_read
  func.func @rank1_contiguous_regression(
      %src: memref<64xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]}
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}
