// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Contiguity of a `vector.transfer_read` is decided by the *tile* extents and
// the physical strides, not by the extents of the whole source memref.
//
// Review finding on PR #4100: judging "packed" from the source shape made the
// new rank>1 gather path swallow reads whose tile is in fact contiguous.  The
// gather path then rejects dynamic or non-zero padding (`createPaddingValue`
// reports a fatal error, `VecGatherOpPattern` refuses a non-zero
// passthrough), so an input that compiled correctly before started failing --
// even though the padding is unobservable in a full-mask read.

// -----

// The tile IS contiguous: strides [32, 1] on a 2x32 read touches 0..31 and
// 32..63, i.e. lanes 0..63 back to back.  It must stay on the contiguous load
// path even though the *source* memref<2x64> is not densely packed.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @contiguous_tile_in_strided_source
  // CHECK-NOT: vector.gather
  // CHECK: vector.transfer_read
  func.func @contiguous_tile_in_strided_source(
      %src: memref<2x64xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x64xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
          vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Same contiguous tile, but with a *non-zero constant* padding.  The gather
// path would reject the non-zero passthrough; the contiguous load does not
// care, because a full-mask read never observes the padding.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @contiguous_tile_nonzero_padding
  // CHECK-NOT: vector.gather
  // CHECK: vector.transfer_read
  func.func @contiguous_tile_nonzero_padding(
      %src: memref<2x64xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 1.5 : f32
    %v = vector.transfer_read %src[%c0, %c0], %pad
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x64xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
          vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Same contiguous tile with a *dynamic* padding operand.  This is the case
// that used to hit the `report_fatal_error` inside `createPaddingValue`.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @contiguous_tile_dynamic_padding
  // CHECK-NOT: vector.gather
  // CHECK: vector.transfer_read
  func.func @contiguous_tile_dynamic_padding(
      %src: memref<2x64xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>,
      %pad: f32)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %src[%c0, %c0], %pad
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x64xf32, strided<[32, 1]>, #hivm.address_space<ub>>,
          vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// The mirror image: a *packed* source memref whose tile is NOT contiguous.
// memref<2x64> has strides [64, 1], so reading vector<2x32> must touch
// 0..31 and 64..95.  Judging packedness from the source shape called this
// contiguous and emitted a load that wrongly read 0..63.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @noncontiguous_tile_in_packed_source
  // CHECK-DAG: %[[IDX:.*]] = arith.constant dense<[0, 1, 2,{{.*}}30, 31, 64, 65,{{.*}}94, 95]> : vector<64xi32>
  // CHECK-NOT: vector.transfer_read
  // CHECK: vector.gather {{.*}}[%[[IDX]]]
  func.func @noncontiguous_tile_in_packed_source(
      %src: memref<2x64xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x64xf32, #hivm.address_space<ub>>, vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Reading the full extent of a packed source stays contiguous: tile shape ==
// source shape, so the tile-based and the source-based check agree here.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @full_packed_read_regression
  // CHECK-NOT: vector.gather
  // CHECK: vector.transfer_read
  func.func @full_packed_read_regression(
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
