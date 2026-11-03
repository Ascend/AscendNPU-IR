// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Cases that -normalize-vector must leave alone because converting them would
// change semantics.  They are kept out of the AVE conversion RUN line of
// normalize_vector_multidim_strided_read.mlir on purpose: the AVE dialect has
// no representation for a non-prefix mask or an out-of-bounds padded read, so
// -convert-vector-to-hivmave is expected to reject them loudly.

// Issue #588 use-case 2: a rectangular partial mask.
// `constant_mask [2, 16]` on <2x32xi1> is active at lanes 0..15 and 32..47.
// Flattening it to `constant_mask [32]` (the 2 * 16 bound product) would claim
// lanes 0..31, moving the active positions.  The flattening must therefore be
// rejected and the mask kept multi-dimensional.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @rank2_partial_mask
  // CHECK-NOT: vector.constant_mask [32] : vector<64xi1>
  // CHECK: vector.constant_mask [2, 16] : vector<2x32xi1>
  func.func @rank2_partial_mask(
      %src: memref<2x32xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [2, 16] : vector<2x32xi1>
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

// An out-of-bounds multi-dim strided read keeps the transfer_read padding
// semantics instead of being turned into a gather.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @rank2_out_of_bounds
  // CHECK-NOT: vector.gather
  // CHECK: vector.transfer_read
  func.func @rank2_out_of_bounds(
      %src: memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
      -> vector<4x32xf32> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [false, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>,
          vector<4x32xf32>
    return %v : vector<4x32xf32>
  }
}
