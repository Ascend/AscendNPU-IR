// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Review finding on PR #4100: every check that can reject the rewrite must run
// *before* any IR is created.
//
// `TransferReadToGatheringLoadPattern` used to materialize the gather index
// constant first and only then test whether the flattened mask is a prefix.
// For a non-prefix mask that test fails, so the pattern returned failure with
// a dead index constant already inserted.  The greedy driver then dropped the
// unused constant, re-matched the same `transfer_read`, inserted it again, and
// kept spinning until it hit the rewrite iteration limit -- an unbounded
// compile-time cost rather than a wrong result.

// -----

// The exact reproducer from the review.  `constant_mask [2, 1]` on
// <2x2xi1> enables lanes 0 and 2 of the flattened vector:
//
//   lane coords : (0,0) (0,1) (1,0) (1,1)
//   offsets     :   0     1     4     5
//   flat mask   :   T     F     T     F
//
// so firstEnabledBit=0, lastEnabledBit=2, firstDisabledBit=1 and the prefix
// test `lastEnabledBit + 1 == firstDisabledBit` (3 == 1) fails.  The pattern
// must bail out without leaving an index constant behind.
func.func @nonprefix_mask_leaves_no_dead_index(
    %src: memref<2x2xf32, strided<[4, 1]>>) -> vector<2x2xf32> {
  // CHECK-LABEL: func.func @nonprefix_mask_leaves_no_dead_index
  // CHECK-NOT: vector.gather
  // CHECK-NOT: arith.constant dense<[0, 0, 4, 0]>
  // CHECK: vector.constant_mask [2, 1] : vector<2x2xi1>
  // CHECK: vector.transfer_read
  // CHECK: return
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %mask = vector.constant_mask [2, 1] : vector<2x2xi1>
  %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
      {in_bounds = [true, true]}
      : memref<2x2xf32, strided<[4, 1]>>, vector<2x2xf32>
  return %v : vector<2x2xf32>
}

// -----

// A larger non-prefix mask on a strided source: `constant_mask [3, 16]` on
// <4x16xi1> enables lanes 0..47 and disables 48..63, which *is* a prefix, so
// this one is still rewritten.  Kept next to the negative case so the guard
// cannot be tightened into rejecting valid prefixes.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @prefix_mask_still_rewritten
  // CHECK: vector.gather
  // CHECK: return
  func.func @prefix_mask_still_rewritten(
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

// A wider non-prefix mask on the identity path: `constant_mask [4, 2]` on
// <4x4xi1> enables the first two lanes of every row, i.e. flattened lanes
// 0, 1, 4, 5, 8, 9, 12, 13.  firstDisabledBit is 2 while lastEnabledBit is 13,
// so the prefix test fails and the rewrite must be abandoned before the index
// constant is materialized.
func.func @nonprefix_mask_column_slice(
    %src: memref<4x4xf32, strided<[8, 1]>>) -> vector<4x4xf32> {
  // CHECK-LABEL: func.func @nonprefix_mask_column_slice
  // CHECK-NOT: vector.gather
  // CHECK-NOT: arith.constant dense<[0, 1, 0, 0
  // CHECK: vector.constant_mask [4, 2] : vector<4x4xi1>
  // CHECK: vector.transfer_read
  // CHECK: return
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %mask = vector.constant_mask [4, 2] : vector<4x4xi1>
  %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
      {in_bounds = [true, true],
       permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
      : memref<4x4xf32, strided<[8, 1]>>, vector<4x4xf32>
  return %v : vector<4x4xf32>
}
