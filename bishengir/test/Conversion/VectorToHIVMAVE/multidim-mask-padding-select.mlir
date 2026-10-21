// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector \
// RUN: -convert-vector-to-hivmave -split-input-file | FileCheck %s --check-prefix=AVE

// Review finding on PR #4100: a rank>1 `vector.transfer_read` mask is lowered
// by flattening it through `vector.shape_cast` and selecting between the
// loaded data and a padding splat, while a rank-1 mask keeps the pre-existing
// contract (the mask is propagated to the consuming ops by
// arith-vector-mask-analyze).  The requested checks were the shape_cast
// equivalence and the elimination of a useless select for a full mask.
//
// NOTE: in the pipeline as it stands the rank>1 branch of
// `TransferReadOpPattern` is not reachable for the inputs below -- a
// multi-dim `vector.constant_mask` is always rewritten to a rank-1 predicate
// (`ave.hir.pge`) before the transfer_read is converted, and a multi-dim
// `vector.create_mask` is rejected outright by -convert-vector-to-hivmave.
// These tests therefore pin the reachable halves: the row-major prefix
// flattening that the select relies on, the rank-1 no-select contract, and
// the full-mask elimination.  Reaching the select branch itself needs an
// input the pass pipeline does not currently produce; see the PR discussion.

// -----

// shape_cast equivalence, producer side: `constant_mask [1, 16]` on
// <2x32xi1> and `constant_mask [16]` on <64xi1> denote the same 16 active
// lanes in row-major order, so flattening must yield exactly bound 16.  This
// is the invariant the rank>1 select depends on when it shape_casts the mask.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @prefix_mask_flattens_to_same_region
  // NORMALIZE: vector.constant_mask [16] : vector<64xi1>
  // NORMALIZE-NOT: vector.constant_mask [32] : vector<64xi1>
  func.func @prefix_mask_flattens_to_same_region(
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

// The rank-1 spelling of the same active region reaches AVE through the
// mask-propagation path, with no select inserted.  Together with the previous
// case this is the shape_cast equivalence: same region, same lowering.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // AVE-LABEL: func.func @rank1_equivalent_mask_no_select
  // AVE-NOT: ave.hir.vsel
  // AVE: ave.hir.pge <VL16>
  // AVE: ave.hir.vload
  func.func @rank1_equivalent_mask_no_select(
      %src: memref<64xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [16] : vector<64xi1>
    %v = vector.transfer_read %src[%c0], %zero, %mask {in_bounds = [true]}
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// A fully-active rank-2 mask carries no information: it flattens to an
// all-true rank-1 mask which the transfer_read folder then drops, so no
// select and no mask op survive.  Without that folding the select would cost
// a splat plus a vsel on every lane for nothing.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @full_multidim_mask_eliminated
  // NORMALIZE-NOT: vector.constant_mask
  // NORMALIZE: vector.transfer_read

  // AVE-LABEL: func.func @full_multidim_mask_eliminated
  // AVE-NOT: ave.hir.vsel
  // AVE: ave.hir.vload
  func.func @full_multidim_mask_eliminated(
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

// Review comment #12: -normalize-vector flattens a multi-dim prefix mask and
// casts it back to the read's shape, so the mask operand is still rank>1 even
// though it carries exactly the information of its rank-1 source.  Inserting a
// padding splat plus a select for it used to abort the compile for i64
// (`HIVMSelectOpLowering` has no predicate vector for a 64-bit element), and
// it is redundant anyway: the select's mask folds back to the very mask the
// rank-1 contract propagates.  Such a mask must therefore take the rank-1
// path.
//
// The residual i64 gap in -convert-hivmave-to-ave-intrin is pre-existing: a
// plain rank-1 i64 masked copy, which never reaches the code above, fails the
// same way.  That is why this RUN line stops at -convert-vector-to-hivmave.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // AVE-LABEL: func.func @i64_flattened_mask_no_select
  // AVE-NOT: ave.hir.vsel
  // AVE-NOT: ave.hir.vbr
  // AVE: ave.hir.vload
  // AVE: ave.hir.masked_store
  func.func @i64_flattened_mask_no_select(
      %src: memref<2x16xi64, #hivm.address_space<ub>>,
      %dst: memref<32xi64, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i64
    %mask = vector.constant_mask [1, 8] : vector<2x16xi1>
    %v = vector.transfer_read %src[%c0, %c0], %pad, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x16xi64, #hivm.address_space<ub>>, vector<2x16xi64>
    %flat = vector.shape_cast %v : vector<2x16xi64> to vector<32xi64>
    %fm = vector.constant_mask [8] : vector<32xi1>
    vector.maskedstore %dst[%c0], %fm, %flat
        : memref<32xi64, #hivm.address_space<ub>>, vector<32xi1>, vector<32xi64>
    return
  }
}

// -----

// Same shape with a dynamic padding operand: the broadcast that used to be
// emitted here has no i64 lowering either, so it must not be emitted.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // AVE-LABEL: func.func @i64_flattened_mask_dynamic_padding
  // AVE-NOT: ave.hir.vsel
  // AVE-NOT: ave.hir.vbr
  // AVE: ave.hir.vload
  func.func @i64_flattened_mask_dynamic_padding(
      %src: memref<2x16xi64, #hivm.address_space<ub>>,
      %dst: memref<32xi64, #hivm.address_space<ub>>,
      %pad: i64)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %mask = vector.constant_mask [1, 8] : vector<2x16xi1>
    %v = vector.transfer_read %src[%c0, %c0], %pad, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x16xi64, #hivm.address_space<ub>>, vector<2x16xi64>
    %flat = vector.shape_cast %v : vector<2x16xi64> to vector<32xi64>
    %fm = vector.constant_mask [8] : vector<32xi1>
    vector.maskedstore %dst[%c0], %fm, %flat
        : memref<32xi64, #hivm.address_space<ub>>, vector<32xi1>, vector<32xi64>
    return
  }
}
