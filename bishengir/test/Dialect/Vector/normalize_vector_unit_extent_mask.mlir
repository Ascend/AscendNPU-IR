// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector \
// RUN: -convert-vector-to-hivmave -split-input-file | FileCheck %s --check-prefix=AVE

// A dim of extent 1 holds a single slice, so a non-empty mask pins its
// coordinate to 0 and that dim can never break the row-major prefix property
// -- whether or not its bound is a compile-time constant.  Requiring a
// *constant* 1 there rejected the mask the Triton reducer below emits, which
// then reached -convert-vector-to-hivmave still multi-dimensional and failed
// with "mask should be a 1-D vector."
//
// The length has to multiply in each leading dim's bound clamped to its
// extent: clamped, because a bound of 0 must empty the whole mask while a
// bound above the extent must contribute a plain 1 rather than scale the
// length.

// -----

// The reported failure, reduced.  `triton_sum_3D_dim1` walks a 7x17x15 buffer
// with two loop-carried bounds that the frontend has already clamped to 1, so
// the mask is `[1, 1, 64]` with bounds (dynamic<=1, dynamic<=1, 15) and the
// active lanes are 0..14 -- a prefix of length 15.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @unit_extent_dynamic_bounds
  // NORMALIZE-NOT: vector.create_mask %{{.*}}, %{{.*}}, %{{.*}} : vector<1x1x64xi1>
  // NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>

  // AVE-LABEL: func.func @unit_extent_dynamic_bounds
  // AVE: ave.hir.vload
  func.func @unit_extent_dynamic_bounds(
      %src: memref<7x17x15xf32, #hivm.address_space<ub>>,
      %dst: memref<7x17x15xf32, #hivm.address_space<ub>>,
      %i: index, %j: index) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c15 = arith.constant 15 : index
    %c17 = arith.constant 17 : index
    %cst = arith.constant 0.0 : f32
    %rows = arith.subi %c7, %i : index
    %rowsLt = arith.cmpi slt, %rows, %c1 : index
    %b0 = arith.select %rowsLt, %rows, %c1 : index
    %cols = arith.subi %c17, %j : index
    %colsLt = arith.cmpi slt, %cols, %c1 : index
    %b1 = arith.select %colsLt, %cols, %c1 : index
    %mask = vector.create_mask %b0, %b1, %c15 : vector<1x1x64xi1>
    %v = vector.transfer_read %src[%i, %j, %c0], %cst, %mask
        : memref<7x17x15xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
    vector.transfer_write %v, %dst[%i, %j, %c0], %mask
        : vector<1x1x64xf32>, memref<7x17x15xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Every bound dynamic on `[1, 1, 64]`: the two unit dims are still pinned, so
// the innermost bound is the pivot and the length is the clamped product.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @unit_extent_all_bounds_dynamic
  // NORMALIZE-NOT: vector.create_mask %{{.*}}, %{{.*}}, %{{.*}} : vector<1x1x64xi1>
  // NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>
  func.func @unit_extent_all_bounds_dynamic(
      %src: memref<1x1x64xf32, #hivm.address_space<ub>>,
      %dst: memref<1x1x64xf32, #hivm.address_space<ub>>,
      %b0: index, %b1: index, %b2: index)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %mask = vector.create_mask %b0, %b1, %b2 : vector<1x1x64xi1>
    %v = vector.transfer_read %src[%c0, %c0, %c0], %cst, %mask
        : memref<1x1x64xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
    vector.transfer_write %v, %dst[%c0, %c0, %c0], %mask
        : vector<1x1x64xf32>, memref<1x1x64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// The bound of a unit dim must be clamped into the length, not dropped: a
// bound of 0 empties the mask, so the flattened length has to be able to
// reach 0 even though the pivot's own bound is a non-zero constant.  The
// `maxsi`/`minsi` pair is what pins the factor to {0, 1}.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @unit_extent_bound_reaches_length
  // NORMALIZE-DAG: %[[Z:.*]] = arith.constant 0 : index
  // NORMALIZE-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // NORMALIZE: %[[LO:.*]] = arith.maxsi %{{.*}}, %[[Z]] : index
  // NORMALIZE: %[[HI:.*]] = arith.minsi %[[LO]], %[[ONE]] : index
  // NORMALIZE: arith.muli
  // NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>
  func.func @unit_extent_bound_reaches_length(
      %src: memref<1x1x64xf32, #hivm.address_space<ub>>,
      %dst: memref<1x1x64xf32, #hivm.address_space<ub>>,
      %b0: index) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %cst = arith.constant 0.0 : f32
    %mask = vector.create_mask %b0, %c1, %c15 : vector<1x1x64xi1>
    %v = vector.transfer_read %src[%c0, %c0, %c0], %cst, %mask
        : memref<1x1x64xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
    vector.transfer_write %v, %dst[%c0, %c0, %c0], %mask
        : vector<1x1x64xf32>, memref<1x1x64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Regression guard for the shape this pattern already supported: leading
// bounds that are constant 1 contribute a factor of 1 that folds away, so the
// innermost bound is used verbatim and no clamping arithmetic appears.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @leading_constant_one_unchanged
  // NORMALIZE-NOT: arith.maxsi
  // NORMALIZE-NOT: arith.minsi
  // NORMALIZE-NOT: arith.muli
  // NORMALIZE: vector.create_mask %arg2 : vector<64xi1>
  func.func @leading_constant_one_unchanged(
      %src: memref<1x1x64xf32, #hivm.address_space<ub>>,
      %dst: memref<1x1x64xf32, #hivm.address_space<ub>>,
      %n: index) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %mask = vector.create_mask %c1, %c1, %n : vector<1x1x64xi1>
    %v = vector.transfer_read %src[%c0, %c0, %c0], %cst, %mask
        : memref<1x1x64xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
    vector.transfer_write %v, %dst[%c0, %c0, %c0], %mask
        : vector<1x1x64xf32>, memref<1x1x64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Regression guard for the dynamic-pivot shape fixed in the previous round
// (review finding A): a non-unit pivot dim keeps using the clamp-and-scale
// path, unaffected by the leading-dim relaxation.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @dynamic_pivot_still_flattens
  // NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>
  func.func @dynamic_pivot_still_flattens(
      %src: memref<2x32xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>,
      %rows: index) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.0 : f32
    %mask = vector.create_mask %rows, %c32 : vector<2x32xi1>
    %v = vector.transfer_read %src[%c0, %c0], %cst, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, #hivm.address_space<ub>>, vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}
