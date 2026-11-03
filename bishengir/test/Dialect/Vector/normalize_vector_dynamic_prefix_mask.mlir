// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector -convert-vector-to-hivmave -split-input-file | FileCheck %s --check-prefix=AVE

// Review finding 10-23 [P1]: a `vector.create_mask` whose active region is a
// row-major prefix must be flattened even when the deciding bound is dynamic.
//
// Flattening is valid exactly when the bounds look like
// `[1, ..., 1, b_k, extent_k+1, ..., extent_n-1]`: the leading dims pin their
// coordinate to 0, every dim inside the pivot is covered in full, and the
// pivot's bound alone sets the prefix length.  Refusing all non-unit outer dims
// (the earlier conservative guard) rejected `(%rows, %c32)` over <2x32xi1>,
// which *is* such a prefix, and the rank-2 `create_mask` then reached
// -convert-vector-to-hivmave and failed with "mask should be a 1-D vector."
// -- a compile regression, not a missed optimization.



// The reported case: the inner dim covers all 32 lanes, so the flattened
// prefix is `%rows * 32` lanes for any `%rows`.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @dynamic_prefix_outer_dim
  // NORMALIZE-NOT: vector.create_mask {{.*}}, {{.*}} : vector<2x32xi1>
  // NORMALIZE-DAG: %[[C32:.*]] = arith.constant 32 : index
  // NORMALIZE: %[[LEN:.*]] = arith.muli %{{.*}}, %[[C32]] : index
  // NORMALIZE: vector.create_mask %[[LEN]] : vector<64xi1>

  // AVE-LABEL: func.func @dynamic_prefix_outer_dim
  // AVE-NOT: vector.create_mask
  func.func @dynamic_prefix_outer_dim(
      %src: memref<2x32xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>,
      %rows: index) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.create_mask %rows, %c32 : vector<2x32xi1>
    %v = vector.transfer_read %src[%c0, %c0], %zero, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x32xf32, #hivm.address_space<ub>>, vector<2x32xf32>
    %flat = vector.shape_cast %v : vector<2x32xf32> to vector<64xf32>
    %flat_mask = vector.shape_cast %mask : vector<2x32xi1> to vector<64xi1>
    vector.maskedstore %dst[%c0], %flat_mask, %flat
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}

// -----

// The pivot does not have to be the outermost dim: `(1, %q, 32)` over
// <2x3x32xi1> pins d0, lets d1 decide, and covers d2 in full.
// NORMALIZE-LABEL: func.func @dynamic_prefix_middle_dim
// NORMALIZE: %[[LEN:.*]] = arith.muli
// NORMALIZE: vector.create_mask %[[LEN]] : vector<192xi1>
func.func @dynamic_prefix_middle_dim(%q: index) -> vector<2x3x32xi1> {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %m = vector.create_mask %c1, %q, %c32 : vector<2x3x32xi1>
  return %m : vector<2x3x32xi1>
}

// -----

// Regression: the previously supported shape (all outer dims unit, bound a
// constant 1) is the pivot rule's innermost-pivot case and must still flatten
// with the bound used directly -- no clamp and no multiply, since the dims
// inside the pivot have a combined extent of 1.
// NORMALIZE-LABEL: func.func @unit_outer_dims_regression
// NORMALIZE-NOT: arith.muli
// NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>
func.func @unit_outer_dims_regression(%n: index) -> vector<1x1x64xi1> {
  %c1 = arith.constant 1 : index
  %m = vector.create_mask %c1, %c1, %n : vector<1x1x64xi1>
  return %m : vector<1x1x64xi1>
}
