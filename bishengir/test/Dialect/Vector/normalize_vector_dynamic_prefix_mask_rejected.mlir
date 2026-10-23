// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE

// Review finding 10-23 [P1], negative half: masks whose active region cannot
// be proven a row-major prefix must stay multi-dimensional.  They are kept out
// of the lowering RUN line of normalize_vector_dynamic_prefix_mask.mlir on
// purpose: -convert-vector-to-hivmave has no lowering for a rank>1
// `create_mask` and is expected to reject them loudly.

// -----

// Two dynamic bounds cannot be proven a prefix: `%c` may leave its dim
// partially covered, which would move active lanes rather than truncate them.
// NORMALIZE-LABEL: func.func @two_dynamic_bounds_rejected
// NORMALIZE-NOT: vector<64xi1>
// NORMALIZE: vector.create_mask %{{.*}}, %{{.*}} : vector<2x32xi1>
func.func @two_dynamic_bounds_rejected(%r: index, %c: index)
    -> vector<2x32xi1> {
  %m = vector.create_mask %r, %c : vector<2x32xi1>
  return %m : vector<2x32xi1>
}


// -----

// A constant inner bound that does not cover its dim is the non-prefix case
// the guard exists for: `(%r, 16)` over <2x32xi1> activates lanes 0..15 and
// 32..47, which no single prefix length describes.
// NORMALIZE-LABEL: func.func @partial_inner_bound_rejected
// NORMALIZE-NOT: vector<64xi1>
// NORMALIZE: vector.create_mask %{{.*}}, %{{.*}} : vector<2x32xi1>
func.func @partial_inner_bound_rejected(%r: index) -> vector<2x32xi1> {
  %c16 = arith.constant 16 : index
  %m = vector.create_mask %r, %c16 : vector<2x32xi1>
  return %m : vector<2x32xi1>
}

