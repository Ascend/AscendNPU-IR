// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Shapes that must keep their rank: relaxing the leading-dim rule to accept a
// dynamic bound on a dim of extent 1 must not start accepting masks whose
// active region is genuinely not a row-major prefix.  These stay
// multi-dimensional so the consumer rejects them loudly rather than reading
// the wrong lanes.
//
// The RUN line stops at -normalize-vector on purpose: -convert-vector-to-hivmave
// has no lowering for a rank>1 mask and errors out, which would make the whole
// file fail.

// -----

// A non-unit dim between the dynamic leading dim and a partially covered
// innermost dim.  `[1, 2, 64]` with bounds (dynamic, 2, 15) is active at lanes
// 0..14 and 64..78 -- two runs, not a prefix -- so the bound product would
// move the active lanes instead of preserving them.
func.func @nonunit_middle_dim_rejected(%b0: index) -> vector<1x2x64xi1> {
  // CHECK-LABEL: func.func @nonunit_middle_dim_rejected
  // CHECK-NOT: vector.create_mask %{{.*}} : vector<128xi1>
  // CHECK: vector.create_mask %{{.*}}, %c2, %c15 : vector<1x2x64xi1>
  // CHECK: return
  %c2 = arith.constant 2 : index
  %c15 = arith.constant 15 : index
  %m = vector.create_mask %b0, %c2, %c15 : vector<1x2x64xi1>
  return %m : vector<1x2x64xi1>
}

// -----

// The pivot is dynamic but its inner dim is only partially covered, so the
// pivot's coordinate does not advance the lane index contiguously.
func.func @partial_inner_bound_rejected(%rows: index) -> vector<2x32xi1> {
  // CHECK-LABEL: func.func @partial_inner_bound_rejected
  // CHECK-NOT: vector.create_mask %{{.*}} : vector<64xi1>
  // CHECK: vector.create_mask %{{.*}}, %c16 : vector<2x32xi1>
  // CHECK: return
  %c16 = arith.constant 16 : index
  %m = vector.create_mask %rows, %c16 : vector<2x32xi1>
  return %m : vector<2x32xi1>
}

// -----

// Two dynamic bounds on dims of extent > 1: the inner one cannot be proven to
// cover its dim, so the region cannot be shown to be a prefix statically.
func.func @two_dynamic_nonunit_bounds_rejected(%rows: index,
                                               %cols: index) -> vector<2x32xi1> {
  // CHECK-LABEL: func.func @two_dynamic_nonunit_bounds_rejected
  // CHECK-NOT: vector.create_mask %{{.*}} : vector<64xi1>
  // CHECK: vector.create_mask %{{.*}}, %{{.*}} : vector<2x32xi1>
  // CHECK: return
  %m = vector.create_mask %rows, %cols : vector<2x32xi1>
  return %m : vector<2x32xi1>
}
