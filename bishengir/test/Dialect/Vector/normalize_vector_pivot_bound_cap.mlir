// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// Review finding on !4571 [13]: the pivot dim's bound has to be capped at that
// dim's own extent before it becomes the flat length.
//
// `vector.create_mask` does cap the bound it is handed, but only against the
// length of the vector it builds.  After flattening that length is the product
// of every extent, which can be larger than the pivot dim's extent, so a bound
// in between the two would activate lanes the multi-dim mask leaves inactive --
// silently wrong data rather than a diagnostic.
//
// The shape that exposes it needs all three of: the pivot is the innermost dim
// (so nothing is multiplied in afterwards), a leading dim of extent > 1 whose
// bound is a constant 1 (so the flat length exceeds the pivot extent), and a
// pivot bound above the pivot extent.

// -----

// `<2x64xi1>` flattens to 128 lanes while the pivot dim holds 64, so a dynamic
// bound has to be capped at 64 and not left to `create_mask`'s own cap of 128.
// Lane 64 upwards must stay inactive: the mask activates dim-0 slice 0 only.
func.func @pivot_innermost_dynamic_bound(%b: index) -> vector<2x64xi1> {
  // CHECK-LABEL: func.func @pivot_innermost_dynamic_bound
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
  // CHECK: %[[LO:.*]] = arith.maxsi %arg0, %[[C0]]
  // CHECK: %[[HI:.*]] = arith.minsi %[[LO]], %[[C64]]
  // CHECK: %[[M:.*]] = vector.create_mask %[[HI]] : vector<128xi1>
  // CHECK: vector.shape_cast %[[M]]
  %c1 = arith.constant 1 : index
  %m = vector.create_mask %c1, %b : vector<2x64xi1>
  return %m : vector<2x64xi1>
}

// -----

// Same shape with a constant bound above the pivot extent.  Capping a constant
// needs no arithmetic, so the length is just the extent.
func.func @pivot_innermost_constant_over_extent() -> vector<2x64xi1> {
  // CHECK-LABEL: func.func @pivot_innermost_constant_over_extent
  // CHECK-NOT: arith.maxsi
  // CHECK-NOT: arith.minsi
  // CHECK: %[[C64:.*]] = arith.constant 64 : index
  // CHECK: vector.create_mask %[[C64]] : vector<128xi1>
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %m = vector.create_mask %c1, %c100 : vector<2x64xi1>
  return %m : vector<2x64xi1>
}

// -----

// Compatibility: a constant bound already within the pivot extent is handed
// through untouched, so this shape keeps the IR it had before the cap existed.
func.func @pivot_innermost_constant_in_range() -> vector<2x64xi1> {
  // CHECK-LABEL: func.func @pivot_innermost_constant_in_range
  // CHECK-NOT: arith.maxsi
  // CHECK-NOT: arith.minsi
  // CHECK-NOT: arith.muli
  // CHECK: %[[C40:.*]] = arith.constant 40 : index
  // CHECK: vector.create_mask %[[C40]] : vector<128xi1>
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %m = vector.create_mask %c1, %c40 : vector<2x64xi1>
  return %m : vector<2x64xi1>
}

// -----

// Compatibility: the pre-existing path, where the pivot is an outer dim and the
// inner extent is multiplied in, still clamps and still scales.
func.func @pivot_outer_dim_scales(%r: index) -> vector<2x32xi1> {
  // CHECK-LABEL: func.func @pivot_outer_dim_scales
  // CHECK: %[[LO:.*]] = arith.maxsi %arg0
  // CHECK: %[[HI:.*]] = arith.minsi %[[LO]]
  // CHECK: %[[N:.*]] = arith.muli %[[HI]]
  // CHECK: vector.create_mask %[[N]] : vector<64xi1>
  %c32 = arith.constant 32 : index
  %m = vector.create_mask %r, %c32 : vector<2x32xi1>
  return %m : vector<2x32xi1>
}
