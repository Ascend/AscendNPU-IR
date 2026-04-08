// RUN: bishengir-opt --eliminate-single-iteration-scf-for="dump-affine-minmax-bounds=true" %s -o - | FileCheck --check-prefixes=CHECK,CHECK-BOUNDS %s

// Bounds dump is printed at the start of stdout; CHECK-BOUNDS must run before any
// CHECK-LABEL so FileCheck's scan cursor is still at the beginning of the tool output.

// CHECK-BOUNDS: === AFFINE_MINMAX_BOUNDS DUMP START ===
// CHECK-BOUNDS-DAG: AFFINE_MINMAX_BOUNDS func=discrete_iv_negative_step_for{{.*}} LB=1 UB=5
// CHECK-BOUNDS-DAG: AFFINE_MINMAX_BOUNDS func=discrete_max_two_results{{.*}} LB=5 UB=10
// CHECK-BOUNDS-DAG: AFFINE_MINMAX_BOUNDS func=discrete_min_ceildiv_mod{{.*}} LB=0 UB=6
// CHECK-BOUNDS-DAG: AFFINE_MINMAX_BOUNDS func=discrete_min_two_results_floordiv{{.*}} LB=0 UB=1
// CHECK-BOUNDS-DAG: AFFINE_MINMAX_BOUNDS func=header_proof_interval_upper{{.*}} LB=1 UB=4
// CHECK-BOUNDS: === AFFINE_MINMAX_BOUNDS DUMP END ===

// Coverage for `AffineMinMaxValueBoundsCollector` + `provesSingleIterationScfFor`:
// trip-count (positive/negative step), discrete `affine.min`/`affine.max` over IVs,
// multi-result affine maps (floordiv/ceilDiv/mod), multi-dim fallback, header proof
// via interval bounds on loop operands.

// -----

// `computeTripCountFromHeader` + negative step: single iteration (lb=4, ub=3, step=-1).

module {
  func.func @trip_count_neg_step_one(%arg : i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %cm1 = arith.constant -1 : index
    %r = scf.for %iv = %c4 to %c3 step %cm1 iter_args(%a = %arg) -> (i32) {
      %x = arith.addi %a, %c0 : i32
      scf.yield %x : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @trip_count_neg_step_one
// CHECK-NOT: scf.for

// -----

// Discrete `affine.min` with two map results -> min(floordiv, linear) over IV samples.

#map_min_fd = affine_map<(d0) -> (d0 floordiv 2, d0 + 1)>

module {
  func.func @discrete_min_two_results_floordiv(%init : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %init) -> (index) {
      %m = affine.min #map_min_fd(%i)
      scf.yield %m : index
    }
    return %r : index
  }
}

// CHECK-LABEL: func.func @discrete_min_two_results_floordiv
// CHECK: affine.min

// -----

// Discrete `affine.min` with ceildiv + mod (evalAffineExprWithDims CeilDiv/Mod paths).

#map_min_ceil_mod = affine_map<(d0) -> (d0 ceildiv 3, d0 mod 7)>

module {
  func.func @discrete_min_ceildiv_mod(%init : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %init) -> (index) {
      %m = affine.min #map_min_ceil_mod(%i)
      scf.yield %m : index
    }
    return %r : index
  }
}

// CHECK-LABEL: func.func @discrete_min_ceildiv_mod
// CHECK: affine.min

// -----

// Discrete `affine.max` with two results (isMin=false reduction uses max).

#map_max_two = affine_map<(d0) -> (d0 - 1, 10 - d0)>

module {
  func.func @discrete_max_two_results(%init : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %init) -> (index) {
      %m = affine.max #map_max_two(%i)
      scf.yield %m : index
    }
    return %r : index
  }
}

// CHECK-LABEL: func.func @discrete_max_two_results
// CHECK: affine.max

// -----

// Negative-step `scf.for` IV in discrete enumeration + `affine.min`.

#map_min_simple = affine_map<(d0) -> (d0, 5)>

module {
  func.func @discrete_iv_negative_step_for(%init : index) -> index {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %cm2 = arith.constant -2 : index
    %r = scf.for %i = %c5 to %c0 step %cm2 iter_args(%a = %init) -> (index) {
      %m = affine.min #map_min_simple(%i)
      scf.yield %m : index
    }
    return %r : index
  }
}

// CHECK-LABEL: func.func @discrete_iv_negative_step_for
// CHECK: affine.min

// -----

// Multi-dim `affine.min` -> `tryRecordConstantBounds` only (no discrete IV path).

#map2d = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @multi_dim_min_fallback(%x : index, %y : index) -> index {
    %m = affine.min #map2d(%x, %y)
    return %m : index
  }
}

// CHECK-LABEL: func.func @multi_dim_min_fallback
// CHECK: affine.min

// -----

// `provesSingleIterationViaAffineMinMaxHeader` with step>0 and upper bound an
// `affine.min` interval (not a single constant): inner eliminated, outer kept.

module {
  func.func @header_proof_interval_upper(%init : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (index) {
      %ub = affine.min affine_map<(d0) -> (4, d0 + 1)>(%i)
      %v = scf.for %j = %c0 to %ub step %c4 iter_args(%b = %acc) -> (index) {
        scf.yield %b : index
      }
      scf.yield %v : index
    }
    return %out : index
  }
}

// CHECK-LABEL: func.func @header_proof_interval_upper
// CHECK-COUNT-1: scf.for
// CHECK-NOT: iter_args(%b

// -----

// Constant trip count > 1 => loop must remain (trip-count path rejects elimination).

module {
  func.func @trip_count_two_keeps_loop(%arg : i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %iv = %c0 to %c3 step %c2 iter_args(%a = %arg) -> (i32) {
      scf.yield %a : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @trip_count_two_keeps_loop
// CHECK: scf.for
