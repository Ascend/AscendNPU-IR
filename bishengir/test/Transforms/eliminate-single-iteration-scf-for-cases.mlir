// RUN: bishengir-opt --eliminate-single-iteration-scf-for %s -split-input-file -o - | FileCheck %s

// Trip-count path: constant lower/upper/step => exactly one iteration.

module {
  func.func @trip_count_one(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %r = scf.for %iv = %c0 to %c1 step %c2 iter_args(%acc = %arg0) -> (i32) {
      %sum = arith.addi %acc, %arg0 : i32
      scf.yield %sum : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @trip_count_one
// CHECK-NOT: scf.for
// CHECK: arith.addi
// CHECK: return

// -----

// Trip-count path: two iterations => loop must remain.

module {
  func.func @trip_count_two(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c65 = arith.constant 65 : index
    %c64 = arith.constant 64 : index
    %r = scf.for %iv = %c0 to %c65 step %c64 iter_args(%acc = %arg0) -> (i32) {
      %sum = arith.addi %acc, %arg0 : i32
      scf.yield %sum : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @trip_count_two
// CHECK: scf.for

// -----

// Inner `scf.for` upper bound = `affine.min` over outer IV. ValueBounds alone
// does not collapse `%ub` to a constant; `AffineMinMaxValueBoundsCollector`
// records a tight interval (enumerate outer IV 0..2 step 1), so the header
// check proves exactly one inner iteration per outer trip. Inner eliminated.

module {
  func.func @nested_affine_min_upper(%init : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (index) {
      // i=0 -> min(4,1)=1 ; i=1 -> min(4,2)=2 ; inner always runs once with step 4.
      %ub = affine.min affine_map<(d0) -> (4, d0 + 1)>(%i)
      %v = scf.for %j = %c0 to %ub step %c4 iter_args(%b = %acc) -> (index) {
        scf.yield %b : index
      }
      scf.yield %v : index
    }
    return %r : index
  }
}

// CHECK-LABEL: func.func @nested_affine_min_upper
// CHECK-COUNT-1: scf.for
// CHECK-NOT: iter_args(%b

// -----

// Affine max + discrete bounds: outer IV 0,1 -> ub = max(d0,3) is always 3;
// inner 0..3 step 5 runs once.

module {
  func.func @nested_affine_max_upper(%init : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (index) {
      %ub = affine.max affine_map<(d0) -> (d0, 3)>(%i)
      %v = scf.for %j = %c0 to %ub step %c5 iter_args(%b = %acc) -> (index) {
        scf.yield %b : index
      }
      scf.yield %v : index
    }
    return %r : index
  }
}

// CHECK-LABEL: func.func @nested_affine_max_upper
// CHECK-COUNT-1: scf.for
// CHECK-NOT: iter_args(%b

// -----

// Both loops single-iteration by constant headers => both eliminated.

module {
  func.func @nested_both_constant_one_iter(%arg : i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %r = scf.for %i = %c0 to %c1 step %c2 iter_args(%a = %arg) -> (i32) {
      %inner = scf.for %j = %c0 to %c1 step %c2 iter_args(%b = %a) -> (i32) {
        %x = arith.addi %b, %arg : i32
        scf.yield %x : i32
      }
      scf.yield %inner : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @nested_both_constant_one_iter
// CHECK-NOT: scf.for
// CHECK: arith.addi
// CHECK: return

// -----

// Zero result `scf.for`: still eliminated when trip count is one.

module {
  func.func @zero_result_for() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k = arith.constant 0 : i32
    scf.for %iv = %c0 to %c1 step %c2 {
      %unused = arith.addi %k, %k : i32
    }
    return
  }
}

// CHECK-LABEL: func.func @zero_result_for
// CHECK-NOT: scf.for
// CHECK: return

// -----

// Multi-dim affine.min: collector falls back to ValueBounds-only; pipeline must
// not fail.

#map2d = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @affine_min_multi_dim(%a : index, %b : index) -> index {
    %m = affine.min #map2d(%a, %b)
    return %m : index
  }
}

// CHECK-LABEL: func.func @affine_min_multi_dim
// CHECK: affine.min

// -----

// Rescaled header: tiling feeds an affine.min remainder through an
// affine.apply before using it as the upper bound. The outer loop clamps the
// remainder to 64, so the rescaled bound is at most 32 and the inner loop runs
// exactly once, even though no bound here is a constant.

#clamp = affine_map<(d0) -> (-d0 + 2062, 64)>
#halve = affine_map<()[s0] -> (s0 floordiv 2)>

module {
  func.func @apply_rescaled_upper_bound(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2062 = arith.constant 2062 : index
    %r = scf.for %outer = %c0 to %c2062 step %c64 iter_args(%acc = %arg0) -> (i32) {
      %rem = affine.min #clamp(%outer)
      %half = affine.apply #halve()[%rem]
      %inner = scf.for %iv = %c0 to %half step %c64 iter_args(%a = %acc) -> (i32) {
        %sum = arith.addi %a, %arg0 : i32
        scf.yield %sum : i32
      }
      scf.yield %inner : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @apply_rescaled_upper_bound
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: arith.addi
// CHECK: return

// -----

// Same rescaling, but the clamp leaves room for two iterations of the inner
// loop (bound is at most 64 while the step is 32), so it must survive.

#clamp2 = affine_map<(d0) -> (-d0 + 2062, 128)>
#halve2 = affine_map<()[s0] -> (s0 floordiv 2)>

module {
  func.func @apply_rescaled_two_iterations(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c2062 = arith.constant 2062 : index
    %r = scf.for %outer = %c0 to %c2062 step %c128 iter_args(%acc = %arg0) -> (i32) {
      %rem = affine.min #clamp2(%outer)
      %half = affine.apply #halve2()[%rem]
      %inner = scf.for %iv = %c0 to %half step %c32 iter_args(%a = %acc) -> (i32) {
        %sum = arith.addi %a, %arg0 : i32
        scf.yield %sum : i32
      }
      scf.yield %inner : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @apply_rescaled_two_iterations
// CHECK: scf.for
// CHECK: scf.for

// -----

// `mod` is not monotone, so its interval cannot be read off the corners of the
// operand interval. The bound stays unknown and the loop survives.

#clamp3 = affine_map<(d0) -> (-d0 + 2062, 64)>
#wrap = affine_map<()[s0] -> (s0 mod 17)>

module {
  func.func @apply_mod_is_rejected(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2062 = arith.constant 2062 : index
    %r = scf.for %outer = %c0 to %c2062 step %c64 iter_args(%acc = %arg0) -> (i32) {
      %rem = affine.min #clamp3(%outer)
      %wrapped = affine.apply #wrap()[%rem]
      %inner = scf.for %iv = %c0 to %wrapped step %c64 iter_args(%a = %acc) -> (i32) {
        %sum = arith.addi %a, %arg0 : i32
        scf.yield %sum : i32
      }
      scf.yield %inner : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @apply_mod_is_rejected
// CHECK: scf.for
// CHECK: scf.for

// -----

// The rescaled bound may be zero when the remainder is small, so the loop can
// be empty and must not be promoted: `affine.apply` of a fully dynamic operand
// gives no interval to reason with.

#halve3 = affine_map<()[s0] -> (s0 floordiv 2)>

module {
  func.func @apply_unknown_operand(%n: index, %arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %half = affine.apply #halve3()[%n]
    %r = scf.for %iv = %c0 to %half step %c64 iter_args(%acc = %arg0) -> (i32) {
      %sum = arith.addi %acc, %arg0 : i32
      scf.yield %sum : i32
    }
    return %r : i32
  }
}

// CHECK-LABEL: func.func @apply_unknown_operand
// CHECK: scf.for

// -----

// A rescaled header whose operand is an induction variable. Only `affine.min`
// and `affine.max` results carry a recorded interval, so bounding this
// `affine.apply` has to fall back to the induction variable's own range.
// Tiling feeds induction variables straight into the maps it emits, and the
// slice cleanup turns an `affine.min` with a dominated candidate into exactly
// this shape, so a header like it must stay provable.

#iv_id = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @apply_over_induction_variable
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: arith.addi
// CHECK: return
module {
  func.func @apply_over_induction_variable(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %r = scf.for %i = %c1 to %c32 step %c1 iter_args(%a = %arg0) -> (i32) {
      %ub = affine.apply #iv_id(%i)
      %inner = scf.for %j = %c0 to %ub step %c64 iter_args(%b = %a) -> (i32) {
        %s = arith.addi %b, %arg0 : i32
        scf.yield %s : i32
      }
      scf.yield %inner : i32
    }
    return %r : i32
  }
}
