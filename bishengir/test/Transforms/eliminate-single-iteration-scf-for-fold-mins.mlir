// RUN: bishengir-opt \
// RUN:   --eliminate-single-iteration-scf-for=fold-dominated-mins=false %s \
// RUN:   -split-input-file -o - | FileCheck %s --check-prefix=OFF
// RUN: bishengir-opt --eliminate-single-iteration-scf-for %s -split-input-file \
// RUN:   -o - | FileCheck %s --check-prefix=ON
// RUN: bishengir-opt --eliminate-single-iteration-scf-for -cse %s \
// RUN:   -split-input-file -o - | FileCheck %s --check-prefix=ON-CSE

// A tail bound that tiling clamps twice: the outer `min` holds the remainder
// at or below 64, so halving it can never exceed 64 and the clamp inside the
// inner `min` is dead. The fold drops that candidate and leaves an
// `affine.apply`. It runs by default, so the ON prefix pins the fold and the
// OFF prefix, which disables it explicitly, pins the untransformed shape.

#clamp = affine_map<(d0) -> (-d0 + 2062, 64)>
#halve_clamped = affine_map<(d0)[s0] -> (64, -d0 + s0 floordiv 2)>

// OFF-LABEL: func.func @fold_dead_clamp
// OFF: affine.min #[[CLAMP:.*]](%{{.*}})
// OFF: affine.min #{{.*}}(%{{.*}})[%{{.*}}]
//
// ON-LABEL: func.func @fold_dead_clamp
// ON: affine.min #[[CLAMP2:.*]](%{{.*}})
// ON: affine.apply #{{.*}}(%{{.*}})[%{{.*}}]
// ON-NOT: affine.min
module {
  func.func @fold_dead_clamp(%arg0: index) -> index {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2062 = arith.constant 2062 : index
    %acc = arith.constant 0 : index
    %r = scf.for %outer = %c0 to %c2062 step %c64 iter_args(%a = %acc) -> (index) {
      %rem = affine.min #clamp(%outer)
      %tail = affine.min #halve_clamped(%c0)[%rem]
      %sum = arith.addi %a, %tail : index
      scf.yield %sum : index
    }
    return %r : index
  }
}

// -----

// Two unrelated dynamic candidates: neither is provably no greater than the
// other, so the `min` has to survive even with the fold enabled.

#pick = affine_map<()[s0, s1] -> (s0, s1)>

// OFF-LABEL: func.func @keep_incomparable_candidates
// OFF: affine.min
//
// ON-LABEL: func.func @keep_incomparable_candidates
// ON: affine.min
module {
  func.func @keep_incomparable_candidates(%a: index, %b: index) -> index {
    %m = affine.min #pick()[%a, %b]
    return %m : index
  }
}

// -----

// An `affine.apply` that already computes the surviving candidate, with the
// same map and operands, is left in place: the fold materializes its own and
// lets CSE merge the two rather than searching for one to reuse. The ON runs
// therefore pin one `affine.apply` only once CSE has run.

#halve = affine_map<()[s0] -> (s0 floordiv 2)>
#halve_clamped_1d = affine_map<()[s0] -> (64, s0 floordiv 2)>
#outer_clamp = affine_map<(d0) -> (-d0 + 2062, 64)>

// OFF-LABEL: func.func @reuse_existing_apply
// OFF: affine.apply
// OFF: affine.min #{{.*}}()[%{{.*}}]
//
// The outer clamp keeps both its candidates, so one `affine.min` stays in all
// three runs; only the inner one folds.
// ON-LABEL: func.func @reuse_existing_apply
// ON: affine.min
// ON-COUNT-2: affine.apply
//
// ON-CSE-LABEL: func.func @reuse_existing_apply
// ON-CSE: affine.min
// ON-CSE: affine.apply
// ON-CSE-NOT: affine.apply
module {
  func.func @reuse_existing_apply(%arg0: index) -> (index, index) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2062 = arith.constant 2062 : index
    %acc = arith.constant 0 : index
    %r:2 = scf.for %outer = %c0 to %c2062 step %c64
        iter_args(%a = %acc, %b = %acc) -> (index, index) {
      %rem = affine.min #outer_clamp(%outer)
      %half = affine.apply #halve()[%rem]
      %tail = affine.min #halve_clamped_1d()[%rem]
      %suma = arith.addi %a, %half : index
      %sumb = arith.addi %b, %tail : index
      scf.yield %suma, %sumb : index, index
    }
    return %r#0, %r#1 : index, index
  }
}
