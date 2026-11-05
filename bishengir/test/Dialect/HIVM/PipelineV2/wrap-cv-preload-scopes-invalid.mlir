// RUN: bishengir-opt %s -wrap-cv-preload-scopes -split-input-file -verify-diagnostics

func.func @inconsistent_max(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () { scope.return } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
    // expected-error@+1 {{expected consistent max_preload_num}}
    scope.scope : () -> () { scope.return } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

func.func @noncontiguous(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () { scope.return } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
    scope.scope : () -> () { scope.return } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
    // expected-error@+1 {{expected scopes ordered by descending preload_num with contiguous stages}}
    scope.scope : () -> () { scope.return } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

func.func @missing_stage(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    // expected-error@+1 {{expected valid preload_num and max_preload_num}}
    scope.scope : () -> () { scope.return }
  } {hivm.cv_pipelined_loop}
  return
}
