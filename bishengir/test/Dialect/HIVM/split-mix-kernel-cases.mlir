// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -hivm-mark-tightly-coupled-buffer -hivm-hoist-tightly-coupled-alloc -hivm-split-mix-kernel -split-input-file | FileCheck %s
//
// New SplitMixKernel cases that must run on published builds.
// The original split-mix-kernel.mlir is skipped in published builds.
// Add later cases here as extra split-input sections.

// Preload scopes are kept as skeletons on both split sides. Their scalar
// results are loop counters that the enclosing control flow still reads.
// Stubbing them to 0 made AIC/AIV disagree on trip counts and deadlock.
// CHECK-LABEL: func.func @preload_scope_keeps_scalar_counter_mix_aic(
// CHECK:         %[[FOR:.*]]:2 = scf.for
// CHECK:           %[[CUBE:.*]] = scope.scope : () -> i32
// CHECK:             arith.addi
// CHECK:             scope.return
// CHECK:           %[[VEC:.*]] = scope.scope : () -> i32
// CHECK:           scf.yield %[[CUBE]], %[[VEC]]
// CHECK:         return %[[FOR]]#0
// CHECK-LABEL: func.func @preload_scope_keeps_scalar_counter_mix_aiv(
// CHECK:         %[[FOR:.*]]:2 = scf.for
// CHECK:           %[[CUBE:.*]] = scope.scope : () -> i32
// CHECK:             arith.addi
// CHECK:             scope.return
// CHECK:           %[[VEC:.*]] = scope.scope : () -> i32
// CHECK:           scf.yield %[[CUBE]], %[[VEC]]
// CHECK:         return %[[FOR]]#0
module {
  func.func @preload_scope_keeps_scalar_counter(%init: i32) -> i32
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %step = arith.constant 32 : i32
    %0:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%cube = %init, %vec = %init) -> (i32, i32) : i32 {
      %cube_out = scope.scope : () -> i32 {
        %add = arith.addi %cube, %step : i32
        scope.return %add : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      %vec_out = scope.scope : () -> i32 {
        %add = arith.addi %vec, %step : i32
        scope.return %add : i32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %cube_out, %vec_out : i32, i32
    }
    return %0#0 : i32
  }
}
