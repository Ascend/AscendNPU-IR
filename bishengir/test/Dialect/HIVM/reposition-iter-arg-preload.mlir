// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true enable-preload=true" -split-input-file -verify-diagnostics

// A preload loop must not leave one iter_arg in scopes of different stages.
func.func @iter_arg_two_preload_scopes(%g: memref<?xi64>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  // expected-error @+1 {{iter_arg #0 is used by scopes with different hivm.preload_num}}
  scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %c0_i64) -> (i64) {
    %a = scope.scope : () -> i64 {
      %n = arith.addi %p, %c1_i64 : i64
      scope.return %n : i64
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    %b = scope.scope : () -> i64 {
      %n = arith.addi %p, %c1_i64 : i64
      scope.return %n : i64
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %a : i64
  }
  return
}
