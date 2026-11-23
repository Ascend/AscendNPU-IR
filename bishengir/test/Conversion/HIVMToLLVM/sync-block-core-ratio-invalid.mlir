// RUN: bishengir-opt %s -split-input-file -verify-diagnostics \
// RUN:   -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-llvm

// The operation must specify its core even when the function is AIC.
func.func @unresolved_scalar_core() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
  // expected-error@+2 {{CV11 FFTS SET requires an explicit CUBE or VECTOR core type}}
  // expected-error@+1 {{failed to legalize operation 'hivm.hir.sync_block_set'}}
  hivm.hir.sync_block_set[<CUBE_OR_VECTOR>, <PIPE_S>, <PIPE_V>] flag = 0
  return
}

// -----

// A supported pipe does not make a combined core type valid for this SET.
func.func @combined_core() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // expected-error@+2 {{CV11 FFTS SET requires an explicit CUBE or VECTOR core type}}
  // expected-error@+1 {{failed to legalize operation 'hivm.hir.sync_block_set'}}
  hivm.hir.sync_block_set[<CUBE_AND_VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
  return
}

// -----

// PIPE_ALL is not a supported sender pipe for the CV11 FFTS fallback.
func.func @all_set_pipe() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
  // expected-error@+2 {{CV11 FFTS SET does not support PIPE_ALL}}
  // expected-error@+1 {{failed to legalize operation 'hivm.hir.sync_block_set'}}
  hivm.hir.sync_block_set[<CUBE>, <PIPE_ALL>, <PIPE_S>] flag = 0
  return
}

// -----

// PIPE_ALL must not be silently mapped to PIPE_S on the receiver.
func.func @all_wait_pipe() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
  // expected-error@+2 {{CV11 FFTS WAIT does not support PIPE_ALL}}
  // expected-error@+1 {{failed to legalize operation 'hivm.hir.sync_block_wait'}}
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_ALL>] flag = 0
  return
}
