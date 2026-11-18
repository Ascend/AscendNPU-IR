// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @constant_vector_range() {
  %id = arith.constant 16 : i64
  // expected-error@+1 {{sync ID out of range for the core type}}
  hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_S>, %id]
  return
}
// -----

func.func @vector_range() {
  // expected-error@+1 {{sync ID out of range for the core type}}
  hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE3>, 16]
  return
}
// -----
func.func @cube_range() {
  // expected-error@+1 {{sync ID out of range for the core type}}
  hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, 32]
  return
}
// -----
func.func @negative() {
  // expected-error@+1 {{sync ID out of range for the core type}}
  hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, -1]
  return
}
// -----
func.func @wrong_pipe() {
  // expected-error@+1 {{pipe must match the core type}}
  hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_FIX>, 0]
  return
}
// -----
func.func @ambiguous_core() {
  // expected-error@+1 {{requires an explicit CUBE or VECTOR core type}}
  hivm.hir.sync_intra_block_wait[<CUBE_OR_VECTOR>, <PIPE_S>, 0]
  return
}
// -----
func.func @missing_id() {
  // expected-error@+1 {{exactly one static or dynamic sync ID is required}}
  "hivm.hir.sync_intra_block_set"() {tcore_type = #hivm.tcore_type<CUBE>, pipe = #hivm.pipe<PIPE_FIX>} : () -> ()
  return
}
// -----
func.func @two_ids(%id: i64) {
  // expected-error@+1 {{exactly one static or dynamic sync ID is required}}
  "hivm.hir.sync_intra_block_wait"(%id) {tcore_type = #hivm.tcore_type<VECTOR>, pipe = #hivm.pipe<PIPE_S>, static_sync_id = 0 : i64} : (i64) -> ()
  return
}
