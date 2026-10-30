// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @constant_range() {
  %id = arith.constant 32 : i64
  // expected-error@+1 {{buffer ID must be in [0, 31]}}
  hivm.hir.get_buf[<PIPE_V>, %id, false]
  return
}
// -----

func.func @range() {
  // expected-error@+1 {{buffer ID must be in [0, 31]}}
  hivm.hir.get_buf[<PIPE_V>, 32, false]
  return
}
// -----
func.func @negative() {
  // expected-error@+1 {{buffer ID must be in [0, 31]}}
  hivm.hir.rls_buf[<PIPE_FIX>, -1, true]
  return
}
// -----
func.func @pipe() {
  // expected-error@+1 {{unsupported buffer synchronization pipe}}
  hivm.hir.get_buf[<PIPE_ALL>, 0, false]
  return
}
// -----
func.func @all_pipe() {
  // expected-error@+1 {{unsupported buffer synchronization pipe}}
  hivm.hir.rls_buf[<PIPE_ALL>, 0, false]
  return
}
// -----
func.func @missing() {
  // expected-error@+1 {{exactly one static or dynamic buffer ID is required}}
  "hivm.hir.get_buf"() {pipe = #hivm.pipe<PIPE_V>, mode = false} : () -> ()
  return
}
// -----
func.func @duplicate(%id: i64) {
  // expected-error@+1 {{exactly one static or dynamic buffer ID is required}}
  "hivm.hir.rls_buf"(%id) {pipe = #hivm.pipe<PIPE_FIX>, mode = true, static_buf_id = 0 : i64} : (i64) -> ()
  return
}
