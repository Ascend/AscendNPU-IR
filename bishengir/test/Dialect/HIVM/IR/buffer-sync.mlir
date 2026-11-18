// RUN: bishengir-opt %s -cse | FileCheck %s

func.func @preserve_tokens(%id: i64) {
  // CHECK-COUNT-2: hivm.hir.get_buf[<PIPE_V>, 31, false]
  hivm.hir.get_buf[<PIPE_V>, 31, false]
  hivm.hir.get_buf[<PIPE_V>, 31, false]
  // CHECK-COUNT-2: hivm.hir.rls_buf[<PIPE_FIX>, %arg0, true]
  hivm.hir.rls_buf[<PIPE_FIX>, %id, true]
  hivm.hir.rls_buf[<PIPE_FIX>, %id, true]
  return
}
