// RUN: bishengir-opt %s -cse | FileCheck %s

func.func @physical_ids(%id: i64) {
  // CHECK-COUNT-2: hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, 31]
  hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, 31]
  hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, 31]
  // CHECK: hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, %arg0]
  hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, %id]
  // CHECK: hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE3>, 15]
  hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE3>, 15]
  // CHECK: hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_MTE2>, %arg0]
  hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_MTE2>, %id]
  return
}
