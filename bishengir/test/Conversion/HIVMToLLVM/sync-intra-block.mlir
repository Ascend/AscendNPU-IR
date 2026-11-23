// RUN: bishengir-opt %s -convert-hivm-to-llvm | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: llvm.func @constant_operand
  func.func @constant_operand() {
    %id = arith.constant 31 : i64
    // CHECK: hivm.SET.INTRA.BLOCKI.mode
    // CHECK: hivm.WAIT.INTRA.BLOCKI.mode
    hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, %id]
    hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, %id]
    return
  }
  // CHECK-LABEL: llvm.func @cube
  func.func @cube(%id: i64) {
    // CHECK: hivm.SET.INTRA.BLOCKI.mode
    // CHECK-SAME: sync_id = 31
    // CHECK-NOT: hivm.SET.INTRA.BLOCKI.mode
    // CHECK: hivm.WAIT.INTRA.BLOCKI.mode
    // CHECK-SAME: sync_id = 16
    // CHECK-NOT: hivm.WAIT.INTRA.BLOCKI.mode
    // CHECK: hivm.SET.INTRA.BLOCK.mode
    // CHECK: hivm.WAIT.INTRA.BLOCK.mode
    hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, 31]
    hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, 16]
    hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, %id]
    hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, %id]
    return
  }
  // CHECK-LABEL: llvm.func @vector
  func.func @vector(%id: i64) {
    // CHECK: hivm.SET.INTRA.BLOCKI.mode
    // CHECK-SAME: sync_id = 15
    // CHECK: hivm.WAIT.INTRA.BLOCKI.mode
    // CHECK-SAME: sync_id = 15
    // CHECK: hivm.SET.INTRA.BLOCK.mode
    // CHECK: hivm.WAIT.INTRA.BLOCK.mode
    hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE3>, 15]
    hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_MTE2>, 15]
    hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE3>, %id]
    hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_MTE2>, %id]
    return
  }
}
