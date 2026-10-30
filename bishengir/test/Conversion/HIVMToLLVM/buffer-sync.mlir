// RUN: bishengir-opt %s -convert-hivm-to-llvm | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: llvm.func @buffer_tokens
  func.func @buffer_tokens(%id: i64) {
    // CHECK: %[[ID:.*]] = llvm.mlir.constant(31 : i64)
    // CHECK: %[[PIPE:.*]] = llvm.mlir.constant(1 : i64)
    // CHECK: %[[MODE:.*]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @llvm.hivm.GET.BUFI.mode(%[[PIPE]], %[[ID]], %[[MODE]])
    hivm.hir.get_buf[<PIPE_V>, 31, false]
    // CHECK: llvm.call @llvm.hivm.RLS.BUFI.mode
    hivm.hir.rls_buf[<PIPE_MTE3>, 0, true]
    // CHECK: llvm.call @llvm.hivm.GET.BUF.mode(%{{.*}}, %arg0, %{{.*}})
    hivm.hir.get_buf[<PIPE_FIX>, %id, true]
    // CHECK: llvm.call @llvm.hivm.RLS.BUF.mode(%{{.*}}, %arg0, %{{.*}})
    hivm.hir.rls_buf[<PIPE_M>, %id, false]
    return
  }
  // CHECK-LABEL: llvm.func @constant_operand
  func.func @constant_operand() {
    %id = arith.constant 31 : i64
    // CHECK: llvm.call @llvm.hivm.GET.BUFI.mode
    hivm.hir.get_buf[<PIPE_V>, %id, false]
    // CHECK: llvm.call @llvm.hivm.RLS.BUFI.mode
    hivm.hir.rls_buf[<PIPE_V>, %id, false]
    return
  }
}
