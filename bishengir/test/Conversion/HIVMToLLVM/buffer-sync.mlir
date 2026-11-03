// RUN: bishengir-opt %s -convert-hivm-to-llvm | FileCheck %s
// RUN: bishengir-opt %s -convert-hivm-to-llvm | bishengir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: llvm.func @buffer_tokens
  func.func @buffer_tokens(%id: i64) {
    // CHECK: "hivm.intr.hivm.GET.BUFI.mode"() <{buf_id = 31 : i64, mode = 0 : i64, pipe = 1 : i64}>
    // LLVM-LABEL: define void @buffer_tokens
    // LLVM: call void @llvm.hivm.GET.BUFI.mode(i64 1, i64 31, i64 0)
    hivm.hir.get_buf[<PIPE_V>, 31, false]
    // CHECK: "hivm.intr.hivm.RLS.BUFI.mode"() <{buf_id = 0 : i64, mode = 1 : i64, pipe = 5 : i64}>
    // LLVM: call void @llvm.hivm.RLS.BUFI.mode(i64 5, i64 0, i64 1)
    hivm.hir.rls_buf[<PIPE_MTE3>, 0, true]
    // CHECK: "hivm.intr.hivm.GET.BUF.mode"(%arg0) <{mode = 1 : i64, pipe = 10 : i64}>
    // LLVM: call void @llvm.hivm.GET.BUF.mode(i64 10, i64 %{{.*}}, i64 1)
    hivm.hir.get_buf[<PIPE_FIX>, %id, true]
    // CHECK: "hivm.intr.hivm.RLS.BUF.mode"(%arg0) <{mode = 0 : i64, pipe = 2 : i64}>
    // LLVM: call void @llvm.hivm.RLS.BUF.mode(i64 2, i64 %{{.*}}, i64 0)
    hivm.hir.rls_buf[<PIPE_M>, %id, false]
    return
  }
  // CHECK-LABEL: llvm.func @constant_operand
  func.func @constant_operand() {
    %id = arith.constant 31 : i64
    // CHECK: "hivm.intr.hivm.GET.BUFI.mode"() <{buf_id = 31 : i64, mode = 0 : i64, pipe = 1 : i64}>
    // LLVM-LABEL: define void @constant_operand
    // LLVM: call void @llvm.hivm.GET.BUFI.mode(i64 1, i64 31, i64 0)
    hivm.hir.get_buf[<PIPE_V>, %id, false]
    // CHECK: "hivm.intr.hivm.RLS.BUFI.mode"() <{buf_id = 31 : i64, mode = 0 : i64, pipe = 1 : i64}>
    // LLVM: call void @llvm.hivm.RLS.BUFI.mode(i64 1, i64 31, i64 0)
    hivm.hir.rls_buf[<PIPE_V>, %id, false]
    return
  }
}
