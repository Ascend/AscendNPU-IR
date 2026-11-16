// RUN: split-file %s %t
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-llvm | FileCheck %t/a5.mlir
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-llvm | bishengir-translate -mlir-to-llvmir | FileCheck %t/a5.mlir --check-prefix=LLVM
// RUN: not bishengir-opt %t/a3.mlir -convert-hivm-to-llvm 2>&1 | FileCheck %s --check-prefix=REJECT-A3
// RUN: bishengir-opt %t/a6.mlir -convert-hivm-to-llvm | FileCheck %s --check-prefix=A6

//--- a5.mlir
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

//--- a3.mlir
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @rejected_get_buf() {
    // REJECT-A3: 'hivm.hir.get_buf' op is not natively supported on the enclosing target
    hivm.hir.get_buf[<PIPE_V>, 31, false]
    return
  }
}

// dav-920r1 is not surveyed for the sync intrinsics: the confirmed buffer
// rows stay open under the forward-compatibility policy and the ops lower.
//--- a6.mlir
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @open_rls_buf() {
    // A6: "hivm.intr.hivm.RLS.BUFI.mode"() <{buf_id = 31 : i64, mode = 0 : i64, pipe = 1 : i64}>
    hivm.hir.rls_buf[<PIPE_V>, 31, false]
    return
  }
}
