// RUN: bishengir-opt --hivm-verify-capability=strict=false --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @report_mode
// CHECK: hivm.hir.try_wait
// CHECK: hivm.hir.get_buf
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @report_mode(%flag: i64) -> i64 {
    // expected-remark @below {{result is native; target is Ascend950 (dav-c310); coverage is partial}}
    %0 = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    // expected-remark @below {{result is native; target is Ascend950 (dav-c310); coverage is partial}}
    hivm.hir.get_buf[<PIPE_V>, 3, false]
    return %0 : i64
  }
}
