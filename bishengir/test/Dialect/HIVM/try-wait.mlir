// RUN: bishengir-opt %s | FileCheck %s
// RUN: bishengir-opt %s -canonicalize -cse | FileCheck %s

module {
  // CHECK-LABEL: func.func @try_wait_static
  func.func @try_wait_static() -> i64 {
    // CHECK: %[[COUNTER:.*]] = hivm.hir.try_wait flag = 3, sync_mode = <intra_block> : i64
    %counter = hivm.hir.try_wait flag = 3, sync_mode = <intra_block> : i64
    return %counter : i64
  }

  // CHECK-LABEL: func.func @try_wait_dynamic
  func.func @try_wait_dynamic(%buffer_id: i64) -> i64 {
    // CHECK: %[[COUNTER:.*]] = hivm.hir.try_wait flag = %arg0, sync_mode = <intra_block> : i64
    %counter = hivm.hir.try_wait flag = %buffer_id, sync_mode = <intra_block> : i64
    return %counter : i64
  }
}

// Queries observe asynchronous hardware state. Even identical or unused queries
// must not be merged, speculated, or removed as pure arithmetic.
// CHECK-LABEL: func.func @repeated_queries
func.func @repeated_queries() -> (i64, i64) {
  // CHECK: %[[FIRST:.*]] = hivm.hir.try_wait flag = 0, sync_mode = <intra_block>
  // CHECK: %[[SECOND:.*]] = hivm.hir.try_wait flag = 0, sync_mode = <intra_block>
  // CHECK: hivm.hir.try_wait flag = 0, sync_mode = <intra_block>
  // CHECK: return %[[FIRST]], %[[SECOND]]
  %a = hivm.hir.try_wait flag = 0, sync_mode = <intra_block> : i64
  %b = hivm.hir.try_wait flag = 0, sync_mode = <intra_block> : i64
  %unused = hivm.hir.try_wait flag = 0, sync_mode = <intra_block> : i64
  return %a, %b : i64, i64
}

// Dynamic queries also observe asynchronous state and must survive CSE.
// CHECK-LABEL: func.func @repeated_intra_block_queries
func.func @repeated_intra_block_queries(%flag: i64) -> (i64, i64) {
  // CHECK: %[[FIRST:.*]] = hivm.hir.try_wait flag = %arg0, sync_mode = <intra_block>
  // CHECK: %[[SECOND:.*]] = hivm.hir.try_wait flag = %arg0, sync_mode = <intra_block>
  // CHECK: hivm.hir.try_wait flag = %arg0, sync_mode = <intra_block>
  // CHECK: return %[[FIRST]], %[[SECOND]]
  %a = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
  %b = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
  %unused = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
  return %a, %b : i64, i64
}
