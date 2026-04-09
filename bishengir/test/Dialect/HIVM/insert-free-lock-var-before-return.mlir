// RUN: bishengir-opt %s --hivm-insert-free-lock-var-before-return -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @inserts_free_lock_before_return
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] :
// CHECK: hivm.hir.free_lock_var lock_var(%[[LOCK]] :
// CHECK-NEXT: return
func.func @inserts_free_lock_before_return() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
  hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
  return
}

// -----

// CHECK-LABEL: func.func @skips_without_hacc_entry
// CHECK: hivm.hir.sync_block_lock
// CHECK-NOT: hivm.hir.free_lock_var
// CHECK: return
func.func @skips_without_hacc_entry() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
  return
}

// -----

// CHECK-LABEL: func.func @skips_when_no_sync_block_lock
// CHECK-NOT: hivm.hir.free_lock_var
// CHECK: return
func.func @skips_when_no_sync_block_lock() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  return
}

// -----

// CHECK-LABEL: func.func @propagates_subblock_attr_to_free_lock_var
// CHECK: hivm.hir.sync_block_lock {hivm.sync_block_lock_with_subblock}
// CHECK: hivm.hir.free_lock_var {hivm.sync_block_lock_with_subblock}
func.func @propagates_subblock_attr_to_free_lock_var() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock {hivm.sync_block_lock_with_subblock} lock_var(%lock : memref<1xi64>)
  return
}

// -----

// CHECK-LABEL: func.func @inserts_for_each_distinct_lock_var
// CHECK-DAG: hivm.hir.free_lock_var lock_var(%{{.*}} : memref<1xi64>)
// CHECK-DAG: hivm.hir.free_lock_var lock_var(%{{.*}} : memref<1xi64>)
// CHECK: return
func.func @inserts_for_each_distinct_lock_var() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %a = hivm.hir.create_sync_block_lock : memref<1xi64>
  %b = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock lock_var(%a : memref<1xi64>)
  hivm.hir.sync_block_lock lock_var(%b : memref<1xi64>)
  return
}

// -----

// Mutex-style branches: only one of then/else runs, but return is shared — pass
// must emit free_lock_var for every lock_var seen on any sync_block_lock.
// CHECK-LABEL: func.func @inserts_after_scf_if_else_distinct_locks
// CHECK: scf.if
// CHECK: hivm.hir.sync_block_lock lock_var(
// CHECK: } else {
// CHECK: hivm.hir.sync_block_lock lock_var(
// CHECK-DAG: hivm.hir.free_lock_var lock_var(%{{.*}} : memref<1xi64>)
// CHECK-DAG: hivm.hir.free_lock_var lock_var(%{{.*}} : memref<1xi64>)
// CHECK: return
func.func @inserts_after_scf_if_else_distinct_locks(%cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %lock1 = hivm.hir.create_sync_block_lock : memref<1xi64>
  %lock2 = hivm.hir.create_sync_block_lock : memref<1xi64>
  scf.if %cond {
    hivm.hir.sync_block_lock lock_var(%lock1 : memref<1xi64>)
    hivm.hir.sync_block_unlock lock_var(%lock1 : memref<1xi64>)
    scf.yield
  } else {
    hivm.hir.sync_block_lock lock_var(%lock2 : memref<1xi64>)
    hivm.hir.sync_block_unlock lock_var(%lock2 : memref<1xi64>)
    scf.yield
  }
  return
}

