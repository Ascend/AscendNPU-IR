// RUN: bishengir-opt %s -hivm-sync-block-hoisting -hivm-insert-infer-sync-block-lock-num-and-init-func -split-input-file | FileCheck %s

// Two unordered creates become one, so the host callback reports one
// unordered lock (1 << 32) and one ordered lock (8 bytes).
//
// CHECK: func.func @unordered_and_ordered_infer_sync_block_lock_num_function() -> i64
// CHECK: %[[NUM:.*]] = arith.constant 4294967304 : i64
// CHECK: return %[[NUM]]
// CHECK-LABEL: func.func @unordered_and_ordered(
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock lock_var
// CHECK-NOT: hivm.hir.create_sync_block_lock
func.func @unordered_and_ordered() {
  %0 = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
  hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
  %1 = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%1 : memref<1xi64>)
  hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%1 : memref<1xi64>)
  %2 = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock lock_var(%2 : memref<1xi64>)
  hivm.hir.sync_block_unlock lock_var(%2 : memref<1xi64>)
  return
}

// -----

// The nested unordered pair is gone, so the callback still reports one lock.
//
// CHECK: func.func @unordered_nested_infer_sync_block_lock_num_function() -> i64
// CHECK: %[[NUM:.*]] = arith.constant 4294967296 : i64
// CHECK: return %[[NUM]]
// CHECK-LABEL: func.func @unordered_nested(
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
// CHECK-NOT: hivm.hir.create_sync_block_lock
func.func @unordered_nested(%arg0: i1) {
  %0 = hivm.hir.create_sync_block_lock : memref<1xi64>
  hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
  scf.if %arg0 {
    %1 = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%1 : memref<1xi64>)
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%1 : memref<1xi64>)
    scf.yield
  }
  hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
  return
}
