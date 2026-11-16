// RUN: bishengir-opt %s -hivm-sync-block-hoisting -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @hoist_sync_in_for_loop(
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_unlock
func.func @hoist_sync_in_for_loop(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
    scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %0 = hivm.hir.create_sync_block_lock : memref<1xi64>
      hivm.hir.sync_block_lock lock_var(%0 : memref<1xi64>)
      %alloc = memref.alloc() : memref<16xi32>
      hivm.hir.vadd ins(%arg0, %arg1 : memref<16xi32>, memref<16xi32>) outs(%alloc : memref<16xi32>)
      hivm.hir.sync_block_unlock lock_var(%0 : memref<1xi64>)
    }
  }
  return
}

// CHECK-LABEL: func.func @hoist_sync_in_while_loop(
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock
// CHECK: scf.while
// CHECK: scf.while
// CHECK: hivm.hir.sync_block_unlock
func.func @hoist_sync_in_while_loop(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:2 = scf.while (%arg2 = %c0_i32, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
    %1 = arith.cmpi slt, %arg2, %arg3 : i32
    scf.condition(%1) %arg2, %arg3 : i32, i32
  } do {
  ^bb0(%arg2: i32, %arg3: i32):
    %1:2 = scf.while (%arg4 = %arg2, %arg5 = %arg3) : (i32, i32) -> (i32, i32) {
      %3 = arith.cmpi slt, %arg4, %arg5 : i32
      scf.condition(%3) %arg4, %arg5 : i32, i32
    } do {
    ^bb0(%arg4: i32, %arg5: i32):
      %3 = hivm.hir.create_sync_block_lock : memref<1xi64>
      hivm.hir.sync_block_lock lock_var(%3 : memref<1xi64>)
      %alloc = memref.alloc() : memref<16xi32>
      hivm.hir.vadd ins(%arg0, %arg1 : memref<16xi32>, memref<16xi32>) outs(%alloc : memref<16xi32>)
      hivm.hir.sync_block_unlock lock_var(%3 : memref<1xi64>)
      %4 = arith.addi %arg4, %c1_i32 : i32
      scf.yield %4, %arg5 : i32, i32
    }
    %2 = arith.addi %arg2, %c1_i32 : i32
    scf.yield %2, %arg3 : i32, i32
  }
  return
}


// CHECK-LABEL: func.func @hoist_sync_in_for_loop_if_statement(
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock
// CHECK: scf.for
// CHECK: scf.if
// CHECK: hivm.hir.sync_block_unlock
func.func @hoist_sync_in_for_loop_if_statement(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: i1) {
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %true = arith.constant 1 : i1
  scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
    scf.if %arg2 {
      %0 = hivm.hir.create_sync_block_lock : memref<1xi64>
      hivm.hir.sync_block_lock lock_var(%0 : memref<1xi64>)
      %alloc = memref.alloc() : memref<16xi32>
      hivm.hir.vadd ins(%arg0, %arg1 : memref<16xi32>, memref<16xi32>) outs(%alloc : memref<16xi32>)
      hivm.hir.sync_block_unlock lock_var(%0 : memref<1xi64>)
    }
  }
  return
}

// -----

// Unordered lock (native $ordering): create may hoist; lock/unlock stay in loop.
// CHECK-LABEL: func.func @unordered_lock_stays_in_for_loop(
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
func.func @unordered_lock_stays_in_for_loop(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
    %0 = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
    %alloc = memref.alloc() : memref<16xi32>
    hivm.hir.vadd ins(%arg0, %arg1 : memref<16xi32>, memref<16xi32>) outs(%alloc : memref<16xi32>)
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
  }
  return
}

// -----

// Unordered sections share one lock variable. Each section keeps its own
// lock/unlock. Ordered locks stay on their own variables.
//
// CHECK-LABEL: func.func @unordered_locks_share_one_var(
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK-NOT: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ORDERED:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[ORDERED]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[ORDERED]] : memref<1xi64>)
func.func @unordered_locks_share_one_var() {
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

// After the shared variable is substituted, a lock already inside that
// critical section is removed with its unlock.
//
// CHECK-LABEL: func.func @unordered_nested_lock_is_removed(
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK-NOT: hivm.hir.create_sync_block_lock
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hivm.hir.sync_block_unlock
func.func @unordered_nested_lock_is_removed(%arg0: i1) {
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

// -----

// Creates inside the loop are hoisted, then folded onto one variable.
// The two critical sections stay inside the loop.
//
// CHECK-LABEL: func.func @unordered_locks_in_loop_share_one_var(
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK-NOT: hivm.hir.create_sync_block_lock
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
func.func @unordered_locks_in_loop_share_one_var() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %0 = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%0 : memref<1xi64>)
    %1 = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%1 : memref<1xi64>)
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%1 : memref<1xi64>)
  }
  return
}
