// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file | FileCheck %s

// The CAS is already between lock and unlock. Decompose it in that section
// and do not nest another lock.
//
// CHECK-LABEL: func.func @test_atomic_cas_return_already_inside_critical_region
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hivm.hir.create_sync_block_lock
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hfusion.atomic_cas
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_atomic_cas_return_already_inside_critical_region(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %gm: memref<1xi32>, %cmp: memref<1xi32>, %new: memref<1xi32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%lock : memref<1xi64>)
    %old = hfusion.atomic_cas ins(%cmp, %new : memref<1xi32>, memref<1xi32>) outs(%gm : memref<1xi32>) -> tensor<1xi32>
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%lock : memref<1xi64>)
    %extracted = tensor.extract %old[%c0] : tensor<1xi32>
    return
  }
}

// -----

// The open lock is in the parent block. The CAS inside the loop is still in
// that critical section.
//
// CHECK-LABEL: func.func @test_atomic_cas_return_inside_ancestor_critical_region
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hivm.hir.create_sync_block_lock
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hfusion.atomic_cas
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_atomic_cas_return_inside_ancestor_critical_region(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %gm: memref<1xi32>, %cmp: memref<1xi32>, %new: memref<1xi32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%lock : memref<1xi64>)
    scf.while : () -> () {
      %old = hfusion.atomic_cas ins(%cmp, %new : memref<1xi32>, memref<1xi32>) outs(%gm : memref<1xi32>) -> tensor<1xi32>
      %extracted = tensor.extract %old[%c0] : tensor<1xi32>
      %cond = arith.cmpi eq, %extracted, %extracted : i32
      scf.condition(%cond)
    } do {
      scf.yield
    }
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%lock : memref<1xi64>)
    return
  }
}

// -----

// Exchange already sits in a critical section. Do not nest another lock.
//
// CHECK-LABEL: func.func @test_atomic_xchg_already_inside_critical_region
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hivm.hir.create_sync_block_lock
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_atomic_xchg_already_inside_critical_region(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %gm: memref<1xi32>, %val: memref<1xi32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%lock : memref<1xi64>)
    hfusion.atomic_xchg ins(%val : memref<1xi32>) outs(%gm : memref<1xi32>)
    hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%lock : memref<1xi64>)
    return
  }
}
