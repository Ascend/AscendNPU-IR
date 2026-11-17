// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file | FileCheck %s

// Spin-lock style CAS: snapshot GM into UB, then atomic_cas on the same GM word.
// The return snapshot must stay inside the same sync-block critical section as CAS.
//
// CHECK-LABEL: func.func @test_atomic_cas_return_snapshot_outer_lock
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %arg2, %{{.*}} : memref<1xi32> to memref<1xi32>
// CHECK-NOT: hfusion.atomic_cas
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_atomic_cas_return_snapshot_outer_lock(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %lock: memref<1xi32>,
      %cmp: memref<1xi32>,
      %new: memref<1xi32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %snapshot = memref.alloc() : memref<1xi32>
    memref.copy %lock, %snapshot : memref<1xi32> to memref<1xi32>
    %old = memref.load %snapshot[%c0] : memref<1xi32>
    %old_tensor = bufferization.to_tensor %snapshot restrict writable : memref<1xi32>
    %cond = arith.cmpi ne, %old, %old : i32
    hfusion.atomic_cas ins(%cmp, %new : memref<1xi32>, memref<1xi32>) outs(%lock : memref<1xi32>)
    scf.if %cond {
      %t = tensor.extract %old_tensor[%c0] : tensor<1xi32>
      %dead = arith.addi %t, %t : i32
    }
    return
  }
}

// -----

// Hardware add is left as hfusion.store. The GM snapshot used as the old
// value still has to be inside the same unordered critical section.
//
// CHECK-LABEL: func.func @test_hardware_atomic_add_return_snapshot
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %arg2, %{{.*}} : memref<1xf16> to memref<1xf16>
// CHECK: hfusion.store {{.*}}already_sync{{.*}} atomic_kind = #hfusion.atomic_kind<add>
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_hardware_atomic_add_return_snapshot(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %gm: memref<1xf16>, %val: memref<1xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %snapshot = memref.alloc() : memref<1xf16>
    memref.copy %gm, %snapshot : memref<1xf16> to memref<1xf16>
    %old = memref.load %snapshot[%c0] : memref<1xf16>
    %old_tensor = bufferization.to_tensor %snapshot restrict writable : memref<1xf16>
    %cond = arith.cmpf une, %old, %old : f16
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%val : memref<1xf16>) outs(%gm : memref<1xf16>)
    scf.if %cond {
      %t = tensor.extract %old_tensor[%c0] : tensor<1xf16>
      %dead = arith.addf %t, %t : f16
    }
    return
  }
}

// -----

// Exchange has no result. The preceding GM snapshot is the returned old value
// and must be inside the exchange critical section, without a nested lock.
//
// CHECK-LABEL: func.func @test_atomic_xchg_return_snapshot_outer_lock
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %arg2, %{{.*}} : memref<1xi32> to memref<1xi32>
// CHECK: scope.scope
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: scope.return
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_atomic_xchg_return_snapshot_outer_lock(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %gm: memref<1xi32>, %val: memref<1xi32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %snapshot = memref.alloc() : memref<1xi32>
    memref.copy %gm, %snapshot : memref<1xi32> to memref<1xi32>
    %old = memref.load %snapshot[%c0] : memref<1xi32>
    %old_tensor = bufferization.to_tensor %snapshot restrict writable : memref<1xi32>
    %cond = arith.cmpi ne, %old, %old : i32
    hfusion.atomic_xchg ins(%val : memref<1xi32>) outs(%gm : memref<1xi32>)
    scf.if %cond {
      %t = tensor.extract %old_tensor[%c0] : tensor<1xi32>
      %dead = arith.addi %t, %t : i32
    }
    return
  }
}
