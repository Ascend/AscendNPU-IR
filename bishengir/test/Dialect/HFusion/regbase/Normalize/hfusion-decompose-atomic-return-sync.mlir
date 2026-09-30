// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file | FileCheck %s

// Spin-lock style CAS: snapshot GM into UB, then atomic_cas on the same GM word.
// The return snapshot must stay inside the same sync-block critical section as CAS.
//
// CHECK-LABEL: func.func @test_atomic_cas_return_snapshot_outer_lock
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<1xi32> to memref<1xi32>
// CHECK-NOT: hfusion.atomic_cas
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
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

// CHECK-LABEL: func.func @test_linalg_cas_return_snapshot_outer_lock
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy
// CHECK: linalg.generic {{.*}} attrs = {GenericAtomicRMW = "cas", {{.*}} already_sync}
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_linalg_cas_return_snapshot_outer_lock(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %lock: memref<1xi32>,
      %cmp: memref<1xi32>,
      %new: memref<1xi32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %snapshot = memref.alloc() : memref<1xi32>
    memref.copy %lock, %snapshot : memref<1xi32> to memref<1xi32>
    %old = memref.load %snapshot[%c0] : memref<1xi32>
    %old_tensor = bufferization.to_tensor %snapshot restrict writable : memref<1xi32>
    %cond = arith.cmpi ne, %old, %old : i32
    linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]
    } ins(%lock, %cmp, %new : memref<1xi32>, memref<1xi32>, memref<1xi32>)
      outs(%lock : memref<1xi32>)
      attrs = {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
      ^bb0(%in: i32, %expected: i32, %desired: i32, %out: i32):
        %eq = arith.cmpi eq, %in, %expected : i32
        %sel = arith.select %eq, %desired, %in : i32
        linalg.yield %sel : i32
    }
    scf.if %cond {
      %t = tensor.extract %old_tensor[%c0] : tensor<1xi32>
      %dead = arith.addi %t, %t : i32
    }
    return
  }
}
