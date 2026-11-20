// RUN: bishengir-opt %s --hfusion-normalize-ops -split-input-file -verify-diagnostics | FileCheck %s

// A3 runs the same hfusion atomic decompose as regbase. Locks are unordered.
// Hardware add stays an atomic store. atomic_rmw with no result becomes that
// store, so a returned GM snapshot is locked by the store patterns.

// CHECK-LABEL: func.func @a3_atomic_cas_result
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: hfusion.atomic_cas
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: tensor.extract
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_atomic_cas_result(
      %gm: memref<1xi32>, %cmp: memref<1xi32>, %new: memref<1xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %old = hfusion.atomic_cas ins(%cmp, %new : memref<1xi32>, memref<1xi32>) outs(%gm : memref<1xi32>) -> tensor<1xi32>
    %extracted = tensor.extract %old[%c0] : tensor<1xi32>
    return %extracted : i32
  }
}

// -----

// CHECK-LABEL: func.func @a3_atomic_store_or
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK: #hfusion.binary_fn<vor>
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
// CHECK-NOT: hfusion.store
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_atomic_store_or(%gm: memref<16xi32>, %src: memref<16xi32>) {
    hfusion.store {atomic_kind = #hfusion.atomic_kind<or>} ins(%src : memref<16xi32>) outs(%gm : memref<16xi32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @a3_atomic_store_add
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK: hfusion.store
// CHECK-SAME: atomic_kind = #hfusion.atomic_kind<add>
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_atomic_store_add(%gm: memref<16xi32>, %src: memref<16xi32>) {
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%src : memref<16xi32>) outs(%gm : memref<16xi32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @a3_atomic_rmw_add_unchanged
// CHECK-NOT: hivm.hir.sync_block_lock
// CHECK-NOT: hfusion.atomic_rmw
// CHECK: hfusion.store
// CHECK-SAME: atomic_kind = #hfusion.atomic_kind<add>
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_atomic_rmw_add_unchanged(%gm: memref<16xi32>, %src: memref<16xi32>) {
    hfusion.atomic_rmw ins(%src : memref<16xi32>) outs(%gm : memref<16xi32>) atomic_kind = <add>
    return
  }
}

// -----

// Hardware add with a returned snapshot stays a store. The copy shares its lock.
// CHECK-LABEL: func.func @a3_atomic_rmw_add_i32_snapshot
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK: memref.copy
// CHECK-NOT: hfusion.atomic_rmw
// CHECK: hfusion.store
// CHECK-SAME: already_sync
// CHECK-SAME: atomic_kind = #hfusion.atomic_kind<add>
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
// CHECK: tensor.extract
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_atomic_rmw_add_i32_snapshot(
      %gm: memref<1xi32>, %src: memref<1xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1xi32>
    memref.copy %gm, %alloc : memref<1xi32> to memref<1xi32>
    %old = bufferization.to_tensor %alloc restrict writable : memref<1xi32>
    hfusion.atomic_rmw ins(%src : memref<1xi32>) outs(%gm : memref<1xi32>) atomic_kind = <add>
    %extracted = tensor.extract %old[%c0] : tensor<1xi32>
    return %extracted : i32
  }
}

// -----

// i64 add is not a hardware atomic. The preceding copy is the returned old
// value and has to sit in the same unordered section as the software update.
// CHECK-LABEL: func.func @a3_atomic_rmw_add_returned_snapshot
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}
// CHECK: memref.copy
// CHECK-NOT: hfusion.atomic_rmw
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>}
// CHECK: tensor.extract
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_atomic_rmw_add_returned_snapshot(
      %gm: memref<1xi64>, %src: memref<1xi64>) -> i64 {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1xi64>
    memref.copy %gm, %alloc : memref<1xi64> to memref<1xi64>
    %old = bufferization.to_tensor %alloc restrict writable : memref<1xi64>
    hfusion.atomic_rmw ins(%src : memref<1xi64>) outs(%gm : memref<1xi64>) atomic_kind = <add>
    %extracted = tensor.extract %old[%c0] : tensor<1xi64>
    return %extracted : i64
  }
}
