// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// Regression test for the `StaticSizedBuffer` generalization in
// NormalizeAtomicTemplate.h - checks if hfusion.store is decomposed by hfusion-normalize-ops
// as expected.
//
// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_and_expand_shape
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[GM:.*]], %[[UB:.*]] : memref<?x?xi32, strided<[?, 1], offset: ?>> to memref<?x?xi32>
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<1024xi32>
// CHECK: %[[DST:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[RES:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[LHS]], %{{.*}} : tensor<1024xi32>, tensor<1024xi32>) outs(%[[DST]] : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK: %[[SL:.*]] = tensor.extract_slice %[[RES]][0] [%{{.*}}] [1] : tensor<1024xi32> to tensor<?xi32>
// CHECK: %[[EX:.*]] = tensor.expand_shape %[[SL]]{{.*}} output_shape [%{{.*}}, %{{.*}}] : tensor<?xi32> into tensor<?x?xi32>
// CHECK: bufferization.materialize_in_destination %[[EX]] in writable %[[GM]] : (tensor<?x?xi32>, memref<?x?xi32, strided<[?, 1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_and_expand_shape(%arg0 : memref<?xi32> {tt.divisibility = 16 : i32}, %arg1 : tensor<1024xi32>, %row : index, %col : index, %off : index) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %subview = memref.subview %reinterpret_cast[%off] [%col] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [%row, %col] : memref<?xi32, strided<[1], offset: ?>> into memref<?x?xi32, strided<[?, 1], offset: ?>>
    %0 = bufferization.to_memref %arg1 : memref<1024xi32, strided<[1]>>
    %subview_0 = memref.subview %0[%off] [%col] [1] : memref<1024xi32, strided<[1]>> to memref<?xi32, strided<[1], offset: ?>>
    %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] output_shape [%row, %col] : memref<?xi32, strided<[1], offset: ?>> into memref<?x?xi32, strided<[?, 1], offset: ?>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%expand_shape_1 : memref<?x?xi32, strided<[?, 1], offset: ?>>) outs(%expand_shape : memref<?x?xi32, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----

// Atomic exchange with a tensor input and a GM destination. The
// XCHG input is already in the GM logical shape, so it must be materialized
// into GM directly.
//
// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_atomic_xchg_expand_shape
// CHECK: %[[UBT:.*]] = bufferization.to_tensor %arg3 restrict writable : memref<?x?xi32>
// CHECK: scope.scope : () -> () {
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[EXP:.*]], %{{.*}} : memref<?x?xi32, strided<[?, 1], offset: ?>> to memref<?x?xi32>
// CHECK: bufferization.materialize_in_destination %[[UBT]] in writable %[[EXP]] : (tensor<?x?xi32>, memref<?x?xi32, strided<[?, 1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: scope.return
// CHECK: } {hivm.allow_flatten, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_atomic_xchg_expand_shape(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32>, %arg3: memref<?x?xi32>, %row: index, %col: index, %off: index) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%off], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %subview = memref.subview %reinterpret_cast[%off] [%col] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [%row, %col] : memref<?xi32, strided<[1], offset: ?>> into memref<?x?xi32, strided<[?, 1], offset: ?>>
    %ub_tensor = bufferization.to_tensor %arg3 restrict writable : memref<?x?xi32>
    %ub = bufferization.to_memref %ub_tensor : memref<?x?xi32>
    hfusion.atomic_xchg ins(%ub : memref<?x?xi32>) outs(%expand_shape : memref<?x?xi32, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----

// CAS test
//
// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_atomic_cas_expand_shape
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[GM:.*]], %{{.*}} : memref<?x?xi32, strided<[?, 1], offset: ?>> to memref<?x?xi32>
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%{{.*}}, %{{.*}} : tensor<1024xi32>, tensor<1024xi32>) outs(%{{.*}} : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[SEL:.*]] = hfusion.select ins(%[[CMP]], %{{.*}}, %{{.*}} : tensor<1024xi1>, tensor<1024xi32>, tensor<1024xi32>) outs(%{{.*}} : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK: %[[SL:.*]] = tensor.extract_slice %[[SEL]][0] [%{{.*}}] [1] : tensor<1024xi32> to tensor<?xi32>
// CHECK: %[[EX:.*]] = tensor.expand_shape %[[SL]]{{.*}} output_shape [%{{.*}}, %{{.*}}] : tensor<?xi32> into tensor<?x?xi32>
// CHECK: bufferization.materialize_in_destination %[[EX]] in writable %[[GM]] : (tensor<?x?xi32>, memref<?x?xi32, strided<[?, 1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.atomic_cas
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_atomic_cas_expand_shape(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %row: index, %col: index, %off: index) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %gm_reinterpret = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %gm_subview = memref.subview %gm_reinterpret[%off] [%col] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %gm = memref.expand_shape %gm_subview [[0, 1]] output_shape [%row, %col] : memref<?xi32, strided<[1], offset: ?>> into memref<?x?xi32, strided<[?, 1], offset: ?>>
    %cmp_reinterpret = memref.reinterpret_cast %arg1 to offset: [%off], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %cmp_subview = memref.subview %cmp_reinterpret[%off] [%col] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %cmp = memref.expand_shape %cmp_subview [[0, 1]] output_shape [%row, %col] : memref<?xi32, strided<[1], offset: ?>> into memref<?x?xi32, strided<[?, 1], offset: ?>>
    %store_reinterpret = memref.reinterpret_cast %arg2 to offset: [%off], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %store_subview = memref.subview %store_reinterpret[%off] [%col] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %store = memref.expand_shape %store_subview [[0, 1]] output_shape [%row, %col] : memref<?xi32, strided<[1], offset: ?>> into memref<?x?xi32, strided<[?, 1], offset: ?>>
    hfusion.atomic_cas ins(%cmp, %store : memref<?x?xi32, strided<[?, 1], offset: ?>>, memref<?x?xi32, strided<[?, 1], offset: ?>>) outs(%gm : memref<?x?xi32, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----

// Test for nested subview
//
// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_and_nested_subview
// CHECK: %[[RHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<1024xi32>
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[GM:.*]], %{{.*}} : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<1024xi32>
// CHECK: %[[DST:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[RES:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[LHS]], %[[RHS]] : tensor<1024xi32>, tensor<1024xi32>) outs(%[[DST]] : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK: %[[SL:.*]] = tensor.extract_slice %[[RES]][0] [%{{.*}}] [1] : tensor<1024xi32> to tensor<?xi32>
// CHECK: bufferization.materialize_in_destination %[[SL]] in writable %[[GM]] : (tensor<?xi32>, memref<?xi32, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock {ordering = #hivm.ordering<unordered>} lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_and_nested_subview(%gm: memref<?xi32>, %src: memref<?xi32>, %off: index, %n: index) {
    %0 = memref.subview %gm[%off] [1024] [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %1 = memref.subview %0[0] [%n] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%src : memref<?xi32>) outs(%1 : memref<?xi32, strided<[1], offset: ?>>)
    return
  }
}
