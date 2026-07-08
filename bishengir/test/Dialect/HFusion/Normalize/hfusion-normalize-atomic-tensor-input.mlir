// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_no_return
// CHECK: scope.scope : () -> () {
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[GM:.*]], %[[TMP:.*]] : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>> to memref<1x1x4x4xi32>
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %[[GM]] : (tensor<1x1x4x4xi32>, memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>) -> ()
// CHECK-NOT: memref.copy %[[TMP]], %{{.*}}
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_no_return(%arg0: memref<?xi32>, %arg1: tensor<1x1x4x4xi32>) {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [1, 1, 4, 4], strides: [16, 16, 4, 1] : memref<?xi32> to memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>
    hfusion.atomic_xchg ins(%arg1 : tensor<1x1x4x4xi32>) outs(%reinterpret_cast : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_with_return
// CHECK: %[[OLD:.*]] = scope.scope : () -> tensor<4x4xi32> {
// CHECK: memref.copy %[[GM:.*]], %[[TMP:.*]] : memref<4x4xi32> to memref<4x4xi32>
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %[[GM]] : (tensor<4x4xi32>, memref<4x4xi32>) -> ()
// CHECK: scope.return %{{.*}} : tensor<4x4xi32>
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_with_return(%arg0: memref<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %old = hfusion.atomic_xchg ins(%arg1 : tensor<4x4xi32>) outs(%arg0 : memref<4x4xi32>) -> tensor<4x4xi32>
    return %old : tensor<4x4xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_4d_subview
// CHECK: scope.scope : () -> () {
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>> to memref<1x1x4x4xi32>
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<1x1x4x4xi32>, memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>) -> ()
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_4d_subview(%arg0: memref<?xi32>, %arg1: tensor<1x1x4x4xi32>) {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [4, 4], strides: [4, 1] : memref<?xi32> to memref<4x4xi32, strided<[4, 1], offset: ?>>
    %expand_shape = memref.expand_shape %reinterpret_cast [[0, 1, 2], [3]] output_shape [1, 1, 4, 4] : memref<4x4xi32, strided<[4, 1], offset: ?>> into memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>
    hfusion.atomic_xchg ins(%arg1 : tensor<1x1x4x4xi32>) outs(%expand_shape : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_dyn_subview_with_return
// CHECK: %[[OLD:.*]] = scope.scope : () -> tensor<?x?xi32> {
// CHECK: memref.copy %[[GM:.*]], %[[TMP:.*]] : memref<?x?xi32, strided<[512, 1], offset: ?>> to memref<?x?xi32, strided<[256, 1]>>
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %[[GM]] : (tensor<?x?xi32>, memref<?x?xi32, strided<[512, 1], offset: ?>>) -> ()
// CHECK: %{{.*}} = bufferization.to_tensor %[[TMP]] restrict writable : memref<?x?xi32, strided<[256, 1]>>
// CHECK: scope.return %{{.*}} : tensor<?x?xi32>
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_dyn_subview_with_return(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?xi32> {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg4], sizes: [12, 256], strides: [512, 1] : memref<?xi32> to memref<12x256xi32, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<12x256xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<12x256xi32>
    %dyn_tensor = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [1, 1] : tensor<12x256xi32> to tensor<?x?xi32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi32, strided<[512, 1], offset: ?>> to memref<?x?xi32, strided<[512, 1], offset: ?>>
    %old = hfusion.atomic_xchg ins(%dyn_tensor : tensor<?x?xi32>) outs(%subview : memref<?x?xi32, strided<[512, 1], offset: ?>>) -> tensor<?x?xi32>
    return %old : tensor<?x?xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_tensor_ins
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<4xi16>, memref<4xi16>) -> ()
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<4xi16>, memref<4xi16>) -> ()
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%{{.*}}, %{{.*}} : tensor<4xi16>, tensor<4xi16>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK: %[[SEL:.*]] = hfusion.select ins(%[[CMP]], %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi16>, tensor<4xi16>) outs(%{{.*}} : tensor<4xi16>) -> tensor<4xi16>
// CHECK: bufferization.materialize_in_destination %[[SEL]] in writable %{{.*}} : (tensor<4xi16>, memref<4xi16>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_tensor_ins(%arg0: memref<4xi16>, %arg1: tensor<4xi16>, %arg2: tensor<4xi16>) {
    hfusion.atomic_cas ins(%arg1, %arg2 : tensor<4xi16>, tensor<4xi16>) outs(%arg0 : memref<4xi16>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_fold_to_memref_4d_subview
// CHECK: scope.scope : () -> () {
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>> to memref<1x1x4x4xi32>
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %{{.*}} : (tensor<1x1x4x4xi32>, memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>) -> ()
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_fold_to_memref_4d_subview(%arg0: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x2x4x4xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2x2x4x4xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<2x2x4x4xi32> to tensor<4x4xi32>
    %expanded_3 = tensor.expand_shape %extracted_slice [[0, 1, 2], [3]] output_shape [1, 1, 4, 4] : tensor<4x4xi32> into tensor<1x1x4x4xi32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [4, 4], strides: [4, 1] : memref<?xi32> to memref<4x4xi32, strided<[4, 1], offset: ?>>
    %expand_shape = memref.expand_shape %reinterpret_cast_2 [[0, 1, 2], [3]] output_shape [1, 1, 4, 4] : memref<4x4xi32, strided<[4, 1], offset: ?>> into memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>
    %3 = bufferization.to_memref %expanded_3 : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>
    hfusion.atomic_xchg ins(%3 : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>) outs(%expand_shape : memref<1x1x4x4xi32, strided<[16, 16, 4, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_fold_to_memref_dyn_subview
// CHECK: %{{.*}} = scope.scope : () -> tensor<?x?xi32> {
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %{{.*}} : (tensor<?x?xi32>, memref<?x?xi32, strided<[512, 1], offset: ?>>) -> ()
// CHECK: scope.return %{{.*}} : tensor<?x?xi32>
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_fold_to_memref_dyn_subview(%arg0: memref<?xi32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<?x?xi32> {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg3], sizes: [12, 256], strides: [512, 1] : memref<?xi32> to memref<12x256xi32, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<12x256xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<12x256xi32>
    %dyn_tensor = tensor.extract_slice %0[0, 0] [%arg1, %arg2] [1, 1] : tensor<12x256xi32> to tensor<?x?xi32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%arg1, %arg2] [1, 1] : memref<12x256xi32, strided<[512, 1], offset: ?>> to memref<?x?xi32, strided<[512, 1], offset: ?>>
    %3 = bufferization.to_memref %dyn_tensor : memref<?x?xi32, strided<[256, 1]>>
    %old = hfusion.atomic_xchg ins(%3 : memref<?x?xi32, strided<[256, 1]>>) outs(%subview : memref<?x?xi32, strided<[512, 1], offset: ?>>) -> tensor<?x?xi32>
    return %old : tensor<?x?xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_fold_to_memref
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<4xi16>, memref<4xi16>) -> ()
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<4xi16>, memref<4xi16>) -> ()
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: hfusion.atomic_cas
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_fold_to_memref(%arg0: memref<4xi16>, %arg1: tensor<4xi16>, %arg2: tensor<4xi16>) {
    %cmp = bufferization.to_memref %arg1 : memref<4xi16>
    %store = bufferization.to_memref %arg2 : memref<4xi16>
    hfusion.atomic_cas ins(%cmp, %store : memref<4xi16>, memref<4xi16>) outs(%arg0 : memref<4xi16>)
    return
  }
}
