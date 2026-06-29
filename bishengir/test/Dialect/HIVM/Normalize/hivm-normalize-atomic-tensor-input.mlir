// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Store_tensor_ins_atomic_or
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %{{.*}} : (tensor<256xi16>, memref<256xi16>) -> ()
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<256xi16>
// CHECK: %[[DST:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[RES:.*]] = hivm.hir.vor ins(%[[LHS]], %{{.*}} : tensor<256xi16>, tensor<256xi16>) outs(%[[DST]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: bufferization.materialize_in_destination %[[RES]] in writable %{{.*}} : (tensor<256xi16>, memref<256xi16, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Store_tensor_ins_atomic_or(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
    hivm.hir.store ins(%arg1 : tensor<256xi16>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>) atomic = <or>
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
// CHECK-NOT: hivm.hir.atomic_xchg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_tensor_ins_with_return(%arg0: memref<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %old = hivm.hir.atomic_xchg ins(%arg1 : tensor<4x4xi32>) outs(%arg0 : memref<4x4xi32>) -> tensor<4x4xi32>
    return %old : tensor<4x4xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Store_tensor_ins_atomic_dyn_and
// CHECK: bufferization.materialize_in_destination %[[UB:.*]] in writable %{{.*}} : (tensor<?x?xi16>, memref<?x?xi16, strided<[256, 1]>>) -> ()
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[GM:.*]], %{{.*}} : memref<?x?xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[256, 1]>>
// CHECK: %[[DST:.*]] = tensor.empty() : tensor<12x256xi16>
// CHECK: %[[RES:.*]] = hivm.hir.vand ins(%{{.*}}, %{{.*}} : tensor<12x256xi16>, tensor<12x256xi16>) outs(%[[DST]] : tensor<12x256xi16>) -> tensor<12x256xi16>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[RES]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<12x256xi16> to tensor<?x?xi16>
// CHECK: bufferization.materialize_in_destination %[[SLICE]] in writable %[[GM]] : (tensor<?x?xi16>, memref<?x?xi16, strided<[512, 1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Store_tensor_ins_atomic_dyn_and(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: index, %arg3: index, %arg4: index) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg4], sizes: [12, 256], strides: [512, 1] : memref<?xi16> to memref<12x256xi16, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<12x256xi16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<12x256xi16>
    %dyn_tensor = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [1, 1] : tensor<12x256xi16> to tensor<?x?xi16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[512, 1], offset: ?>>
    hivm.hir.store ins(%dyn_tensor : tensor<?x?xi16>) outs(%subview : memref<?x?xi16, strided<[512, 1], offset: ?>>) atomic = <and>
    return
  }
}
