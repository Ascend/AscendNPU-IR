// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_or_i16
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<256xi16>
// CHECK: %[[RES:.*]] = hivm.hir.vor ins(%[[LHS]], %{{.*}} : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: bufferization.materialize_in_destination %[[RES]] in writable %{{.*}} : (tensor<256xi16>, memref<256xi16, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_or_i16(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
    hivm.hir.store ins(%1 : memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>) atomic = <or>
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_atomic_cas_i16
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%[[CMP]], %{{.*}}, %{{.*}} : tensor<256xi1>, tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: bufferization.materialize_in_destination %[[SEL]] in writable %{{.*}} : (tensor<256xi16>, memref<256xi16, strided<[1]>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_atomic_cas_i16(%arg0: memref<?xi16>) {
    %alloc = memref.alloc() : memref<256xi16>
    %alloc_0 = memref.alloc() : memref<256xi16>
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
    hivm.hir.atomic_cas ins(%alloc, %alloc_0 : memref<256xi16>, memref<256xi16>) outs(%reinterpret_cast : memref<256xi16, strided<[1]>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_atomic_xchg_i16
// CHECK: scope.scope : () -> () {
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<256xi16, strided<[1]>> to memref<256xi16>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<256xi16> to memref<256xi16, strided<[1]>>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<256xi16> to memref<256xi16>
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_atomic_xchg_i16(%arg0: memref<?xi16>) {
    %alloc = memref.alloc() : memref<256xi16>
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
    hivm.hir.atomic_xchg ins(%alloc : memref<256xi16>) outs(%reinterpret_cast : memref<256xi16, strided<[1]>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSort_vsort_i8
// CHECK: %[[CAST_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<8xi8>) outs(%{{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[CAST_F32:.*]] = hivm.hir.vcast ins(%[[CAST_F16]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[SORT:.*]] = hivm.hir.vsort ins(%[[CAST_F32]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) descending = true sort_axis = 0 -> tensor<8xf32>
// CHECK: %[[CAST_I32:.*]] = hivm.hir.vcast{{.*}} ins(%[[SORT]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xi32>) {{.*}} -> tensor<8xi32>
// CHECK: %[[CAST_I8:.*]] = hivm.hir.vcast{{.*}} ins(%[[CAST_I32]] : tensor<8xi32>) outs(%{{.*}} : tensor<8xi8>) {{.*}} -> tensor<8xi8>
func.func @test_NormalizeSort_vsort_i8(%arg0 : tensor<8xi8>) -> tensor<8xi8> {
  %empty = tensor.empty() : tensor<8xi8>
  %0 = hivm.hir.vsort ins(%arg0 : tensor<8xi8>) outs(%empty : tensor<8xi8>) descending = true sort_axis = 0 -> tensor<8xi8>
  return %0 : tensor<8xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSort_vsort_f16_keep_original
// CHECK: %[[SORT:.*]] = hivm.hir.vsort ins(%arg0 : tensor<8xf16>) outs(%{{.*}} : tensor<8xf16>) descending = false sort_axis = 0 -> tensor<8xf16>
// CHECK-NOT: hivm.hir.vcast
func.func @test_NormalizeSort_vsort_f16_keep_original(%arg0 : tensor<8xf16>) -> tensor<8xf16> {
  %empty = tensor.empty() : tensor<8xf16>
  %0 = hivm.hir.vsort ins(%arg0 : tensor<8xf16>) outs(%empty : tensor<8xf16>) descending = false sort_axis = 0 -> tensor<8xf16>
  return %0 : tensor<8xf16>
}
