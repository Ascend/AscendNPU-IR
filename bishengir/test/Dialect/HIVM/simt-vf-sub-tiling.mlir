// RUN: bishengir-opt --hivm-simt-vf-sub-tiling=max-tile-size=4 %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tile_local_store(
// CHECK: scf.for
// CHECK: %[[OFF:.*]] = affine.apply
// CHECK: %[[DSTUB:.*]] = memref.subview %arg1[%[[OFF]]] [4] [1]
// CHECK: %[[SRCGM:.*]] = memref.subview %reinterpret_cast[%[[OFF]]] [4] [1]
// CHECK: hivm.hir.load ins(%[[SRCGM]] : memref<4xf32
// CHECK: outs(%[[DSTUB]] : memref<4xf32
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[DSTUB]] restrict writable {tiled_op} : memref<4xf32
// CHECK: %[[DST:.*]] = memref.subview %arg2[%[[OFF]]] [4] [1]
// CHECK: hivm.hir.local_store ins(%[[DST]] : memref<4xf32
// CHECK-SAME: %[[T]] : tensor<4xf32>) {tiled_op}
module {
  func.func @tile_local_store(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<8xf32, #hivm.address_space<ub>>, %arg2: memref<8xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>, %1 : tensor<8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @tile_gather_local_store_with_indices_chain(
// CHECK: scf.for
// CHECK: %[[IDXUB:.*]] = memref.subview %arg1[%{{.*}}, 0] [4, 8] [1, 1]
// CHECK: %[[IDX:.*]] = bufferization.to_tensor %[[IDXUB]] restrict writable {tiled_op} : memref<4x8xi64
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vadd
// CHECK: %[[G:.*]] = hivm.hir.gather_load
// CHECK-SAME: outs(%{{.*}} : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[DST:.*]] = memref.subview %arg4[%{{.*}}, 0] [4, 8] [1, 1]
// CHECK: hivm.hir.local_store ins(%[[DST]] : memref<4x8xf32
// CHECK-SAME: %[[G]] : tensor<4x8xf32>) {tiled_op}
module {
  func.func @tile_gather_local_store_with_indices_chain(%arg0: memref<?xi64, #hivm.address_space<gm>>, %arg1: memref<8x8xi64, #hivm.address_space<ub>>, %arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: i32, %arg4: memref<8x8xf32, #hivm.address_space<ub>>, %arg5: i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<?xi64, #hivm.address_space<gm>> to memref<8x8xi64, strided<[8, 1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8x8xi64, strided<[8, 1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8x8xi64, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8x8xi64, #hivm.address_space<ub>>
    %2 = tensor.empty() : tensor<8xi32>
    %3 = hivm.hir.varange offset[%c0] strides[%c1] outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %4 = tensor.empty() : tensor<8xi32>
    %5 = hivm.hir.vmul ins(%3, %arg5 : tensor<8xi32>, i32) outs(%4 : tensor<8xi32>) -> tensor<8xi32>
    %6 = tensor.empty() : tensor<8xi64>
    %7 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%5 : tensor<8xi32>) outs(%6 : tensor<8xi64>) -> tensor<8xi64>
    %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [8, 1] : tensor<8xi64> into tensor<8x1xi64>
    %8 = tensor.empty() : tensor<8x8xi64>
    %9 = hivm.hir.vbrc ins(%expanded : tensor<8x1xi64>) outs(%8 : tensor<8x8xi64>) broadcast_dims = [1] -> tensor<8x8xi64>
    %10 = tensor.empty() : tensor<8x8xi64>
    %11 = hivm.hir.vadd ins(%9, %1 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%10 : tensor<8x8xi64>) -> tensor<8x8xi64>
    %12 = tensor.empty() : tensor<8x8xf32>
    %13 = hivm.hir.gather_load ins(%arg2 : memref<?xf32, #hivm.address_space<gm>>, %11 : tensor<8x8xi64>, %arg3 : i32) outs(%12 : tensor<8x8xf32>) -> tensor<8x8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8x8xf32, #hivm.address_space<ub>>, %13 : tensor<8x8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skip_mismatched_local_stores(
// CHECK-NOT: scf.for
// CHECK: hivm.hir.local_store ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>, %[[T0:.*]] : tensor<8xf32>)
// CHECK: hivm.hir.local_store ins(%arg5 : memref<10xf32, #hivm.address_space<ub>>, %[[T1:.*]] : tensor<10xf32>)
// CHECK: return
module {
  func.func @skip_mismatched_local_stores(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<8xf32, #hivm.address_space<ub>>, %arg2: memref<8xf32, #hivm.address_space<ub>>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<10xf32, #hivm.address_space<ub>>, %arg5: memref<10xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>, %1 : tensor<8xf32>)
    %2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [10], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<10xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%2 : memref<10xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%arg4 : memref<10xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %arg4 restrict writable : memref<10xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg5 : memref<10xf32, #hivm.address_space<ub>>, %3 : tensor<10xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skip_vtranspose_local_store(
// CHECK-NOT: scf.for
// CHECK: hivm.hir.vtranspose
// CHECK: hivm.hir.local_store ins(%arg2 : memref<4x8xf32, #hivm.address_space<ub>>, %[[T:.*]] : tensor<4x8xf32>)
module {
  func.func @skip_vtranspose_local_store(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<8x4xf32, #hivm.address_space<ub>>, %arg2: memref<4x8xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 4], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8x4xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8x4xf32, #hivm.address_space<ub>>
    %2 = tensor.empty() : tensor<4x8xf32>
    %3 = hivm.hir.vtranspose ins(%1 : tensor<8x4xf32>) outs(%2 : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
    hivm.hir.local_store ins(%arg2 : memref<4x8xf32, #hivm.address_space<ub>>, %3 : tensor<4x8xf32>)
    return
  }
}
