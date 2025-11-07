// RUN: bishengir-opt %s --lower-hfusion-pipeline="enable-triton-kernel-compile=true" --cse --canonicalize-ext | FileCheck %s

// CHECK-LABEL: @triton_collapse_transpose(
// CHECK: %[[VAL_24:.*]] = linalg.transpose
// CHECK-SAME: permutation = [1, 2, 0]
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary
// CHECK-SAME: ins(%{{.*}}, %[[VAL_24]] : tensor<2x25x16xf32>, tensor<2x25x16xf32>)
// CHECK: %[[VAL_26:.*]] = tensor.collapse_shape %[[VAL_25]] {{\[\[}}0, 1], [2]] : tensor<2x25x16xf32> into tensor<50x16xf32>
// CHECK: return
module {
  func.func @triton_collapse_transpose(%arg0: memref<?xf32>, %arg1: tensor<16x50xi1>, %arg2: memref<16x2x25xf32, strided<[25, 400, 1], offset: ?>>, %arg3: tensor<50x16xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: tensor<16x50xf32>, %arg8: tensor<2xi64>) -> (memref<50x16xf32, strided<[16, 1], offset: ?>>, tensor<?x?xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %alloc = memref.alloc() : memref<16x2x25xf32>
    memref.copy %arg2, %alloc : memref<16x2x25xf32, strided<[25, 400, 1], offset: ?>> to memref<16x2x25xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<16x2x25xf32>
    %reshape = tensor.reshape %0(%arg8) : (tensor<16x2x25xf32>, tensor<2xi64>) -> tensor<16x50xf32>
    %1 = arith.select %arg1, %reshape, %arg7 : tensor<16x50xi1>, tensor<16x50xf32>
    %2 = tensor.empty() : tensor<50x16xf32>
    %transposed = linalg.transpose ins(%1 : tensor<16x50xf32>) outs(%2 : tensor<50x16xf32>) permutation = [1, 0]
    %3 = arith.addf %arg3, %transposed : tensor<50x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg6], sizes: [50, 16], strides: [16, 1] : memref<?xf32> to memref<50x16xf32, strided<[16, 1], offset: ?>>
    %extracted_slice = tensor.extract_slice %3[0, 0] [%arg4, %arg5] [1, 1] : tensor<50x16xf32> to tensor<?x?xf32>
    return %reinterpret_cast, %extracted_slice : memref<50x16xf32, strided<[16, 1], offset: ?>>, tensor<?x?xf32>
  }
}