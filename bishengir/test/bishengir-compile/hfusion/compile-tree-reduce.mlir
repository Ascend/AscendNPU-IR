// RUN: bishengir-compile -enable-lir-compile=false --enable-tree-reduce=true -enable-hfusion-compile=true -enable-hivm-compile=true --enable-triton-kernel-compile=true     --target=Ascend950PR_9589  %s | FileCheck %s

// -----

// CHECK: define dso_local void @triton_sum_3D_dim0

module {
  func.func @triton_sum_3D_dim0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [5, 8, 12], strides: [96, 12, 1] : memref<?xi32> to memref<5x8x12xi32, strided<[96, 12, 1]>>
    %alloc = memref.alloc() : memref<5x8x12xi32>
    memref.copy %reinterpret_cast, %alloc : memref<5x8x12xi32, strided<[96, 12, 1]>> to memref<5x8x12xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<5x8x12xi32>
    %3 = tensor.empty() : tensor<8x12xi32>
    %4 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%3 : tensor<8x12xi32>) -> tensor<8x12xi32>
    %reduced_0 = linalg.reduce ins(%0 : tensor<5x8x12xi32>) outs(%4 : tensor<8x12xi32>) dimensions = [0] 
      (%in: i32, %init: i32) {
        %5 = arith.maxsi %in, %init : i32
        linalg.yield %5 : i32
      }
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 12], strides: [12, 1] : memref<?xi32> to memref<8x12xi32, strided<[12, 1]>>
    bufferization.materialize_in_destination %reduced_0 in writable %reinterpret_cast_1 : (tensor<8x12xi32>, memref<8x12xi32, strided<[12, 1]>>) -> ()
    return
  }
}