// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir && bishengir-compile -enable-hfusion-compile=true -target=Ascend910_9589 -enable-triton-kernel-compile=true %s -o %t

// CHECK: LLVMDialectModule
module {
  func.func @triton_broadcast_to_3d(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i64 = arith.constant 0 : i64
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [7, 1, 3], strides: [3, 3, 1] : memref<?xi64> to memref<7x1x3xi64, strided<[3, 3, 1]>>
    %alloc = memref.alloc() : memref<7x1x3xi64>
    memref.copy %reinterpret_cast, %alloc : memref<7x1x3xi64, strided<[3, 3, 1]>> to memref<7x1x3xi64>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<7x1x3xi64>
    %1 = math.absi %0 : tensor<7x1x3xi64>
    %2 = tensor.empty() : tensor<7x1xi64>
    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<7x1xi64>) -> tensor<7x1xi64>
    %reduced = linalg.reduce ins(%1 : tensor<7x1x3xi64>) outs(%3 : tensor<7x1xi64>) dimensions = [2]
      (%in: i64 , %init: i64 ) {
        %5 = arith.addi %in, %init : i64
        linalg.yield %5 : i64
      }
    %collapsed = tensor.collapse_shape %reduced [[0, 1]] : tensor<7x1xi64> into tensor<7xi64>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [7], strides: [1] : memref<?xi64> to memref<7xi64, strided<[1]>>
    bufferization.materialize_in_destination %collapsed in writable %reinterpret_cast_0 : (tensor<7xi64>, memref<7xi64, strided<[1]>>) -> ()
    return
  }
}
