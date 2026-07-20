// RUN: bishengir-opt --auto-scope --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @simple_indirect_load_kernel
  // CHECK:         %[[REINTERPRETCAST:.*]] = memref.reinterpret_cast %[[ARG3:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<8xi64>
  // CHECK:         hivm.hir.load ins(%[[REINTERPRETCAST]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC]] : memref<8xi64>)
  // CHECK:         %[[SCOPE:.*]] = scope.scope
  // CHECK-NOT:       memref.reinterpret_cast
  // CHECK-NOT:       hivm.hir.load
  // CHECK:           %[[TOTENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xi64>
  // CHECK-NEXT:      %[[EMPTY:.*]] = tensor.empty() : tensor<8xf32>
  // CHECK-NEXT:      %[[GATHERLOAD:.*]] = hivm.hir.gather_load ins(%[[ARG2:.*]] : memref<?xf32>, %[[TOTENSOR]] : tensor<8xi64>, %[[C1:.*]] : i32) outs(%[[EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT:      scope.return %[[GATHERLOAD]] : tensor<8xf32>
  func.func @simple_indirect_load_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %3 = tensor.empty() : tensor<8xf32>
    %4 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %2 : tensor<8xi64>, %c1_i32 : i32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    hivm.hir.store ins(%4 : tensor<8xf32>) outs(%reinterpret_cast_0 : memref<8xf32, strided<[1]>>)
    return
  }
}

