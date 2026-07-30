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
  // CHECK-NEXT:      %[[GATHERLOAD:.*]] = hivm.hir.gather_load ins(%[[ARG2:.*]] : memref<?xf32>, %[[TOTENSOR]] : tensor<8xi64>, %[[c:.*]] : i32) outs(%[[EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
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

// -----

module {
  // CHECK-LABEL: func.func @shared_alloc_backed_indices_two_gathers
  // CHECK:         %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[INDICES:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<8xi64>
  // CHECK:         hivm.hir.load ins(%[[REINTERPRET]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC]] : memref<8xi64>)
  // CHECK:         %[[SCOPE0:.*]] = scope.scope
  // CHECK-NOT:      memref.reinterpret_cast
  // CHECK-NOT:      hivm.hir.load
  // CHECK:           %[[TOTENSOR0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xi64>
  // CHECK:           %[[GATHER0:.*]] = hivm.hir.gather_load ins(%[[BASE0:.*]] : memref<?xf32>, %[[TOTENSOR0]] : tensor<8xi64>, %[[C1:.*]] : i32) outs(%[[EMPTY0:.*]] : tensor<8xf32>) -> tensor<8xf32>
  // CHECK:           scope.return %[[GATHER0]] : tensor<8xf32>
  // CHECK:         %[[SCOPE1:.*]] = scope.scope
  // CHECK-NOT:      memref.reinterpret_cast
  // CHECK-NOT:      hivm.hir.load
  // CHECK:           %[[TOTENSOR1:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xi64>
  // CHECK:           %[[GATHER1:.*]] = hivm.hir.gather_load ins(%[[BASE1:.*]] : memref<?xf32>, %[[TOTENSOR1]] : tensor<8xi64>, %[[C1]] : i32) outs(%[[EMPTY1:.*]] : tensor<8xf32>) -> tensor<8xf32>
  // CHECK:           scope.return %[[GATHER1]] : tensor<8xf32>
  func.func @shared_alloc_backed_indices_two_gathers(%base0: memref<?xf32>, %base1: memref<?xf32>, %indices_gm: memref<?xi64>) {
    %c1_i32 = arith.constant 1 : i32
    %reinterpret_cast = memref.reinterpret_cast %indices_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%base0 : memref<?xf32>, %0 : tensor<8xi64>, %c1_i32 : i32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    %3 = tensor.empty() : tensor<8xf32>
    %4 = hivm.hir.gather_load ins(%base1 : memref<?xf32>, %0 : tensor<8xi64>, %c1_i32 : i32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
    %5 = tensor.empty() : tensor<8xf32>
    %6 = hivm.hir.vadd ins(%2, %4 : tensor<8xf32>, tensor<8xf32>) outs(%5 : tensor<8xf32>) -> tensor<8xf32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @shared_alloc_backed_indices_gather_and_scatter
  // CHECK:         %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[INDICES:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<8xi64>
  // CHECK:         hivm.hir.load ins(%[[REINTERPRET]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC]] : memref<8xi64>)
  // CHECK:         %[[SCOPE0:.*]] = scope.scope
  // CHECK-NOT:       memref.reinterpret_cast
  // CHECK-NOT:       hivm.hir.load
  // CHECK:           %[[TOTENSOR0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xi64>
  // CHECK:           %[[GATHER0:.*]] = hivm.hir.gather_load ins(%[[BASE0:.*]] : memref<?xf32>, %[[TOTENSOR0]] : tensor<8xi64>, %[[C1:.*]] : i32) outs(%[[EMPTY0:.*]] : tensor<8xf32>) -> tensor<8xf32>
  // CHECK:           scope.return %[[GATHER0]] : tensor<8xf32>
  // CHECK:         scope.scope
  // CHECK-NOT:       memref.reinterpret_cast
  // CHECK-NOT:       hivm.hir.load
  // CHECK:           %[[TOTENSOR1:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xi64>
  // CHECK:           hivm.hir.scatter_store ins(%[[TOTENSOR1]] : tensor<8xi64>, %[[SCOPE0]] : tensor<8xf32>, %[[C1]] : i32) outs(%[[BASE1:.*]] : memref<?xf32>)
  // CHECK:           scope.return
  func.func @shared_alloc_backed_indices_gather_and_scatter(%base0: memref<?xf32>, %base1: memref<?xf32>, %indices_gm: memref<?xi64>) -> tensor<8xf32> {
    %c1_i32 = arith.constant 1 : i32
    %reinterpret_cast = memref.reinterpret_cast %indices_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%base0 : memref<?xf32>, %0 : tensor<8xi64>, %c1_i32 : i32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%0 : tensor<8xi64>, %2 : tensor<8xf32>, %c1_i32 : i32) outs(%base1 : memref<?xf32>)
    return %2 : tensor<8xf32>
  }
}
