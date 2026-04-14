// RUN: bishengir-compile --enable-auto-multi-buffer=true \
// RUN: --enable-hfusion-compile=true --enable-triton-kernel-compile=true \
// RUN: --enable-lir-compile=true --target=Ascend950PR_9589 \
// RUN: %s -save-linked-ir -o %t1.ll
// RUN: cat %t1_linked.ll | FileCheck %s

// CEHCK: define internal void @_mlir_ciface_cast_int32_t_to_int64_t
// CHECK: define internal <64 x i32> @_mlir_ciface_cast_int64_t_to_int32_t

module {
  func.func @triton_unk_fused_eq_where_zeros_like_1(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c0_i32 = arith.constant 0 : i32
    %c2112 = arith.constant 2112 : index
    %c1408_i32 = arith.constant 1408 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<2112xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<2112xi64>) -> tensor<2112xi64>
    %2 = linalg.fill ins(%c-1_i64 : i64) outs(%0 : tensor<2112xi64>) -> tensor<2112xi64>
    %3 = arith.muli %arg8, %c1408_i32 : i32
    %4 = arith.addi %3, %c1408_i32 : i32
    %5 = arith.minsi %4, %arg4 : i32
    %6 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [2112], strides: [1] : memref<?xi32> to memref<2112xi32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<2112xi32>
    %7 = arith.addi %6, %c2112 : index
    %8 = arith.index_cast %5 : i32 to index
    %9 = arith.maxsi %6, %8 : index
    %10 = arith.minsi %7, %9 : index
    %11 = arith.subi %10, %6 : index
    %12 = arith.cmpi slt, %11, %c2112 : index
    scf.if %12 {
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<2112xi32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%11] [1] : memref<2112xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%11] [1] : memref<2112xi32> to memref<?xi32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
    %13 = bufferization.to_tensor %alloc restrict writable : memref<2112xi32>
    %14 = arith.extsi %13 : tensor<2112xi32> to tensor<2112xi64>
    %15 = arith.cmpi eq, %14, %2 : tensor<2112xi64>
    %16 = arith.select %15, %1, %14 : tensor<2112xi1>, tensor<2112xi64>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [2112], strides: [1] : memref<?xi32> to memref<2112xi32, strided<[1], offset: ?>>
    %17 = arith.trunci %16 : tensor<2112xi64> to tensor<2112xi32>
    %extracted_slice = tensor.extract_slice %17[0] [%11] [1] : tensor<2112xi32> to tensor<?xi32>
    %subview_2 = memref.subview %reinterpret_cast_1[0] [%11] [1] : memref<2112xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xi32>, memref<?xi32, strided<[1], offset: ?>>) -> ()
    return
  }
}
