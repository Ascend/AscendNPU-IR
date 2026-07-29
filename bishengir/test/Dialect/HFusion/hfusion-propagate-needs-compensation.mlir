// RUN: bishengir-opt %s -hfusion-generalize -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @chunk_local_cumsum_vector_kernel
func.func @chunk_local_cumsum_vector_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64_i32 = arith.constant 64 : i32
  %0 = arith.divsi %arg10, %c4_i32 : i32
  %1 = arith.remsi %arg10, %c4_i32 : i32
  %2 = arith.muli %0, %arg4 : i32
  %3 = arith.muli %2, %c4_i32 : i32
  %4 = arith.addi %3, %1 : i32
  %5 = arith.muli %4, %c64_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = arith.muli %arg9, %c64_i32 : i32
  %8 = arith.muli %arg8, %c64_i32 : i32
  %9 = arith.maxsi %7, %c0_i32 : i32
  %10 = arith.index_cast %9 : i32 to index
  %11 = arith.maxsi %8, %c0_i32 : i32
  %12 = arith.index_cast %11 : i32 to index
  %13 = arith.muli %10, %c256 : index
  %14 = arith.addi %13, %6 : index
  %15 = arith.index_cast %arg4 : i32 to index
  %16 = arith.addi %14, %12 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
  %alloc = memref.alloc() : memref<64x64xf32>
  %17 = arith.subi %16, %6 : index
  %18 = arith.divsi %17, %c256 : index
  %19 = arith.subi %15, %18 : index
  %20 = arith.maxsi %19, %c0 : index
  %21 = arith.minsi %20, %c64 : index
  %22 = arith.remsi %17, %c256 : index
  %23 = arith.subi %c64, %22 : index
  %24 = arith.maxsi %23, %c0 : index
  %25 = arith.minsi %24, %c64 : index
  %26 = arith.subi %c0_i32, %7 : i32
  %27 = arith.maxsi %26, %c0_i32 : i32
  %28 = arith.index_cast %27 : i32 to index
  %29 = arith.minsi %28, %21 : index
  %30 = arith.subi %21, %29 : index
  %31 = arith.subi %c0_i32, %8 : i32
  %32 = arith.maxsi %31, %c0_i32 : i32
  %33 = arith.index_cast %32 : i32 to index
  %34 = arith.minsi %33, %25 : index
  %35 = arith.subi %25, %34 : index
  %36 = arith.cmpi slt, %30, %c64 : index
  %37 = arith.cmpi slt, %35, %c64 : index
  %38 = arith.ori %36, %37 : i1
  scf.if %38 {
    linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
  } {hivm.unlikely_condition}
  %subview = memref.subview %reinterpret_cast[0, 0] [%30, %35] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
  %subview_1 = memref.subview %alloc[%29, %34] [%30, %35] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
  memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
  // CHECK:      %[[ALLOC_TENSOR:[0-9]+]] = bufferization.to_tensor
  // CHECK-NEXT: %[[CUMSUM_RES:[0-9]+]] = hfusion.cumsum %[[ALLOC_TENSOR]] {needs_compensation} : tensor<64x64xf32> cum_dims = [0] reverse = true -> tensor<64x64xf32>
  // CHECK-NEXT: %{{.*}} = tensor.extract_slice %[[CUMSUM_RES]][%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1]
  %39 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
  %40 = hfusion.cumsum %39 : tensor<64x64xf32> cum_dims = [0] reverse = true -> tensor<64x64xf32>
  %extracted_slice = tensor.extract_slice %40[%29, %34] [%30, %35] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%30, %35] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x?xf32>, memref<?x?xf32, strided<[256, 1], offset: ?>>) -> ()
  return
}
