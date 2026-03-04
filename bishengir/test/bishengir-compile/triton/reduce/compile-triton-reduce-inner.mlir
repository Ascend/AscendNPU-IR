// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-triton-kernel-compile %s | FileCheck %s

// CHECK: LLVMDialectModule
module{
func.func @triton_test_fn_reduce_inner(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {global_kernel = "local"} {
  %c256 = arith.constant 256 : index
  %cst = arith.constant dense<[256, 64]> : tensor<2xi64>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c16384 = arith.constant 16384 : index
  %c16384_i32 = arith.constant 16384 : i32
  %c256_i32 = arith.constant 256 : i32
  %0 = arith.muli %arg7, %c16384_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<16384xf32>
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.addi %2, %c16384 : index
  %4 = arith.index_cast %arg2 : i32 to index
  %5 = arith.maxsi %2, %4 : index
  %6 = arith.minsi %3, %5 : index
  %7 = arith.subi %6, %2 : index
  %8 = arith.cmpi slt, %7, %c16384 : index
  scf.if %8 {
    linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<16384xf32>)
  }
  %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<16384xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %subview_1 = memref.subview %alloc[0] [%7] [1] : memref<16384xf32> to memref<?xf32, strided<[1]>>
  memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  %9 = bufferization.to_tensor %alloc restrict writable : memref<16384xf32>
  %reshape = tensor.reshape %9(%cst) : (tensor<16384xf32>, tensor<2xi64>) -> tensor<256x64xf32>
  %10 = tensor.empty() : tensor<256xf32>
  %reduced = linalg.reduce ins(%reshape : tensor<256x64xf32>) outs(%10 : tensor<256xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %19 = arith.addf %in, %init : f32
      linalg.yield %19 : f32
    }
  %11 = arith.muli %arg7, %c256_i32 : i32
  %12 = arith.index_cast %11 : i32 to index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
  %13 = arith.index_cast %11 : i32 to index
  %14 = arith.addi %13, %c256 : index
  %15 = arith.index_cast %arg3 : i32 to index
  %16 = arith.maxsi %13, %15 : index
  %17 = arith.minsi %14, %16 : index
  %18 = arith.subi %17, %13 : index
  %extracted_slice = tensor.extract_slice %reduced[0] [%18] [1] : tensor<256xf32> to tensor<?xf32>
  %subview_3 = memref.subview %reinterpret_cast_2[0] [%18] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
  return
}
}

