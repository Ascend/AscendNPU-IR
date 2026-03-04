// RUN: bishengir-compile --enable-lir-compile=false --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-debug-info=true %s

module {
  func.func @triton_add(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %0 = arith.muli %arg7, %c32768_i32 : i32
    scf.for %arg10 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %1 = arith.muli %arg10, %c1024_i32 : i32
      %2 = arith.addi %0, %1 : i32
      %3 = arith.index_cast %2 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1024xf32>
      memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
      %4 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc_1 = memref.alloc() : memref<1024xf32>
      memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
      %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xf32>
      %6 = arith.addf %4, %5 : tensor<1024xf32>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

