// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-triton-kernel-compile %s

module {
  func.func @test_basic__kernel0(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {global_kernel = "local", hivm.func_core_type=#hivm.func_core_type<AIV>} {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.muli %arg6, %c256_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %3 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<256xf32>, memref<256xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
