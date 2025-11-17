// RUN: bishengir-compile --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true %s

module {
  func.func private @triton_print_0(tensor<8xf32>) attributes {hex = false, prefix = " x: "}
  func.func @vector_kernel(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<8xf32>
    memref.copy %reinterpret_cast, %alloc : memref<8xf32, strided<[1]>> to memref<8xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xf32>
    gpu.barrier
    call @triton_print_0(%0) : (tensor<8xf32>) -> ()
    return
  }
}