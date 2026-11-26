// REQUIRES: enable-lir-compile
// A runtime scalar filled into a tensor and truncated to FP8 must use a vector
// conversion. Scalar fptrunc returns zero on Ascend950, and introducing a VCast
// after vectorization instead leaves an unresolved template library call.
// RUN: split-file %s %t
// RUN: bishengir-compile %t/e4m3.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --disable-ffts --save-temps=%t/e4m3 -o %t/e4m3/kernel.o
// RUN: FileCheck %s --check-prefix=E4M3 --implicit-check-not=fptrunc --implicit-check-not=vcast_float_to_float8 --input-file=%t/e4m3/kernel.ll
// RUN: bishengir-compile %t/e5m2.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --disable-ffts --save-temps=%t/e5m2 -o %t/e5m2/kernel.o
// RUN: FileCheck %s --check-prefix=E5M2 --implicit-check-not=fptrunc --implicit-check-not=vcast_float_to_float8 --input-file=%t/e5m2/kernel.ll

// E4M3: call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(
// E5M2: call <256 x float8e5m2> @llvm.hivm.vcvtff.f322f8e5m2.x(

//--- e4m3.mlir
module {
  func.func @scalar_fp8_broadcast(%src: memref<?xf32>, %dst: memref<?xf8E4M3FN>,
      %grid_x: i32, %grid_y: i32, %grid_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32)
      attributes {global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %value = memref.load %src[%c0] : memref<?xf32>
    %empty = tensor.empty() : tensor<64xf32>
    %values = linalg.fill ins(%value : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
    %converted = arith.truncf %values : tensor<64xf32> to tensor<64xf8E4M3FN>
    %out = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1]
        : memref<?xf8E4M3FN> to memref<64xf8E4M3FN, strided<[1]>>
    bufferization.materialize_in_destination %converted in writable %out
        : (tensor<64xf8E4M3FN>, memref<64xf8E4M3FN, strided<[1]>>) -> ()
    return
  }
}

//--- e5m2.mlir
module {
  func.func @scalar_fp8_broadcast(%src: memref<?xf32>, %dst: memref<?xf8E5M2>,
      %grid_x: i32, %grid_y: i32, %grid_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32)
      attributes {global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %value = memref.load %src[%c0] : memref<?xf32>
    %empty = tensor.empty() : tensor<64xf32>
    %values = linalg.fill ins(%value : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
    %converted = arith.truncf %values : tensor<64xf32> to tensor<64xf8E5M2>
    %out = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1]
        : memref<?xf8E5M2> to memref<64xf8E5M2, strided<[1]>>
    bufferization.materialize_in_destination %converted in writable %out
        : (tensor<64xf8E5M2>, memref<64xf8E5M2, strided<[1]>>) -> ()
    return
  }
}
