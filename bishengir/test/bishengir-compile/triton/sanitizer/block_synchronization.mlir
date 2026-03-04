// RUN: bishengir-compile --enable-lir-compile=false --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-sanitizer=true %s | FileCheck %s

// CHECK-DAG: call void @llvm.hivm.SET.CROSS.CORE({{.*}}, {{.*}}), !dbg {{.*}}, !asan.cce.api.name ![[S_API:.*]], !asan.stub.mangling.name ![[S_STUB:.*]]
// CHECK-DAG: call void @llvm.hivm.WAIT.FLAG.DEV.REG({{.*}}), !dbg {{.*}}, !asan.cce.api.name ![[W_API:.*]], !asan.stub.mangling.name ![[W_STUB:.*]]
// CHECK-DAG: ![[S_API]] = !{!"ffts_cross_core_sync"}
// CHECK-DAG: ![[S_STUB]] = !{!"_Z39__sanitizer_report_ffts_cross_core_syncPU3AS1hmmljm"}
// CHECK-DAG: ![[W_API]] = !{!"wait_flag_dev"}
// CHECK-DAG: ![[W_STUB]] = !{!"_Z32__sanitizer_report_wait_flag_devPU3AS1hmmll"}

module {
  func.func @fn_dot(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "mix"} {
    %cst = arith.constant dense<4096> : tensor<1xi64>
    %cst_0 = arith.constant dense<[64, 32]> : tensor<2xi64>
    %cst_1 = arith.constant dense<[128, 64]> : tensor<2xi64>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x32xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8192], strides: [1] : memref<?xf32> to memref<8192xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<8192xf32>
    memref.copy %reinterpret_cast, %alloc : memref<8192xf32, strided<[1]>> to memref<8192xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<8192xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1]>>
    %alloc_4 = memref.alloc() : memref<2048xf32>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<2048xf32, strided<[1]>> to memref<2048xf32>
    %3 = bufferization.to_tensor %alloc_4 restrict writable : memref<2048xf32>
    %reshape = tensor.reshape %2(%cst_1) : (tensor<8192xf32>, tensor<2xi64>) -> tensor<128x64xf32>
    %reshape_5 = tensor.reshape %3(%cst_0) : (tensor<2048xf32>, tensor<2xi64>) -> tensor<64x32xf32>
    %4 = linalg.matmul {input_precison = "ieee"} ins(%reshape, %reshape_5 : tensor<128x64xf32>, tensor<64x32xf32>) outs(%1 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %5 = math.absf %4 : tensor<128x32xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
    %reshape_7 = tensor.reshape %5(%cst) : (tensor<128x32xf32>, tensor<1xi64>) -> tensor<4096xf32>
    bufferization.materialize_in_destination %reshape_7 in writable %reinterpret_cast_6 : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
    return
  }
}

