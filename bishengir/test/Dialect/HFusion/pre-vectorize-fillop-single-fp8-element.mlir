// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK-LABEL: func.func @test_scalar_fp8e4m3fn_fill_to_generic(
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f8E4M3FN
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf8E4M3FN>
//
// CHECK: linalg.generic
//
// CHECK: ^bb0(%{{.*}}: f8E4M3FN):
// CHECK:   linalg.yield %[[CST]] : f8E4M3FN
//
// CHECK-NOT: linalg.fill
func.func @test_scalar_fp8e4m3fn_fill_to_generic(
    %arg0: memref<?xi8>, 
    %arg1: memref<?xi8>, 
    %arg2: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, 
    %arg3: f8E4M3FN, 
    %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32
) attributes {
    SyncBlockLockArgIdx = 0 : i64, 
    WorkspaceArgIdx = 1 : i64, 
    global_kernel = "local", 
    mix_mode = "aiv", 
    parallel_mode = "simd"
} {
  %cst = arith.constant 0.000000e+00 : f8E4M3FN
  %0 = tensor.empty() : tensor<1xf8E4M3FN>
  %1 = linalg.fill ins(%cst : f8E4M3FN) outs(%0 : tensor<1xf8E4M3FN>) -> tensor<1xf8E4M3FN>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] 
      : memref<?xf8E4M3FN> to memref<1xf8E4M3FN, strided<[1]>>
  bufferization.materialize_in_destination %1 in writable %reinterpret_cast 
      : (tensor<1xf8E4M3FN>, memref<1xf8E4M3FN, strided<[1]>>) -> ()
  return
}