// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK-LABEL: func.func @test_scalar_i1_fill_to_generic(
// CHECK: %[[CST:.*]] = arith.constant false
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xi1>
//
// CHECK: linalg.generic
//
// CHECK: ^bb0(%{{.*}}: i1):
// CHECK:   linalg.yield %[[CST]] : i1
//
// CHECK-NOT: linalg.fill
func.func @test_scalar_i1_fill_to_generic(
    %arg0: memref<?xi8>,
    %arg1: memref<?xi8>,
    %arg2: memref<?xi1> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
    %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32
) attributes {
    SyncBlockLockArgIdx = 0 : i64,
    WorkspaceArgIdx = 1 : i64,
    global_kernel = "local",
    mix_mode = "aiv",
    parallel_mode = "simd"
} {
  %cst = arith.constant false
  %0 = tensor.empty() : tensor<1xi1>
  %1 = linalg.fill ins(%cst : i1) outs(%0 : tensor<1xi1>) -> tensor<1xi1>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1]
      : memref<?xi1> to memref<1xi1, strided<[1]>>
  bufferization.materialize_in_destination %1 in writable %reinterpret_cast
      : (tensor<1xi1>, memref<1xi1, strided<[1]>>) -> ()
  return
}
