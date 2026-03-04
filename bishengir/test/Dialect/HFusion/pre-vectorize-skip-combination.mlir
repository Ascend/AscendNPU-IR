// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

func.func @test_pre_vectorize_skip_combination(%arg0: memref<2048xf32>) -> () {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant -1.000000e+00 : f32
  %0 = tensor.empty() : tensor<2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
  %alloc = memref.alloc() : memref<2048xf32>
  %subview = memref.subview %alloc[0] [2048] [1] : memref<2048xf32> to memref<2048xf32>
  %subview_1 = memref.subview %arg0[0] [2048] [1] : memref<2048xf32> to memref<2048xf32>
  memref.copy %subview_1, %subview : memref<2048xf32> to memref<2048xf32>
  %2 = bufferization.to_tensor %subview restrict writable : memref<2048xf32>
  // CHECK: %{{.*}} = tensor.empty()
  // CHECK-NEXT: %{{.*}} = linalg.generic
  // CHECK-NEXT: ^bb0(%in
  // CHECK-NEXT: %{{.*}} = math.log
  %3 = tensor.empty() : tensor<2048xf32>
  %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%2 : tensor<2048xf32>) outs(%3 : tensor<2048xf32>) -> tensor<2048xf32>
  scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
    %5 = tensor.empty() : tensor<2048xf32>
    // CHECK: %{{.*}} = tensor.empty()
    // CHECK-NEXT: %{{.*}} = linalg.generic
    // CHECK-NEXT: ^bb0(%in
    // CHECK-NEXT: %{{.*}} = arith.mulf %in, %cst
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %1 : tensor<2048xf32>, tensor<2048xf32>) outs(%5 : tensor<2048xf32>) -> tensor<2048xf32>
    bufferization.materialize_in_destination %6 in writable %subview_1 : (tensor<2048xf32>, memref<2048xf32>) -> ()
  }
  return
}