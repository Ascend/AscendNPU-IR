// RUN: bishengir-opt -convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_bitcast_f32_i32_tensor
// CHECK: hivm.hir.bitcast
func.func @test_bitcast_f32_i32_tensor(%arg0 : tensor<128x128xf32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = hfusion.bitcast
    ins(%arg0 : tensor<128x128xf32>) 
    outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>  // Bitcast the f32 value to i32
  %2 = tensor.empty() : tensor<128x128xi32>
  %3 = linalg.matmul ins(%1, %arg1 : tensor<128x128xi32>, tensor<128x128xi32>) outs(%2 : tensor<128x128xi32>) -> tensor<128x128xi32>        // Another constant integer
  return %3 : tensor<128x128xi32>
}

// -----

func.func @test_bitcast_f32_i32_memref(
  %src : memref<6xf32>, %dst : memref<6xi32>){
// expected-error@+2 {{'hfusion.bitcast' op hfusion.bitcast must be in Pure Tensor Semantics}}
// expected-error@+1 {{failed to legalize operation 'hfusion.bitcast' that was explicitly marked illegal}}
  hfusion.bitcast 
  ins(%src : memref<6xf32>) 
  outs(%dst : memref<6xi32>)
  return
}

// -----

// CHECK-LABEL: @test_bitcast_f32_i32_tensor
// CHECK: hivm.hir.bitcast
func.func @test_bitcast_f32_i32_tensor(%arg0 : tensor<128x128xf32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = hfusion.bitcast
    ins(%arg0 : tensor<128x128xf32>)
    outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>  // Bitcast the f32 value to i32
  return %1 : tensor<128x128xi32>
}

// -----

// CHECK-LABEL: @test_complicated_bitcast
// CHECK: hivm.hir.bitcast
func.func @test_complicated_bitcast(%arg0: tensor<32xf32>, %arg1: tensor<32xi32>) -> tensor<32xi32> {
  %0 = tensor.empty() : tensor<32xi32>
  %1 = hfusion.bitcast ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg1, %1 : tensor<32xi32>, tensor<32xi32>) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
  return %2 : tensor<32xi32>
}

