// RUN: bishengir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_load_store
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_load_store(%arg: tensor<f32>) -> tensor<f32> {
  %0 = tensor.empty() : tensor<f32>
  %1 = tensor.empty() : tensor<f32>
  %2 = hfusion.load ins(%arg: tensor<f32>) outs(%0: tensor<f32>) -> tensor<f32>
  %3 = hfusion.store ins(%2: tensor<f32>) outs(%1: tensor<f32>) -> tensor<f32>
  return %3 : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @test_memref_load_store
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_memref_load_store(%arg0: memref<f32>, %arg1: memref<f32>) {
  hfusion.load ins(%arg0: memref<f32>) outs(%arg1: memref<f32>)
  hfusion.store ins(%arg0: memref<f32>) outs(%arg1: memref<f32>)
  return
}