// RUN: bishengir-opt -split-input-file -mlir-print-op-generic %s | FileCheck %s

// CHECK-LABEL: test_tensor
func.func @test_tensor(%src:tensor<16x16xf16>, %idx:tensor<16x4xi32>) -> tensor<16x4xf16>{
  %init = tensor.empty() : tensor<16x4xf16>
  %res = hfusion.gather ins(%src, %idx : tensor<16x16xf16>, tensor<16x4xi32>) outs(%init:tensor<16x4xf16>) axis = 1 -> tensor<16x4xf16>
  // CHECK: linalg.index
  // CHECK-SAME: dim = 2
  // CHECK: arith.cmpi
  // CHECK: arith.select
  return %res : tensor<16x4xf16>
}

// -----

// CHECK-LABEL: test_memref
func.func @test_memref(%src:memref<16xf16>, %idx:memref<4xi32>) {
  %init = memref.alloc() : memref<4xf16>
  // CHECK: linalg.index
  // CHECK-SAME: dim = 1
  hfusion.gather ins(%src, %idx : memref<16xf16>, memref<4xi32>) outs(%init:memref<4xf16>) axis = 0
  return
}
