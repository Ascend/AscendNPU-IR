// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-convert-hivm-to-upstream -split-input-file %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_vbrc_tensor_normal
func.func @test_vbrc_tensor_normal(%arg1: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK-NOT: generic
  // CHECK: %1 = linalg.fill ins(%cst{{ *}}: f32) outs(%0{{ *}}: tensor<128xf32>) -> tensor<128xf32>

  %cst = arith.constant 0.000000e+00 : f32
  %empty1 = tensor.empty() : tensor<128xf32>
  %ret = hivm.hir.vbrc ins(%cst: f32) outs(%empty1: tensor<128xf32>) -> tensor<128xf32>
  return %ret: tensor<128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vbrc_tensor_inline
func.func @test_vbrc_tensor_inline(%arg0: tensor<32x64xf32>, %arg1: tensor<32x1xf32>) -> tensor<32x64xf32> {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK-NOT: generic
  // CHECK: %broadcasted = linalg.broadcast ins(%collapsed{{ *}}: tensor<32xf32>) outs(%{{.*}}{{ *}}: tensor<32x64xf32>) dimensions = [1]
  // CHECK: %{{.*}} = linalg.sub ins(%arg0, %broadcasted{{ *}}: tensor<32x64xf32>, tensor<32x64xf32>) outs(%{{.*}}{{ *}}: tensor<32x64xf32>) -> tensor<32x64xf32>

  %empty0 = tensor.empty() : tensor<32x64xf32>
  %ret = hivm.hir.vsub ins(%arg0, %arg1: tensor<32x64xf32>, tensor<32x1xf32>) outs(%empty0 : tensor<32x64xf32>) broadcast = [1] -> tensor<32x64xf32>
  return %ret: tensor<32x64xf32>
}

// -----
// CHECK-LABEL: func.func @test_vbrc_memref_normal
func.func @test_vbrc_memref_normal(%arg1: memref<128xf32>) -> memref<128xf32> {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK-NOT: generic
  // CHECK: linalg.fill

  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() : memref<128xf32>
  hivm.hir.vbrc ins(%cst: f32) outs(%alloc_1: memref<128xf32>)
  return %alloc_1: memref<128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vbrc_memref_broadcast
func.func @test_vbrc_memref_broadcast(%arg0: memref<32x1xf32>) -> memref<32x64xf32> {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK-NOT: generic

  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<32x64xf32>
  // CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<32x1xf32> into memref<32xf32>
  // CHECK: linalg.broadcast ins(%[[COLLAPSED]] : memref<32xf32>) outs(%[[ALLOC]] : memref<32x64xf32>) dimensions = [1]
  // CHECK: return %[[ALLOC]] : memref<32x64xf32>

  %alloc = memref.alloc() : memref<32x64xf32>
  hivm.hir.vbrc ins(%arg0: memref<32x1xf32>) outs(%alloc: memref<32x64xf32>) broadcast = [1]
  return %alloc: memref<32x64xf32>
}
