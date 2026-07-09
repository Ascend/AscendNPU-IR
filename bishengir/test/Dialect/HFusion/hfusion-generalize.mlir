// RUN: bishengir-opt --canonicalize --cse --hfusion-generalize %s --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_gather_normal
// CHECK: tensor.empty() : tensor<5x3x1xf16>
// CHECK: linalg.generic
// CHECK: linalg.index 2 : index
// CHECK: arith.index_cast {{.*}} : index to i32
// CHECK: arith.cmpi eq, {{.*}}, {{.*}} : i32
// CHECK: arith.select {{.*}}, {{.*}}, {{.*}} : f16
// CHECK: linalg.yield {{.*}} : f16
func.func @test_gather_normal(%arg0: tensor<5x6x1xf16>, %arg1: tensor<5x3x1xi32>) -> tensor<5x3x1xf16> {
  %0 = tensor.empty() : tensor<5x3x1xf16>
  %1 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%arg0, %arg1 : tensor<5x6x1xf16>, tensor<5x3x1xi32>) outs(%0 : tensor<5x3x1xf16>) axis = 1 -> tensor<5x3x1xf16>
  return %1 : tensor<5x3x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_gather_case0
// CHECK: linalg.generic
// CHECK: linalg.index 3 : index
func.func @test_gather_case0(%src:tensor<5x6x1xf16>, %idx:tensor<5x6x1xi32>) -> tensor<5x6x1xf16> {
  %init = tensor.empty() : tensor<5x6x1xf16>
  %res = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%src, %idx : tensor<5x6x1xf16>, tensor<5x6x1xi32>) outs(%init:tensor<5x6x1xf16>) axis = 2 -> tensor<5x6x1xf16>
  return %res : tensor<5x6x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_gather_case1
// CHECK: linalg.generic
// CHECK: tensor<5x6x3xf16>
// CHECK: linalg.index 3 : index
func.func @test_gather_case1(%src:tensor<5x6x1xf16>, %idx:tensor<5x6x3xi32>) -> tensor<5x6x3xf16> {
  %init = tensor.empty() : tensor<5x6x3xf16>
  %res = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%src, %idx : tensor<5x6x1xf16>, tensor<5x6x3xi32>) outs(%init:tensor<5x6x3xf16>) axis = 2 -> tensor<5x6x3xf16>
  return %res : tensor<5x6x3xf16>
}

// -----

// CHECK-LABEL: func.func @test_cumsum_test_cumsum_compensated
// CHECK: hfusion.cumsum %{{.*}} {needs_compensation} : tensor<32x32xf32> cum_dims = [0] reverse = false -> tensor<32x32xf32>
func.func @test_cumsum_test_cumsum_compensated(%src:tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = tensor.empty() : tensor<32xf32>
  %cumres = hfusion.cumsum %src : tensor<32x32xf32> cum_dims = [0] reverse = false -> tensor<32x32xf32>
  %temp = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%src, %cumres : tensor<32x32xf32>, tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %reduced = linalg.reduce ins(%src : tensor<32x32xf32>) outs(%1 : tensor<32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %2 = arith.addf %in, %init : f32
      linalg.yield %2 : f32
    }
  %broadcasted = linalg.broadcast ins(%reduced : tensor<32xf32>) outs(%0 : tensor<32x32xf32>) dimensions = [0]
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%temp, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %res : tensor<32x32xf32>
}
