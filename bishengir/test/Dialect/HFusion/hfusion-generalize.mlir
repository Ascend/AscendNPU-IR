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
// CHECK-NOT: linalg.generic
func.func @test_gather_case0(%src:tensor<5x6x1xf16>, %idx:tensor<5x6x1xi32>) -> tensor<5x6x1xf16> {
  %init = tensor.empty() : tensor<5x6x1xf16>
  %res = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%src, %idx : tensor<5x6x1xf16>, tensor<5x6x1xi32>) outs(%init:tensor<5x6x1xf16>) axis = 2 -> tensor<5x6x1xf16>
  return %res : tensor<5x6x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_gather_case1
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] : tensor<5x6x1xf16> into tensor<5x6xf16>
// CHECK: linalg.broadcast ins({{.*}} : tensor<5x6xf16>) outs({{.*}} : tensor<5x6x3xf16>) dimensions = {{\[}}2]
func.func @test_gather_case1(%src:tensor<5x6x1xf16>, %idx:tensor<5x6x3xi32>) -> tensor<5x6x3xf16> {
  %init = tensor.empty() : tensor<5x6x3xf16>
  %res = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%src, %idx : tensor<5x6x1xf16>, tensor<5x6x3xi32>) outs(%init:tensor<5x6x3xf16>) axis = 2 -> tensor<5x6x3xf16>
  return %res : tensor<5x6x3xf16>
}