// RUN: bishengir-opt -convert-tensor-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_splat
func.func @test_splat() -> (tensor<2x128x688xf32>, tensor<?x20x?xf32>) {
  // CHECK-NEXT:       %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  %cst = arith.constant 1.000000e+00 : f32
  // CHECK:       %[[EMPTY1:.*]] = tensor.empty() : tensor<2x128x688xf32>
  // CHECK:       %[[FILL1:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY1]] : tensor<2x128x688xf32>)
  %splat = tensor.splat %cst : tensor<2x128x688xf32>
  // CHECK-NEXT:       %[[M:.*]] = arith.constant 10 : index
  // CHECK-NEXT:       %[[N:.*]] = arith.constant 30 : index
  %m = arith.constant 10 : index
  %n = arith.constant 30 : index
  // CHECK:       %[[EMPTY2:.*]] = tensor.empty(%[[M]], %[[N]]) : tensor<?x20x?xf32>
  // CHECK:       %[[FILL2:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY2]] : tensor<?x20x?xf32>)
  %dynamic_splat = tensor.splat %cst[%m, %n] : tensor<?x20x?xf32>
  return %splat, %dynamic_splat : tensor<2x128x688xf32>, tensor<?x20x?xf32>
}
