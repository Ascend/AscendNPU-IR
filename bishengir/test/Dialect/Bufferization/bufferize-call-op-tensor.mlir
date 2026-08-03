// RUN: bishengir-opt %s -func-bufferize -split-input-file | FileCheck %s

// CHECK-LABEL: func private @callee_returning_tensor
func.func private @callee_returning_tensor() -> tensor<16xf32>

// CHECK-LABEL: func @test_call_tensor
func.func @test_call_tensor() -> tensor<16xf32> {
  // CHECK: %[[VAL:.*]] = call @callee_returning_tensor()
  // CHECK: () -> memref<16xf32>
  %0 = call @callee_returning_tensor() : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
