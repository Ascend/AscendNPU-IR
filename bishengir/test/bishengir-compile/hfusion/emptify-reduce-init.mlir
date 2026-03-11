// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

func.func @test_emptify_reduce_init(%arg0: tensor<4x142xi32>) -> tensor<4xi32> {
  %empty = tensor.empty(): tensor<4xi32>
  // CHECK: %0 = tensor.empty() : tensor<4xi32>
  %zero = arith.constant 0 : i32
  // CHECK-NOT: linalg.fill
  %filled = linalg.fill ins(%zero : i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
  %res = linalg.reduce {arith.addi} ins(%arg0 : tensor<4x142xi32>) outs(%filled : tensor<4xi32>) dimensions = [1] {}
  return %res : tensor<4xi32>
}