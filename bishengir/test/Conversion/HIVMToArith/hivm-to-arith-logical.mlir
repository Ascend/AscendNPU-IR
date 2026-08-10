// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_vxor_i32
func.func @test_vxor_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.empty() : tensor<4xi32>
  // CHECK: %[[RET:.*]] = arith.xori %arg0, %arg1 : tensor<4xi32>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<4xi32>, tensor<4xi32>) outs(%0 : tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @test_vxor_i1
func.func @test_vxor_i1(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = tensor.empty() : tensor<4xi1>
  // CHECK: %[[RET:.*]] = arith.xori %arg0, %arg1 : tensor<4xi1>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<4xi1>, tensor<4xi1>) outs(%0 : tensor<4xi1>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// -----

// CHECK-LABEL: func.func @test_vnot_i32
func.func @test_vnot_i32(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.empty() : tensor<4xi32>
  // CHECK: %[[ALL_ONES:.*]] = arith.constant dense<-1> : tensor<4xi32>
  // CHECK: %[[RET:.*]] = arith.xori %arg0, %[[ALL_ONES]] : tensor<4xi32>
  %1 = hivm.hir.vnot ins(%arg0 : tensor<4xi32>) outs(%0 : tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @test_vnot_i1
func.func @test_vnot_i1(%arg0: tensor<4xi1>) -> tensor<4xi1> {
  %0 = tensor.empty() : tensor<4xi1>
  // CHECK: %[[ALL_ONES:.*]] = arith.constant dense<true> : tensor<4xi1>
  // CHECK: %[[RET:.*]] = arith.xori %arg0, %[[ALL_ONES]] : tensor<4xi1>
  %1 = hivm.hir.vnot ins(%arg0 : tensor<4xi1>) outs(%0 : tensor<4xi1>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}
