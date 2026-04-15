// RUN: bishengir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

module {
  // CHECK-LABEL: func.func @print_test
  // CHECK: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: tensor<32xf32>, %[[ARG2:.*]]: tensor<32xi8>)
  // CHECK: hfusion.print "PID: " {hex = true} %[[ARG0]] : i32
  // CHECK: hfusion.print "Val: " {hex = false} %[[ARG1]] : tensor<32xf32>
  // CHECK: hfusion.print "" {hex = false} %[[ARG2]] : tensor<32xi8>
  func.func @print_test(%arg0 : i32, %arg1 : tensor<32xf32>, %arg2 : tensor<32xi8>) {
    hfusion.print "PID: " {hex = true} %arg0 : i32
    hfusion.print "Val: " {hex = false} %arg1 : tensor<32xf32>
    hfusion.print "" {hex = false} %arg2 : tensor<32xi8>
    func.return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @gather_load_test
  // CHECK: hfusion.gather_load
  func.func @gather_load_test(%base : memref<?xf32>, %indices: tensor<16x400xi32>, %mask: tensor<16x400xi1>, %other: f32) {
    %c1_i64 = arith.constant 1 : i64
    %init = tensor.empty() : tensor<16x400xf32>
    %output = hfusion.gather_load  ins(%base : memref<?xf32>, %indices: tensor<16x400xi32>, %c1_i64: i64, %mask: tensor<16x400xi1>, %other: f32) outs(%init : tensor<16x400xf32>) {cache = #hfusion.cache_modifier<none>, evict = #hfusion.eviction_policy<EvictNormal>, isVolatile = false} -> tensor<16x400xf32>
    return
  }

  // CHECK-LABEL: func.func @scatter_store_test
  // CHECK: hfusion.scatter_store
  func.func @scatter_store_test(%base : memref<?xf32>, %indices: tensor<16x400xi32>, %data: tensor<16x400xf32>, %mask: tensor<16x400xi1>) {
    %c1_i64 = arith.constant 1 : i64
    hfusion.scatter_store ins(%indices: tensor<16x400xi32>, %data: tensor<16x400xf32>, %c1_i64: i64, %mask: tensor<16x400xi1>) outs(%base : memref<?xf32>) {cache = #hfusion.cache_modifier<none>, evict = #hfusion.eviction_policy<EvictNormal>}
    return
  }
}

// -----
module {
  func.func @gather_load_shape_mismatch(%base : memref<?xf32>, %indices: tensor<16x400xi32>, %mask: tensor<16x400xi1>, %other: f32) {
    %c1_i64 = arith.constant 1 : i64
    %init = tensor.empty() : tensor<8x800xf32>
    // expected-error @+1 {{'hfusion.gather_load' op failed to verify that all of {indices, dst} have same shape}}
    %output = hfusion.gather_load ins(%base : memref<?xf32>, %indices: tensor<16x400xi32>, %c1_i64: i64, %mask: tensor<16x400xi1>, %other: f32) outs(%init : tensor<8x800xf32>) -> tensor<8x800xf32>
    return
  }
}

// -----
module {
  func.func @scatter_store_shape_mismatch(%base : memref<?xf32>, %indices: tensor<16x400xi32>, %data: tensor<8x800xf32>, %mask: tensor<8x800xi1>) {
    %c1_i64 = arith.constant 1 : i64
    // expected-error @+1 {{'hfusion.scatter_store' op failed to verify that all of {indices, data} have same shape}}
    hfusion.scatter_store ins(%indices: tensor<16x400xi32>, %data: tensor<8x800xf32>, %c1_i64: i64, %mask: tensor<8x800xi1>) outs(%base : memref<?xf32>)
    return
  }
}

// -----
// CHECK-LABEL: @test_group_matmul
func.func @test_group_matmul(%w1 : tensor<2x?x?xf32>, %tokens : tensor<?x?xf32>, %tpe : tensor<2xi64>, %out : tensor<?x?xf32>) {
  %res = hfusion.group_matmul
    ins(%w1, %tokens, %tpe : tensor<2x?x?xf32>, tensor<?x?xf32>, tensor<2xi64>)
    outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    return
}

// -----
module {
  // CHECK-LABEL: func.func @embeddinggather_test
  // CHECK: hfusion.embedding_gather
  func.func @embeddinggather_test(%arg0: memref<?xf32>, %arg1: tensor<16x400xi32>, %arg2: tensor<16x400x32xf32>) attributes {DirectlyUsedGMArgIdxList = [0]} {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %c9000_i32 = arith.constant 9000 : i32
    %0 = tensor.empty() : tensor<16x400x32xf32>
    %1 = hfusion.embedding_gather ins(%arg0 : memref<?xf32>, %arg1 : tensor<16x400xi32>, %c9000_i32 : i32, [%c0_i32, %c0_i32, %c0_i32 : i32, i32, i32], [%c32_i32, %c4000_i32, %c32_i32 : i32, i32, i32]) outs(%0 : tensor<16x400x32xf32>) -> tensor<16x400x32xf32>
    return
  }
}
