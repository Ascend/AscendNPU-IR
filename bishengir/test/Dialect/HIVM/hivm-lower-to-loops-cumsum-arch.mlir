// RUN: bishengir-opt %s -hivm-lower-to-loops -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_cumsum_a3_1d_i16() {
    // CHECK-LABEL: func.func @test_cumsum_a3_1d_i16
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK-NOT: hivm.hir.vcumsum
    // CHECK: return
    %src = memref.alloc() : memref<16xi16>
    %dst = memref.alloc() : memref<16xi16>
    hivm.hir.vcumsum ins(%src : memref<16xi16>) outs(%dst : memref<16xi16>) cum_dims = [0] reverse = false
    return
  }

  func.func @test_cumsum_a3_1d_i64() {
    // CHECK-LABEL: func.func @test_cumsum_a3_1d_i64
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK-NOT: hivm.hir.vcumsum
    // CHECK: return
    %src = memref.alloc() : memref<16xi64>
    %dst = memref.alloc() : memref<16xi64>
    hivm.hir.vcumsum ins(%src : memref<16xi64>) outs(%dst : memref<16xi64>) cum_dims = [0] reverse = false
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_cumsum_a5_1d_i16() {
    // CHECK-LABEL: func.func @test_cumsum_a5_1d_i16
    // CHECK: hivm.hir.vcumsum
    // CHECK-NOT: scf.for
    // CHECK: return
    %src = memref.alloc() : memref<16xi16>
    %dst = memref.alloc() : memref<16xi16>
    hivm.hir.vcumsum ins(%src : memref<16xi16>) outs(%dst : memref<16xi16>) cum_dims = [0] reverse = false
    return
  }

  func.func @test_cumsum_a5_1d_i64() {
    // CHECK-LABEL: func.func @test_cumsum_a5_1d_i64
    // CHECK: hivm.hir.vcumsum
    // CHECK-NOT: scf.for
    // CHECK: return
    %src = memref.alloc() : memref<16xi64>
    %dst = memref.alloc() : memref<16xi64>
    hivm.hir.vcumsum ins(%src : memref<16xi64>) outs(%dst : memref<16xi64>) cum_dims = [0] reverse = false
    return
  }

  func.func @test_cumsum_a5_2d_i64() {
    // CHECK-LABEL: func.func @test_cumsum_a5_2d_i64
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK-NOT: hivm.hir.vcumsum
    // CHECK: return
    %src = memref.alloc() : memref<2x16xi64>
    %dst = memref.alloc() : memref<2x16xi64>
    hivm.hir.vcumsum ins(%src : memref<2x16xi64>) outs(%dst : memref<2x16xi64>) cum_dims = [0] reverse = false
    return
  }
}
