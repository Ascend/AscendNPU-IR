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

  // 1D fp32 cumsum bypasses scalar lowering → stays as hivm.hir.vcumsum.
  func.func @test_cumsum_a3_1d_f32() {
    // CHECK-LABEL: func.func @test_cumsum_a3_1d_f32
    // CHECK: hivm.hir.vcumsum
    // CHECK-NOT: scf.for
    // CHECK: return
    %src = memref.alloc() : memref<16xf32>
    %dst = memref.alloc() : memref<16xf32>
    hivm.hir.vcumsum ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) cum_dims = [0] reverse = false
    return
  }

  // 1D fp16 cumsum still lowers to scalar loops (only fp32 uses library call).
  func.func @test_cumsum_a3_1d_f16() {
    // CHECK-LABEL: func.func @test_cumsum_a3_1d_f16
    // CHECK: scf.for
    // CHECK: arith.addf
    // CHECK-NOT: hivm.hir.vcumsum
    // CHECK: return
    %src = memref.alloc() : memref<16xf16>
    %dst = memref.alloc() : memref<16xf16>
    hivm.hir.vcumsum ins(%src : memref<16xf16>) outs(%dst : memref<16xf16>) cum_dims = [0] reverse = false
    return
  }

  // 1D fp32 reverse cumsum still lowers to scalar loops.
  func.func @test_cumsum_a3_1d_f32_reverse() {
    // CHECK-LABEL: func.func @test_cumsum_a3_1d_f32_reverse
    // CHECK: scf.for
    // CHECK: arith.addf
    // CHECK-NOT: hivm.hir.vcumsum
    // CHECK: return
    %src = memref.alloc() : memref<16xf32>
    %dst = memref.alloc() : memref<16xf32>
    hivm.hir.vcumsum ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) cum_dims = [0] reverse = true
    return
  }

  // 2D fp32 cumsum with -1-axis reduce still lowers to scalar loops (only 1D uses library call).
  func.func @test_cumsum_a3_2d_f32() {
    // CHECK-LABEL: func.func @test_cumsum_a3_2d_f32
    // CHECK: scf.for
    // CHECK: arith.addf
    // CHECK-NOT: hivm.hir.vcumsum
    // CHECK: return
    %src = memref.alloc() : memref<2x16xf32>
    %dst = memref.alloc() : memref<2x16xf32>
    hivm.hir.vcumsum ins(%src : memref<2x16xf32>) outs(%dst : memref<2x16xf32>) cum_dims = [1] reverse = false
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
