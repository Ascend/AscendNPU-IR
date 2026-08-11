// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// Test vcumsum on float32 tensor along axis 0
// CHECK-LABEL: func.func @test_vcumsum_f32
func.func @test_vcumsum_f32(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tensor.empty():tensor<10xf32>
    // CHECK: tt.scan
    // CHECK: arith.addf
    // CHECK: tt.scan.return
    %1 = hivm.hir.vcumsum ins(%arg0 : tensor<10xf32>) outs(%0 : tensor<10xf32>) cum_dims = [0] reverse = false -> tensor<10xf32>
    return %1 : tensor<10xf32>
}

// -----

// Test vcumsum on integer tensor along axis 0
// CHECK-LABEL: func.func @test_vcumsum_i32
func.func @test_vcumsum_i32(%arg0: tensor<10xi32>) -> tensor<10xi32> {
    %0 = tensor.empty():tensor<10xi32>
    // CHECK: tt.scan
    // CHECK: arith.addi
    // CHECK: tt.scan.return
    %1 = hivm.hir.vcumsum ins(%arg0 : tensor<10xi32>) outs(%0 : tensor<10xi32>) cum_dims = [0] reverse = false -> tensor<10xi32>
    return %1 : tensor<10xi32>
}

// -----

// Test vcumsum with reverse=true
// CHECK-LABEL: func.func @test_vcumsum_reverse_f32
func.func @test_vcumsum_reverse_f32(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tensor.empty():tensor<10xf32>
    // CHECK: tt.scan
    // CHECK-SAME: reverse = true
    // CHECK: arith.addf
    // CHECK: tt.scan.return
    %1 = hivm.hir.vcumsum ins(%arg0 : tensor<10xf32>) outs(%0 : tensor<10xf32>) cum_dims = [0] reverse = true -> tensor<10xf32>
    return %1 : tensor<10xf32>
}

// -----

// Test vcumsum on 2D tensor along axis 1
// CHECK-LABEL: func.func @test_vcumsum_2d_f32
func.func @test_vcumsum_2d_f32(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = tensor.empty():tensor<8x16xf32>
    // CHECK: tt.scan
    // CHECK-SAME: axis = 1
    // CHECK: arith.addf
    // CHECK: tt.scan.return
    %1 = hivm.hir.vcumsum ins(%arg0 : tensor<8x16xf32>) outs(%0 : tensor<8x16xf32>) cum_dims = [1] reverse = false -> tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
}

// -----

// Test vcumsum on 2D integer tensor along axis 1
// CHECK-LABEL: func.func @test_vcumsum_2d_i32
func.func @test_vcumsum_2d_i32(%arg0: tensor<8x16xi32>) -> tensor<8x16xi32> {
    %0 = tensor.empty():tensor<8x16xi32>
    // CHECK: tt.scan
    // CHECK-SAME: axis = 1
    // CHECK: arith.addi
    // CHECK: tt.scan.return
    %1 = hivm.hir.vcumsum ins(%arg0 : tensor<8x16xi32>) outs(%0 : tensor<8x16xi32>) cum_dims = [1] reverse = false -> tensor<8x16xi32>
    return %1 : tensor<8x16xi32>
}
