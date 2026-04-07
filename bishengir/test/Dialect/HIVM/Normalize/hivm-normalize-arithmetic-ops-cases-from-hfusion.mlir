// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// This test verifies the correctness of HIVM normalization transformations
// applied to HFusion operators after conversion to HIVM dialect.

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf32>)
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<5x1xf32>)
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<5x1xf32>)
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> {
    %0 = tensor.empty() : tensor<5x1xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
    return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>)
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<16xf16>)
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<16xf16>)
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
    %0 = tensor.empty() : tensor<16xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
    return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt_dynshape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x?xf32>, %[[ARG1:.*]]: index)
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<5x?xf32>)
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<5x?xf32>)
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt_dynshape(%s: tensor<5x?xf32>, %d : index) -> tensor<5x?xf32> {
    %0 = tensor.empty(%d) : tensor<5x?xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%s : tensor<5x?xf32>) outs(%0 : tensor<5x?xf32>) -> tensor<5x?xf32>
    return %1 : tensor<5x?xf32>
}