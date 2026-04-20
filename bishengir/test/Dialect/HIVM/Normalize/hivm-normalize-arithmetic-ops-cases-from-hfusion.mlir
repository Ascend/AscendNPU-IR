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

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_hivm_mul_div_by_one
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[VDIV:.*]] = hivm.hir.vdiv ins(%[[VBRC]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[VMUL0:.*]] = hivm.hir.vmul ins(%[[VDIV]], %[[ARG1]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[VMUL1:.*]] = hivm.hir.vmul ins(%[[VMUL0]], %[[VDIV]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[VMUL1]]
func.func @test_NormalizeMulRec_hivm_mul_div_by_one(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_hivm_mul_div_by_one_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV0:.*]] = hivm.hir.vdiv ins(%[[ARG1]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV1:.*]] = hivm.hir.vdiv ins(%[[DIV0]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV1]]
func.func @test_NormalizeMulRec_hivm_mul_div_by_one_rec(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}
