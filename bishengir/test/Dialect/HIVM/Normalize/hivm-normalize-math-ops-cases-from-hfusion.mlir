// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt --hfusion-normalize-ops --convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics
//
// Excluded source HFusion testcases from
// `bishengir/test/Dialect/HFusion/Normalize/hfusion-normalize-math-ops.mlir`:
// - `test_NormalizeExp2_hfusion_elemwise_unary_exp2`
//   Excluded because `exp2` currently depends on the brand-new HIVM normalize-
//   source op `hivm.hir.vexp2`, and this branch does not keep the
//   `convert-hfusion-to-hivm` mapping from HFusion `exp2` to `vexp2`.
// - `test_NormalizeExp2_hfusion_elemwise_unary_exp2_f16`
//   Same reason as above.

// CHECK-LABEL: func.func @test_NormalizeErf_hfusion_elemwise_erf
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 3.920000
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -3.920000
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[ARG0]], %[[UPPER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[MIN]], %[[LOWER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[SQUARE_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SQUARE:.*]] = hivm.hir.vmul ins(%[[MAX]], %[[MAX]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[SQUARE_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<1024xf32>, tensor<1024xf32>) outs(%{{.*}} : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[DIV]]
func.func @test_NormalizeErf_hfusion_elemwise_erf(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeErf_hfusion_elemwise_erf_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 3.920000
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -3.920000
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1024xf16>) outs(%[[EMPTY_F32]] : tensor<1024xf32>){{( round_mode = <round>)?}} -> tensor<1024xf32>
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[CAST]], %[[UPPER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[MIN]], %[[LOWER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[SQUARE_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SQUARE:.*]] = hivm.hir.vmul ins(%[[MAX]], %[[MAX]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[SQUARE_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<1024xf32>, tensor<1024xf32>) outs(%{{.*}} : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST_BACK:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%{{.*}} : tensor<1024xf16>) round_mode = <round> -> tensor<1024xf16>
// CHECK: return %[[CAST_BACK]]
func.func @test_NormalizeErf_hfusion_elemwise_erf_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}
