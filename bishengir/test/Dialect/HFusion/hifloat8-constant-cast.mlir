// REQUIRES: bishengir-hifloat8
// RUN: bishengir-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @keep_hifloat8_cast
// CHECK: {{hfusion.cast|arith.truncf}}
// CHECK-SAME: round_mode = #hfusion.round_mode<round>
// CHECK-SAME: tensor<64xf32>
// CHECK-SAME: tensor<64xf8HIF>
func.func @keep_hifloat8_cast() -> tensor<64xf8HIF> {
  %input = arith.constant dense<1.0> : tensor<64xf32>
  %empty = tensor.empty() : tensor<64xf8HIF>
  %cast = hfusion.cast {round_mode = #hfusion.round_mode<round>}
      ins(%input : tensor<64xf32>) outs(%empty : tensor<64xf8HIF>)
      -> tensor<64xf8HIF>
  return %cast : tensor<64xf8HIF>
}

// CHECK-LABEL: func.func @fold_supported_cast
// CHECK: arith.constant dense<1.000000e+00> : tensor<64xf16>
// CHECK-NOT: hfusion.cast
func.func @fold_supported_cast() -> tensor<64xf16> {
  %input = arith.constant dense<1.0> : tensor<64xf32>
  %empty = tensor.empty() : tensor<64xf16>
  %cast = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
      ins(%input : tensor<64xf32>) outs(%empty : tensor<64xf16>)
      -> tensor<64xf16>
  return %cast : tensor<64xf16>
}
