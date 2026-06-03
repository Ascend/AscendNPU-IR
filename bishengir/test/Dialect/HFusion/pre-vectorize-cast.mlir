// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_cast_signed_i1_to_f16
// CHECK-NOT: arith.uitofp
// CHECK: arith.sitofp
func.func @test_cast_signed_i1_to_f16(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xi1> {
  %cst = arith.constant -1.000000e+00 : f16
  %0 = tensor.empty() : tensor<128xi1>
  %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xi1>) -> tensor<128xi1>
  %2 = tensor.empty() : tensor<128xf16>
  %3 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%1 : tensor<128xi1>) outs(%2 : tensor<128xf16>) -> tensor<128xf16>
  %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%3, %cst : tensor<128xf16>, f16) outs(%0 : tensor<128xi1>) -> tensor<128xi1>
  return %4 : tensor<128xi1>
}