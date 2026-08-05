// RUN: bishengir-opt --hfusion-legalize-bf16 %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_legalize_bf16_keeps_copysign(
// CHECK-SAME: %[[MAG:.*]]: tensor<8xbf16>, %[[SIGN:.*]]: tensor<8xbf16>) -> tensor<8xbf16> {
// CHECK-NOT: hfusion.cast
// CHECK: %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<copysign>} ins(%[[MAG]], %[[SIGN]] : tensor<8xbf16>, tensor<8xbf16>) outs(%{{.*}} : tensor<8xbf16>) -> tensor<8xbf16>
// CHECK: return %[[RET]] : tensor<8xbf16>
// CHECK: }
func.func @test_legalize_bf16_keeps_copysign(%arg0: tensor<8xbf16>, %arg1: tensor<8xbf16>) -> tensor<8xbf16> {
  %0 = tensor.empty() : tensor<8xbf16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<copysign>} ins(%arg0, %arg1 : tensor<8xbf16>, tensor<8xbf16>) outs(%0 : tensor<8xbf16>) -> tensor<8xbf16>
  return %1 : tensor<8xbf16>
}
