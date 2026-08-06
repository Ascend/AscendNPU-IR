// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeCopysign_hfusion_elemwise_binary_f32(
// CHECK-SAME: %[[MAG:.*]]: tensor<8xf32>, %[[SIGN:.*]]: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NOT: #hfusion.binary_fn<copysign>
// CHECK-DAG: %[[SIGN_MASK:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG: %[[MAG_MASK:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[MAG_BITS:.*]] = hfusion.bitcast ins(%[[MAG]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xi32>) -> tensor<8xi32>
// CHECK: %[[SIGN_BITS:.*]] = hfusion.bitcast ins(%[[SIGN]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xi32>) -> tensor<8xi32>
// CHECK: %[[MASKED_MAG:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG_BITS]], %[[MAG_MASK]] : tensor<8xi32>, i32)
// CHECK: %[[MASKED_SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SIGN_BITS]], %[[SIGN_MASK]] : tensor<8xi32>, i32)
// CHECK: %[[RET_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[MASKED_MAG]], %[[MASKED_SIGN]] : tensor<8xi32>, tensor<8xi32>)
// CHECK: %[[RET:.*]] = hfusion.bitcast ins(%[[RET_BITS]] : tensor<8xi32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: return %[[RET]] : tensor<8xf32>
// CHECK: }
func.func @test_NormalizeCopysign_hfusion_elemwise_binary_f32(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tensor.empty() : tensor<8xf32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<copysign>} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCopysign_hfusion_elemwise_binary_f16(
// CHECK-SAME: %[[MAG:.*]]: tensor<8xf16>, %[[SIGN:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK-NOT: #hfusion.binary_fn<copysign>
// CHECK-DAG: %[[SIGN_MASK:.*]] = arith.constant -32768 : i16
// CHECK-DAG: %[[MAG_MASK:.*]] = arith.constant 32767 : i16
// CHECK: %[[MAG_BITS:.*]] = hfusion.bitcast ins(%[[MAG]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[SIGN_BITS:.*]] = hfusion.bitcast ins(%[[SIGN]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[MASKED_MAG:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG_BITS]], %[[MAG_MASK]] : tensor<8xi16>, i16)
// CHECK: %[[MASKED_SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SIGN_BITS]], %[[SIGN_MASK]] : tensor<8xi16>, i16)
// CHECK: %[[RET_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[MASKED_MAG]], %[[MASKED_SIGN]] : tensor<8xi16>, tensor<8xi16>)
// CHECK: %[[RET:.*]] = hfusion.bitcast ins(%[[RET_BITS]] : tensor<8xi16>) outs(%{{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: return %[[RET]] : tensor<8xf16>
// CHECK: }
func.func @test_NormalizeCopysign_hfusion_elemwise_binary_f16(%arg0: tensor<8xf16>, %arg1: tensor<8xf16>) -> tensor<8xf16> {
  %0 = tensor.empty() : tensor<8xf16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<copysign>} ins(%arg0, %arg1 : tensor<8xf16>, tensor<8xf16>) outs(%0 : tensor<8xf16>) -> tensor<8xf16>
  return %1 : tensor<8xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCopysign_hfusion_elemwise_binary_bf16(
// CHECK-SAME: %[[MAG:.*]]: tensor<8xbf16>, %[[SIGN:.*]]: tensor<8xbf16>) -> tensor<8xbf16> {
// CHECK-NOT: #hfusion.binary_fn<copysign>
// CHECK-DAG: %[[SIGN_MASK:.*]] = arith.constant -32768 : i16
// CHECK-DAG: %[[MAG_MASK:.*]] = arith.constant 32767 : i16
// CHECK: %[[MAG_BITS:.*]] = hfusion.bitcast ins(%[[MAG]] : tensor<8xbf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[SIGN_BITS:.*]] = hfusion.bitcast ins(%[[SIGN]] : tensor<8xbf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[MASKED_MAG:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG_BITS]], %[[MAG_MASK]] : tensor<8xi16>, i16)
// CHECK: %[[MASKED_SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SIGN_BITS]], %[[SIGN_MASK]] : tensor<8xi16>, i16)
// CHECK: %[[RET_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[MASKED_MAG]], %[[MASKED_SIGN]] : tensor<8xi16>, tensor<8xi16>)
// CHECK: %[[RET:.*]] = hfusion.bitcast ins(%[[RET_BITS]] : tensor<8xi16>) outs(%{{.*}} : tensor<8xbf16>) -> tensor<8xbf16>
// CHECK: return %[[RET]] : tensor<8xbf16>
// CHECK: }
func.func @test_NormalizeCopysign_hfusion_elemwise_binary_bf16(%arg0: tensor<8xbf16>, %arg1: tensor<8xbf16>) -> tensor<8xbf16> {
  %0 = tensor.empty() : tensor<8xbf16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<copysign>} ins(%arg0, %arg1 : tensor<8xbf16>, tensor<8xbf16>) outs(%0 : tensor<8xbf16>) -> tensor<8xbf16>
  return %1 : tensor<8xbf16>
}
