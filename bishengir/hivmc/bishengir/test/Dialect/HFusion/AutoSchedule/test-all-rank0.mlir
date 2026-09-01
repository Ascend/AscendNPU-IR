// RUN: bishengir-opt %s -hfusion-auto-schedule --split-input-file | FileCheck %s

// CHECK: @all_rank0_tiling_function
// CHECK-NEXT: return

// CHECK: @all_rank0
// CHECK: linalg.fill
// CHECK: hfusion.store
func.func @all_rank0(%arg0: tensor<f32>) -> (tensor<f32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

// CHECK: @anchor_rank0
// CHECK: linalg.fill
// CHECK: hfusion.store

func.func @anchor_rank0(%arg0: tensor<1x1xi64>, %arg1: tensor<1x1xi32>) -> tensor<1xi64> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %c-9223372036854775808_i64 = arith.constant -9223372036854775808 : i64
  %0 = tensor.empty() : tensor<i64>
  %1 = linalg.fill ins(%c-9223372036854775808_i64 : i64) outs(%0 : tensor<i64>) -> tensor<i64>
  %collapsed = tensor.collapse_shape %arg0 [] : tensor<1x1xi64> into tensor<i64>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%collapsed, %1 : tensor<i64>, tensor<i64>) outs(%0 : tensor<i64>) -> tensor<i64>
  %expanded = tensor.expand_shape %2 [] output_shape [1] : tensor<i64> into tensor<1xi64>
  return %expanded : tensor<1xi64>
}
