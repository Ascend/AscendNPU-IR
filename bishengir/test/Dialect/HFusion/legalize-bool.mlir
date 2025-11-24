// RUN: bishengir-opt -hfusion-legalize-bool %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @some_kernel_entry(%arg0: i8, %arg1: i32, %arg2: i8, %arg3: f32, %arg4: i8, %arg5: i32) -> (i8, i32)
func.func @some_kernel_entry(%arg0: i1, %arg1: i32, %arg2: i1, %arg3: f32, %arg4: i1, %arg5: i32) -> (i1, i32)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-NEXT: %[[arg4casted:.*]] = arith.trunci %arg4 : i8 to i1
  // CHECK-NEXT: %[[arg0casted:.*]] = arith.trunci %arg0 : i8 to i1
  // CHECK-NEXT: %[[first_and:.*]] = arith.andi %[[arg0casted]], %[[arg4casted]] : i1
  // CHECK-NEXT: %[[exted:.*]] = arith.extsi %[[first_and]] : i1 to i8
  // CHECK: return %[[exted]], %{{.*}} : i8, i32
  %2 = arith.andi %arg0, %arg4 : i1
  %3 = arith.addi %arg1, %arg5 : i32
  func.return %2, %3 : i1, i32
}

// CHECK-LABEL: @some_other_func(%arg0: i1) -> i1
func.func @some_other_func(%arg0: i1) -> i1 {
  // CHECK: return %arg0 : i1
  func.return %arg0 : i1
}

// -----

// CHECK-LABEL: @legalize_tensor_bool
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<24x1x1x1xi8>) outs({{.*}} : tensor<24x1x1x1xf32>)
func.func @legalize_tensor_bool(%arg0: tensor<24xi1>, %arg1: tensor<24x3x256x192xf32>) -> tensor<24x3x256x192xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [24, 1, 1, 1] : tensor<24xi1> into tensor<24x1x1x1xi1>
  %0 = tensor.empty() : tensor<24x1x1x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<24x1x1x1xi1>) outs(%0 : tensor<24x1x1x1xf32>) -> tensor<24x1x1x1xf32>
  %2 = tensor.empty() : tensor<24x3x256x192xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1, 2, 3]] : tensor<24x1x1x1xf32> into tensor<24xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<24xf32>) outs(%2 : tensor<24x3x256x192xf32>) dimensions = [1, 2, 3] 
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg1 : tensor<24x3x256x192xf32>, tensor<24x3x256x192xf32>) outs(%2 : tensor<24x3x256x192xf32>) -> tensor<24x3x256x192xf32>
  return %3 : tensor<24x3x256x192xf32>
}

// -----

module {
  // CHECK-LABEL: @Fused_Cast_Mul_Add_Add_fusion_0
  // CHECK: %[[ARG0:.*]]: tensor<200x200xi8>
  // CHECK: %[[ONE:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<40000xi8>) outs({{.*}} : tensor<40000xi64>)
  func.func @Fused_Cast_Mul_Add_Add_fusion_0(%arg0: tensor<200x200xi1>) -> tensor<40000xi64> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<40000xi64>
    %c16_i64 = arith.constant 16 : i64
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<200x200xi1> into tensor<40000xi1>
    %1 = hfusion.cast {fun = #hfusion.round_mode<rint>} ins(%collapsed : tensor<40000xi1>) outs(%0 : tensor<40000xi64>) -> tensor<40000xi64>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %c16_i64 : tensor<40000xi64>, i64) outs(%0 : tensor<40000xi64>) -> tensor<40000xi64>
    return %2 : tensor<40000xi64>
  }
  // CHECK-LABEL: @Fused_Cast_Mul_Add_Add_fusion
  // CHECK: %[[ARG0:.*]]: tensor<200x200xi8>
  func.func @Fused_Cast_Mul_Add_Add_fusion(%arg0: tensor<200x200xi1>) -> tensor<40000xi64> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = call @Fused_Cast_Mul_Add_Add_fusion_0(%arg0) : (tensor<200x200xi1>) -> tensor<40000xi64>
    return %0 : tensor<40000xi64>
  }
}

// -----

module {
  // CHECK-LABEL: @Fused_SubExt_ReLU_Minimum_Equal_fusion_0
  func.func @Fused_SubExt_ReLU_Minimum_Equal_fusion_0(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
    %0 = tensor.empty(%dim) : tensor<?xi1>
    %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg0 : tensor<?xi32>, tensor<?xi32>) outs(%0 : tensor<?xi1>) -> tensor<?xi1>
    return %arg0, %1 : tensor<?xi32>, tensor<?xi1>
  }
  // Fused_SubExt_ReLU_Minimum_Equal_fusion
  // CHECK: call @Fused_SubExt_ReLU_Minimum_Equal_fusion_0[[_:.*]] tensor<?xi8>
  func.func @Fused_SubExt_ReLU_Minimum_Equal_fusion(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi1>) attributes {OperatorType = "Broadcast", compute_capability = "", frontend_symbol = {input_0 = ["s50", "s51"], output_0 = ["s50", "s51"], output_1 = ["s50", "s51"]}, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, mindspore_kernel, process = "aicore"} {
    %0:2 = call @Fused_SubExt_ReLU_Minimum_Equal_fusion_0(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi1>)
    return %0#0, %0#1 : tensor<?xi32>, tensor<?xi1>
  }

}

// -----
// CHECK-LABEL: collapse_before_process(
// CHECK: collapse
// CHECK: dim
// CHECK: dim
// CHECK: cast
// CHECK: return
module {
  func.func @collapse_before_process(%arg0: tensor<?x?x1xi1>) -> tensor<?x?xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<?x?x1xi1> into tensor<?x?xi1>
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x1xi1>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x1xi1>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xbf16>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%collapsed : tensor<?x?xi1>) outs(%0 : tensor<?x?xbf16>) -> tensor<?x?xbf16>
    return %1 : tensor<?x?xbf16>
  }
  func.func @collapse_before_process_1(%arg0: tensor<?x?x1xi1>, %arg1: tensor<?x?x7168xbf16>) -> tensor<?x?xbf16> attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["s91", "s92", "1"], input_1 = ["s91", "s92", "7168"], output_0 = ["s91", "s92", "7168"]}, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, mindspore_kernel, process = "aicore"} {
    %0 = call @collapse_before_process(%arg0) : (tensor<?x?x1xi1>) -> tensor<?x?xbf16>
    return %0 : tensor<?x?xbf16>
  }
}

// -----
// CHECK-LABEL: fold_cast(
// CHECK: hfusion.cast
// CHECK-NOT: hfusion.cast
func.func @fold_cast(%arg0: tensor<1x32768x3584xbf16>, %arg1: tensor<1x32768x1xi1>) ->  tensor<1x32768x1xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = tensor.empty() : tensor<1x32768x1xbf16>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1x32768x1xi1>) outs(%0 : tensor<1x32768x1xbf16>) -> tensor<1x32768x1xbf16>
  return %1 :  tensor<1x32768x1xbf16>
}

// -----
// CHECK: bool_use1_expand_use2(
// CHECK-SAME: %[[arg0:.*]]: tensor<1x16384xi64>
// CHECK-SAME: %[[arg1:.*]]: tensor<1x16384xi8>
// CHECK: hfusion.cast
// CHECK-SAME: ins(%[[arg1]] : tensor<1x16384xi8>)
// CHECK: %[[expand:.*]] = tensor.expand_shape %[[arg1]]
// CHECK: hfusion.cast
// CHECK-SAME: ins(%[[expand]] : tensor<1x16384x1xi8>)
module {
  func.func @bool_use1_expand_use2(%arg0: tensor<1x16384xi64>, %arg1: tensor<1x16384xi1>) -> (tensor<1x16384xi1>, tensor<1x16384x1xi1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %5 = tensor.empty() : tensor<1x16384xi1>
    %6 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%arg1 : tensor<1x16384xi1>) outs(%5 : tensor<1x16384xi1>) -> tensor<1x16384xi1>
    %expanded = tensor.expand_shape %arg1 [[0], [1, 2]] output_shape [1, 16384, 1] : tensor<1x16384xi1> into tensor<1x16384x1xi1>
    %14 = tensor.empty() : tensor<1x16384x1xi1>
    %15 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%expanded : tensor<1x16384x1xi1>) outs(%14 : tensor<1x16384x1xi1>) -> tensor<1x16384x1xi1>
    return %6, %15 : tensor<1x16384xi1>, tensor<1x16384x1xi1>
  }
}

// -----

// CHECK-LABEL: @legalize_tensor_chain
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<24x9xi8>) outs(%{{.*}} : tensor<24x9xi8>) -> tensor<24x9xi8>
func.func @legalize_tensor_chain(%arg0: tensor<24x9xi1>) -> tensor<3x2x1x12x3xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3], [4, 5]] output_shape [3, 2, 1, 4, 3, 3] : tensor<24x9xi1> into tensor<3x2x1x4x3x3xi1>
  %collapsed_2 = tensor.collapse_shape %expanded [[0], [1], [2], [3, 4], [5]] : tensor<3x2x1x4x3x3xi1> into tensor<3x2x1x12x3xi1>
  return %collapsed_2 : tensor<3x2x1x12x3xi1>
}
