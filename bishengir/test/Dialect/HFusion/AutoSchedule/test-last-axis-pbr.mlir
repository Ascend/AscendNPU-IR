// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// CHECK: @rms_norm_f32
module {
  // CHECK: hacc.block_dim = 20
  func.func @rms_norm_f32(%arg0: tensor<256x128xf32>, %arg1: tensor<128xf32>, %arg2: f32, %arg3: tensor<256x128xf32>, %arg4: tensor<256xf32>) -> (tensor<256x128xf32>, tensor<256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<256x128xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<256x128xf32>
    %0 = tensor.empty() : tensor<256x128xf32>
    %1 = linalg.elemwise_binary {__1__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<256x128xf32>, tensor<256x128xf32>) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %2 = tensor.empty() : tensor<256xf32>
    %reduced = linalg.reduce { arith.addf } ins(%1 : tensor<256x128xf32>) outs(%2 : tensor<256xf32>) dimensions = [1]  {__2__}
    %3 = tensor.empty() : tensor<256xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %4 = arith.index_castui %dim_0 : index to i32
    %5 = arith.uitofp %4 : i32 to f32
    %6 = arith.divf %cst, %5 : f32
    %7 = linalg.elemwise_binary {__3__, fun = #linalg.binary_fn<mul>} ins(%reduced, %6 : tensor<256xf32>, f32) outs(%3 : tensor<256xf32>) -> tensor<256xf32>
    %8 = tensor.empty() : tensor<256xf32>
    %9 = linalg.elemwise_binary {__4__, fun = #linalg.binary_fn<add>} ins(%7, %arg2 : tensor<256xf32>, f32) outs(%8 : tensor<256xf32>) -> tensor<256xf32>
    %10 = tensor.empty() : tensor<256xf32>
    %11 = hfusion.elemwise_unary {__5__, fun = #hfusion.unary_fn<sqrt>} ins(%9 : tensor<256xf32>) outs(%10 : tensor<256xf32>) -> tensor<256xf32>
    %12 = hfusion.elemwise_unary {__6__, fun = #hfusion.unary_fn<rec>} ins(%11 : tensor<256xf32>) outs(%arg4 : tensor<256xf32>) -> tensor<256xf32>
    %13 = tensor.empty() : tensor<256x128xf32>
    %broadcasted = linalg.broadcast ins(%12 : tensor<256xf32>) outs(%13 : tensor<256x128xf32>) dimensions = [1]  {__7__}
    %14 = tensor.empty() : tensor<256x128xf32>
    %15 = linalg.elemwise_binary {__8__, fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg0 : tensor<256x128xf32>, tensor<256x128xf32>) outs(%14 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %16 = tensor.empty() : tensor<256x128xf32>
    %broadcasted_1 = linalg.broadcast ins(%arg1 : tensor<128xf32>) outs(%16 : tensor<256x128xf32>) dimensions = [0]  {__9__}
    %17 = linalg.elemwise_binary {__10__, fun = #linalg.binary_fn<mul>} ins(%broadcasted_1, %15 : tensor<256x128xf32>, tensor<256x128xf32>) outs(%arg3 : tensor<256x128xf32>) -> tensor<256x128xf32>
    return %17, %12 : tensor<256x128xf32>, tensor<256xf32>
  }
}

// -----

// CHECK: @rms_norm_f32
module {
  // CHECK: hacc.block_dim = 20
  func.func @rms_norm_f32(%arg0: tensor<256x12800xf32>, %arg1: tensor<12800xf32>, %arg2: f32, %arg3: tensor<256x12800xf32>, %arg4: tensor<256xf32>) -> (tensor<256x12800xf32>, tensor<256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<256x12800xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<256x12800xf32>
    %0 = tensor.empty() : tensor<256x12800xf32>
    %1 = linalg.elemwise_binary {__1__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<256x12800xf32>, tensor<256x12800xf32>) outs(%0 : tensor<256x12800xf32>) -> tensor<256x12800xf32>
    %2 = tensor.empty() : tensor<256xf32>
    %reduced = linalg.reduce { arith.addf } ins(%1 : tensor<256x12800xf32>) outs(%2 : tensor<256xf32>) dimensions = [1]  {__2__}
    %3 = tensor.empty() : tensor<256xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %4 = arith.index_castui %dim_0 : index to i32
    %5 = arith.uitofp %4 : i32 to f32
    %6 = arith.divf %cst, %5 : f32
    %7 = linalg.elemwise_binary {__3__, fun = #linalg.binary_fn<mul>} ins(%reduced, %6 : tensor<256xf32>, f32) outs(%3 : tensor<256xf32>) -> tensor<256xf32>
    %8 = tensor.empty() : tensor<256xf32>
    %9 = linalg.elemwise_binary {__4__, fun = #linalg.binary_fn<add>} ins(%7, %arg2 : tensor<256xf32>, f32) outs(%8 : tensor<256xf32>) -> tensor<256xf32>
    %10 = tensor.empty() : tensor<256xf32>
    %11 = hfusion.elemwise_unary {__5__, fun = #hfusion.unary_fn<sqrt>} ins(%9 : tensor<256xf32>) outs(%10 : tensor<256xf32>) -> tensor<256xf32>
    %12 = hfusion.elemwise_unary {__6__, fun = #hfusion.unary_fn<rec>} ins(%11 : tensor<256xf32>) outs(%arg4 : tensor<256xf32>) -> tensor<256xf32>
    %13 = tensor.empty() : tensor<256x12800xf32>
    %broadcasted = linalg.broadcast ins(%12 : tensor<256xf32>) outs(%13 : tensor<256x12800xf32>) dimensions = [1]  {__7__}
    %14 = tensor.empty() : tensor<256x12800xf32>
    %15 = linalg.elemwise_binary {__8__, fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg0 : tensor<256x12800xf32>, tensor<256x12800xf32>) outs(%14 : tensor<256x12800xf32>) -> tensor<256x12800xf32>
    %16 = tensor.empty() : tensor<256x12800xf32>
    %broadcasted_1 = linalg.broadcast ins(%arg1 : tensor<12800xf32>) outs(%16 : tensor<256x12800xf32>) dimensions = [0]  {__9__}
    %17 = linalg.elemwise_binary {__10__, fun = #linalg.binary_fn<mul>} ins(%broadcasted_1, %15 : tensor<256x12800xf32>, tensor<256x12800xf32>) outs(%arg3 : tensor<256x12800xf32>) -> tensor<256x12800xf32>
    return %17, %12 : tensor<256x12800xf32>, tensor<256xf32>
  }
}


// -----
// CHECK-LABEL: @model_0
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
func.func @model_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xf32>, %arg2: tensor<24x128xf32>) -> tensor<24x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %collapsed = tensor.collapse_shape %arg2 [[0, 1]] : tensor<24x128xf32> into tensor<3072xf32>
  %collapsed_0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x128x256x192xbf16> into tensor<3072x49152xbf16>
  %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<3072x49152xf32>
  %0 = tensor.empty() : tensor<3072x49152xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<3072x49152xbf16>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %collapsed_1 : tensor<3072x49152xf32>, tensor<3072x49152xf32>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
  %reduced = linalg.reduce ins(%2 : tensor<3072x49152xf32>) outs(%collapsed : tensor<3072xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %3 = arith.addf %in, %init : f32
      linalg.yield %3 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [24, 128] : tensor<3072xf32> into tensor<24x128xf32>
  return %expanded : tensor<24x128xf32>
}

// -----
// CHECK-LABEL: @model_1
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
func.func @model_1(%arg0: tensor<24x128x256xf32>,
                   %arg1: tensor<24xf32>,
                   %arg2: tensor<128xf32>,
                   %arg3: tensor<256xf32>,
                   %arg4: tensor<24x128x256xf32>) -> tensor<24x128x256xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  
  %empty = tensor.empty() : tensor<24x128xf32>
  %sum0 = linalg.reduce {arith.addf} ins(%arg0 : tensor<24x128x256xf32>) outs(%empty : tensor<24x128xf32>) dimensions = [2]
  
  %empty1 = tensor.empty() : tensor<24x128x256xf32>
  %broadcasted = linalg.broadcast ins(%sum0 : tensor<24x128xf32>) outs(%empty1 : tensor<24x128x256xf32>) dimensions = [2]
  %broadcasted_0 = linalg.broadcast ins(%arg1 : tensor<24xf32>) outs(%empty1 : tensor<24x128x256xf32>) dimensions = [1, 2]
  %broadcasted_1 = linalg.broadcast ins(%arg2 : tensor<128xf32>) outs(%empty1 : tensor<24x128x256xf32>) dimensions = [0, 2]
  %broadcasted_2 = linalg.broadcast ins(%arg3 : tensor<256xf32>) outs(%empty1 : tensor<24x128x256xf32>) dimensions = [0, 1]

  %4 = linalg.elemwise_binary { fun = #linalg.binary_fn<add> } ins(%broadcasted, %broadcasted_0 : tensor<24x128x256xf32>, tensor<24x128x256xf32>) outs(%empty1 : tensor<24x128x256xf32>) -> tensor<24x128x256xf32>
  %5 = linalg.elemwise_binary { fun = #linalg.binary_fn<add> } ins(%4, %broadcasted_1 : tensor<24x128x256xf32>, tensor<24x128x256xf32>) outs(%empty1 : tensor<24x128x256xf32>) -> tensor<24x128x256xf32>
  %6 = linalg.elemwise_binary { fun = #linalg.binary_fn<add> } ins(%5, %broadcasted_2 : tensor<24x128x256xf32>, tensor<24x128x256xf32>) outs(%arg4 : tensor<24x128x256xf32>) -> tensor<24x128x256xf32>
  return %6 : tensor<24x128x256xf32>
}

// -----
// CHECK-LABEL: @model_2
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
func.func @model_2(%arg0: tensor<24x128x25600xf32>,
                   %arg1: tensor<24xf32>,
                   %arg2: tensor<128xf32>,
                   %arg3: tensor<25600xf32>,
                   %arg4: tensor<24x128x25600xf32>) -> tensor<24x128x25600xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  
  %empty = tensor.empty() : tensor<24x128xf32>
  %sum0 = linalg.reduce {arith.addf} ins(%arg0 : tensor<24x128x25600xf32>) outs(%empty : tensor<24x128xf32>) dimensions = [2]
  
  %empty1 = tensor.empty() : tensor<24x128x25600xf32>
  %broadcasted = linalg.broadcast ins(%sum0 : tensor<24x128xf32>) outs(%empty1 : tensor<24x128x25600xf32>) dimensions = [2]
  %broadcasted_0 = linalg.broadcast ins(%arg1 : tensor<24xf32>) outs(%empty1 : tensor<24x128x25600xf32>) dimensions = [1, 2]
  %broadcasted_1 = linalg.broadcast ins(%arg2 : tensor<128xf32>) outs(%empty1 : tensor<24x128x25600xf32>) dimensions = [0, 2]
  %broadcasted_2 = linalg.broadcast ins(%arg3 : tensor<25600xf32>) outs(%empty1 : tensor<24x128x25600xf32>) dimensions = [0, 1]

  %4 = linalg.elemwise_binary { fun = #linalg.binary_fn<add> } ins(%broadcasted, %broadcasted_0 : tensor<24x128x25600xf32>, tensor<24x128x25600xf32>) outs(%empty1 : tensor<24x128x25600xf32>) -> tensor<24x128x25600xf32>
  %5 = linalg.elemwise_binary { fun = #linalg.binary_fn<add> } ins(%4, %broadcasted_1 : tensor<24x128x25600xf32>, tensor<24x128x25600xf32>) outs(%empty1 : tensor<24x128x25600xf32>) -> tensor<24x128x25600xf32>
  %6 = linalg.elemwise_binary { fun = #linalg.binary_fn<add> } ins(%5, %broadcasted_2 : tensor<24x128x25600xf32>, tensor<24x128x25600xf32>) outs(%arg4 : tensor<24x128x25600xf32>) -> tensor<24x128x25600xf32>
  return %6 : tensor<24x128x25600xf32>
}

// -----
// CHECK-LABEL: @model_28
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
func.func @model_28(%arg0: tensor<24x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<24x256xf32>, %arg3: tensor<24x32xf32>, %arg4: tensor<24x32xf32>) -> (tensor<24x32xf32>, tensor<24x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [32, 8] : tensor<256xf32> into tensor<32x8xf32>
  %0 = tensor.empty() : tensor<24x32x8xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [24, 32, 8] : tensor<24x256xf32> into tensor<24x32x8xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0], [1, 2]] output_shape [24, 32, 8] : tensor<24x256xf32> into tensor<24x32x8xf32>
  %1 = tensor.empty() : tensor<24x32x8xf32>
  %broadcasted = linalg.broadcast ins(%expanded : tensor<32x8xf32>) outs(%0 : tensor<24x32x8xf32>) dimensions = [0] 
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_0, %broadcasted : tensor<24x32x8xf32>, tensor<24x32x8xf32>) outs(%1 : tensor<24x32x8xf32>) -> tensor<24x32x8xf32>
  %reduced = linalg.reduce ins(%2 : tensor<24x32x8xf32>) outs(%arg3 : tensor<24x32xf32>) dimensions = [2] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_1, %broadcasted : tensor<24x32x8xf32>, tensor<24x32x8xf32>) outs(%1 : tensor<24x32x8xf32>) -> tensor<24x32x8xf32>
  %reduced_2 = linalg.reduce ins(%3 : tensor<24x32x8xf32>) outs(%arg4 : tensor<24x32xf32>) dimensions = [2] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  return %reduced, %reduced_2 : tensor<24x32xf32>, tensor<24x32xf32>
}

// -----

// CHECK-LABEL: @mlir_fused_convert_element_type_native_group_norm_0
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
module {
  // CHECK-NOT: hfusion.load
  // CHECK: scf.for
  // CHECK-NOT: hfusion.store
  // CHECK: scf.for
  // CHECK: hfusion.load
  // CHECK: hivm.block
  func.func @mlir_fused_convert_element_type_native_group_norm_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xf32>, %arg2: tensor<24x32x1x1xf32>) -> (tensor<24x128x256x192xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %collapsed = tensor.collapse_shape %arg2 [[0, 1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<768xf32>
    %expanded = tensor.expand_shape %arg1 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xf32> into tensor<24x32x1x1x4x256x192xf32>
    %collapsed_0 = tensor.collapse_shape %expanded [[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x4x256x192xf32> into tensor<768x196608xf32>
    %expanded_1 = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %collapsed_2 = tensor.collapse_shape %expanded_1 [[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x4x256x192xbf16> into tensor<768x196608xbf16>
    %0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_2 : tensor<768x196608xbf16>) outs(%collapsed_0 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %expanded_3 = tensor.expand_shape %0 [[0, 1, 2, 3], [4, 5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<768x196608xf32> into tensor<24x32x1x1x4x256x192xf32>
    %collapsed_4 = tensor.collapse_shape %expanded_3 [[0], [1, 2, 3, 4], [5], [6]] : tensor<24x32x1x1x4x256x192xf32> into tensor<24x128x256x192xf32>
    %reduced = linalg.reduce ins(%0 : tensor<768x196608xf32>) outs(%collapsed : tensor<768xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %1 = arith.addf %in, %init : f32
        linalg.yield %1 : f32
      }
    %expanded_5 = tensor.expand_shape %reduced [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
    return %collapsed_4, %expanded_5 : tensor<24x128x256x192xf32>, tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK-LABEL: @mlir_fused_native_group_norm_backward_0
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
module {
  func.func @mlir_fused_native_group_norm_backward_0(%arg0: tensor<24x256x128x96xbf16>, %arg1: tensor<24x256x1x1xbf16>, %arg2: tensor<24x256x1x1xbf16>, %arg3: tensor<24x256xf32>) -> tensor<24x256xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 1.000000e+00 : bf16
    %cst_0 = arith.constant 1.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg2 [[0, 1, 2, 3]] : tensor<24x256x1x1xbf16> into tensor<6144xbf16>
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<24x256x1x1xbf16> into tensor<6144xbf16>
    %collapsed_2 = tensor.collapse_shape %arg3 [[0, 1]] : tensor<24x256xf32> into tensor<6144xf32>
    %collapsed_3 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x256x128x96xbf16> into tensor<6144x12288xbf16>
    %0 = tensor.empty() : tensor<6144x12288xbf16>
    %1 = tensor.empty() : tensor<6144xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_1 : tensor<6144xbf16>) outs(%1 : tensor<6144xf32>) -> tensor<6144xf32>
    %3 = tensor.empty() : tensor<6144x12288xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<6144xf32>) outs(%3 : tensor<6144x12288xf32>) dimensions = [1]
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_3 : tensor<6144x12288xbf16>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %broadcasted : tensor<6144x12288xf32>, tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<6144xbf16>) outs(%1 : tensor<6144xf32>) -> tensor<6144xf32>
    %broadcasted_4 = linalg.broadcast ins(%6 : tensor<6144xf32>) outs(%3 : tensor<6144x12288xf32>) dimensions = [1]
    %7 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<6144x12288xbf16>) -> tensor<6144x12288xbf16>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<6144x12288xbf16>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_4, %8 : tensor<6144x12288xf32>, tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %9 : tensor<6144x12288xf32>, tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%10 : tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%11 : tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%12, %cst_0 : tensor<6144x12288xf32>, f32) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst_0, %13 : f32, tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %8 : tensor<6144x12288xf32>, tensor<6144x12288xf32>) outs(%3 : tensor<6144x12288xf32>) -> tensor<6144x12288xf32>
    %reduced = linalg.reduce ins(%15 : tensor<6144x12288xf32>) outs(%collapsed_2 : tensor<6144xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %16 = arith.addf %in, %init : f32
        linalg.yield %16 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [24, 256] : tensor<6144xf32> into tensor<24x256xf32>
    return %expanded : tensor<24x256xf32>
  }
}

// -----
// CHECK-DAG: dynamic_shape_0
// CHECK-DAG: dynamic_shape_1
func.func @dynamic_shape(%arg0: tensor<?x?xf32>,
                         %arg1: tensor<?xf32>) -> tensor<?xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %0 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) dimensions = [1]
  return %0 : tensor<?xf32>
}

// -----

// CHECK: test_dyn_shape
module {
  func.func @test_dyn_shape(%arg0: tensor<?x256xf32>, %arg1: tensor<256xf32>) -> (tensor<?xf32>, tensor<?x1xf32>, tensor<?x256xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 2.560000e+02 : f32
    %cst_3 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x256xf32>
    %0 = tensor.empty(%dim) : tensor<?x256xf32>
    %1 = tensor.empty(%dim) : tensor<?x1x1x256xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<?x1x1x256xf32> into tensor<?x256xf32>
    %2 = tensor.empty(%dim) : tensor<?xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
    %4 = tensor.empty(%dim) : tensor<?x1xf32>
    %collapsed_4 = tensor.collapse_shape %4 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %5 = linalg.fill ins(%cst_2 : f32) outs(%collapsed_4 : tensor<?xf32>) -> tensor<?xf32>
    %6 = tensor.empty(%dim) : tensor<?x1x1xf32>
    %collapsed_5 = tensor.collapse_shape %6 [[0, 1, 2]] : tensor<?x1x1xf32> into tensor<?xf32>
    %7 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_1 : tensor<?x256xf32>, f32) outs(%0 : tensor<?x256xf32>) -> tensor<?x256xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<256xf32>) outs(%0 : tensor<?x256xf32>) dimensions = [0]
    %reduced = linalg.reduce { arith.addf } ins(%7 : tensor<?x256xf32>) outs(%3 : tensor<?xf32>) dimensions = [1]
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %5 : tensor<?xf32>, tensor<?xf32>) outs(%collapsed_4 : tensor<?xf32>) -> tensor<?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %cst_3 : tensor<?xf32>, f32) outs(%collapsed_5 : tensor<?xf32>) -> tensor<?xf32>
    %10 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%9 : tensor<?xf32>) outs(%collapsed_5 : tensor<?xf32>) -> tensor<?xf32>
    %11 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%10 : tensor<?xf32>) outs(%collapsed_5 : tensor<?xf32>) -> tensor<?xf32>
    %broadcasted_6 = linalg.broadcast ins(%11 : tensor<?xf32>) outs(%collapsed : tensor<?x256xf32>) dimensions = [1]
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted_6 : tensor<?x256xf32>, tensor<?x256xf32>) outs(%collapsed : tensor<?x256xf32>) -> tensor<?x256xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %broadcasted : tensor<?x256xf32>, tensor<?x256xf32>) outs(%collapsed : tensor<?x256xf32>) -> tensor<?x256xf32>
    %14 = linalg.fill ins(%cst_2 : f32) outs(%4 : tensor<?x1xf32>) -> tensor<?x1xf32>
    return %3, %14, %13 : tensor<?xf32>, tensor<?x1xf32>, tensor<?x256xf32>
  }
}

// -----

module {
  // CHECK: %[[BLOCKDIM_RESULTS:.*]]:3 = scf.for
  // CHECK: %[[UB_AXIS_N_RESULTS:.*]]:3 = scf.for
  // CHECK: %[[UB_AXIS_PARALLEL_AXIS_D_RESULTS:.*]]:2 = scf.for
  func.func @test_fuse_loop_for_parallel_axis_d(%arg0: tensor<3072x49152xf32>, %arg1: tensor<3072x49152xf32>) -> (tensor<3072x49152xf32>, tensor<3072x49152xf32>, tensor<3072xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = tensor.empty() : tensor<3072x49152xf32>
    %1 = tensor.empty() : tensor<3072xf32>
    %2 = linalg.elemwise_unary ins(%arg0 : tensor<3072x49152xf32>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
    %3 = linalg.elemwise_unary ins(%2 : tensor<3072x49152xf32>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
    %reduced = linalg.reduce ins(%3 : tensor<3072x49152xf32>) outs(%1 : tensor<3072xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    return %2, %3, %reduced : tensor<3072x49152xf32>, tensor<3072x49152xf32>, tensor<3072xf32>
  }
}

// -----

func.func @rank1_reduce(%arg0: tensor<5222400xf32>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  %reduced = linalg.reduce ins(%arg0 : tensor<5222400xf32>) outs(%1 : tensor<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %2 = arith.addf %in, %init : f32
      linalg.yield %2 : f32
    }
  return %reduced : tensor<f32>
}
