// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// REQUIRES: issue2438
// -----
// CHECK-LABEL: @model_0
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
func.func @model_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xf32>) -> tensor<1x49152xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %collapsed_0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x128x256x192xbf16> into tensor<3072x49152xbf16>
  %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<3072x49152xf32>
  %0 = tensor.empty() : tensor<3072x49152xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<3072x49152xbf16>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %collapsed_1 : tensor<3072x49152xf32>, tensor<3072x49152xf32>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
  %3 = tensor.empty() : tensor<49152xf32>
  %reduced = linalg.reduce ins(%2 : tensor<3072x49152xf32>) outs(%3 : tensor<49152xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 49152] : tensor<49152xf32> into tensor<1x49152xf32>
  return %expanded : tensor<1x49152xf32>
}

// -----
// CHECK-LABEL: @model_2
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
// CHECK-DEBUG: tile_reduction_using_for {{.*}} by tile_sizes = {{\[}}0, 1, {{.*}}]
func.func @model_2(%arg0: tensor<24x128x25600xf32>,
                   %arg1: tensor<24xf32>,
                   %arg2: tensor<128xf32>,
                   %arg3: tensor<25600xf32>,
                   %arg4: tensor<24x128x25600xf32>) -> tensor<24x128x25600xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  
  %empty = tensor.empty() : tensor<24x25600xf32>
  %sum0 = linalg.reduce {arith.addf} ins(%arg0 : tensor<24x128x25600xf32>) outs(%empty : tensor<24x25600xf32>) dimensions = [1]
  
  %empty1 = tensor.empty() : tensor<24x128x25600xf32>
  %broadcasted = linalg.broadcast ins(%sum0 : tensor<24x25600xf32>) outs(%empty1 : tensor<24x128x25600xf32>) dimensions = [1]
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
// CHECK-DEBUG: %[[tile_factor_0:.*]] = transform.func.get_func_argument {{.*}}{{\[}}6]
// CHECK-DEBUG: %[[tile_factor_1:.*]] = transform.func.get_func_argument {{.*}}{{\[}}8]
// CHECK-DEBUG: transform.structured.tile_using_for {{.*}}{{\[}}%[[tile_factor_0]], %[[tile_factor_1]]]
// CHECK-DEBUG: transform.loop.tile
// CHECK-DEBUG: transform.loop.for_to_forall {{.*}} {annotate_only = true, mapping = [#hivm.block]}
// CHECK-DEBUG-NOT: tile_reduction_using_for
func.func @model_28(%arg0: tensor<24x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<24x256xf32>, %arg3: tensor<24x8xf32>, %arg4: tensor<24x8xf32>) -> (tensor<24x8xf32>, tensor<24x8xf32>) 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [32, 8] : tensor<256xf32> into tensor<32x8xf32>
  %0 = tensor.empty() : tensor<24x32x8xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [24, 32, 8] : tensor<24x256xf32> into tensor<24x32x8xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0], [1, 2]] output_shape [24, 32, 8] : tensor<24x256xf32> into tensor<24x32x8xf32>
  %1 = tensor.empty() : tensor<24x32x8xf32>
  %broadcasted = linalg.broadcast ins(%expanded : tensor<32x8xf32>) outs(%0 : tensor<24x32x8xf32>) dimensions = [0] 
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_0, %broadcasted : tensor<24x32x8xf32>, tensor<24x32x8xf32>) outs(%1 : tensor<24x32x8xf32>) -> tensor<24x32x8xf32>
  %reduced = linalg.reduce ins(%2 : tensor<24x32x8xf32>) outs(%arg3 : tensor<24x8xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_1, %broadcasted : tensor<24x32x8xf32>, tensor<24x32x8xf32>) outs(%1 : tensor<24x32x8xf32>) -> tensor<24x32x8xf32>
  %reduced_2 = linalg.reduce ins(%3 : tensor<24x32x8xf32>) outs(%arg4 : tensor<24x8xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  return %reduced, %reduced_2 : tensor<24x8xf32>, tensor<24x8xf32>
}


// -----
// CHECK-LABEL: func.func @test_anypbr_without_reduce_0(
// CHECK-SAME: %[[arg3:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}, 
// CHECK-SAME: %[[arg4:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[arg5:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[arg6:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[arg7:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[arg8:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
// CHECK-DAG: arith.index_cast %[[arg4]] : i64 to index
// CHECK-DAG: arith.index_cast %[[arg5]] : i64 to index
func.func @test_anypbr_without_reduce_0(%arg0: tensor<256xf32>, %arg1: tensor<100x256xf32>) -> tensor<100x256xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %0 = tensor.empty() : tensor<100x256xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<256xf32>) outs(%0 : tensor<100x256xf32>) dimensions = [0] 
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg1, %broadcasted : tensor<100x256xf32>, tensor<100x256xf32>) outs(%0 : tensor<100x256xf32>) -> tensor<100x256xf32>
  return %1 : tensor<100x256xf32>
}

// -----
// CHECK-LABEL: func.func @test_reduce_multi_dims_0(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.concat
// CHECK: linalg.reduce
func.func @test_reduce_multi_dims_0(%arg0: tensor<768x16x192xf32>, %arg1: tensor<768x8x192xf32>) -> tensor<768xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<768xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<768xf32>) -> tensor<768xf32>
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<768x16x192xf32>, tensor<768x8x192xf32>) -> tensor<768x24x192xf32>
  %reduced = linalg.reduce ins(%concat : tensor<768x24x192xf32>) outs(%1 : tensor<768xf32>) dimensions = [1, 2] 
    (%in: f32, %init: f32) {
      %2 = arith.addf %in, %init : f32
      linalg.yield %2 : f32
    }
  return %reduced : tensor<768xf32>
}

// -----
// CHECK-LABEL: func.func @test_same_size_diff_type_reduces(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.reduce
// CHECK: linalg.reduce
func.func @test_same_size_diff_type_reduces(%arg0: tensor<4x128x50xf16>, %arg1: tensor<4x128x50xf32>) -> tensor<4x128xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %0 = tensor.empty() : tensor<4x128xf32>
  %1 = tensor.empty() : tensor<4x128xf16>
  %2 = tensor.empty() : tensor<4x128xf32>
  %reduced = linalg.reduce { arith.maximumf } ins(%arg0 : tensor<4x128x50xf16>) outs(%1 : tensor<4x128xf16>) dimensions = [2] 
  %reduced_0 = linalg.reduce { arith.addf } ins(%arg1 : tensor<4x128x50xf32>) outs(%2 : tensor<4x128xf32>) dimensions = [2] 
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<4x128xf16>) outs(%2 : tensor<4x128xf32>) -> tensor<4x128xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %reduced_0 : tensor<4x128xf32>, tensor<4x128xf32>) outs(%0 : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %4 : tensor<4x128xf32>
}

// -----

// CHECK-LABEL: func.func @test_non_increasing_brc_dims(
// CHECK: linalg.broadcast
func.func @test_non_increasing_brc_dims(%arg0: tensor<1xf32>, %arg1: tensor<16x1x1xf32>) -> tensor<32xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %0 = tensor.empty() : tensor<16x1x1xf32>
  %broadcasted_0 = linalg.broadcast ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<16x1x1xf32>) dimensions = [1, 0] 
  %concat = tensor.concat dim(0) %broadcasted_0, %arg1 : (tensor<16x1x1xf32>, tensor<16x1x1xf32>) -> tensor<32x1x1xf32>
  %collapsed = tensor.collapse_shape %concat [[0, 1, 2]] : tensor<32x1x1xf32> into tensor<32xf32>
  return %collapsed : tensor<32xf32>
}
