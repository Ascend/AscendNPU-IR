// RUN: bishengir-opt %s --hfusion-decompose-multi --split-input-file | FileCheck %s

module {
// CHECK-LABEL: @reduction_tile_2
// CHECK: %[[out1:.*]] = linalg.reduce
// CHECK: %[[out2:.*]] = linalg.reduce
// CHECK: %[[out3:.*]] = linalg.reduce
// CHECK: %[[out4:.*]] = linalg.reduce
// CHECK: %[[out5:.*]] = linalg.reduce
// CHECK: %[[out6:.*]] = linalg.reduce
// CHECK: return %[[out2]], %[[out3]], %[[out4]], %[[out6]], %[[out5]], %[[out1]]
  func.func @reduction_tile_2(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<4x5xf16>, %arg5: tensor<4x5xf16>, %arg6: tensor<4x6xf32>, %arg7: tensor<4x6xf32>, %arg8: tensor<4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4x6xf16>, %arg11: tensor<4x6xf16>, %arg12: tensor<4xf16>, %arg13: tensor<4xf16>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) {
    %reduced:5 = linalg.reduce ins(%arg10, %arg6, %arg7, %arg10, %arg10 : tensor<4x6xf16>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf16>, tensor<4x6xf16>) outs(%arg12, %arg8, %arg9, %arg12, %arg12 : tensor<4xf16>, tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>) dimensions = [1]  {hfusion.reduce_composed = ""}
      (%in: f16, %in_1: f32, %in_2: f32, %in_3: f16, %in_4: f16, %init: f16, %init_5: f32, %init_6: f32, %init_7: f16, %init_8: f16) {
        %0 = arith.addf %in, %init : f16
        %1 = arith.mulf %in, %0 : f16
        %2 = arith.mulf %in_1, %init_5 : f32
        %3 = arith.addf %in_2, %init_6 : f32
        %4 = arith.subf %in_3, %init_7 : f16
        %5 = arith.addf %in_4, %init_8 : f16
        linalg.yield %1, %2, %3, %4, %5 : f16, f32, f32, f16, f16
      }
    %reduced_0:8 = linalg.reduce ins(%arg0, %arg1, %arg4, %arg4, %arg0, %arg1, %arg4, %arg4 : tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf16>, tensor<4x5xf16>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf16>, tensor<4x5xf16>) outs(%arg2, %arg3, %arg12, %arg12, %arg2, %arg3, %arg12, %arg12 : tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>, tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>) dimensions = [1]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_1: f32, %in_2: f16, %in_3: f16, %in_4: f32, %in_5: f32, %in_6: f16, %in_7: f16, %init: f32, %init_8: f32, %init_9: f16, %init_10: f16, %init_11: f32, %init_12: f32, %init_13: f16, %init_14: f16) {
        %0 = arith.addf %in, %init : f32
        %1 = arith.addf %in_1, %init_8 : f32
        %2 = arith.addf %in_2, %init_9 : f16
        %3 = arith.addf %in_3, %init_10 : f16
        %4 = arith.mulf %in_3, %3 : f16
        %5 = arith.mulf %in_4, %init_11 : f32
        %6 = arith.addf %in_5, %init_12 : f32
        %7 = arith.subf %in_6, %init_13 : f16
        %8 = arith.addf %in_7, %init_14 : f16
        linalg.yield %0, %1, %2, %4, %5, %6, %7, %8 : f32, f32, f16, f16, f32, f32, f16, f16
      }
    return %reduced_0#0, %reduced_0#1, %reduced_0#2, %reduced_0#3, %reduced_0#6, %reduced#0 : tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>
  }
}

// -----

// CHECK-LABEL: @generic_tile
// CHECK: %[[out1:.*]] = linalg.generic
// CHECK: %[[out2:.*]] = linalg.generic
// CHECK: %[[out3:.*]] = linalg.generic
// CHECK: %[[out4:.*]] = linalg.generic
// CHECK: return %[[out1]], %[[out2]], %[[out3]], %[[out4]]
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @generic_tile(%arg0: tensor<4x6x9xf32>, %arg1: tensor<4x6x9xf32>, %arg2: tensor<4x6x9xf32>, %arg3: tensor<4x6x9xf32>, %arg4: tensor<4x6x9xf32>, %arg5: tensor<4x6x9xf32>, %arg6: tensor<4x6x9xf32>, %arg7: tensor<4x6x9xf32>, %arg8: tensor<4x6x9xf32>, %arg9: tensor<4x6x9xf32>, %arg10: tensor<4x6x9xf32>, %arg11: tensor<4x6x9xf32>, %arg12: tensor<4x6x9xf32>, %arg13: tensor<4x6x9xf32>) -> (tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>) {
    %0:4 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>) outs(%arg4, %arg5, %arg6, %arg7 : tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>) attrs =  {__partial_reduction_op__} {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32, %out_4: f32, %out_5: f32):
      %1 = arith.addf %in, %out : f32
      %2 = arith.addf %in_0, %out_3 : f32
      %3 = arith.addf %in_1, %out_4 : f32
      %4 = arith.addf %in_2, %out_5 : f32
      linalg.yield %1, %2, %3, %4 : f32, f32, f32, f32
    } -> (tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>)
    return %0#0, %0#1, %0#2, %0#3, %arg12, %arg13 : tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>, tensor<4x6x9xf32>
  }
}
// -----
#map = affine_map<()[s0] -> (s0 * -3072 + 3072, 3072)>
#map1 = affine_map<(d0) -> (-d0 + 49152, 2456)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @decomposing
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: %reduced = linalg.reduce
// CHECK: return
module {
  func.func @mlir_fused_add_convert_element_type_fill_mul_native_group_norm_backward_sigmoid_sub_sum_0_tiling_func(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xbf16>, %arg2: tensor<24x128x1x1xbf16>, %arg3: tensor<24x128x1x1xbf16>, %arg4: tensor<24x128x256x192xf32>, %arg5: tensor<24x32xf32>, %arg6: tensor<24x32xf32>, %arg7: tensor<24x128xf32>, %arg8: tensor<24x128xf32>, %arg9: tensor<24x128x1x1xf32>, %arg10: tensor<24x128x1x1xf32>, %arg11: tensor<24x128x1x1xbf16>, %arg12: tensor<24x128x1x1xbf16>, %arg13: tensor<24x32x4xf32>) -> (i64, i64, i64, i64, i64) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2456_i64 = arith.constant 2456 : i64
    %c9824_i64 = arith.constant 9824 : i64
    return %c2_i64, %c1_i64, %c1_i64, %c2456_i64, %c9824_i64 : i64, i64, i64, i64, i64
  }
  func.func @decomposing(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xbf16>, %arg2: tensor<24x128x1x1xbf16>, %arg3: tensor<24x128x1x1xbf16>, %arg4: tensor<24x128x256x192xf32>, %arg5: tensor<24x32xf32>, %arg6: tensor<24x32xf32>, %arg7: tensor<24x128xf32>, %arg8: tensor<24x128xf32>, %arg9: tensor<24x128x1x1xf32>, %arg10: tensor<24x128x1x1xf32>, %arg11: tensor<24x128x1x1xbf16>, %arg12: tensor<24x128x1x1xbf16>, %arg13: tensor<24x32x4xf32>, %arg14: i64 {hacc.tiling_data, hacc.tiling_key}, %arg15: i64 {hacc.tiling_data}, %arg16: i64 {hacc.tiling_data}, %arg17: i64 {hacc.tiling_data}, %arg18: i64 {hacc.tiling_data}) -> (tensor<24x128xf32>, tensor<24x128xf32>, tensor<24x128x1x1xf32>, tensor<24x128x1x1xf32>, tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>, tensor<24x32x4xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "mlir_fused_add_convert_element_type_fill_mul_native_group_norm_backward_sigmoid_sub_sum_0_tiling_func", hacc.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %c2456 = arith.constant 2456 : index
    %c49152 = arith.constant 49152 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c3072 = arith.constant 3072 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant -1.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : bf16
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<768x4x49152xbf16>
    %1 = tensor.empty() : tensor<768x4xf32>
    %2 = tensor.empty() : tensor<768x4x49152xf32>
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 4, 1, 1, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x4x1x1x256x192xbf16>
    %collapsed = tensor.collapse_shape %expanded [[0, 1], [2, 3, 4], [5, 6]] : tensor<24x32x4x1x1x256x192xbf16> into tensor<768x4x49152xbf16>
    %expanded_3 = tensor.expand_shape %arg1 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 4, 1, 1, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x4x1x1x256x192xbf16>
    %collapsed_4 = tensor.collapse_shape %expanded_3 [[0, 1], [2, 3, 4], [5, 6]] : tensor<24x32x4x1x1x256x192xbf16> into tensor<768x4x49152xbf16>
    %expanded_5 = tensor.expand_shape %arg2 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_6 = tensor.collapse_shape %expanded_5 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xbf16> into tensor<768x4xbf16>
    %3 = tensor.empty() : tensor<768x4xbf16>
    %expanded_7 = tensor.expand_shape %arg3 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_8 = tensor.collapse_shape %expanded_7 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xbf16> into tensor<768x4xbf16>
    %expanded_9 = tensor.expand_shape %arg4 [[0], [1, 2], [3, 4, 5], [6]] output_shape [24, 32, 4, 1, 1, 256, 192] : tensor<24x128x256x192xf32> into tensor<24x32x4x1x1x256x192xf32>
    %collapsed_10 = tensor.collapse_shape %expanded_9 [[0, 1], [2, 3, 4], [5, 6]] : tensor<24x32x4x1x1x256x192xf32> into tensor<768x4x49152xf32>
    %collapsed_11 = tensor.collapse_shape %arg5 [[0, 1]] : tensor<24x32xf32> into tensor<768xf32>
    %4 = tensor.empty() : tensor<768xf32>
    %collapsed_12 = tensor.collapse_shape %arg6 [[0, 1]] : tensor<24x32xf32> into tensor<768xf32>
    %expanded_13 = tensor.expand_shape %arg7 [[0], [1, 2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_14 = tensor.collapse_shape %expanded_13 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<768x4xf32>
    %expanded_15 = tensor.expand_shape %arg8 [[0], [1, 2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_16 = tensor.collapse_shape %expanded_15 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<768x4xf32>
    %expanded_17 = tensor.expand_shape %arg9 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_18 = tensor.collapse_shape %expanded_17 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<768x4xf32>
    %expanded_19 = tensor.expand_shape %arg10 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_20 = tensor.collapse_shape %expanded_19 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<768x4xf32>
    %expanded_21 = tensor.expand_shape %arg11 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_22 = tensor.collapse_shape %expanded_21 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xbf16> into tensor<768x4xbf16>
    %expanded_23 = tensor.expand_shape %arg12 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_24 = tensor.collapse_shape %expanded_23 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xbf16> into tensor<768x4xbf16>
    %collapsed_25 = tensor.collapse_shape %arg13 [[0, 1], [2]] : tensor<24x32x4xf32> into tensor<768x4xf32>
    %5:7 = scf.for %arg19 = %c0 to %c1 step %c1 iter_args(%arg20 = %collapsed_20, %arg21 = %collapsed_18, %arg22 = %collapsed_16, %arg23 = %collapsed_14, %arg24 = %collapsed_24, %arg25 = %collapsed_22, %arg26 = %collapsed_25) -> (tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xbf16>, tensor<768x4xbf16>, tensor<768x4xf32>) {
      %6 = affine.min #map()[%arg19]
      %7:7 = scf.for %arg27 = %c0 to %6 step %c1 iter_args(%arg28 = %arg20, %arg29 = %arg21, %arg30 = %arg22, %arg31 = %arg23, %arg32 = %arg24, %arg33 = %arg25, %arg34 = %arg26) -> (tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xbf16>, tensor<768x4xbf16>, tensor<768x4xf32>) {
        %8 = arith.muli %arg19, %c3072 : index
        %9 = arith.addi %8, %arg27 : index
        %10 = arith.remsi %9, %c4 : index
        %11 = arith.divsi %9, %c4 : index
        %extracted_slice = tensor.extract_slice %collapsed_6[%11, %10] [1, 1] [1, 1] : tensor<768x4xbf16> to tensor<1x1xbf16>
        %extracted_slice_39 = tensor.extract_slice %3[%11, %10] [1, 1] [1, 1] : tensor<768x4xbf16> to tensor<1x1xbf16>
        %12 = hfusion.load {__arg2__} ins(%extracted_slice : tensor<1x1xbf16>) outs(%extracted_slice_39 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
        %extracted_slice_40 = tensor.extract_slice %1[%11, %10] [1, 1] [1, 1] : tensor<768x4xf32> to tensor<1x1xf32>
        %13 = hfusion.cast {__intermediate_producer__, round_mode = #hfusion.round_mode<rint>} ins(%12 : tensor<1x1xbf16>) outs(%extracted_slice_40 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %extracted_slice_41 = tensor.extract_slice %collapsed_8[%11, %10] [1, 1] [1, 1] : tensor<768x4xbf16> to tensor<1x1xbf16>
        %14 = hfusion.load {__arg3__} ins(%extracted_slice_41 : tensor<1x1xbf16>) outs(%extracted_slice_39 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
        %15 = hfusion.cast {__intermediate_producer__, round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<1x1xbf16>) outs(%extracted_slice_40 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %16 = tensor.empty() : tensor<1x1x2456xf32>
        %17 = linalg.fill {__reduction_init_op__} ins(%cst : f32) outs(%16 : tensor<1x1x2456xf32>) -> tensor<1x1x2456xf32>
        %18 = linalg.fill {__reduction_init_op___1} ins(%cst : f32) outs(%16 : tensor<1x1x2456xf32>) -> tensor<1x1x2456xf32>
        %19 = linalg.fill {__reduction_init_op___2} ins(%cst : f32) outs(%16 : tensor<1x1x2456xf32>) -> tensor<1x1x2456xf32>
        %20 = linalg.fill {__reduction_init_op___3} ins(%cst : f32) outs(%16 : tensor<1x1x2456xf32>) -> tensor<1x1x2456xf32>
        %21:4 = scf.for %arg35 = %c0 to %c49152 step %c2456 iter_args(%arg36 = %17, %arg37 = %18, %arg38 = %19, %arg39 = %20) -> (tensor<1x1x2456xf32>, tensor<1x1x2456xf32>, tensor<1x1x2456xf32>, tensor<1x1x2456xf32>) {
          %35 = affine.min #map1(%arg35)
          %extracted_slice_59 = tensor.extract_slice %collapsed[%11, %10, %arg35] [1, 1, %35] [1, 1, 1] : tensor<768x4x49152xbf16> to tensor<1x1x?xbf16>
          %extracted_slice_60 = tensor.extract_slice %0[%11, %10, %arg35] [1, 1, %35] [1, 1, 1] : tensor<768x4x49152xbf16> to tensor<1x1x?xbf16>
          %36 = hfusion.load {__arg0__, __reduction0_fusible_producer__} ins(%extracted_slice_59 : tensor<1x1x?xbf16>) outs(%extracted_slice_60 : tensor<1x1x?xbf16>) -> tensor<1x1x?xbf16>
          annotation.mark %36 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xbf16>
          %extracted_slice_61 = tensor.extract_slice %2[%11, %10, %arg35] [1, 1, %35] [1, 1, 1] : tensor<768x4x49152xf32> to tensor<1x1x?xf32>
          %37 = hfusion.cast {__intermediate_producer__, __reduction0_fusible_producer__, round_mode = #hfusion.round_mode<rint>} ins(%36 : tensor<1x1x?xbf16>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %37 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %extracted_slice_62 = tensor.extract_slice %collapsed_4[%11, %10, %arg35] [1, 1, %35] [1, 1, 1] : tensor<768x4x49152xbf16> to tensor<1x1x?xbf16>
          %38 = hfusion.load {__arg1__, __reduction0_fusible_producer__} ins(%extracted_slice_62 : tensor<1x1x?xbf16>) outs(%extracted_slice_60 : tensor<1x1x?xbf16>) -> tensor<1x1x?xbf16>
          annotation.mark %38 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xbf16>
          %39 = hfusion.cast {__intermediate_producer__, __reduction0_fusible_producer__, round_mode = #hfusion.round_mode<rint>} ins(%38 : tensor<1x1x?xbf16>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %39 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %broadcasted_63 = linalg.broadcast ins(%13 : tensor<1x1xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) dimensions = [2]  {__intermediate_producer__, __reduction0_fusible_producer__}
          annotation.mark %broadcasted_63 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %40 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%39, %broadcasted_63 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %40 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %broadcasted_64 = linalg.broadcast ins(%15 : tensor<1x1xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) dimensions = [2]  {__intermediate_producer__, __reduction0_fusible_producer__}
          annotation.mark %broadcasted_64 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %41 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<add>} ins(%40, %broadcasted_64 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %41 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %42 = linalg.fill {__intermediate_producer__, __reduction0_fusible_producer__} ins(%cst_1 : bf16) outs(%extracted_slice_60 : tensor<1x1x?xbf16>) -> tensor<1x1x?xbf16>
          annotation.mark %42 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xbf16>
          %43 = hfusion.cast {__intermediate_producer__, __reduction0_fusible_producer__, round_mode = #hfusion.round_mode<rint>} ins(%42 : tensor<1x1x?xbf16>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %43 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %44 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%41, %cst_0 : tensor<1x1x?xf32>, f32) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %44 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %45 = linalg.elemwise_unary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.unary_fn<exp>} ins(%44 : tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %45 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %46 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<add>} ins(%45, %cst_2 : tensor<1x1x?xf32>, f32) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %46 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %47 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<div>} ins(%43, %46 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %47 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %48 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<sub>} ins(%43, %47 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %48 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %49 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%41, %48 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %49 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %50 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<add>} ins(%49, %cst_2 : tensor<1x1x?xf32>, f32) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %50 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %51 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<div>} ins(%50, %46 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %51 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %52 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%37, %51 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %52 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %53 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%52, %broadcasted_63 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %53 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %54 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%52, %39 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %54 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %extracted_slice_65 = tensor.extract_slice %collapsed_10[%11, %10, %arg35] [1, 1, %35] [1, 1, 1] : tensor<768x4x49152xf32> to tensor<1x1x?xf32>
          %55 = hfusion.load {__arg4__, __reduction0_fusible_producer__} ins(%extracted_slice_65 : tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %55 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %56 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>} ins(%53, %55 : tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_61 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
          annotation.mark %56 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %extracted_slice_66 = tensor.extract_slice %arg36[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x2456xf32> to tensor<1x1x?xf32>
          %extracted_slice_67 = tensor.extract_slice %arg37[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x2456xf32> to tensor<1x1x?xf32>
          %extracted_slice_68 = tensor.extract_slice %arg38[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x2456xf32> to tensor<1x1x?xf32>
          %extracted_slice_69 = tensor.extract_slice %arg39[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x2456xf32> to tensor<1x1x?xf32>
          %57:4 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2, #map2, #map2, #map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %53, %54, %56, %extracted_slice_66, %extracted_slice_67, %extracted_slice_68, %extracted_slice_69 : tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>) outs(%extracted_slice_66, %extracted_slice_67, %extracted_slice_68, %extracted_slice_69 : tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>) attrs =  {__partial_reduction_op__} {
          ^bb0(%in: f32, %in_74: f32, %in_75: f32, %in_76: f32, %in_77: f32, %in_78: f32, %in_79: f32, %in_80: f32, %out: f32, %out_81: f32, %out_82: f32, %out_83: f32):
            %58 = arith.addf %in, %in_77 : f32
            %59 = arith.addf %in_74, %in_78 : f32
            %60 = arith.addf %in_75, %in_79 : f32
            %61 = arith.addf %in_76, %in_80 : f32
            linalg.yield %58, %59, %60, %61 : f32, f32, f32, f32
          } -> (tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>, tensor<1x1x?xf32>)
          annotation.mark %57#0 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          annotation.mark %57#1 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          annotation.mark %57#2 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          annotation.mark %57#3 {buffer_size_in_byte = 9824 : i64} : tensor<1x1x?xf32>
          %inserted_slice_70 = tensor.insert_slice %57#0 into %arg36[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x?xf32> into tensor<1x1x2456xf32>
          %inserted_slice_71 = tensor.insert_slice %57#1 into %arg37[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x?xf32> into tensor<1x1x2456xf32>
          %inserted_slice_72 = tensor.insert_slice %57#2 into %arg38[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x?xf32> into tensor<1x1x2456xf32>
          %inserted_slice_73 = tensor.insert_slice %57#3 into %arg39[0, 0, 0] [1, 1, %35] [1, 1, 1] : tensor<1x1x?xf32> into tensor<1x1x2456xf32>
          scf.yield %inserted_slice_70, %inserted_slice_71, %inserted_slice_72, %inserted_slice_73 : tensor<1x1x2456xf32>, tensor<1x1x2456xf32>, tensor<1x1x2456xf32>, tensor<1x1x2456xf32>
        } {__reduction_loop__}
        %reduced:4 = linalg.reduce ins(%21#0, %21#1, %21#2, %21#3 : tensor<1x1x2456xf32>, tensor<1x1x2456xf32>, tensor<1x1x2456xf32>, tensor<1x1x2456xf32>) outs(%extracted_slice_40, %extracted_slice_40, %extracted_slice_40, %extracted_slice_40 : tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) dimensions = [2]  {__final_reduction_op__}
          (%in: f32, %in_59: f32, %in_60: f32, %in_61: f32, %init: f32, %init_62: f32, %init_63: f32, %init_64: f32) {
            %35 = arith.addf %in, %init : f32
            %36 = arith.addf %in_59, %init_62 : f32
            %37 = arith.addf %in_60, %init_63 : f32
            %38 = arith.addf %in_61, %init_64 : f32
            linalg.yield %35, %36, %37, %38 : f32, f32, f32, f32
          }
        %extracted_slice_42 = tensor.extract_slice %arg28[%11, %10] [1, 1] [1, 1] : tensor<768x4xf32> to tensor<1x1xf32>
        %22 = hfusion.store ins(%reduced#0 : tensor<1x1xf32>) outs(%extracted_slice_42 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %inserted_slice = tensor.insert_slice %22 into %arg28[%11, %10] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<768x4xf32>
        %extracted_slice_43 = tensor.extract_slice %arg29[%11, %10] [1, 1] [1, 1] : tensor<768x4xf32> to tensor<1x1xf32>
        %23 = hfusion.store ins(%reduced#0 : tensor<1x1xf32>) outs(%extracted_slice_43 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %inserted_slice_44 = tensor.insert_slice %23 into %arg29[%11, %10] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<768x4xf32>
        %extracted_slice_45 = tensor.extract_slice %arg30[%11, %10] [1, 1] [1, 1] : tensor<768x4xf32> to tensor<1x1xf32>
        %24 = hfusion.store ins(%reduced#0 : tensor<1x1xf32>) outs(%extracted_slice_45 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %inserted_slice_46 = tensor.insert_slice %24 into %arg30[%11, %10] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<768x4xf32>
        %extracted_slice_47 = tensor.extract_slice %arg31[%11, %10] [1, 1] [1, 1] : tensor<768x4xf32> to tensor<1x1xf32>
        %25 = hfusion.store ins(%reduced#0 : tensor<1x1xf32>) outs(%extracted_slice_47 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %inserted_slice_48 = tensor.insert_slice %25 into %arg31[%11, %10] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<768x4xf32>
        %26 = hfusion.cast {__intermediate_producer__, round_mode = #hfusion.round_mode<rint>} ins(%reduced#0 : tensor<1x1xf32>) outs(%extracted_slice_39 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
        %extracted_slice_49 = tensor.extract_slice %arg32[%11, %10] [1, 1] [1, 1] : tensor<768x4xbf16> to tensor<1x1xbf16>
        %27 = hfusion.store ins(%26 : tensor<1x1xbf16>) outs(%extracted_slice_49 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
        %inserted_slice_50 = tensor.insert_slice %27 into %arg32[%11, %10] [1, 1] [1, 1] : tensor<1x1xbf16> into tensor<768x4xbf16>
        %extracted_slice_51 = tensor.extract_slice %arg33[%11, %10] [1, 1] [1, 1] : tensor<768x4xbf16> to tensor<1x1xbf16>
        %28 = hfusion.store ins(%26 : tensor<1x1xbf16>) outs(%extracted_slice_51 : tensor<1x1xbf16>) -> tensor<1x1xbf16>
        %inserted_slice_52 = tensor.insert_slice %28 into %arg33[%11, %10] [1, 1] [1, 1] : tensor<1x1xbf16> into tensor<768x4xbf16>
        %extracted_slice_53 = tensor.extract_slice %collapsed_11[%11] [1] [1] : tensor<768xf32> to tensor<1xf32>
        %extracted_slice_54 = tensor.extract_slice %4[%11] [1] [1] : tensor<768xf32> to tensor<1xf32>
        %29 = hfusion.load {__arg5__} ins(%extracted_slice_53 : tensor<1xf32>) outs(%extracted_slice_54 : tensor<1xf32>) -> tensor<1xf32>
        %broadcasted = linalg.broadcast ins(%29 : tensor<1xf32>) outs(%extracted_slice_40 : tensor<1x1xf32>) dimensions = [1]  {__intermediate_producer__}
        %30 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%reduced#0, %broadcasted : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_40 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %31 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<sub>} ins(%reduced#0, %30 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_40 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %extracted_slice_55 = tensor.extract_slice %collapsed_12[%11] [1] [1] : tensor<768xf32> to tensor<1xf32>
        %32 = hfusion.load {__arg6__} ins(%extracted_slice_55 : tensor<1xf32>) outs(%extracted_slice_54 : tensor<1xf32>) -> tensor<1xf32>
        %broadcasted_56 = linalg.broadcast ins(%32 : tensor<1xf32>) outs(%extracted_slice_40 : tensor<1x1xf32>) dimensions = [1]  {__intermediate_producer__}
        %33 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%31, %broadcasted_56 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_40 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %extracted_slice_57 = tensor.extract_slice %arg34[%11, %10] [1, 1] [1, 1] : tensor<768x4xf32> to tensor<1x1xf32>
        %34 = hfusion.store ins(%33 : tensor<1x1xf32>) outs(%extracted_slice_57 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %inserted_slice_58 = tensor.insert_slice %34 into %arg34[%11, %10] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<768x4xf32>
        scf.yield %inserted_slice, %inserted_slice_44, %inserted_slice_46, %inserted_slice_48, %inserted_slice_50, %inserted_slice_52, %inserted_slice_58 : tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xbf16>, tensor<768x4xbf16>, tensor<768x4xf32>
      } {__tiled_for___16}
      scf.yield %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6 : tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xf32>, tensor<768x4xbf16>, tensor<768x4xbf16>, tensor<768x4xf32>
    } {__tiled_for___15, map_for_to_forall, mapping = [#hivm.block]}
    %expanded_26 = tensor.expand_shape %5#0 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_27 = tensor.collapse_shape %expanded_26 [[0], [1, 2], [3], [4]] : tensor<24x32x4x1x1xf32> into tensor<24x128x1x1xf32>
    %expanded_28 = tensor.expand_shape %5#1 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_29 = tensor.collapse_shape %expanded_28 [[0], [1, 2], [3], [4]] : tensor<24x32x4x1x1xf32> into tensor<24x128x1x1xf32>
    %expanded_30 = tensor.expand_shape %5#2 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_31 = tensor.collapse_shape %expanded_30 [[0], [1, 2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<24x128xf32>
    %expanded_32 = tensor.expand_shape %5#3 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_33 = tensor.collapse_shape %expanded_32 [[0], [1, 2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<24x128xf32>
    %expanded_34 = tensor.expand_shape %5#4 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_35 = tensor.collapse_shape %expanded_34 [[0], [1, 2], [3], [4]] : tensor<24x32x4x1x1xbf16> into tensor<24x128x1x1xbf16>
    %expanded_36 = tensor.expand_shape %5#5 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_37 = tensor.collapse_shape %expanded_36 [[0], [1, 2], [3], [4]] : tensor<24x32x4x1x1xbf16> into tensor<24x128x1x1xbf16>
    %expanded_38 = tensor.expand_shape %5#6 [[0, 1], [2]] output_shape [24, 32, 4] : tensor<768x4xf32> into tensor<24x32x4xf32>
    return %collapsed_33, %collapsed_31, %collapsed_29, %collapsed_27, %collapsed_37, %collapsed_35, %expanded_38 : tensor<24x128xf32>, tensor<24x128xf32>, tensor<24x128x1x1xf32>, tensor<24x128x1x1xf32>, tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>, tensor<24x32x4xf32>
  }
}
