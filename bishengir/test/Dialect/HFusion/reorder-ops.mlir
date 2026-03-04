// RUN: bishengir-opt %s -hfusion-reorder-ops -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-reorder-ops -split-input-file | md5sum > %t.hash1
// RUN: bishengir-opt %s -hfusion-reorder-ops -split-input-file | md5sum > %t.hash2
// RUN: cmp %t.hash1 %t.hash2

// CHECK: func.func @mlir_fused_convert_element_type_native_group_norm_silu_1(
func.func @mlir_fused_convert_element_type_native_group_norm_silu_1(%arg0: tensor<24x128x256x192xf32>, %arg1: tensor<24x32x1x1xf32>, %arg2: tensor<24x32x1x1xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<24x128x256x192xbf16>, %arg6: tensor<24x128x256x192xf16>, %arg7: i64 {hacc.tiling_data, hacc.tiling_key}, %arg8: i64 {hacc.tiling_data}, %arg9: i64 {hacc.tiling_data}, %arg10: i64 {hacc.tiling_data}, %arg11: i64 {hacc.tiling_data}, %arg12: i64 {hacc.tiling_data}) -> (tensor<24x128x256x192xbf16>, tensor<24x128x256x192xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "mlir_fused_convert_element_type_native_group_norm_silu_1_tiling_func", hacc.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %c3072 = arith.constant 3072 : index
  %c49152 = arith.constant 49152 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant 9.99999974E-6 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_2 = arith.constant 1.966080e+05 : f32
  %0 = tensor.empty() : tensor<24x32x4x49152xf32>
  %1 = tensor.empty() : tensor<24x32xf32>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xf32> into tensor<24x32x1x1x4x256x192xf32>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4], [5, 6]] : tensor<24x32x1x1x4x256x192xf32> into tensor<24x32x4x49152xf32>
  %collapsed_3 = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
  %collapsed_4 = tensor.collapse_shape %arg2 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1]] output_shape [32, 4] : tensor<128xf32> into tensor<32x4xf32>
  %2 = tensor.empty() : tensor<32x4xf32>
  %expanded_6 = tensor.expand_shape %arg4 [[0, 1]] output_shape [32, 4] : tensor<128xf32> into tensor<32x4xf32>
  %expanded_7 = tensor.expand_shape %arg5 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
  %collapsed_8 = tensor.collapse_shape %expanded_7 [[0], [1, 2, 3], [4], [5, 6]] : tensor<24x32x1x1x4x256x192xbf16> into tensor<24x32x4x49152xbf16>
  %expanded_9 = tensor.expand_shape %arg6 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xf16> into tensor<24x32x1x1x4x256x192xf16>
  %collapsed_10 = tensor.collapse_shape %expanded_9 [[0], [1, 2, 3], [4], [5, 6]] : tensor<24x32x1x1x4x256x192xf16> into tensor<24x32x4x49152xf16>
  %3 = tensor.empty() : tensor<24x32x4x49152xbf16>
  %4 = arith.index_cast %arg11 : i64 to index
  %5 = tensor.empty() : tensor<24x32x4x49152xf16>
  %6 = arith.ceildivsi %c49152, %4 : index
  %7 = arith.muli %6, %c3072 : index
  %8:2 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %collapsed_8, %arg15 = %collapsed_10) -> (tensor<24x32x4x49152xbf16>, tensor<24x32x4x49152xf16>) {
    %9 = arith.remsi %arg13, %6 : index
    %10 = arith.divsi %arg13, %6 : index
    %11 = arith.remsi %10, %c4 : index
    %12 = arith.divsi %10, %c4 : index
    %13 = arith.remsi %12, %c32 : index
    %14 = arith.divsi %12, %c32 : index
    %15 = arith.muli %9, %4 : index
    %16 = affine.min affine_map<(d0)[s0] -> (-d0 + 49152, s0)>(%15)[%4]
    %extracted_slice = tensor.extract_slice %collapsed[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<24x32x4x49152xf32> to tensor<1x1x1x?xf32>
    %extracted_slice_15 = tensor.extract_slice %0[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<24x32x4x49152xf32> to tensor<1x1x1x?xf32>
    %17 = hfusion.load {__arg0__} ins(%extracted_slice : tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %17 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %extracted_slice_16 = tensor.extract_slice %collapsed_3[%14, %13] [1, 1] [1, 1] : tensor<24x32xf32> to tensor<1x1xf32>
    %extracted_slice_17 = tensor.extract_slice %1[%14, %13] [1, 1] [1, 1] : tensor<24x32xf32> to tensor<1x1xf32>
    %18 = hfusion.load {__arg1__} ins(%extracted_slice_16 : tensor<1x1xf32>) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted = linalg.broadcast ins(%18 : tensor<1x1xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) dimensions = [2, 3]  {__intermediate_producer__}
    annotation.mark %broadcasted {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %19 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<sub>} ins(%17, %broadcasted : tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %19 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %extracted_slice_18 = tensor.extract_slice %collapsed_4[%14, %13] [1, 1] [1, 1] : tensor<24x32xf32> to tensor<1x1xf32>
    %20 = hfusion.load {__arg2__} ins(%extracted_slice_18 : tensor<1x1xf32>) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %21 = linalg.fill {__intermediate_producer__} ins(%cst_2 : f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %22 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<div>} ins(%20, %21 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %23 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<add>} ins(%22, %cst_0 : tensor<1x1xf32>, f32) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %24 = hfusion.elemwise_unary {__intermediate_producer__, fun = #hfusion.unary_fn<sqrt>} ins(%23 : tensor<1x1xf32>) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %25 = hfusion.elemwise_unary {__intermediate_producer__, fun = #hfusion.unary_fn<rec>} ins(%24 : tensor<1x1xf32>) outs(%extracted_slice_17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted_19 = linalg.broadcast ins(%25 : tensor<1x1xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) dimensions = [2, 3]  {__intermediate_producer__}
    annotation.mark %broadcasted_19 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %26 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%19, %broadcasted_19 : tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %26 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %extracted_slice_20 = tensor.extract_slice %expanded_5[%13, %11] [1, 1] [1, 1] : tensor<32x4xf32> to tensor<1x1xf32>
    %extracted_slice_21 = tensor.extract_slice %2[%13, %11] [1, 1] [1, 1] : tensor<32x4xf32> to tensor<1x1xf32>
    %27 = hfusion.load {__arg3__} ins(%extracted_slice_20 : tensor<1x1xf32>) outs(%extracted_slice_21 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted_22 = linalg.broadcast ins(%27 : tensor<1x1xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) dimensions = [0, 3]  {__intermediate_producer__}
    annotation.mark %broadcasted_22 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %28 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%26, %broadcasted_22 : tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %28 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %extracted_slice_23 = tensor.extract_slice %expanded_6[%13, %11] [1, 1] [1, 1] : tensor<32x4xf32> to tensor<1x1xf32>
    %29 = hfusion.load {__arg4__} ins(%extracted_slice_23 : tensor<1x1xf32>) outs(%extracted_slice_21 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted_24 = linalg.broadcast ins(%29 : tensor<1x1xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) dimensions = [0, 3]  {__intermediate_producer__}
    annotation.mark %broadcasted_24 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %30 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<add>} ins(%28, %broadcasted_24 : tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %30 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %extracted_slice_25 = tensor.extract_slice %3[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<24x32x4x49152xbf16> to tensor<1x1x1x?xbf16>
    %31 = hfusion.cast {__intermediate_producer__, round_mode = #hfusion.round_mode<rint>} ins(%30 : tensor<1x1x1x?xf32>) outs(%extracted_slice_25 : tensor<1x1x1x?xbf16>) -> tensor<1x1x1x?xbf16>
    annotation.mark %31 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xbf16>
    %extracted_slice_26 = tensor.extract_slice %arg14[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<24x32x4x49152xbf16> to tensor<1x1x1x?xbf16>
    %32 = hfusion.store ins(%31 : tensor<1x1x1x?xbf16>) outs(%extracted_slice_26 : tensor<1x1x1x?xbf16>) -> tensor<1x1x1x?xbf16>
    %inserted_slice = tensor.insert_slice %32 into %arg14[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<1x1x1x?xbf16> into tensor<24x32x4x49152xbf16>
    %33 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%30, %cst : tensor<1x1x1x?xf32>, f32) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %33 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %34 = linalg.elemwise_unary {__intermediate_producer__, fun = #linalg.unary_fn<exp>} ins(%33 : tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %34 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %35 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<add>} ins(%34, %cst_1 : tensor<1x1x1x?xf32>, f32) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %35 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %36 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<div>} ins(%30, %35 : tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) outs(%extracted_slice_15 : tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32>
    annotation.mark %36 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf32>
    %extracted_slice_27 = tensor.extract_slice %5[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<24x32x4x49152xf16> to tensor<1x1x1x?xf16>
    %37 = hfusion.cast {__intermediate_producer__, round_mode = #hfusion.round_mode<rint>} ins(%36 : tensor<1x1x1x?xf32>) outs(%extracted_slice_27 : tensor<1x1x1x?xf16>) -> tensor<1x1x1x?xf16>
    annotation.mark %37 {buffer_size_in_byte = 15104 : i64} : tensor<1x1x1x?xf16>
    %extracted_slice_28 = tensor.extract_slice %arg15[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<24x32x4x49152xf16> to tensor<1x1x1x?xf16>
    %38 = hfusion.store ins(%37 : tensor<1x1x1x?xf16>) outs(%extracted_slice_28 : tensor<1x1x1x?xf16>) -> tensor<1x1x1x?xf16>
    %inserted_slice_29 = tensor.insert_slice %38 into %arg15[%14, %13, %11, %15] [1, 1, 1, %16] [1, 1, 1, 1] : tensor<1x1x1x?xf16> into tensor<24x32x4x49152xf16>
    scf.yield %inserted_slice, %inserted_slice_29 : tensor<24x32x4x49152xbf16>, tensor<24x32x4x49152xf16>
  } {__tiled_for___10}
  %expanded_11 = tensor.expand_shape %8#0 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xbf16> into tensor<24x32x1x1x4x256x192xbf16>
  %collapsed_12 = tensor.collapse_shape %expanded_11 [[0], [1, 2, 3, 4], [5], [6]] : tensor<24x32x1x1x4x256x192xbf16> into tensor<24x128x256x192xbf16>
  %expanded_13 = tensor.expand_shape %8#1 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xf16> into tensor<24x32x1x1x4x256x192xf16>
  %collapsed_14 = tensor.collapse_shape %expanded_13 [[0], [1, 2, 3, 4], [5], [6]] : tensor<24x32x1x1x4x256x192xf16> into tensor<24x128x256x192xf16>
  return %collapsed_12, %collapsed_14 : tensor<24x128x256x192xbf16>, tensor<24x128x256x192xf16>
}