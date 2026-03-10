// RUN: bishengir-opt %s --propagate-reshape="for-regbased=true" -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @reduce_with_index_case(
// CHECK-SAME:                                           %[[ARG_0:.*]]: tensor<1x8x1xf32>,
// CHECK-SAME:                                           %[[ARG_1:.*]]: tensor<1x8x4xi32>) -> (tensor<1x4xf32>, tensor<1x4xi32>) {
// CHECK:           %[[EXPANDED_ARG:.*]] = tensor.expand_shape %[[ARG_1]] {{\[\[}}0], [1, 2], [3]] output_shape [1, 8, 1, 4] : tensor<1x8x4xi32> into tensor<1x8x1x4xi32>
// CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<1x8x1xf32>
// CHECK:           %[[UNARY:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG_0]] : tensor<1x8x1xf32>) outs(%[[VAL_0]] : tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x8x4xf32>
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0], [1, 2], [3]] output_shape [1, 8, 1, 4] : tensor<1x8x4xf32> into tensor<1x8x1x4xf32>
// CHECK:           %[[BRC:.*]] = linalg.broadcast ins(%[[UNARY]] : tensor<1x8x1xf32>) outs(%[[EXPANDED_1]] : tensor<1x8x1x4xf32>) dimensions = [3]
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1x4xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x4xi32>
// CHECK:           %[[EXPANDED_2:.*]] = tensor.expand_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] output_shape [1, 1, 4] : tensor<1x4xf32> into tensor<1x1x4xf32>
// CHECK:           %[[EXPANDED_3:.*]] = tensor.expand_shape %[[VAL_3]] {{\[\[}}0], [1, 2]] output_shape [1, 1, 4] : tensor<1x4xi32> into tensor<1x1x4xi32>
// CHECK:           %[[REDUCED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%[[BRC]], %[[EXPANDED_ARG]] : tensor<1x8x1x4xf32>, tensor<1x8x1x4xi32>) outs(%[[EXPANDED_2]], %[[EXPANDED_3]] : tensor<1x1x4xf32>, tensor<1x1x4xi32>) dimensions = [1]  -> tensor<1x1x4xf32>, tensor<1x1x4xi32>
// CHECK:           %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[REDUCED]]#0 {{\[\[}}0], [1, 2]] : tensor<1x1x4xf32> into tensor<1x4xf32>
// CHECK:           %[[COLLAPSED_1:.*]] = tensor.collapse_shape %[[REDUCED]]#1 {{\[\[}}0], [1, 2]] : tensor<1x1x4xi32> into tensor<1x4xi32>
// CHECK:           return %[[COLLAPSED_0]], %[[COLLAPSED_1]] : tensor<1x4xf32>, tensor<1x4xi32>
module {
  func.func @reduce_with_index_case(%arg0: tensor<1x8x1xf32>, %arg1: tensor<1x8x4xi32>) -> (tensor<1x4xf32>, tensor<1x4xi32>) {
    %0 = tensor.empty() : tensor<1x8x1xf32>
    %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<1x8x1xf32>) outs(%0 : tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    %2 = tensor.empty() : tensor<1x8x4xf32>
    %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<1x8x1xf32> into tensor<1x8xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<1x8xf32>) outs(%2 : tensor<1x8x4xf32>) dimensions = [2]
    %3 = tensor.empty() : tensor<1x4xf32>
    %4 = tensor.empty() : tensor<1x4xi32>
    %5:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%broadcasted, %arg1 : tensor<1x8x4xf32>, tensor<1x8x4xi32>) outs(%3, %4 : tensor<1x4xf32>, tensor<1x4xi32>) dimensions = [1]  -> tensor<1x4xf32>, tensor<1x4xi32>
    return %5#0, %5#1 : tensor<1x4xf32>, tensor<1x4xi32>
  }
}

// -----
// CHECK-LABEL: @mulext_through_collapse(
// CHECK: mulext %{{.*}}, %{{.*}} : tensor<2x3x4xi32>
func.func @mulext_through_collapse(%arg0: tensor<2x3x4xi32>, %arg1: tensor<2x3x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<2x3x4xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%arg0, %arg0 : tensor<2x3x4xi32>, tensor<2x3x4xi32>) outs(%0 : tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<2x3x4xi32> into tensor<6x4xi32>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<2x3x4xi32> into tensor<6x4xi32>
  %low, %high = hfusion.mulext %collapsed, %collapsed_0 : tensor<6x4xi32>
  %2 = tensor.empty() : tensor<6x4xi32>
  %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%low, %low : tensor<6x4xi32>, tensor<6x4xi32>) outs(%2 : tensor<6x4xi32>) -> tensor<6x4xi32>
  return %3 : tensor<6x4xi32>
}

// -----
// CHECK-LABEL: @mulext_and_expand(
// CHECK: mulext %{{.*}}, %{{.*}} : tensor<2x3x4xi32>
// CHECK: return
func.func @mulext_and_expand(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<2x3x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%arg0, %arg0 : tensor<6x4xi32>, tensor<6x4xi32>) outs(%0 : tensor<6x4xi32>) -> tensor<6x4xi32>
  %low, %high = hfusion.mulext %arg1, %1 : tensor<6x4xi32>
  %collapsed_0 = tensor.expand_shape %low [[0, 1], [2]] output_shape [2, 3, 4] :  tensor<6x4xi32> into tensor<2x3x4xi32>
  %2 = tensor.empty() : tensor<2x3x4xi32>
  %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%collapsed_0, %collapsed_0 : tensor<2x3x4xi32>, tensor<2x3x4xi32>) outs(%2 : tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
  return %3 : tensor<2x3x4xi32>
}

// -----

// CHECK-LABEL: func.func @triton_max_5d_dim1(
// CHECK: %[[VAL_55:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<2x1x3x1xf32>) outs(%{{.*}} : tensor<2x1x3x1xf16>) -> tensor<2x1x3x1xf16>
// CHECK: math.absf %[[VAL_55]] : tensor<2x1x3x1xf16> 
// CHECK: return
func.func @triton_max_5d_dim1(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %reduced : tensor<2x3x1xf32>, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %16 = tensor.empty() : tensor<2x3x1xf16>  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 1, 3, 1], strides: [3, 3, 1, 1] : memref<?xf16> to memref<2x1x3x1xf16, strided<[3, 3, 1, 1]>>
  %17 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<2x3x1xf32>) outs(%16 : tensor<2x3x1xf16>) -> tensor<2x3x1xf16>
  %expanded = tensor.expand_shape %17 [[0], [1, 2], [3]] output_shape [2, 1, 3, 1] : tensor<2x3x1xf16> into tensor<2x1x3x1xf16>
  %2 = math.absf %expanded: tensor<2x1x3x1xf16>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_1 : (tensor<2x1x3x1xf16>, memref<2x1x3x1xf16, strided<[3, 3, 1, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @reduce(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x1x1xf32>)
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<16x1x1xf32>
// CHECK: %[[EMPTY_16I32:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[EXP_INPUT_I32:.*]] = tensor.expand_shape %[[EMPTY_16I32]] {{\[\[}}0, 1, 2]] output_shape {{\[}}16, 1, 1] : tensor<16xi32> into tensor<16x1x1xi32>
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<f32>
// CHECK: %[[EMPTY_I32:.*]] = tensor.empty() : tensor<i32>
// CHECK: %[[EXP_OUT_F32:.*]] = tensor.expand_shape %[[EMPTY_F32]] [] output_shape {{\[}}1, 1] : tensor<f32> into tensor<1x1xf32>
// CHECK: %[[EXP_OUT_I32:.*]] = tensor.expand_shape %[[EMPTY_I32]] [] output_shape {{\[}}1, 1] : tensor<i32> into tensor<1x1xi32>
// CHECK: %[[RES:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%[[TENSOR]], %[[EXP_INPUT_I32]] : tensor<16x1x1xf32>, tensor<16x1x1xi32>) outs(%[[EXP_OUT_F32]], %[[EXP_OUT_I32]] : tensor<1x1xf32>, tensor<1x1xi32>) dimensions = [0] -> tensor<1x1xf32>, tensor<1x1xi32>
// CHECK: %[[FINAL_F32:.*]] = tensor.collapse_shape %[[RES]]#0 [] : tensor<1x1xf32> into tensor<f32>
// CHECK: %[[FINAL_I32:.*]] = tensor.collapse_shape %[[RES]]#1 [] : tensor<1x1xi32> into tensor<i32>
// CHECK: return %[[FINAL_F32]], %[[FINAL_I32]] : tensor<f32>, tensor<i32>
func.func @reduce(%arg0: tensor<16x1x1xf32>) -> (tensor<f32>, tensor<i32>) {
  %alloc = memref.alloc() : memref<16x1x1xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x1x1xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<16x1x1xf32> into tensor<16xf32>
  %1 = tensor.empty() : tensor<16xi32>
  %2 = tensor.empty() : tensor<f32>
  %3 = tensor.empty() : tensor<i32>
  %4:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%collapsed, %1 : tensor<16xf32>, tensor<16xi32>) outs(%2, %3 : tensor<f32>, tensor<i32>) dimensions = [0]  -> tensor<f32>, tensor<i32>
  return %4#0, %4#1 : tensor<f32>, tensor<i32>
}

// -----

// CHECK-LABEL: func.func @cast_expand(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<7x15x13xf16>)
// CHECK: %[[EXP_INPUT:.*]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] output_shape {{\[}}7, 3, 5, 13] : tensor<7x15x13xf16> into tensor<7x3x5x13xf16>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7x15x13xf32>
// CHECK: %[[EXP_OUT:.*]] = tensor.expand_shape %[[EMPTY]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] output_shape {{\[}}7, 3, 5, 13] : tensor<7x15x13xf32> into tensor<7x3x5x13xf32>
// CHECK: %[[CASTED:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[EXP_INPUT]] : tensor<7x3x5x13xf16>) outs(%[[EXP_OUT]] : tensor<7x3x5x13xf32>) -> tensor<7x3x5x13xf32>
// CHECK: math.absf %[[CASTED]] : tensor<7x3x5x13xf32>
func.func @cast_expand(%arg0: tensor<7x15x13xf16>) -> (tensor<7x3x5x13xf32>) {
    %0 = tensor.empty() : tensor<7x15x13xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<7x15x13xf16>) outs(%0 : tensor<7x15x13xf32>) -> tensor<7x15x13xf32>
    %2 = tensor.expand_shape %1 [[0], [1, 2], [3]] output_shape [7, 3, 5, 13] : tensor<7x15x13xf32> into tensor<7x3x5x13xf32>
    %3 = math.absf %2 : tensor<7x3x5x13xf32>
    return %3 : tensor<7x3x5x13xf32>
}

// -----

// CHECK-LABEL: func.func @collapse_cast(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<7x3x5x13xf32>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<7x3x5x13xf16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0]] : tensor<7x3x5x13xf32>) outs(%[[EMPTY0]] : tensor<7x3x5x13xf16>) -> tensor<7x3x5x13xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<7x15x13xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EMPTY1]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] output_shape {{\[}}7, 3, 5, 13] : tensor<7x15x13xf32> into tensor<7x3x5x13xf32>
// CHECK: %[[CAST2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[CAST1]] : tensor<7x3x5x13xf16>) outs(%[[EXPANDED]] : tensor<7x3x5x13xf32>) -> tensor<7x3x5x13xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CAST2]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] : tensor<7x3x5x13xf32> into tensor<7x15x13xf32>
// CHECK: return %[[COLLAPSED]] : tensor<7x15x13xf32>
func.func @collapse_cast(%arg0: tensor<7x3x5x13xf32>) -> (tensor<7x15x13xf32>) {
    %0 = tensor.empty() : tensor<7x3x5x13xf16>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<7x3x5x13xf32>) outs(%0 : tensor<7x3x5x13xf16>) -> tensor<7x3x5x13xf16>
    %2 = tensor.collapse_shape %1 [[0], [1, 2], [3]] : tensor<7x3x5x13xf16> into tensor<7x15x13xf16>
    %3 = tensor.empty() : tensor<7x15x13xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<7x15x13xf16>) outs(%3 : tensor<7x15x13xf32>) -> tensor<7x15x13xf32>
    return %4 : tensor<7x15x13xf32>
}

// -----

// CHECK-LABEL: func.func @interleave_expand(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<4x8x8xf32>, %[[ARG1:.*]]: tensor<4x8x8xf32>)
// CHECK: %[[EXP_1:.*]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3, 4]] output_shape {{\[}}4, 4, 2, 2, 4] : tensor<4x8x8xf32> into tensor<4x4x2x2x4xf32>
// CHECK: %[[EXP_0:.*]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3, 4]] output_shape {{\[}}4, 4, 2, 2, 4] : tensor<4x8x8xf32> into tensor<4x4x2x2x4xf32>
// CHECK: %[[RES:.*]] = hfusion.interleave %[[EXP_0]], %[[EXP_1]] : tensor<4x4x2x2x4xf32>, tensor<4x4x2x2x4xf32> -> tensor<4x4x2x2x8xf32>
func.func @interleave_expand(%arg0: tensor<4x8x8xf32>, %arg1: tensor<4x8x8xf32>) -> (tensor<4x4x2x2x8xf32>) {
    %0 = hfusion.interleave %arg0, %arg1 : tensor<4x8x8xf32>, tensor<4x8x8xf32> -> tensor<4x8x16xf32>
    %1 = tensor.expand_shape %0 [[0], [1, 2], [3, 4]] output_shape [4, 4, 2, 2, 8] : tensor<4x8x16xf32> into tensor<4x4x2x2x8xf32>
    %2 = math.absf %1: tensor<4x4x2x2x8xf32>
    return %2 : tensor<4x4x2x2x8xf32>
}

// -----

// CHECK-LABEL: func.func @collapse_interleave(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<4x8x8xf32>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8x4x2xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0], {{\[}}1], {{\[}}2, 3]] : tensor<4x8x4x2xf32> into tensor<4x8x8xf32>
// CHECK: %[[RES:.*]] = hfusion.interleave %[[ARG0]], %[[COLLAPSED]] : tensor<4x8x8xf32>, tensor<4x8x8xf32> -> tensor<4x8x16xf32>
// CHECK: return %[[RES]] : tensor<4x8x16xf32>
func.func @collapse_interleave(%arg0: tensor<4x8x8xf32>) -> (tensor<4x8x16xf32>) {
    %0 = tensor.empty() : tensor<4x8x4x2xf32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2, 3]] : tensor<4x8x4x2xf32> into tensor<4x8x8xf32>
    %2 = hfusion.interleave %arg0, %1 : tensor<4x8x8xf32>, tensor<4x8x8xf32> -> tensor<4x8x16xf32>
    return %2 : tensor<4x8x16xf32>
}

// -----
 
// CHECK-LABEL: func.func @expand_insertSlice(
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4096, 1, 128, 4, 8] : tensor<4096x1x128x32xbf16> into tensor<4096x1x128x4x8xbf16>
// CHECK:           %[[EXPANDED_2:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4096, 1, 128, 8, 8] : tensor<4096x1x128x64xbf16> into tensor<4096x1x128x8x8xbf16>
// CHECK:           %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into {{.*}}[0, 0, 0, 0, 0] [4096, 1, 128, 4, 8] [1, 1, 1, 1, 1] : tensor<4096x1x128x4x8xbf16> into tensor<4096x1x128x8x8xbf16>
func.func @expand_insertSlice(%cst : f32, %cst2 : f32) -> tensor<4096x1x128x8x8xbf16> {
  %0 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<4096x1x128x32xbf16>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x1x128x32xbf16>) -> tensor<4096x1x128x32xbf16>
  %2 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<4096x1x128x64xbf16>
  %3 = linalg.fill ins(%cst2 : f32) outs(%2 : tensor<4096x1x128x64xbf16>) -> tensor<4096x1x128x64xbf16>
  %extracted_slice = tensor.insert_slice %1 into %3[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x1x128x32xbf16> into tensor<4096x1x128x64xbf16>
  %expanded = tensor.expand_shape %extracted_slice [[0], [1] ,[2], [3,4]] output_shape [4096, 1, 128, 8 ,8] {debug_instruction_number = 2 : i32} : tensor<4096x1x128x64xbf16> into tensor<4096x1x128x8x8xbf16>
  %4 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<4096x1x128x8x8xbf16>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%expanded : tensor<4096x1x128x8x8xbf16>) outs(%4 : tensor<4096x1x128x8x8xbf16>) -> tensor<4096x1x128x8x8xbf16>
  return %5 : tensor<4096x1x128x8x8xbf16>
}
 
// -----
 
// CHECK-LABEL: func.func @collapse_insertSlice(
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4096, 1, 128, 1, 32] : tensor<4096x1x128x32xbf16> into tensor<4096x1x128x1x32xbf16>
// CHECK:           %[[INSERTED:.*]] = tensor.insert_slice %[[EXPANDED_1]] into {{.*}}[0, 0, 0, 0, 0] [4096, 1, 128, 1, 32] [1, 1, 1, 1, 1] : tensor<4096x1x128x1x32xbf16> into tensor<4096x1x128x2x32xbf16>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[INSERTED]] {{\[\[}}0], [1], [2], [3, 4]] : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
func.func @collapse_insertSlice(%arg0: tensor<4096x1x128x2x32xf32>, %val: tensor<4096x1x128x32xbf16>) -> tensor<4096x1x128x64xbf16> {
  %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4096x1x128x2x32xbf16>
  %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x1x128x2x32xf32>) outs(%0 : tensor<4096x1x128x2x32xbf16>) -> tensor<4096x1x128x2x32xbf16>
  %collapsed = tensor.collapse_shape %1 [[0], [1], [2], [3, 4]] {debug_instruction_number = 4 : i32} : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
  %extracted_slice = tensor.insert_slice %val into %collapsed[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x1x128x32xbf16> into tensor<4096x1x128x64xbf16>
  return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<4096x1x128x64xbf16>
}
 
// -----
 
// CHECK-LABEL: func.func @expand_extractSlice(
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1, 2]] output_shape [24, 64, 4] : tensor<24x256xbf16> into tensor<24x64x4xbf16> 
// CHECK:           %[[EXTRACTED:.*]] = tensor.extract_slice %[[EXPANDED_1]][0, 32, 0] [24, 32, 4] [1, 1, 1] : tensor<24x64x4xbf16> to tensor<24x32x4xbf16>
func.func @expand_extractSlice(%arg0: tensor<24x256xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}) -> tensor<24x32x4xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[0, 128] [24, 128] [1, 1] {debug_instruction_number = 0 : i32} : tensor<24x256xbf16> to tensor<24x128xbf16>
  %0 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<24x32x4xbf16>
  %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [24, 32, 4] {debug_instruction_number = 2 : i32} : tensor<24x128xbf16> into tensor<24x32x4xbf16>
  %1 = hfusion.load {debug_instruction_number = 4 : i32} ins(%expanded : tensor<24x32x4xbf16>) outs(%0 : tensor<24x32x4xbf16>) -> tensor<24x32x4xbf16>
  return {debug_instruction_number = 5 : i32} %1 : tensor<24x32x4xbf16>
}
 
// -----
 
// CHECK-LABEL: func.func @collapse_extractSlice(
// CHECK:           %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0, 0, 0, 0, 0] [4096, 1, 128, 1, 32] [1, 1, 1, 1, 1] : tensor<4096x1x128x2x32xbf16> to tensor<4096x1x128x1x32xbf16>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[EXTRACTED]] {{\[\[}}0], [1], [2], [3, 4]] : tensor<4096x1x128x1x32xbf16> into tensor<4096x1x128x32xbf16>
func.func @collapse_extractSlice(%arg0: tensor<4096x1x128x2x32xf32>) -> tensor<4096x1x128x32xbf16> {
  %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4096x1x128x2x32xbf16>
  %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x1x128x2x32xf32>) outs(%0 : tensor<4096x1x128x2x32xbf16>) -> tensor<4096x1x128x2x32xbf16>
  %collapsed = tensor.collapse_shape %1 [[0], [1], [2], [3, 4]] {debug_instruction_number = 4 : i32} : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
  %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x1x128x64xbf16> to tensor<4096x1x128x32xbf16>
  return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<4096x1x128x32xbf16>
}

// -----
 
// CHECK-LABEL: func.func @expand_rank_reducing_insert_slice(
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [4096, 128, 4, 8] : tensor<4096x128x32xbf16> into tensor<4096x128x4x8xbf16>
// CHECK:           %[[EXPANDED_2:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4096, 3, 128, 8, 8] : tensor<4096x3x128x64xbf16> into tensor<4096x3x128x8x8xbf16>
// CHECK:           %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into {{.*}}[0, 0, 0, 0, 0] [4096, 1, 128, 4, 8] [1, 1, 1, 1, 1] : tensor<4096x128x4x8xbf16> into tensor<4096x3x128x8x8xbf16>
func.func @expand_rank_reducing_insert_slice(%cst : f32, %cst2 : f32) -> tensor<4096x3x128x8x8xbf16> {
  %0 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<4096x128x32xbf16>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x128x32xbf16>) -> tensor<4096x128x32xbf16>
  %2 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<4096x3x128x64xbf16>
  %3 = linalg.fill ins(%cst2 : f32) outs(%2 : tensor<4096x3x128x64xbf16>) -> tensor<4096x3x128x64xbf16>
  %inserted_slice = tensor.insert_slice %1 into %3[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x128x32xbf16> into tensor<4096x3x128x64xbf16>
  %expanded = tensor.expand_shape %inserted_slice [[0], [1] ,[2], [3,4]] output_shape [4096, 1, 128, 8 ,8] {debug_instruction_number = 2 : i32} : tensor<4096x3x128x64xbf16> into tensor<4096x3x128x8x8xbf16>
  %4 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<4096x3x128x8x8xbf16>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%expanded : tensor<4096x3x128x8x8xbf16>) outs(%4 : tensor<4096x3x128x8x8xbf16>) -> tensor<4096x3x128x8x8xbf16>
  return %5 : tensor<4096x3x128x8x8xbf16>
}
 
// -----
 
// CHECK-LABEL: func.func @collapse_rank_reducing_insert_slice(
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [4096, 128, 1, 32] : tensor<4096x128x32xbf16> into tensor<4096x128x1x32xbf16>
// CHECK:           %[[INSERTED:.*]] = tensor.insert_slice %[[EXPANDED_1]] into {{.*}}[0, 0, 0, 0, 0] [4096, 1, 128, 1, 32] [1, 1, 1, 1, 1] : tensor<4096x128x1x32xbf16> into tensor<4096x1x128x2x32xbf16>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[INSERTED]] {{\[\[}}0], [1], [2], [3, 4]] : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
func.func @collapse_rank_reducing_insert_slice(%arg0: tensor<4096x1x128x2x32xf32>, %val: tensor<4096x128x32xbf16>) -> tensor<4096x1x128x64xbf16> {
  %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4096x1x128x2x32xbf16>
  %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x1x128x2x32xf32>) outs(%0 : tensor<4096x1x128x2x32xbf16>) -> tensor<4096x1x128x2x32xbf16>
  %collapsed = tensor.collapse_shape %1 [[0], [1], [2], [3, 4]] {debug_instruction_number = 4 : i32} : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
  %extracted_slice = tensor.insert_slice %val into %collapsed[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x128x32xbf16> into tensor<4096x1x128x64xbf16>
  return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<4096x1x128x64xbf16>
}

// -----
 
// CHECK-LABEL: func.func @collapse_rank_reducing_insert_slice2(
// CHECK:           %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [16, 15, 4, 4] : tensor<16x15x16xf32> into tensor<16x15x4x4xf32>
// CHECK:           %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into %[[EXPANDED_1]][0, 0, 0, 0] [16, 1, 4, 4] [1, 1, 1, 1] : tensor<16x4x4xf32> into tensor<16x15x4x4xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[INSERTED]] {{\[\[}}0], [1], [2, 3]] : tensor<16x15x4x4xf32> into tensor<16x15x16xf32>
func.func @collapse_rank_reducing_insert_slice2(%arg0: tensor<16x15x16xf32>, %val: tensor<16x4x4xf32>) -> tensor<16x15x16xf32> {
  %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<16x15x16xf32>
  %1 = math.absf %arg0: tensor<16x15x16xf32>
  %val1 = math.absf %val: tensor<16x4x4xf32>
  %collapsed = tensor.collapse_shape %val1 [[0], [1, 2]] : tensor<16x4x4xf32>into tensor<16x16xf32>
  %extracted_slice = tensor.insert_slice %collapsed into %1[0, 0, 0] [16, 1, 16] [1, 1, 1] : tensor<16x16xf32> into tensor<16x15x16xf32>
  return %extracted_slice : tensor<16x15x16xf32>
}

// -----

// CHECK-LABEL: func.func @collapse_rank_reducing_extract_slice(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0, 0, 0, 0] [4096, 1, 1, 32] [1, 1, 1, 1] : tensor<4096x128x2x32xf32> to tensor<4096x1x32xf32>
// CHECK: tensor.collapse_shape %[[EXTRACTED]] {{\[\[}}0], [1, 2]] : tensor<4096x1x32xf32> into tensor<4096x32xf32>
func.func @collapse_rank_reducing_extract_slice(%arg0: tensor<4096x128x2x32xf32>) -> tensor<4096x32xf32> {
  %1 = math.absf %arg0: tensor<4096x128x2x32xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<4096x128x2x32xf32> into tensor<4096x128x64xf32>
  %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0] [4096, 1, 32] [1, 1, 1] : tensor<4096x128x64xf32> to tensor<4096x32xf32>
  return %extracted_slice : tensor<4096x32xf32>
}

// -----

// CHECK: %[[EXPANDED_1:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [64, 24, 64, 8] : tensor<64x24x512xbf16> into tensor<64x24x64x8xbf16>
// CHECK: tensor.extract_slice %[[EXPANDED_1]][0, 0, 16, 0] [1, 24, 32, 4] [1, 1, 1, 2] : tensor<64x24x64x8xbf16> to tensor<24x32x4xbf16>
func.func @expand_rank_reducing_extract_slice(%arg0: tensor<64x24x512xbf16>) -> tensor<24x32x4xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 128] [1, 24, 128] [1, 1, 2] : tensor<64x24x512xbf16> to tensor<24x128xbf16>
  %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xbf16> into tensor<24x32x4xbf16>
  %1 = math.absf %expanded: tensor<24x32x4xbf16>
  return %1 : tensor<24x32x4xbf16>
}
 
// -----
 
// CHECK-LABEL: func.func @collapse_extract(
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%[[C1]], %[[C1]], %[[C0]]] : tensor<2x2x1xf32>
func.func @collapse_extract(%arg0: tensor<2x2x1xf32>) -> f32 {
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<2x2x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x2x1xf32>) outs(%0 : tensor<2x2x1xf32>) -> tensor<2x2x1xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<2x2x1xf32> into tensor<2x2xf32>
  %extracted = tensor.extract %collapsed[%c1, %c1] : tensor<2x2xf32>
  return %extracted : f32
}
 
// -----
 
// CHECK-LABEL: func.func @expand_fill(
// CHECK:           %[[EXTRACTED:.*]] = tensor.expand_shape %{{.*}} {{\[\[}}0], [1, 2]] output_shape [4, 2, 5] : tensor<4x10xf32> into tensor<4x2x5xf32>
// CHECK:           %[[ARG:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x2x5xf32>) -> tensor<4x2x5xf32>
func.func @expand_fill() -> tensor<4x2x5xf32> {
  %f1 = arith.constant 1.0 : f32
  %0 = tensor.empty() : tensor<4x10xf32>
  %1 = linalg.fill ins(%f1 : f32) outs(%0 : tensor<4x10xf32>) -> tensor<4x10xf32>
  %expanded = tensor.expand_shape %1 [[0], [1, 2]] output_shape [4, 2, 5] {debug_instruction_number = 2 : i32} : tensor<4x10xf32> into tensor<4x2x5xf32>
  %2 = math.absf %expanded: tensor<4x2x5xf32>
  return %2 : tensor<4x2x5xf32>
}
 
// -----
 
// CHECK-LABEL: func.func @collapse_fill(
// CHECK:           %[[ARG:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x2x5xf32>) -> tensor<4x2x5xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<4x2x5xf32> into tensor<4x10xf32>
func.func @collapse_fill(%arg0: tensor<4x2x5xf32>) -> tensor<4x10xf32> {
  %f1 = arith.constant 1.0 : f32
  %0 = tensor.empty() : tensor<4x2x5xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<4x2x5xf32>) outs(%0 : tensor<4x2x5xf32>) -> tensor<4x2x5xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<4x2x5xf32> into tensor<4x10xf32>
  %2 = linalg.fill ins(%f1 : f32) outs(%collapsed : tensor<4x10xf32>) -> tensor<4x10xf32>
  return %2 : tensor<4x10xf32>
}


// -----
 
// CHECK-LABEL: func.func @test_compare_propagate_collapse(
// CHECK: %[[VAL_0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_1:.*]], %[[VAL_2:.*]] : tensor<6x6xi32>, tensor<6x6xi32>) outs(%[[VAL_3:.*]] : tensor<6x6xi1>) -> tensor<6x6xi1>
// CHECK: %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<6x6xi1> into tensor<36xi1>
func.func @test_compare_propagate_collapse() -> tensor<36xi1> {
  %zero = arith.constant 0 : i32
  %cst = arith.constant 1 : i32
  %cst1 = arith.constant 2 : i32
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = tensor.empty() : tensor<6x6xi32>
  %2 = tensor.empty() : tensor<6x6xi1>
  %3 = linalg.fill ins(%cst : i32) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  %4 = linalg.fill ins(%cst1 : i32) outs(%1 : tensor<6x6xi32>) -> tensor<6x6xi32>
  %5 = linalg.fill ins(%zero : i32) outs(%2 : tensor<6x6xi1>) -> tensor<6x6xi1>
  %collapsed0 = tensor.collapse_shape %3 [[0, 1]] : tensor<6x6xi32> into tensor<36xi32>
  %collapsed1 = tensor.collapse_shape %4 [[0, 1]] : tensor<6x6xi32> into tensor<36xi32>
  %collapsed2 = tensor.collapse_shape %5 [[0, 1]] : tensor<6x6xi1> into tensor<36xi1>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<veq>}
    ins(%collapsed0, %collapsed1 : tensor<36xi32>, tensor<36xi32>)
    outs(%collapsed2 : tensor<36xi1>) -> tensor<36xi1>
  return %ret : tensor<36xi1>
}
 
// -----
 
// CHECK-LABEL: func.func @test_compare_propagate_expand(
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[VAL_1:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi32> into tensor<6x6xi32>
// CHECK: %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_3:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi32> into tensor<6x6xi32>
// CHECK: %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_5:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi1> into tensor<6x6xi1>
// CHECK: %[[VAL_6:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_7:.*]], %[[VAL_8:.*]] : tensor<6x6xi32>, tensor<6x6xi32>) outs(%[[VAL_9:.*]] : tensor<6x6xi1>) -> tensor<6x6xi1
func.func @test_compare_propagate_expand() -> tensor<6x6xi1> {
  %zero = arith.constant 0 : i32
  %cst = arith.constant 1 : i32
  %cst1 = arith.constant 2 : i32
  %0 = tensor.empty() : tensor<36xi32>
  %1 = tensor.empty() : tensor<36xi32>
  %2 = tensor.empty() : tensor<36xi1>
  %3 = linalg.fill ins(%cst : i32) outs(%0 : tensor<36xi32>) -> tensor<36xi32>
  %4 = linalg.fill ins(%cst1 : i32) outs(%1 : tensor<36xi32>) -> tensor<36xi32>
  %5 = linalg.fill ins(%zero : i32) outs(%2 : tensor<36xi1>) -> tensor<36xi1>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<veq>}
    ins(%3, %4 : tensor<36xi32>, tensor<36xi32>)
    outs(%5 : tensor<36xi1>) -> tensor<36xi1>
  %expanded = tensor.expand_shape %ret [[0, 1]] output_shape [6, 6] : tensor<36xi1> into tensor<6x6xi1>
  %6 = math.absi %expanded: tensor<6x6xi1>
  return %6 : tensor<6x6xi1>
}
 
// -----
 
// CHECK-LABEL: func.func @test_select_propagate_collapse(
// CHECK: %[[VAL_0:.*]] = hfusion.select ins(%[[VAL_1:.*]], %[[VAL_2:.*]], %[[VAL_3:.*]] : tensor<6x6xi1>, tensor<6x6xi32>, tensor<6x6xi32>) outs(%[[VAL_4:.*]] : tensor<6x6xi32>) -> tensor<6x6xi32>
// CHECK: %[[VAL_5:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<6x6xi32> into tensor<36xi32>
func.func @test_select_propagate_collapse() -> tensor<36xi32> {
  %zero = arith.constant 1 : i32
  %cst = arith.constant 1 : i32
  %cst1 = arith.constant 2 : i32
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = tensor.empty() : tensor<6x6xi32>
  %2 = tensor.empty() : tensor<6x6xi1>
  %ret = tensor.empty() : tensor<6x6xi32>
  %3 = linalg.fill ins(%cst : i32) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  %4 = linalg.fill ins(%cst1 : i32) outs(%1 : tensor<6x6xi32>) -> tensor<6x6xi32>
  %5 = linalg.fill ins(%zero : i32) outs(%2 : tensor<6x6xi1>) -> tensor<6x6xi1>
  %collapsed0 = tensor.collapse_shape %3 [[0, 1]] : tensor<6x6xi32> into tensor<36xi32>
  %collapsed1 = tensor.collapse_shape %4 [[0, 1]] : tensor<6x6xi32> into tensor<36xi32>
  %collapsed2 = tensor.collapse_shape %5 [[0, 1]] : tensor<6x6xi1> into tensor<36xi1>
  %collapsed3 = tensor.collapse_shape %ret [[0, 1]] : tensor<6x6xi32> into tensor<36xi32>
  %6 = hfusion.select
    ins(%collapsed2, %collapsed0, %collapsed1 : tensor<36xi1>, tensor<36xi32>, tensor<36xi32>)
    outs(%collapsed3 : tensor<36xi32>) -> tensor<36xi32>
  return %6 : tensor<36xi32>
}
 
// -----
 
// CHECK-LABEL: func.func @test_select_propagate_expand(
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[VAL_1:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi32> into tensor<6x6xi32>
// CHECK: %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_3:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi32> into tensor<6x6xi32>
// CHECK: %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_5:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi1> into tensor<6x6xi1>
// CHECK: %[[VAL_6:.*]] = tensor.expand_shape %[[VAL_7:.*]] {{\[\[}}0, 1]] output_shape [6, 6] : tensor<36xi32> into tensor<6x6xi32>
// CHECK: %[[VAL_8:.*]] = hfusion.select ins(%[[VAL_4]], %[[VAL_0]], %[[VAL_2]] : tensor<6x6xi1>, tensor<6x6xi32>, tensor<6x6xi32>) outs(%expanded_2 : tensor<6x6xi32>) -> tensor<6x6xi32>
func.func @test_select_propagate_expand() -> tensor<6x6xi32> {
  %zero = arith.constant 0 : i32
  %cst = arith.constant 1 : i32
  %cst1 = arith.constant 2 : i32
  %0 = tensor.empty() : tensor<36xi32>
  %1 = tensor.empty() : tensor<36xi32>
  %2 = tensor.empty() : tensor<36xi1>
  %ret = tensor.empty() : tensor<36xi32>
  %3 = linalg.fill ins(%cst : i32) outs(%0 : tensor<36xi32>) -> tensor<36xi32>
  %4 = linalg.fill ins(%cst1 : i32) outs(%1 : tensor<36xi32>) -> tensor<36xi32>
  %5 = linalg.fill ins(%zero : i32) outs(%2 : tensor<36xi1>) -> tensor<36xi1>
  %6 = hfusion.select
    ins(%2, %0, %1 : tensor<36xi1>, tensor<36xi32>, tensor<36xi32>)
    outs(%ret : tensor<36xi32>) -> tensor<36xi32>
  %expanded = tensor.expand_shape %6 [[0, 1]] output_shape [6, 6] : tensor<36xi32> into tensor<6x6xi32>
  %7 = math.absi %expanded: tensor<6x6xi32>
  return %7 : tensor<6x6xi32>
}
 
// -----
 
// CHECK-LABEL: func.func @test_subview_collape_standard_hyperrectangle(
// CHECK: %[[VAL_0:.*]] = memref.subview %[[VAL_1:.*]][0, 0] [2, 16] [1, 1] : memref<8x16xf32, strided<[16, 1]>> to memref<2x16xf32, strided<[16, 1]>>
// CHECK: %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : memref<2x16xf32, strided<[16, 1]>> into memref<32xf32, strided<[1]>>
func.func @test_subview_collape_standard_hyperrectangle(){
  %arg0 = memref.alloc() : memref<8x16xf32, strided<[16, 1]>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 3.14 : f32
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>> into memref<128xf32, strided<[1]>>
  %subview = memref.subview %collapse_shape[0] [32] [1] : memref<128xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
  memref.store %val, %subview[%c0] : memref<32xf32, strided<[1]>>
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_subview_collape_standard_hyperrectangle_1(
// CHECK: %[[VAL_0:.*]] = memref.subview %[[VAL_1:.*]][0, 0] [1, 8] [1, 1] : memref<8x16xf32, strided<[16, 1]>> to memref<1x8xf32, strided<[16, 1]>>
// CHECK: %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : memref<1x8xf32, strided<[16, 1]>> into memref<8xf32, strided<[1]>>
func.func @test_subview_collape_standard_hyperrectangle_1(){
  %arg0 = memref.alloc() : memref<8x16xf32, strided<[16, 1]>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 3.14 : f32
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>> into memref<128xf32, strided<[1]>>
  %subview = memref.subview %collapse_shape[0] [8] [1] : memref<128xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
  memref.store %val, %subview[%c0] : memref<8xf32, strided<[1]>>
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_subview_collape_stride_not_unit(
// CHECK: %[[VAL_0:.*]] = memref.subview %[[VAL_1:.*]][0, 0] [5, 4] [1, 4] : memref<8x16xf32, strided<[32, 2]>> to memref<5x4xf32, strided<[32, 8]>>
// CHECK: %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : memref<5x4xf32, strided<[32, 8]>> into memref<20xf32, strided<[8]>>
func.func @test_subview_collape_stride_not_unit(){
  %arg0 = memref.alloc() : memref<8x16xf32, strided<[32, 2]>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 3.14 : f32
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<8x16xf32, strided<[32, 2]>> into memref<128xf32, strided<[2]>>
  %subview = memref.subview %collapse_shape[0] [20] [4] : memref<128xf32, strided<[2]>> to memref<20xf32, strided<[8]>>
  memref.store %val, %subview[%c0] : memref<20xf32, strided<[8]>>
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_subview_collape_mutiple_dimension(
// CHECK: %[[VAL_0:.*]] = memref.subview %[[VAL_1:.*]][0, 0, 0] [1, 2, 16] [1, 1, 1] : memref<2x4x16xf32, strided<[160, 32, 2]>> to memref<1x2x16xf32, strided<[160, 32, 2]>>
// CHECK: %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_0]] {{\[}}{{\[}}0], {{\[}}1, 2]] : memref<1x2x16xf32, strided<[160, 32, 2]>> into memref<1x32xf32, strided<[160, 2]>>
func.func @test_subview_collape_mutiple_dimension(){
  %arg0 = memref.alloc() : memref<2x4x16xf32, strided<[160, 32, 2]>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 3.14 : f32
  %collapse_shape = memref.collapse_shape %arg0 [[0], [1, 2]] : memref<2x4x16xf32, strided<[160, 32, 2]>> into memref<2x64xf32, strided<[160, 2]>>
  %subview = memref.subview %collapse_shape[0, 0] [1, 32] [1, 1] : memref<2x64xf32, strided<[160, 2]>> to memref<1x32xf32, strided<[160,2]>>
  memref.store %val, %subview[%c0, %c0] : memref<1x32xf32, strided<[160, 2]>>
  return
}
 
 
// -----
 
// CHECK-LABEL: func.func @test_subview_expand_standard_hyperrectangle(
// CHECK: %[[VAL_0:.*]] = memref.subview %[[VAL_1:.*]][0, 0] [4, 8] [1, 1] : memref<16x8xf32> to memref<4x8xf32, strided<[8, 1]>>
func.func @test_subview_expand_standard_hyperrectangle(){
  %arg0 = memref.alloc() : memref<128xf32>
  %c0 = arith.constant 0 : index
  %val = arith.constant 3.14 : f32
  %subview = memref.subview %arg0[0] [32] [1] : memref<128xf32> to memref<32xf32>
  %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [4, 8] : memref<32xf32> into memref<4x8xf32>
  memref.store %val, %expand_shape[%c0, %c0] : memref<4x8xf32>
  return
}
 
 
 
// -----
 
// CHECK-LABEL: func.func @test_subview_expand_stride_not_unit(
// CHECK: %[[VAL_0:.*]] = memref.subview %[[VAL_1:.*]][0, 0] [5, 4] [1, 4] : memref<8x16xf32, strided<[32, 2]>> to memref<5x4xf32, strided<[32, 8]>>
func.func @test_subview_expand_stride_not_unit(){
  %arg0 = memref.alloc() : memref<128xf32, strided<[2]>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 3.14 : f32
  %subview = memref.subview %arg0[0] [20] [4] : memref<128xf32, strided<[2]>> to memref<20xf32, strided<[8]>>
  %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [5, 4] : memref<20xf32, strided<[8]>> into memref<5x4xf32, strided<[32, 8]>>
  memref.store %val, %expand_shape[%c0, %c0] : memref<5x4xf32, strided<[32, 8]>>
  return
}

// -----
// CHECK-LABEL: func.func @avoid_infinite_loop(
// CHECK: return %expanded : tensor<20x1x2x1x21xf32>
func.func @avoid_infinite_loop(%arg0: tensor<20x2x21xf32>, %arg1: tensor<20x21xf32>, %arg2: tensor<20x1x2x1x21xf32>, %arg3: f32, %arg4: tensor<20x1x2x1x21xf32>) -> tensor<20x1x2x1x21xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %collapsed = tensor.collapse_shape %arg4 [[0, 1], [2, 3], [4]] : tensor<20x1x2x1x21xf32> into tensor<20x2x21xf32>
  %collapsed_0 = tensor.collapse_shape %arg2 [[0, 1], [2, 3], [4]] : tensor<20x1x2x1x21xf32> into tensor<20x2x21xf32>
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%collapsed : tensor<20x2x21xf32>) outs(%collapsed_0 : tensor<20x2x21xf32>) -> tensor<20x2x21xf32>
  %1 = linalg.fill ins(%arg3 : f32) outs(%arg1 : tensor<20x21xf32>) -> tensor<20x21xf32>
  %reduced = linalg.reduce ins(%0 : tensor<20x2x21xf32>) outs(%1 : tensor<20x21xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %3 = arith.addf %in, %init : f32
      linalg.yield %3 : f32
    }
  %broadcasted = linalg.broadcast ins(%reduced : tensor<20x21xf32>) outs(%arg0 : tensor<20x2x21xf32>) dimensions = [1] 
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%0, %broadcasted : tensor<20x2x21xf32>, tensor<20x2x21xf32>) outs(%collapsed_0 : tensor<20x2x21xf32>) -> tensor<20x2x21xf32>
  %expanded = tensor.expand_shape %2 [[0, 1], [2, 3], [4]] output_shape [20, 1, 2, 1, 21] : tensor<20x2x21xf32> into tensor<20x1x2x1x21xf32>
  return %expanded : tensor<20x1x2x1x21xf32>
}

// -----
// CHECK-LABEL: func.func @avoid_infinite_loop2(
// CHECK: tensor.insert_slice %[[VAL_1:.*]] into %[[VAL_2:.*]][0, 4] [3, 1] [1, 1] : tensor<3xi64> into tensor<3x10xi64>
func.func @avoid_infinite_loop2(%arg0: tensor<3xi64>, %arg1: tensor<3xi64>, %arg2: tensor<3xi64>, %arg3: tensor<3xi64>, %arg4: tensor<3xi64>, %arg5: tensor<3xi64>, %arg6: i64, %arg7: tensor<3x10xi64>) -> tensor<3x10xi64> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [3, 1] : tensor<3xi64> into tensor<3x1xi64>
  %0 = tensor.empty() : tensor<3xi1>
  %expanded_0 = tensor.expand_shape %0 [[0, 1]] output_shape [3, 1] : tensor<3xi1> into tensor<3x1xi1>
  %1 = tensor.empty() : tensor<3xi64>
  %expanded_1 = tensor.expand_shape %1 [[0, 1]] output_shape [3, 1] : tensor<3xi64> into tensor<3x1xi64>
  %2 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg5 : tensor<3xi64>, tensor<3xi64>) outs(%0 : tensor<3xi1>) -> tensor<3xi1>
  %expanded_2 = tensor.expand_shape %2 [[0, 1]] output_shape [3, 1] : tensor<3xi1> into tensor<3x1xi1>
  %3 = hfusion.select ins(%expanded_2, %arg6, %expanded : tensor<3x1xi1>, i64, tensor<3x1xi64>) outs(%expanded_1 : tensor<3x1xi64>) -> tensor<3x1xi64>
  %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<3x1xi64> into tensor<3xi64>
  %inserted_slice = tensor.insert_slice %collapsed into %arg7[0, 0] [3, 1] [1, 1] : tensor<3xi64> into tensor<3x10xi64>
  %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%expanded, %expanded : tensor<3x1xi64>, tensor<3x1xi64>) outs(%expanded_0 : tensor<3x1xi1>) -> tensor<3x1xi1>
  %5 = hfusion.select ins(%4, %arg6, %expanded : tensor<3x1xi1>, i64, tensor<3x1xi64>) outs(%expanded_1 : tensor<3x1xi64>) -> tensor<3x1xi64>
  %extracted_slice = tensor.extract_slice %inserted_slice[0, 2] [3, 8] [1, 1] : tensor<3x10xi64> to tensor<3x8xi64>
  %concat = tensor.concat dim(1) %3, %5, %extracted_slice : (tensor<3x1xi64>, tensor<3x1xi64>, tensor<3x8xi64>) -> tensor<3x10xi64>
  %extracted_slice_3 = tensor.extract_slice %concat[0, 0] [3, 2] [1, 1] : tensor<3x10xi64> to tensor<3x2xi64>
  %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, 1] [3, 7] [1, 1] : tensor<3x8xi64> to tensor<3x7xi64>
  %concat_5 = tensor.concat dim(1) %extracted_slice_3, %5, %extracted_slice_4 : (tensor<3x2xi64>, tensor<3x1xi64>, tensor<3x7xi64>) -> tensor<3x10xi64>
  %extracted_slice_6 = tensor.extract_slice %concat_5[0, 0] [3, 3] [1, 1] : tensor<3x10xi64> to tensor<3x3xi64>
  %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, 1] [3, 6] [1, 1] : tensor<3x7xi64> to tensor<3x6xi64>
  %concat_8 = tensor.concat dim(1) %extracted_slice_6, %5, %extracted_slice_7 : (tensor<3x3xi64>, tensor<3x1xi64>, tensor<3x6xi64>) -> tensor<3x10xi64>
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg4, %arg5 : tensor<3xi64>, tensor<3xi64>) outs(%0 : tensor<3xi1>) -> tensor<3xi1>
  %7 = hfusion.select ins(%6, %arg6, %arg4 : tensor<3xi1>, i64, tensor<3xi64>) outs(%1 : tensor<3xi64>) -> tensor<3xi64>
  %inserted_slice_9 = tensor.insert_slice %7 into %concat_8[0, 4] [3, 1] [1, 1] : tensor<3xi64> into tensor<3x10xi64>
  return %inserted_slice_9 : tensor<3x10xi64>
}

// -----
// CHECK-LABEL: func.func @avoid_infinite_loop3(
// CHECK: tensor.extract_slice %[[VAL_1:.*]][0, 26] [2, 1] [1, 1] : tensor<2x27xf32> to tensor<2xf32>
func.func @avoid_infinite_loop3(%arg0: tensor<2x24xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2x27xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x27xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x27xf32>) -> tensor<2x27xf32>
  %inserted_slice = tensor.insert_slice %arg0 into %1[0, 0] [2, 24] [1, 1] : tensor<2x24xf32> into tensor<2x27xf32>
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
  %inserted_slice_0 = tensor.insert_slice %expanded into %inserted_slice[0, 24] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x27xf32>
  %extracted_slice = tensor.extract_slice %inserted_slice_0[0, 0] [2, 25] [1, 1] : tensor<2x27xf32> to tensor<2x25xf32>
  %extracted_slice_1 = tensor.extract_slice %inserted_slice_0[0, 26] [2, 1] [1, 1] : tensor<2x27xf32> to tensor<2xf32>
  %expanded_2 = tensor.expand_shape %extracted_slice_1 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
  %expanded_3 = tensor.expand_shape %arg2 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
  %concat = tensor.concat dim(1) %extracted_slice, %expanded_3, %expanded_2 : (tensor<2x25xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x27xf32>
  %expanded_4 = tensor.expand_shape %arg3 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
  %extracted_slice_5 = tensor.extract_slice %concat[0, 0] [2, 26] [1, 1] : tensor<2x27xf32> to tensor<2x26xf32>
  %concat_6 = tensor.concat dim(1) %extracted_slice_5, %expanded_4 : (tensor<2x26xf32>, tensor<2x1xf32>) -> tensor<2x27xf32>
  return %concat_6 : tensor<2x27xf32>
}