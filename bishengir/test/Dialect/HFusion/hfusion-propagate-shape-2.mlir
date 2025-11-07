// REQUIRES: asserts
// RUN: bishengir-opt %s --propagate-reshape --cse --canonicalize-ext --valid-propagate --debug-only="propagate-valid-check" -split-input-file | FileCheck %s

// CHECK: Valid
// CHECK-LABEL: @expand_up_reduce_multi(
// CHECK: %[[RED:.*]]:2 = linalg.reduce
// CHECK: linalg.elemwise_unary ins(%[[RED]]#1
module {
  func.func @expand_up_reduce_multi(%arg0: tensor<24x32x12xf32>, %arg1: tensor<24x32x12xf32>, %arg2: tensor<24x12xf32>, %arg3: tensor<24x12xf32>) -> tensor<4x6x12xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = tensor.empty() : tensor<24x32x12xf32>
    %1 = tensor.empty() : tensor<4x6x12xf32>
    %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg0 : tensor<24x32x12xf32>) outs(%0 : tensor<24x32x12xf32>) -> tensor<24x32x12xf32>
    %3 = tensor.empty() : tensor<f32>
    %reduced:2 = linalg.reduce ins(%arg1, %2 : tensor<24x32x12xf32>, tensor<24x32x12xf32>) outs(%arg2, %arg3 : tensor<24x12xf32>, tensor<24x12xf32>) dimensions = [1]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_0: f32, %init: f32, %init_1: f32) {
        %6 = arith.addf %in, %init : f32
        %7 = arith.addf %in_0, %init_1 : f32
        linalg.yield %6, %7 : f32, f32
      }
    %expanded = tensor.expand_shape %reduced#1 [[0, 1], [2]] output_shape [4, 6, 12] : tensor<24x12xf32> into tensor<4x6x12xf32>
    %4 = tensor.empty() : tensor<1xi32>
    %5 = linalg.elemwise_unary ins(%expanded : tensor<4x6x12xf32>) outs(%1 : tensor<4x6x12xf32>) -> tensor<4x6x12xf32>
    %6 = linalg.elemwise_unary ins(%5 : tensor<4x6x12xf32>) outs(%1 : tensor<4x6x12xf32>) -> tensor<4x6x12xf32>
    return %6 : tensor<4x6x12xf32>
  }
}

// -----

// CHECK: Valid
// CHECK-LABEL: func.func @test_collapse_down
// CHECK: hfusion.cast
// CHECK-NOT: tensor.collapse_shape
// CHECK: tensor.extract
func.func @test_collapse_down(%arg0: tensor<1x1xi32>) -> f32 attributes {OperatorType = "Default", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %0 = tensor.empty() : tensor<1x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<1x1xi32>) outs(%0 : tensor<1x1xf32>) -> tensor<1x1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1x1xf32> into tensor<f32>
  %extracted = tensor.extract %collapsed[] : tensor<f32>
  return %extracted : f32
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[0, 0, 0, 0, 0] [4096, 1, 128, 1, 32] [1, 1, 1, 1, 1]
// CHECK: collapse
// CHECK: return
module {
  func.func @collapse_extract(%arg0: tensor<4096x1x128x2x32xf32>) -> tensor<4096x1x128x32xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4096x1x128x2x32xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x1x128x2x32xf32>) outs(%0 : tensor<4096x1x128x2x32xbf16>) -> tensor<4096x1x128x2x32xbf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2], [3, 4]] {debug_instruction_number = 4 : i32} : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x1x128x64xbf16> to tensor<4096x1x128x32xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<4096x1x128x32xbf16>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_second_half(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[0, 0, 0, 1, 0] [4096, 1, 128, 1, 32] [1, 1, 1, 1, 1]
// CHECK: collapse
// CHECK: return
module {
  func.func @collapse_extract_second_half(%arg0: tensor<4096x1x128x2x32xf32>) -> tensor<4096x1x128x32xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4096x1x128x2x32xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x1x128x2x32xf32>) outs(%0 : tensor<4096x1x128x2x32xbf16>) -> tensor<4096x1x128x2x32xbf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2], [3, 4]] {debug_instruction_number = 4 : i32} : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0, 32] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x1x128x64xbf16> to tensor<4096x1x128x32xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<4096x1x128x32xbf16>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_4_3_aligned(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[0, 0] [2, 3] [1, 1]
// CHECK: collapse
// CHECK: return
// [0, 1, 2, 3, 4, 5, (6, 7, 8, 9, 10, 11)]
//
// [0, 1, 2],
// [3, 4, 5],
// [6, 7, 8],  // Taking this
// [9, 10, 11]
module {
  func.func @collapse_extract_4_3_aligned(%arg0: tensor<4x3xf32>) -> tensor<6xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x3xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x3xf32>) outs(%0 : tensor<4x3xbf16>) -> tensor<4x3xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] {debug_instruction_number = 4 : i32} : tensor<4x3xbf16> into tensor<12xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0] [6] [1] {debug_instruction_number = 5 : i32} : tensor<12xbf16> to tensor<6xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<6xbf16>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_4_3_aligned(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[1, 0] [2, 3] [1, 1]
// CHECK: collapse
// CHECK: return
// [0, 1, 2, (3, 4, 5, 6, 7, 8), 9, 10, 11]
//
// [0, 1, 2],
// [3, 4, 5],  // Taking this
// [6, 7, 8],  // Taking this
// [9, 10, 11]
module {
  func.func @collapse_extract_4_3_aligned(%arg0: tensor<4x3xf32>) -> tensor<6xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x3xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x3xf32>) outs(%0 : tensor<4x3xbf16>) -> tensor<4x3xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] {debug_instruction_number = 4 : i32} : tensor<4x3xbf16> into tensor<12xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[3] [6] [1] {debug_instruction_number = 5 : i32} : tensor<12xbf16> to tensor<6xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<6xbf16>
  }
}

// -----
// CHECK: Failed
// CHECK-LABEL: func.func @collapse_extract_4_3_unaligned(
// CHECK: cast
// CHECK: collapse
// CHECK: extract_slice
// CHECK: return
// [0, 1, (2, 3, 4, 5, 6, 7), 8, 9, 10, 11]
//
// [0, 1, 2],
// [3, 4, 5],
// [6, 7, 8],
// [9, 10, 11]
module {
  func.func @collapse_extract_4_3_unaligned(%arg0: tensor<4x3xf32>) -> tensor<6xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x3xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x3xf32>) outs(%0 : tensor<4x3xbf16>) -> tensor<4x3xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] {debug_instruction_number = 4 : i32} : tensor<4x3xbf16> into tensor<12xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[2] [6] [1] {debug_instruction_number = 5 : i32} : tensor<12xbf16> to tensor<6xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<6xbf16>
  }
}

// -----
// CHECK: Failed
// CHECK-LABEL: func.func @collapse_extract_3_4_unaligned(
// CHECK: cast
// CHECK: collapse
// CHECK: extract_slice
// CHECK: return
// [0, 1, 2, 3, 4, 5, (6, 7, 8, 9, 10, 11)]
//
// [0, 1, 2, 3],
// [4, 5, 6, 7],
// [8, 9, 10, 11]
module {
  func.func @collapse_extract_3_4_unaligned(%arg0: tensor<3x4xf32>) -> tensor<6xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<3x4xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<3x4xf32>) outs(%0 : tensor<3x4xbf16>) -> tensor<3x4xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] {debug_instruction_number = 4 : i32} : tensor<3x4xbf16> into tensor<12xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0] [6] [1] {debug_instruction_number = 5 : i32} : tensor<12xbf16> to tensor<6xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<6xbf16>
  }
}
// -----
// CHECK: Failed
// CHECK-LABEL: func.func @collapse_extract_3d_unaligned(
// CHECK: cast
// CHECK: collapse
// CHECK: extract_slice
// CHECK: return
module {
  func.func @collapse_extract_3d_unaligned(%arg0: tensor<2x2x2xbf16>) -> tensor<6xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<2x2x2xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<2x2x2xbf16>) outs(%0 : tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2]] {debug_instruction_number = 4 : i32} : tensor<2x2x2xbf16> into tensor<8xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0] [6] [1] {debug_instruction_number = 5 : i32} : tensor<8xbf16> to tensor<6xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<6xbf16>
  }
}
// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_3d_aligned(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[1, 0, 0] [2, 2, 2] [1, 1, 1]
// CHECK: collapse
// CHECK: return
module {
  func.func @collapse_extract_3d_aligned(%arg0: tensor<4x2x2xbf16>) -> tensor<8xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x2x2xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x2xbf16>) outs(%0 : tensor<4x2x2xbf16>) -> tensor<4x2x2xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2]] {debug_instruction_number = 4 : i32} : tensor<4x2x2xbf16> into tensor<16xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[4] [8] [1] {debug_instruction_number = 5 : i32} : tensor<16xbf16> to tensor<8xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<8xbf16>
  }
}
// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_3d_aligned(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[1, 0, 0, 0] [2, 2, 2, 5] [1, 1, 1, 1]
// CHECK: collapse
// CHECK: return
module {
  func.func @collapse_extract_3d_aligned(%arg0: tensor<4x2x2x5xbf16>) -> tensor<8x5xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x2x2x5xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x2x5xbf16>) outs(%0 : tensor<4x2x2x5xbf16>) -> tensor<4x2x2x5xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2], [3]] {debug_instruction_number = 4 : i32} : tensor<4x2x2x5xbf16> into tensor<16x5xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[4, 0] [8, 5] [1, 1] {debug_instruction_number = 5 : i32} : tensor<16x5xbf16> to tensor<8x5xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<8x5xbf16>
  }
}
// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_single(
// CHECK: cast
// CHECK: extract_slice %{{.*}}[1, 0, 0] [1, 1, 1] [1, 1, 1]
// CHECK: collapse
// CHECK: return
module {
  func.func @collapse_extract_single(%arg0: tensor<4x2x2xbf16>) -> tensor<1xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x2x2xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x2xbf16>) outs(%0 : tensor<4x2x2xbf16>) -> tensor<4x2x2xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2]] {debug_instruction_number = 4 : i32} : tensor<4x2x2xbf16> into tensor<16xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[4] [1] [1] {debug_instruction_number = 5 : i32} : tensor<16xbf16> to tensor<1xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<1xbf16>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_extract_all(
// CHECK: cast
// CHECK-NOT: extract_slice
// CHECK: collapse
// CHECK: return
module {
  func.func @collapse_extract_all(%arg0: tensor<4x2x2xbf16>) -> tensor<16xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4x2x2xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x2xbf16>) outs(%0 : tensor<4x2x2xbf16>) -> tensor<4x2x2xbf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2]] {debug_instruction_number = 4 : i32} : tensor<4x2x2xbf16> into tensor<16xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0] [16] [1] {debug_instruction_number = 5 : i32} : tensor<16xbf16> to tensor<16xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<16xbf16>
  }
}

// -----

// CHECK: Valid
// CHECK-LABEL: func.func @extract_expand(
// CHECK: expand
// CHECK: extract_slice %{{.*}}[0, 32, 0] [24, 32, 4] [1, 1, 1]
// CHECK: load
// CHECK: return
module {
  func.func @extract_expand(%arg0: tensor<24x256xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}) -> tensor<24x32x4xbf16> attributes {debug_instruction_number = 6 : i32, enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %extracted_slice = tensor.extract_slice %arg0[0, 128] [24, 128] [1, 1] {debug_instruction_number = 0 : i32} : tensor<24x256xbf16> to tensor<24x128xbf16>
    %0 = tensor.empty() {debug_instruction_number = 1 : i32} : tensor<24x32x4xbf16>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [24, 32, 4] {debug_instruction_number = 2 : i32} : tensor<24x128xbf16> into tensor<24x32x4xbf16>
    %1 = hfusion.load {debug_instruction_number = 4 : i32} ins(%expanded : tensor<24x32x4xbf16>) outs(%0 : tensor<24x32x4xbf16>) -> tensor<24x32x4xbf16>
    return {debug_instruction_number = 5 : i32} %1 : tensor<24x32x4xbf16>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_collapsedown_empty_reassoc(
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[\[}}0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, %{{.*}}, 3, 5] : tensor<?x3x5xf32> into tensor<1x1x1x?x3x5xf32>
func.func @main_collapsedown_empty_reassoc(%arg0: tensor<?x4096xf16>, %arg1: tensor<1x1x1xi64>) -> tensor<?x1x3x5xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1x1x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1x1x1xi64>) outs(%0 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1x1x1xf32> into tensor<f32>
  %2 = tensor.empty(%dim) : tensor<?x3x5xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%2 : tensor<?x3x5xf32>) dimensions = [0, 1, 2]
  %expanded = tensor.expand_shape %broadcasted [[0, 1], [2], [3]] output_shape [%dim, 1, 3, 5] : tensor<?x3x5xf32> into tensor<?x1x3x5xf32>
  return %expanded : tensor<?x1x3x5xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_collapsedown_empty_reassoc(
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[\[}}0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, %{{.*}}, 3, 5] : tensor<?x3x5xf32> into tensor<1x1x1x?x3x5xf32>
func.func @main_collapsedown_empty_reassoc(%arg0: tensor<?x4096xf16>, %arg1: tensor<1x1x1xi64>) -> tensor<?x1x3x5xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1x1x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1x1x1xi64>) outs(%0 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1x1x1xf32> into tensor<f32>
  %2 = tensor.empty(%dim) : tensor<?x3x5xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%2 : tensor<?x3x5xf32>) dimensions = [0, 1, 2]
  %expanded = tensor.expand_shape %broadcasted [[0, 1], [2], [3]] output_shape [%dim, 1, 3, 5] : tensor<?x3x5xf32> into tensor<?x1x3x5xf32>
  return %expanded : tensor<?x1x3x5xf32>
}


// -----
// CHECK: Valid
// CHECK-LABEL: @main_collapsedown_reduce(
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[\[}}0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, %{{.*}}, 3, 5] : tensor<?x3x5xf32> into tensor<1x1x1x?x3x5xf32>
func.func @main_collapsedown_reduce(%arg0: tensor<?x4096xf16>, %arg1: tensor<1x1x1xi64>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1x1x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1x1x1xi64>) outs(%0 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1x1x1xf32> into tensor<f32>
  %2 = tensor.empty(%dim) : tensor<?x3x5xf32>
  %reduced = linalg.reduce ins(%2 : tensor<?x3x5xf32>) outs(%collapsed : tensor<f32>) dimensions = [0, 1, 2]
  (%in: f32, %init: f32) {
    %asdasd = arith.addf %in, %init : f32
    linalg.yield %asdasd : f32
  }
  return %reduced : tensor<f32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_insert(
module {
  func.func @collapse_insert(%arg0: tensor<4096x1x128x2x32xf32>, %val: tensor<4096x1x128x32xbf16>) -> tensor<4096x1x128x64xbf16> attributes {debug_instruction_number = 7 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() {debug_instruction_number = 0 : i32} : tensor<4096x1x128x2x32xbf16>
    %1 = hfusion.cast {debug_instruction_number = 3 : i32, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x1x128x2x32xf32>) outs(%0 : tensor<4096x1x128x2x32xbf16>) -> tensor<4096x1x128x2x32xbf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2], [3, 4]] {debug_instruction_number = 4 : i32} : tensor<4096x1x128x2x32xbf16> into tensor<4096x1x128x64xbf16>
    %extracted_slice = tensor.insert_slice %val into %collapsed[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] {debug_instruction_number = 5 : i32} : tensor<4096x1x128x32xbf16> into tensor<4096x1x128x64xbf16>
    return {debug_instruction_number = 6 : i32} %extracted_slice : tensor<4096x1x128x64xbf16>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @mlir_fused_copy_25(
module {
  func.func @mlir_fused_copy_25(%arg0: tensor<4x4x256x513xf32>, %arg1: tensor<4x3x513x512xf32>) -> tensor<4x256x513xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [4, 1, 256, 513] [1, 1, 1, 1] : tensor<4x4x256x513xf32> to tensor<4x1x256x513xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0], [1, 2], [3]] : tensor<4x1x256x513xf32> into tensor<4x256x513xf32>
    %extracted_slice_0 = tensor.extract_slice %collapsed[0, 1, 0] [4, 255, 513] [1, 1, 1] : tensor<4x256x513xf32> to tensor<4x255x513xf32>
    %collapsed_1 = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<4x3x513x512xf32> into tensor<4x3x262656xf32>
    %expanded = tensor.expand_shape %collapsed_1 [[0], [1], [2, 3]] output_shape [4, 3, 512, 513] : tensor<4x3x262656xf32> into tensor<4x3x512x513xf32>
    %extracted_slice_2 = tensor.extract_slice %expanded[0, 0, 0, 0] [4, 1, 512, 513] [1, 1, 1, 1] : tensor<4x3x512x513xf32> to tensor<4x1x512x513xf32>
    %collapsed_3 = tensor.collapse_shape %extracted_slice_2 [[0], [1, 2], [3]] : tensor<4x1x512x513xf32> into tensor<4x512x513xf32>
    %extracted_slice_4 = tensor.extract_slice %collapsed_3[0, 0, 0] [4, 255, 513] [1, 1, 1] : tensor<4x512x513xf32> to tensor<4x255x513xf32>
    %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0, 258] [4, 255, 255] [1, 1, 1] : tensor<4x255x513xf32> to tensor<4x255x255xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice_5 into %extracted_slice_0[0, 0, 1] [4, 255, 255] [1, 1, 1] : tensor<4x255x255xf32> into tensor<4x255x513xf32>
    %inserted_slice_6 = tensor.insert_slice %inserted_slice into %collapsed[0, 1, 0] [4, 255, 513] [1, 1, 1] : tensor<4x255x513xf32> into tensor<4x256x513xf32>
    return %inserted_slice_6 : tensor<4x256x513xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @transpose_error(
module {
  func.func @transpose_error(%arg0: tensor<2x16x16x1280xf32>, %arg1: tensor<2x16x16x1280xf32>) -> tensor<2x1x32x1x1x40x1x1x16x16xf32> {
    %0 = tensor.empty() : tensor<2x16x16x1280xf32>
    %1 = tensor.empty() : tensor<2x1280x16x16xf32>
    %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<2x16x16x1280xf32>) outs(%0 : tensor<2x16x16x1280xf32>) -> tensor<2x16x16x1280xf32>
    %transposed = linalg.transpose ins(%2 : tensor<2x16x16x1280xf32>) outs(%1 : tensor<2x1280x16x16xf32>) permutation = [0, 3, 1, 2]  {debug_instruction_number = 12 : i32}
    %expanded = tensor.expand_shape %transposed [[0], [1, 2, 3, 4, 5], [6, 7, 8], [9]] output_shape [2, 1, 32, 1, 1, 40, 1, 1, 16, 16] : tensor<2x1280x16x16xf32> into tensor<2x1x32x1x1x40x1x1x16x16xf32>
    %3 = tensor.empty() : tensor<2x1x32x1x1x40x1x1x16x16xf32>
    %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%expanded : tensor<2x1x32x1x1x40x1x1x16x16xf32>) outs(%3 : tensor<2x1x32x1x1x40x1x1x16x16xf32>) -> tensor<2x1x32x1x1x40x1x1x16x16xf32>
    %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%4 : tensor<2x1x32x1x1x40x1x1x16x16xf32>) outs(%3 : tensor<2x1x32x1x1x40x1x1x16x16xf32>) -> tensor<2x1x32x1x1x40x1x1x16x16xf32>
    return %5 : tensor<2x1x32x1x1x40x1x1x16x16xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @extract_with_dynamic_only(
module {
  func.func @extract_with_dynamic_only(%arg0: tensor<?x1x9216xbf16>) -> tensor<?x48xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1x9216xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<?x1x9216xbf16> into tensor<?x9216xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0, 3072] [%dim, 3072] [1, 1] : tensor<?x9216xbf16> to tensor<?x3072xbf16>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%dim, 48, 64] : tensor<?x3072xbf16> into tensor<?x48x64xbf16>
    %0 = tensor.empty(%dim) : tensor<?x1x48x64xf32>
    %collapsed_0 = tensor.collapse_shape %0 [[0], [1, 2], [3]] : tensor<?x1x48x64xf32> into tensor<?x48x64xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<?x48x64xbf16>) outs(%collapsed_0 : tensor<?x48x64xf32>) -> tensor<?x48x64xf32>
    %2 = tensor.empty(%dim) : tensor<?x48xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x48xf32>) -> tensor<?x48xf32>
    %reduced = linalg.reduce ins(%1 : tensor<?x48x64xf32>) outs(%3 : tensor<?x48xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    return %reduced : tensor<?x48xf32>
  }
}

// -----

// CHECK-LABEL: @bitcast_collapse_down(
// CHECK: %[[VAND:.*]] = hivm.hir.vand
// CHECK: %[[BITCAST:.*]] = hivm.hir.bitcast %[[VAND]] : tensor<32x2x1xi32> -> tensor<32x2x1xf32>
// CHECK: hivm.hir.store ins(%[[BITCAST]]
module {
  func.func @bitcast_collapse_down() {
    %0 = tensor.empty() : tensor<64xi32>
    %1 = tensor.empty() : tensor<32x2x1xi32>
    %2 = tensor.empty() : tensor<32x2x1xi32>
    %expanded = tensor.expand_shape %0 [[0, 1, 2]] output_shape [32, 2, 1] : tensor<64xi32> into tensor<32x2x1xi32>
    %3 = hivm.hir.vand ins(%2, %1 : tensor<32x2x1xi32>, tensor<32x2x1xi32>) outs(%expanded : tensor<32x2x1xi32>) -> tensor<32x2x1xi32>
    %collapsed = tensor.collapse_shape %3 [[0, 1, 2]] : tensor<32x2x1xi32> into tensor<64xi32>
    %4 = hivm.hir.bitcast %collapsed : tensor<64xi32> -> tensor<64xf32>
    %alloc = memref.alloc() : memref<64xf32, strided<[1]>>
    hivm.hir.store ins(%4 : tensor<64xf32>) outs(%alloc : memref<64xf32, strided<[1]>>)
    return
  }
}

// -----

// CHECK-LABEL: @bitcast_expand_up(
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64xi32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EMPTY]] {{\[}}[0, 1, 2]] output_shape [8, 2, 4] : tensor<64xi32> into tensor<8x2x4xi32>
// CHECK: %[[BITCAST:.*]] = hivm.hir.bitcast %[[EXPANDED]] : tensor<8x2x4xi32> -> tensor<8x2x4xf32>
// CHECK: hivm.hir.store ins(%[[BITCAST]] : tensor<8x2x4xf32>)
module {
  func.func @bitcast_expand_up() {
    %0 = tensor.empty() : tensor<64xi32>
    %1 = hivm.hir.bitcast %0 : tensor<64xi32> -> tensor<64xf32>
    %expanded = tensor.expand_shape %1 [[0, 1, 2]] output_shape [8, 2, 4] : tensor<64xf32> into tensor<8x2x4xf32>
    %alloc = memref.alloc() : memref<8x2x4xf32, strided<[8, 4, 1]>>
    hivm.hir.store ins(%expanded : tensor<8x2x4xf32>) outs(%alloc : memref<8x2x4xf32, strided<[8, 4, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL:   func.func @extract_expand(
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0:.*]] {{\[\[}}0], [1, 2]] output_shape [2, 1, 2560] : tensor<2x2560xbf16> into tensor<2x1x2560xbf16>
// CHECK: %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_1]][1, 0, 0] [1, 1, 2560] [1, 1, 1] : tensor<2x1x2560xbf16> to tensor<1x1x2560xbf16>
module {
  func.func @extract_expand(%arg0: tensor<2x2560xbf16>) -> (tensor<1x1x2560xbf16>) {
    %extracted_slice = tensor.extract_slice %arg0[1, 0] [1, 2560] [1, 1] : tensor<2x2560xbf16> to tensor<1x2560xbf16>
    %0 = tensor.empty() : tensor<1x1x2560xbf16>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [1, 1, 2560] : tensor<1x2560xbf16> into tensor<1x1x2560xbf16>
    %1 = hfusion.load ins(%expanded : tensor<1x1x2560xbf16>) outs(%0 : tensor<1x1x2560xbf16>) -> tensor<1x1x2560xbf16>
    return %1 : tensor<1x1x2560xbf16>
  }
}

// -----
// CHECK-LABEL:   func.func @insert_expand(
// CHECK: %[[VAL_3:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<4096x1x128x32xf32> into tensor<4096x1x32x4x32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<4096x1x128x64xf32> into tensor<4096x1x32x4x64xf32>
// CHECK: %[[VAL_5:.*]] = tensor.insert_slice %[[VAL_3]] into %[[VAL_4]][0, 0, 0, 0, 0] [4096, 1, 32, 4, 32] [1, 1, 1, 1, 1] : tensor<4096x1x32x4x32xf32> into tensor<4096x1x32x4x64xf32>
// CHECK: return
module {
  func.func @insert_expand(%arg0: tensor<2x2560xf32>, %arg1: tensor<4096x1x128x32xf32>, %arg2: tensor<4096x1x128x64xf32>) -> (tensor<4096x1x32x4x64xf32>) {
    %extracted_slice = tensor.insert_slice %arg1 into %arg2[0, 0, 0, 0] [4096, 1, 128, 32] [1, 1, 1, 1] : tensor<4096x1x128x32xf32> into tensor<4096x1x128x64xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1], [2, 3], [4]] output_shape [4096, 1, 32, 4, 64] : tensor<4096x1x128x64xf32> into tensor<4096x1x32x4x64xf32>
    %0 = tensor.empty() : tensor<4096x1x32x4x64xf32>
    %1 = hfusion.load ins(%expanded : tensor<4096x1x32x4x64xf32>) outs(%0 : tensor<4096x1x32x4x64xf32>) -> tensor<4096x1x32x4x64xf32>
    return %1 : tensor<4096x1x32x4x64xf32>
  }
}