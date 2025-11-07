// REQUIRES: asserts
// RUN: bishengir-opt %s --propagate-reshape --cse --canonicalize-ext --valid-propagate --debug-only="propagate-valid-check" -split-input-file | FileCheck %s

// CHECK: Valid
// CHECK-LABEL: func.func @model_0
// CHECK: hfusion.cast
// CHECK: reduce
// CHECK-NOT: collapse_shape
// CHECK: return
module {
  func.func @model_0(%arg0: tensor<24x128x256x192xbf16>) -> tensor<24x32x1x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<24x128x256x192xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x128x256x192xbf16>) outs(%0 : tensor<24x128x256x192xf32>) -> tensor<24x128x256x192xf32>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<24x128x49152xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 4, 49152] : tensor<24x128x49152xf32> into tensor<24x32x4x49152xf32>
    %2 = tensor.empty() : tensor<24x32xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<24x32x4x49152xf32>) outs(%2 : tensor<24x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    %expanded_0 = tensor.expand_shape %reduced [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    return %expanded_0 : tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @model_0_long
// CHECK: hfusion.cast
// CHECK: return

module {
  func.func @model_0_long(%arg0: tensor<24x128x256x192xbf16>) -> (tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 1.966080e+05 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x128x256x192xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x128x256x192xbf16>) outs(%0 : tensor<24x128x256x192xf32>) -> tensor<24x128x256x192xf32>
    %2 = tensor.empty() : tensor<24x128x256x192xf16>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<24x128x256x192xf32>) outs(%2 : tensor<24x128x256x192xf16>) -> tensor<24x128x256x192xf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<24x128x49152xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 4, 49152] : tensor<24x128x49152xf32> into tensor<24x32x4x49152xf32>
    %4 = tensor.empty() : tensor<24x32xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<24x32x4x49152xf32>) outs(%5 : tensor<24x32xf32>) dimensions = [2, 3] 
      (%in: f32, %init: f32) {
        %22 = arith.addf %in, %init : f32
        linalg.yield %22 : f32
      }
    %expanded_4 = tensor.expand_shape %reduced [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %6 = tensor.empty() : tensor<24x32x1x1xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_4, %7 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %9 = tensor.empty() : tensor<24x32x4x49152xf32>
    %collapsed_5 = tensor.collapse_shape %8 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed_5 : tensor<24x32xf32>) outs(%9 : tensor<24x32x4x49152xf32>) dimensions = [2, 3] 
    %10 = linalg.fill ins(%cst_2 : f32) outs(%9 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %10 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%9 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%expanded, %11 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%9 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %13 = linalg.fill ins(%cst_3 : f32) outs(%9 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %14 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%12, %13 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%9 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %reduced_6 = linalg.reduce ins(%14 : tensor<24x32x4x49152xf32>) outs(%5 : tensor<24x32xf32>) dimensions = [2, 3] 
      (%in: f32, %init: f32) {
        %22 = arith.addf %in, %init : f32
        linalg.yield %22 : f32
      }
    %expanded_7 = tensor.expand_shape %reduced_6 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_7, %7 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %16 = arith.truncf %cst_0 : f64 to f32
    %17 = linalg.fill ins(%16 : f32) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %18 = linalg.fill ins(%cst_2 : f32) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %18 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %19 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %21 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%20 : tensor<24x32x1x1xf32>) outs(%6 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    return %1, %3, %expanded_4, %8, %expanded_7, %21 : tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: propagate_parallel
// CHECK: expand_shape
// CHECK: broadcast
// CHECK-NOT: expand_shape
// CHECK: binary
// CHECK-NOT: expand_shape
// CHECK: return
module {
  func.func @propagate_parallel(%arg0: tensor<24x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<24x32xf32>, %arg3: tensor<24x128xf32>, %arg4: tensor<24x32xf32>, %arg5: tensor<24x32xf32>) -> (tensor<24x32xf32>, tensor<24x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x128xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<128xf32>) outs(%0 : tensor<24x128xf32>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted : tensor<24x128xf32>, tensor<24x128xf32>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %expanded = tensor.expand_shape %1 [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xf32> into tensor<24x32x4xf32>
    %2 = tensor.empty() : tensor<24x32xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<24x32x4xf32>) outs(%arg4 : tensor<24x32xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg3, %broadcasted : tensor<24x128xf32>, tensor<24x128xf32>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %expanded_1 = tensor.expand_shape %5 [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xf32> into tensor<24x32x4xf32>
    %reduced_2 = linalg.reduce ins(%expanded_1 : tensor<24x32x4xf32>) outs(%3 : tensor<24x32xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced_2, %cst_0 : tensor<24x32xf32>, f32) outs(%2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%4, %6 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%arg5 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %reduced, %7 : tensor<24x32xf32>, tensor<24x32xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_elemwise

// CHECK: elemwise_unary
// CHECK: elemwise_binary
// CHECK: collapse_shape
// CHECK: return
module {
  func.func @collapse_elemwise(%arg0: tensor<24x32x8x9xf32>, %arg1: tensor<24x32x8x9xf32>, %arg2: tensor<24x32x8x9xf32>) -> (tensor<24x256x9xf32>)  {
    %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<24x32x8x9xf32>) outs(%arg1 : tensor<24x32x8x9xf32>) -> tensor<24x32x8x9xf32>
    %collapsed_unary = tensor.collapse_shape %unary [[0], [1, 2], [3]] : tensor<24x32x8x9xf32> into tensor<24x256x9xf32>
    %collapsed_arg1 = tensor.collapse_shape %arg1 [[0], [1, 2], [3]] : tensor<24x32x8x9xf32> into tensor<24x256x9xf32>
    %collapsed_arg2 = tensor.collapse_shape %arg2 [[0], [1, 2], [3]] : tensor<24x32x8x9xf32> into tensor<24x256x9xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed_unary, %cst_0 : tensor<24x256x9xf32>, f32) outs(%collapsed_arg2 : tensor<24x256x9xf32>) -> tensor<24x256x9xf32>
    return %6 : tensor<24x256x9xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast
// CHECK-NOT: collapse_shape
// CHECK: broadcast
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_broadcast(%arg0: tensor<24x32x8x768xf32>, %arg1: tensor<24x32x6144x9xf32>, %arg2: tensor<24x32x8x9xf32>) -> tensor<24x32x6144x9xf32> {
  %expanded = tensor.expand_shape %arg1 [[0], [1], [2, 3], [4]] output_shape [24, 32, 8, 768, 9] : tensor<24x32x6144x9xf32> into tensor<24x32x8x768x9xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<24x32x8x768xf32>) outs(%expanded : tensor<24x32x8x768x9xf32>) dimensions = [4]
  %collapsed = tensor.collapse_shape %broadcasted [[0], [1], [2, 3], [4]] : tensor<24x32x8x768x9xf32> into tensor<24x32x6144x9xf32>
  return %collapsed : tensor<24x32x6144x9xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_1
// CHECK-NOT: collapse_shape
// CHECK: broadcast
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_broadcast_1(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x12x6x5xf32>) -> tensor<2x12x6x5xf32> {
  %expanded = tensor.expand_shape %arg1 [[0], [1, 2], [3], [4]] output_shape [2, 3, 4, 6, 5] : tensor<2x12x6x5xf32> into tensor<2x3x4x6x5xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<2x3x4x5xf32>) outs(%expanded : tensor<2x3x4x6x5xf32>) dimensions = [3]
  %collapsed = tensor.collapse_shape %broadcasted [[0], [1, 2], [3], [4]] : tensor<2x3x4x6x5xf32> into tensor<2x12x6x5xf32>
  return %collapsed : tensor<2x12x6x5xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_2
// CHECK-NOT: collapse_shape
// CHECK: broadcast
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_broadcast_2(%arg0: tensor<2x3x4x5x6x7xf32>, %arg1: tensor<2x12x210x8xf32>) -> tensor<2x12x210x8xf32> {
  %expanded = tensor.expand_shape %arg1 [[0], [1, 2], [3, 4, 5], [6]] output_shape [2, 3, 4, 5, 6, 7, 8] : tensor<2x12x210x8xf32> into tensor<2x3x4x5x6x7x8xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<2x3x4x5x6x7xf32>) outs(%expanded : tensor<2x3x4x5x6x7x8xf32>) dimensions = [6]
  %collapsed = tensor.collapse_shape %broadcasted [[0], [1, 2], [3, 4, 5], [6]] : tensor<2x3x4x5x6x7x8xf32> into tensor<2x12x210x8xf32>
  return %collapsed : tensor<2x12x210x8xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_3
// CHECK-NOT: collapse_shape
// CHECK: broadcast
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_broadcast_3(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<2x12x5x6x7x8xf32>) -> tensor<2x12x5x6x7x8xf32> {
  %expanded = tensor.expand_shape %arg1 [[0], [1, 2], [3], [4], [5], [6]] output_shape [2, 3, 4, 5, 6, 7, 8] : tensor<2x12x5x6x7x8xf32> into tensor<2x3x4x5x6x7x8xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<2x3x4x5x6xf32>) outs(%expanded : tensor<2x3x4x5x6x7x8xf32>) dimensions = [5, 6]
  %collapsed = tensor.collapse_shape %broadcasted [[0], [1, 2], [3], [4], [5], [6]] : tensor<2x3x4x5x6x7x8xf32> into tensor<2x12x5x6x7x8xf32>
  return %collapsed : tensor<2x12x5x6x7x8xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_reduce
// CHECK-NOT: collapse_shape
// CHECK: reduce
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_reduce(%arg0: tensor<2x3x4x5x6x7x8x9x10xf32>) -> tensor<2x5x336x10xf32> {
  %init = tensor.empty() : tensor<2x5x336x10xf32>
  %zero = arith.constant 0.0 : f32

  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7x8x9x10xf32>) outs(%arg0 : tensor<2x3x4x5x6x7x8x9x10xf32>) -> tensor<2x3x4x5x6x7x8x9x10xf32>
  %collapsed = tensor.collapse_shape %unary [[0], [1, 2], [3], [4, 5, 6], [7], [8]] : 
    tensor<2x3x4x5x6x7x8x9x10xf32> into tensor<2x12x5x336x9x10xf32>

  %reduced = linalg.reduce ins(%collapsed : tensor<2x12x5x336x9x10xf32>)
                            outs(%init : tensor<2x5x336x10xf32>)
                            dimensions = [1, 4]
                      (%asd: f32, %asdInit: f32) {
                        %inside = arith.addf %asd, %asdInit : f32
                        linalg.yield %inside : f32
                      }
  return %reduced : tensor<2x5x336x10xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_reduce_1
// CHECK-NOT: collapse_shape
// CHECK: reduce
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_reduce_1(%arg0: tensor<4x6x8x10x12xf32>) -> tensor<24xf32> {
  %0 = tensor.empty() : tensor<24xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<4x6x8x10x12xf32>) outs(%arg0 : tensor<4x6x8x10x12xf32>) -> tensor<4x6x8x10x12xf32>
  %collapsed = tensor.collapse_shape %unary [[0, 1], [2, 3, 4]] : tensor<4x6x8x10x12xf32> into tensor<24x960xf32>
  %reduced = linalg.reduce ins(%collapsed : tensor<24x960xf32>) outs(%0 : tensor<24xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %1 = arith.addf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<24xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_reduce_2
// CHECK-NOT: collapse_shape
// CHECK: reduce
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_reduce_2(%arg0: tensor<2x3x4x5x6x7xf32>) -> tensor<2x30x7xf32> {
  %0 = tensor.empty() : tensor<2x30x7xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7xf32>) outs(%arg0 : tensor<2x3x4x5x6x7xf32>) -> tensor<2x3x4x5x6x7xf32>
  %collapsed = tensor.collapse_shape %unary [[0], [1, 2], [3, 4], [5]] : tensor<2x3x4x5x6x7xf32> into tensor<2x12x30x7xf32>
  %reduced = linalg.reduce ins(%collapsed : tensor<2x12x30x7xf32>) outs(%0 : tensor<2x30x7xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %1 = arith.mulf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<2x30x7xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_reduce_3
// CHECK-NOT: collapse_shape
// CHECK: reduce
// CHECK: collapse_shape
// CHECK: return
func.func @collapse_reduce_3(%arg0: tensor<2x3x4x5x6x7x8xf32>) -> tensor<6x5x8xf32> {
  %0 = tensor.empty() : tensor<6x5x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7x8xf32>) outs(%arg0 : tensor<2x3x4x5x6x7x8xf32>) -> tensor<2x3x4x5x6x7x8xf32>
  %collapsed = tensor.collapse_shape %unary [[0, 1], [2], [3], [4, 5], [6]] : tensor<2x3x4x5x6x7x8xf32> into tensor<6x4x5x42x8xf32>
  %reduced = linalg.reduce ins(%collapsed : tensor<6x4x5x42x8xf32>) outs(%0 : tensor<6x5x8xf32>) dimensions = [1, 3]
    (%in: f32, %init: f32) {
      %1 = arith.addf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<6x5x8xf32>
}

// -----

// CHECK: Valid
// CHECK-LABEL: @model_2
// CHECK: return
module {
  func.func @model_2(%arg0: tensor<24x256x32x24xbf16>, %arg1: tensor<24x256x32x24xbf16>, %arg2: tensor<24x256x1x1xbf16>, %arg3: tensor<24x256x1x1xbf16>, %arg4: tensor<24x32xf32>, %arg5: tensor<256xf32>, %arg6: tensor<24x256x32x24xf32>, %arg7: tensor<24x32xf32>, %arg8: tensor<24x32xf32>, %arg9: tensor<24x32xf32>, %arg10: tensor<24x32x8x768xf32>, %arg11: tensor<24x256x32x24xbf16>) -> (tensor<24x32x8x768xf32>, tensor<24x256x32x24xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 1.6276041666666666E-4 : f64
    %cst_1 = arith.constant 1.000000e+00 : bf16
    %cst_2 = arith.constant dense<1> : tensor<i64>
    %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<24x256x32x24xbf16> into tensor<24x256x768xbf16>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xbf16> into tensor<24x32x8x768xbf16>
    %collapsed_3 = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<24x256x32x24xbf16> into tensor<24x256x768xbf16>
    %expanded_4 = tensor.expand_shape %collapsed_3 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xbf16> into tensor<24x32x8x768xbf16>
    %collapsed_5 = tensor.collapse_shape %arg10 [[0], [1, 2], [3]] : tensor<24x32x8x768xf32> into tensor<24x256x768xf32>
    %expanded_6 = tensor.expand_shape %collapsed_5 [[0], [1], [2, 3]] output_shape [24, 256, 32, 24] : tensor<24x256x768xf32> into tensor<24x256x32x24xf32>
    %0 = tensor.empty() : tensor<24x256x32x24xbf16>
    %collapsed_7 = tensor.collapse_shape %0 [[0], [1], [2, 3]] : tensor<24x256x32x24xbf16> into tensor<24x256x768xbf16>
    %expanded_8 = tensor.expand_shape %collapsed_7 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xbf16> into tensor<24x32x8x768xbf16>
    %collapsed_9 = tensor.collapse_shape %arg2 [[0], [1, 2, 3]] : tensor<24x256x1x1xbf16> into tensor<24x256xbf16>
    %1 = tensor.empty() : tensor<24x256xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_9 : tensor<24x256xbf16>) outs(%1 : tensor<24x256xf32>) -> tensor<24x256xf32>
    %3 = tensor.empty() : tensor<24x256x32x24xf32>
    %collapsed_10 = tensor.collapse_shape %3 [[0], [1], [2, 3]] : tensor<24x256x32x24xf32> into tensor<24x256x768xf32>
    %expanded_11 = tensor.expand_shape %collapsed_10 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xf32> into tensor<24x32x8x768xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<24x256x32x24xbf16>) outs(%3 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_8 : tensor<24x32x8x768xbf16>) outs(%expanded_11 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<24x256xf32>) outs(%4 : tensor<24x256x32x24xf32>) dimensions = [2, 3]
    %collapsed_12 = tensor.collapse_shape %broadcasted [[0], [1], [2, 3]] : tensor<24x256x32x24xf32> into tensor<24x256x768xf32>
    %expanded_13 = tensor.expand_shape %collapsed_12 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xf32> into tensor<24x32x8x768xf32>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<24x256x32x24xbf16>) outs(%3 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_4 : tensor<24x32x8x768xbf16>) outs(%expanded_11 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %broadcasted : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %expanded_13 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %collapsed_14 = tensor.collapse_shape %arg3 [[0], [1, 2, 3]] : tensor<24x256x1x1xbf16> into tensor<24x256xbf16>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_14 : tensor<24x256xbf16>) outs(%1 : tensor<24x256xf32>) -> tensor<24x256xf32>
    %broadcasted_15 = linalg.broadcast ins(%10 : tensor<24x256xf32>) outs(%4 : tensor<24x256x32x24xf32>) dimensions = [2, 3]
    %collapsed_16 = tensor.collapse_shape %broadcasted_15 [[0], [1], [2, 3]] : tensor<24x256x32x24xf32> into tensor<24x256x768xf32>
    %expanded_17 = tensor.expand_shape %collapsed_16 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xf32> into tensor<24x32x8x768xf32>
    %11 = linalg.fill ins(%cst_1 : bf16) outs(%0 : tensor<24x256x32x24xbf16>) -> tensor<24x256x32x24xbf16>
    %12 = linalg.fill ins(%cst_1 : bf16) outs(%expanded_8 : tensor<24x32x8x768xbf16>) -> tensor<24x32x8x768xbf16>
    %13 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%11 : tensor<24x256x32x24xbf16>) outs(%3 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %14 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%12 : tensor<24x32x8x768xbf16>) outs(%expanded_11 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_15, %13 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_17, %14 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %15 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%9, %16 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%17 : tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%18 : tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %21 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%19 : tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %22 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%20 : tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %cst : tensor<24x256x32x24xf32>, f32) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%22, %cst : tensor<24x32x8x768xf32>, f32) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %23 : f32, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %24 : f32, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %27 = tensor.empty() : tensor<bf16>
    %28 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%cst_2 : tensor<i64>) outs(%27 : tensor<bf16>) -> tensor<bf16>
    %29 = tensor.empty() : tensor<f32>
    %30 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%28 : tensor<bf16>) outs(%29 : tensor<f32>) -> tensor<f32>
    %broadcasted_18 = linalg.broadcast ins(%30 : tensor<f32>) outs(%4 : tensor<24x256x32x24xf32>) dimensions = [0, 1, 2, 3]
    %broadcasted_19 = linalg.broadcast ins(%30 : tensor<f32>) outs(%5 : tensor<24x32x8x768xf32>) dimensions = [0, 1, 2, 3]
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %13 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %14 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted_18, %31 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted_19, %32 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %33 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %34 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %13 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %14 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%35, %37 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%36, %38 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %39 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %40 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %43 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x256x32x24xbf16>) outs(%3 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %44 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<24x32x8x768xbf16>) outs(%expanded_11 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %45 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%43, %41 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %46 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%44, %42 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%45, %broadcasted : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%4 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %48 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%46, %expanded_13 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%5 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %49 = tensor.empty() : tensor<24x32x8xf32>
    %broadcasted_20 = linalg.broadcast ins(%arg4 : tensor<24x32xf32>) outs(%49 : tensor<24x32x8xf32>) dimensions = [2]
    %expanded_21 = tensor.expand_shape %arg5 [[0, 1]] output_shape [32, 8] : tensor<256xf32> into tensor<32x8xf32>
    %broadcasted_22 = linalg.broadcast ins(%expanded_21 : tensor<32x8xf32>) outs(%49 : tensor<24x32x8xf32>) dimensions = [0]
    %50 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_20, %broadcasted_22 : tensor<24x32x8xf32>, tensor<24x32x8xf32>) outs(%49 : tensor<24x32x8xf32>) -> tensor<24x32x8xf32>
    %51 = tensor.empty() : tensor<24x32x8x768xf32>
    %collapsed_23 = tensor.collapse_shape %51 [[0], [1, 2], [3]] : tensor<24x32x8x768xf32> into tensor<24x256x768xf32>
    %expanded_24 = tensor.expand_shape %collapsed_23 [[0], [1], [2, 3]] output_shape [24, 256, 32, 24] : tensor<24x256x768xf32> into tensor<24x256x32x24xf32>
    %broadcasted_25 = linalg.broadcast ins(%50 : tensor<24x32x8xf32>) outs(%51 : tensor<24x32x8x768xf32>) dimensions = [3]
    %collapsed_26 = tensor.collapse_shape %broadcasted_25 [[0], [1, 2], [3]] : tensor<24x32x8x768xf32> into tensor<24x256x768xf32>
    %expanded_27 = tensor.expand_shape %collapsed_26 [[0], [1], [2, 3]] output_shape [24, 256, 32, 24] : tensor<24x256x768xf32> into tensor<24x256x32x24xf32>
    %52 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%48, %broadcasted_25 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%51 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %53 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%47, %expanded_27 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%expanded_24 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %collapsed_28 = tensor.collapse_shape %arg6 [[0], [1], [2, 3]] : tensor<24x256x32x24xf32> into tensor<24x256x768xf32>
    %expanded_29 = tensor.expand_shape %collapsed_28 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xf32> into tensor<24x32x8x768xf32>
    %54 = tensor.empty() : tensor<24x32xf32>
    %55 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg7, %arg8 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %56 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg9, %cst : tensor<24x32xf32>, f32) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %57 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%55, %56 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %58 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%57, %arg4 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %59 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%58, %arg4 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %60 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%59, %arg4 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %61 = arith.truncf %cst_0 : f64 to f32
    %62 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%60, %61 : tensor<24x32xf32>, f32) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %broadcasted_30 = linalg.broadcast ins(%62 : tensor<24x32xf32>) outs(%51 : tensor<24x32x8x768xf32>) dimensions = [2, 3]
    %collapsed_31 = tensor.collapse_shape %broadcasted_30 [[0], [1, 2], [3]] : tensor<24x32x8x768xf32> into tensor<24x256x768xf32>
    %expanded_32 = tensor.expand_shape %collapsed_31 [[0], [1], [2, 3]] output_shape [24, 256, 32, 24] : tensor<24x256x768xf32> into tensor<24x256x32x24xf32>
    %63 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_29, %broadcasted_30 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%51 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %64 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg6, %expanded_32 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%expanded_24 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %65 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%63, %cst : tensor<24x32x8x768xf32>, f32) outs(%51 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %66 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%64, %cst : tensor<24x256x32x24xf32>, f32) outs(%expanded_24 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %67 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%52, %65 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%arg10 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %68 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%53, %66 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%expanded_6 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %69 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%62 : tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %70 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%69, %arg8 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %71 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg7, %arg4 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %72 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%71, %61 : tensor<24x32xf32>, f32) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %73 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%72, %cst : tensor<24x32xf32>, f32) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %74 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%70, %73 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%54 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %broadcasted_33 = linalg.broadcast ins(%74 : tensor<24x32xf32>) outs(%51 : tensor<24x32x8x768xf32>) dimensions = [2, 3]
    %collapsed_34 = tensor.collapse_shape %broadcasted_33 [[0], [1, 2], [3]] : tensor<24x32x8x768xf32> into tensor<24x256x768xf32>
    %expanded_35 = tensor.expand_shape %collapsed_34 [[0], [1], [2, 3]] output_shape [24, 256, 32, 24] : tensor<24x256x768xf32> into tensor<24x256x32x24xf32>
    %75 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_35, %cst : tensor<24x256x32x24xf32>, f32) outs(%expanded_24 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %76 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%68, %75 : tensor<24x256x32x24xf32>, tensor<24x256x32x24xf32>) outs(%expanded_24 : tensor<24x256x32x24xf32>) -> tensor<24x256x32x24xf32>
    %77 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%76 : tensor<24x256x32x24xf32>) outs(%arg11 : tensor<24x256x32x24xbf16>) -> tensor<24x256x32x24xbf16>
    return %67, %77 : tensor<24x32x8x768xf32>, tensor<24x256x32x24xbf16>
  }
}

// -----

// CHECK: Valid
// CHECK-LABEL: model_24
// CHECK: return
func.func @model_24(%arg0: tensor<24x48x48xf32>, %arg1: tensor<24x48x1xf32>, %arg2: tensor<24x48x1xf32>, %arg3: tensor<24x48x48xf32>, %arg4: tensor<24x48x48xbf16>) -> (tensor<24x48x1xf32>, tensor<24x48x48xf32>, tensor<24x48x48xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<24x48x48xf32>
  %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<24x48x1xf32> into tensor<24x48xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<24x48xf32>) outs(%0 : tensor<24x48x48xf32>) dimensions = [2] 
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %cst : tensor<24x48x48xf32>, f32) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %1 : tensor<24x48x48xf32>, tensor<24x48x48xf32>) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%2 : tensor<24x48x48xf32>) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
  %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1, 2]] : tensor<24x48x1xf32> into tensor<24x48xf32>
  %reduced = linalg.reduce ins(%3 : tensor<24x48x48xf32>) outs(%collapsed_0 : tensor<24x48xf32>) dimensions = [2] 
    (%in: f32, %init: f32) {
      %6 = arith.addf %in, %init : f32
      linalg.yield %6 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xf32> into tensor<24x48x1xf32>
  %broadcasted_1 = linalg.broadcast ins(%reduced : tensor<24x48xf32>) outs(%0 : tensor<24x48x48xf32>) dimensions = [2] 
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %broadcasted_1 : tensor<24x48x48xf32>, tensor<24x48x48xf32>) outs(%arg3 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<24x48x48xf32>) outs(%arg4 : tensor<24x48x48xbf16>) -> tensor<24x48x48xbf16>
  return %expanded, %4, %5 : tensor<24x48x1xf32>, tensor<24x48x48xf32>, tensor<24x48x48xbf16>
}

// -----

// CHECK: Valid
// CHECK-LABEL: collapse_down_out
// CHECK: return
func.func @collapse_down_out(%arg0: tensor<24x10x48xf32>, %arg1: tensor<3x1x8x48x2x5x48x4x5xf32>, %arg2: tensor<24x48x10x48x20xf32>, %arg3: tensor<24x69x48x10x48x69x20xf32>) -> (tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1, 2], [3], [4, 5], [6], [7, 8]] : tensor<3x1x8x48x2x5x48x4x5xf32> into tensor<24x48x10x48x20xf32>
  %broadcasted_1 = linalg.broadcast ins(%arg0 : tensor<24x10x48xf32>) outs(%collapsed_0 : tensor<24x48x10x48x20xf32>) dimensions = [1, 4] 
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%collapsed_0 : tensor<24x48x10x48x20xf32>) outs(%collapsed_0 : tensor<24x48x10x48x20xf32>) -> tensor<24x48x10x48x20xf32>
  %reduced = linalg.reduce ins(%arg3 : tensor<24x69x48x10x48x69x20xf32>) outs(%collapsed_0 : tensor<24x48x10x48x20xf32>) dimensions = [1, 5] 
  (%in: f32, %init: f32) {
    %6 = arith.addf %in, %init : f32
    linalg.yield %6 : f32
  }
  return %broadcasted_1, %collapsed_0, %unary, %reduced: tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>
}

// -----

// CHECK: Valid
// CHECK-LABEL: expand_up_out
// CHECK: return
module {
  func.func @expand_up_out(%arg0: tensor<24x10x48xf32>, %arg1: tensor<3x1x8x48x2x5x48x4x5xf32>, %arg2: tensor<24x48x10x48x20xf32>, %arg3: tensor<24x69x48x10x48x69x20xf32>) -> (tensor<24x4x1x3x4x10x48x2x5x2xf32>, tensor<24x4x1x3x4x10x48x2x5x2xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<24x10x48xf32>) outs(%arg3 : tensor<24x69x48x10x48x69x20xf32>) dimensions = [1, 2, 5, 6]
    %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%broadcasted : tensor<24x69x48x10x48x69x20xf32>) outs(%arg3 : tensor<24x69x48x10x48x69x20xf32>) -> tensor<24x69x48x10x48x69x20xf32>
    %reduced = linalg.reduce ins(%0 : tensor<24x69x48x10x48x69x20xf32>) outs(%arg2 : tensor<24x48x10x48x20xf32>) dimensions = [1, 5]
      (%in: f32, %init: f32) {
        %1 = arith.addf %in, %init : f32
        linalg.yield %1 : f32
      }
    %expanded = tensor.expand_shape %arg2 [[0], [1, 2, 3, 4], [5], [6], [7, 8, 9]] output_shape [24, 4, 1, 3, 4, 10, 48, 2, 5, 2] : tensor<24x48x10x48x20xf32> into tensor<24x4x1x3x4x10x48x2x5x2xf32>
    %expanded_0 = tensor.expand_shape %reduced [[0], [1, 2, 3, 4], [5], [6], [7, 8, 9]] output_shape [24, 4, 1, 3, 4, 10, 48, 2, 5, 2] : tensor<24x48x10x48x20xf32> into tensor<24x4x1x3x4x10x48x2x5x2xf32>
    return %expanded, %expanded_0 : tensor<24x4x1x3x4x10x48x2x5x2xf32>, tensor<24x4x1x3x4x10x48x2x5x2xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast
// CHECK: return
func.func @collapse_broadcast(%arg0: tensor<24x32x8x768xf32>, %arg1: tensor<24x32x6144x9xf32>, %arg2: tensor<24x32x8x9xf32>) -> tensor<24x32x8x768x9xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<24x32x8x768xf32> into tensor<24x32x6144xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<24x32x6144xf32>) outs(%arg1 : tensor<24x32x6144x9xf32>) dimensions = [3]
  %expanded = tensor.expand_shape %broadcasted [[0], [1], [2, 3], [4]] output_shape [24, 32, 8, 768, 9] : tensor<24x32x6144x9xf32> into tensor<24x32x8x768x9xf32>
  return %expanded : tensor<24x32x8x768x9xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_1
// CHECK: return
func.func @collapse_broadcast_1(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x12x6x5xf32>) -> tensor<2x3x4x6x5xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<2x12x5xf32>) outs(%arg1 : tensor<2x12x6x5xf32>) dimensions = [2]
  %expanded = tensor.expand_shape %broadcasted [[0], [1, 2], [3], [4]] output_shape [2, 3, 4, 6, 5] : tensor<2x12x6x5xf32> into tensor<2x3x4x6x5xf32>
  return %expanded : tensor<2x3x4x6x5xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_2
// CHECK: return
func.func @collapse_broadcast_2(%arg0: tensor<2x3x4x5x6x7xf32>, %arg1: tensor<2x12x210x8xf32>) -> tensor<2x3x4x5x6x7x8xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3, 4, 5]] : tensor<2x3x4x5x6x7xf32> into tensor<2x12x210xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<2x12x210xf32>) outs(%arg1 : tensor<2x12x210x8xf32>) dimensions = [3]
  %expanded = tensor.expand_shape %broadcasted [[0], [1, 2], [3, 4, 5], [6]] output_shape [2, 3, 4, 5, 6, 7, 8] : tensor<2x12x210x8xf32> into tensor<2x3x4x5x6x7x8xf32>
  return %expanded : tensor<2x3x4x5x6x7x8xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_3
// CHECK: return
func.func @collapse_broadcast_3(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<2x12x5x6x7x8xf32>) -> tensor<2x3x4x5x6x7x8xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4]] : tensor<2x3x4x5x6xf32> into tensor<2x12x5x6xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<2x12x5x6xf32>) outs(%arg1 : tensor<2x12x5x6x7x8xf32>) dimensions = [4, 5]
  %expanded = tensor.expand_shape %broadcasted [[0], [1, 2], [3], [4], [5], [6]] output_shape [2, 3, 4, 5, 6, 7, 8] : tensor<2x12x5x6x7x8xf32> into tensor<2x3x4x5x6x7x8xf32>
  return %expanded : tensor<2x3x4x5x6x7x8xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapsing_empty
// CHECK: return
func.func @collapsing_empty(%arg0: tensor<1xf32>, %arg1: tensor<24xf32>, %arg2: tensor<24xf32>) -> tensor<24xf32> attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 1.16666663 : f32
  %collapsed = tensor.collapse_shape %arg0 [] : tensor<1xf32> into tensor<f32>
  %0 = tensor.empty() : tensor<24xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%0 : tensor<24xf32>) dimensions = [0]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %broadcasted : tensor<24xf32>, tensor<24xf32>) outs(%0 : tensor<24xf32>) -> tensor<24xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %cst : tensor<24xf32>, f32) outs(%arg2 : tensor<24xf32>) -> tensor<24xf32>
  return %2 : tensor<24xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_hivm_reduce_1
// CHECK: return
func.func @collapse_hivm_reduce_1(%arg0: tensor<?x?x?xf32>, %arg1: tensor <?x1xf32>) -> tensor<?x1xf32> {
  %unary =  hivm.hir.vrec ins(%arg0 : tensor<?x?x?xf32>) outs(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %collapsed = tensor.collapse_shape %unary [[0], [1, 2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  %reduced = hivm.hir.vreduce <sum> ins(%collapsed : tensor<?x?xf32>) outs(%arg1 : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
  return %reduced : tensor<?x1xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapsing_empty(
// CHECK: return
module {
  func.func @collapsing_empty(%arg0: tensor<1xf32>, %arg1: tensor<24xf32>, %arg2: tensor<24xf32>) -> tensor<24xf32> attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %cst = arith.constant 1.16666663 : f32
    %collapsed = tensor.collapse_shape %arg0 [] : tensor<1xf32> into tensor<f32>
    %0 = tensor.empty() : tensor<24xf32>
    %expanded = tensor.expand_shape %collapsed [] output_shape [1] : tensor<f32> into tensor<1xf32>
    %1 = hivm.hir.vbrc ins(%expanded : tensor<1xf32>) outs(%0 : tensor<24xf32>) broadcast_dims = [0] -> tensor<24xf32>
    %2 = hivm.hir.vdiv ins(%arg1, %1 : tensor<24xf32>, tensor<24xf32>) outs(%0 : tensor<24xf32>) -> tensor<24xf32>
    %3 = hivm.hir.vmul ins(%2, %cst : tensor<24xf32>, f32) outs(%arg2 : tensor<24xf32>) -> tensor<24xf32>
    return %3 : tensor<24xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast_3(
// CHECK: return
module {
  func.func @collapse_broadcast_3(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<2x12x5x6x7x8xf32>) -> tensor<2x3x4x5x6x7x8xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4]] : tensor<2x3x4x5x6xf32> into tensor<2x12x5x6xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1], [2], [3, 4, 5]] output_shape [2, 12, 5, 6, 1, 1] : tensor<2x12x5x6xf32> into tensor<2x12x5x6x1x1xf32>
    %0 = hivm.hir.vbrc ins(%expanded : tensor<2x12x5x6x1x1xf32>) outs(%arg1 : tensor<2x12x5x6x7x8xf32>) broadcast_dims = [4, 5] -> tensor<2x12x5x6x7x8xf32>
    %expanded_0 = tensor.expand_shape %0 [[0], [1, 2], [3], [4], [5], [6]] output_shape [2, 3, 4, 5, 6, 7, 8] : tensor<2x12x5x6x7x8xf32> into tensor<2x3x4x5x6x7x8xf32>
    return %expanded_0 : tensor<2x3x4x5x6x7x8xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @model_24(
// CHECK: return
module {
  func.func @model_24(%arg0: tensor<24x48x48xf32>, %arg1: tensor<24x48x1xf32>, %arg2: tensor<24x48x1xf32>, %arg3: tensor<24x48x48xf32>, %arg4: tensor<24x48x48xbf16>) -> (tensor<24x48x1xf32>, tensor<24x48x48xf32>, tensor<24x48x48xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x48x48xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<24x48x1xf32> into tensor<24x48xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xf32> into tensor<24x48x1xf32>
    %1 = hivm.hir.vbrc ins(%expanded : tensor<24x48x1xf32>) outs(%0 : tensor<24x48x48xf32>) broadcast_dims = [2] -> tensor<24x48x48xf32>
    %2 = hivm.hir.vmul ins(%1, %cst : tensor<24x48x48xf32>, f32) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
    %3 = hivm.hir.vsub ins(%arg0, %2 : tensor<24x48x48xf32>, tensor<24x48x48xf32>) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
    %4 = hivm.hir.vexp ins(%3 : tensor<24x48x48xf32>) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1, 2]] : tensor<24x48x1xf32> into tensor<24x48xf32>
    %expanded_1 = tensor.expand_shape %collapsed_0 [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xf32> into tensor<24x48x1xf32>
    %5 = hivm.hir.vreduce <sum> ins(%4 : tensor<24x48x48xf32>) outs(%expanded_1 : tensor<24x48x1xf32>) reduce_dims = [2] -> tensor<24x48x1xf32>
    %collapsed_2 = tensor.collapse_shape %5 [[0], [1, 2]] : tensor<24x48x1xf32> into tensor<24x48xf32>
    %expanded_3 = tensor.expand_shape %collapsed_2 [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xf32> into tensor<24x48x1xf32>
    %expanded_4 = tensor.expand_shape %collapsed_2 [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xf32> into tensor<24x48x1xf32>
    %6 = hivm.hir.vbrc ins(%expanded_4 : tensor<24x48x1xf32>) outs(%0 : tensor<24x48x48xf32>) broadcast_dims = [2] -> tensor<24x48x48xf32>
    %7 = hivm.hir.vdiv ins(%4, %6 : tensor<24x48x48xf32>, tensor<24x48x48xf32>) outs(%arg3 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
    %8 = hivm.hir.vcast ins(%7 : tensor<24x48x48xf32>) outs(%arg4 : tensor<24x48x48xbf16>) round_mode = <rint> -> tensor<24x48x48xbf16>
    return %expanded_3, %7, %8 : tensor<24x48x1xf32>, tensor<24x48x48xf32>, tensor<24x48x48xbf16>
  }
}


// -----
// CHECK: Valid
// CHECK-LABEL: func.func @collapse_down_out(
// CHECK: return
module {
  func.func @collapse_down_out(%arg0: tensor<24x10x48xf32>, %arg1: tensor<3x1x8x48x2x5x48x4x5xf32>, %arg2: tensor<24x48x10x48x20xf32>, %arg3: tensor<24x69x48x10x48x69x20xf32>) -> (tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2], [3], [4, 5], [6], [7, 8]] : tensor<3x1x8x48x2x5x48x4x5xf32> into tensor<24x48x10x48x20xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3, 4]] output_shape [24, 1, 10, 48, 1] : tensor<24x10x48xf32> into tensor<24x1x10x48x1xf32>
    %0 = hivm.hir.vbrc ins(%expanded : tensor<24x1x10x48x1xf32>) outs(%collapsed : tensor<24x48x10x48x20xf32>) broadcast_dims = [1, 4] -> tensor<24x48x10x48x20xf32>
    %1 = hivm.hir.vexp ins(%collapsed : tensor<24x48x10x48x20xf32>) outs(%collapsed : tensor<24x48x10x48x20xf32>) -> tensor<24x48x10x48x20xf32>
    %expanded_0 = tensor.expand_shape %collapsed [[0, 1], [2], [3], [4, 5], [6]] output_shape [24, 1, 48, 10, 48, 1, 20] : tensor<24x48x10x48x20xf32> into tensor<24x1x48x10x48x1x20xf32>
    %2 = hivm.hir.vreduce <sum> ins(%arg3 : tensor<24x69x48x10x48x69x20xf32>) outs(%expanded_0 : tensor<24x1x48x10x48x1x20xf32>) reduce_dims = [1, 5] -> tensor<24x1x48x10x48x1x20xf32>
    %collapsed_1 = tensor.collapse_shape %2 [[0, 1], [2], [3], [4, 5], [6]] : tensor<24x1x48x10x48x1x20xf32> into tensor<24x48x10x48x20xf32>
    return %0, %collapsed, %1, %collapsed_1 : tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>, tensor<24x48x10x48x20xf32>
  }
}


// -----

// CHECK: Valid
// CHECK-LABEL: @expand_up_out
// CHECK: return
module {
  func.func @expand_up_out(%arg0: tensor<24x10x48xf32>, %arg1: tensor<3x1x8x48x2x5x48x4x5xf32>, %arg2: tensor<24x48x10x48x20xf32>, %arg3: tensor<24x69x48x10x48x69x20xf32>) -> (tensor<24x4x1x3x4x10x48x2x5x2xf32>, tensor<24x4x1x3x4x10x48x2x5x2xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3], [4, 5, 6]] output_shape [24, 1, 1, 10, 48, 1, 1] : tensor<24x10x48xf32> into tensor<24x1x1x10x48x1x1xf32>
    %0 = hivm.hir.vbrc ins(%expanded : tensor<24x1x1x10x48x1x1xf32>) outs(%arg3 : tensor<24x69x48x10x48x69x20xf32>) broadcast_dims = [1, 2, 5, 6] -> tensor<24x69x48x10x48x69x20xf32>
    %1 = hivm.hir.vexp ins(%0 : tensor<24x69x48x10x48x69x20xf32>) outs(%arg3 : tensor<24x69x48x10x48x69x20xf32>) -> tensor<24x69x48x10x48x69x20xf32>
    %expanded_0 = tensor.expand_shape %arg2 [[0, 1], [2], [3], [4, 5], [6]] output_shape [24, 1, 48, 10, 48, 1, 20] : tensor<24x48x10x48x20xf32> into tensor<24x1x48x10x48x1x20xf32>
    %2 = hivm.hir.vreduce <sum> ins(%1 : tensor<24x69x48x10x48x69x20xf32>) outs(%expanded_0 : tensor<24x1x48x10x48x1x20xf32>) reduce_dims = [1, 5] -> tensor<24x1x48x10x48x1x20xf32>
    %collapsed = tensor.collapse_shape %2 [[0, 1], [2], [3], [4, 5], [6]] : tensor<24x1x48x10x48x1x20xf32> into tensor<24x48x10x48x20xf32>
    %expanded_1 = tensor.expand_shape %arg2 [[0], [1, 2, 3, 4], [5], [6], [7, 8, 9]] output_shape [24, 4, 1, 3, 4, 10, 48, 2, 5, 2] : tensor<24x48x10x48x20xf32> into tensor<24x4x1x3x4x10x48x2x5x2xf32>
    %expanded_2 = tensor.expand_shape %collapsed [[0], [1, 2, 3, 4], [5], [6], [7, 8, 9]] output_shape [24, 4, 1, 3, 4, 10, 48, 2, 5, 2] : tensor<24x48x10x48x20xf32> into tensor<24x4x1x3x4x10x48x2x5x2xf32>
    return %expanded_1, %expanded_2 : tensor<24x4x1x3x4x10x48x2x5x2xf32>, tensor<24x4x1x3x4x10x48x2x5x2xf32>
  }
}


// -----

// CHECK: Valid
// CHECK-LABEL: @collapse_broadcast
// CHECK: return
module {
  func.func @collapse_broadcast(%arg0: tensor<24x32x8x768xf32>, %arg1: tensor<24x32x6144x9xf32>, %arg2: tensor<24x32x8x9xf32>) -> tensor<24x32x8x768x9xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<24x32x8x768xf32> into tensor<24x32x6144xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1], [2, 3]] output_shape [24, 32, 6144, 1] : tensor<24x32x6144xf32> into tensor<24x32x6144x1xf32>
    %0 = hivm.hir.vbrc ins(%expanded : tensor<24x32x6144x1xf32>) outs(%arg1 : tensor<24x32x6144x9xf32>) broadcast_dims = [3] -> tensor<24x32x6144x9xf32>
    %expanded_0 = tensor.expand_shape %0 [[0], [1], [2, 3], [4]] output_shape [24, 32, 8, 768, 9] : tensor<24x32x6144x9xf32> into tensor<24x32x8x768x9xf32>
    return %expanded_0 : tensor<24x32x8x768x9xf32>
  }
}



// -----
// TODO: refactor
// CHECK: Valid
// CHECK-LABEL: @model_0_long
// CHECK: return
module {
  func.func @model_0_long(%arg0: tensor<24x128x256x192xbf16>) -> (tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 1.966080e+05 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x128x256x192xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<24x128x256x192xbf16>) outs(%0 : tensor<24x128x256x192xf32>) -> tensor<24x128x256x192xf32>
    %2 = tensor.empty() : tensor<24x128x256x192xf16>
    %3 = hivm.hir.vcast ins(%1 : tensor<24x128x256x192xf32>) outs(%2 : tensor<24x128x256x192xf16>) round_mode = <rint> -> tensor<24x128x256x192xf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<24x128x49152xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 4, 49152] : tensor<24x128x49152xf32> into tensor<24x32x4x49152xf32>
    %4 = tensor.empty() : tensor<24x32xf32>
    %5 = hivm.hir.vbrc ins(%cst : f32) outs(%4 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %expanded_4 = tensor.expand_shape %5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %6 = hivm.hir.vreduce <sum> ins(%expanded : tensor<24x32x4x49152xf32>) outs(%expanded_4 : tensor<24x32x1x1xf32>) reduce_dims = [2, 3] -> tensor<24x32x1x1xf32>
    %collapsed_5 = tensor.collapse_shape %6 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %expanded_6 = tensor.expand_shape %collapsed_5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %7 = tensor.empty() : tensor<24x32x1x1xf32>
    %8 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %9 = hivm.hir.vmul ins(%expanded_6, %8 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %10 = tensor.empty() : tensor<24x32x4x49152xf32>
    %collapsed_7 = tensor.collapse_shape %9 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %expanded_8 = tensor.expand_shape %collapsed_7 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %11 = hivm.hir.vbrc ins(%expanded_8 : tensor<24x32x1x1xf32>) outs(%10 : tensor<24x32x4x49152xf32>) broadcast_dims = [2, 3] -> tensor<24x32x4x49152xf32>
    %12 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %13 = hivm.hir.vmul ins(%11, %12 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %14 = hivm.hir.vsub ins(%expanded, %13 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %15 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %expanded_9 = tensor.expand_shape %5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %17 = hivm.hir.vreduce <sum> ins(%15 : tensor<24x32x4x49152xf32>) outs(%expanded_9 : tensor<24x32x1x1xf32>) reduce_dims = [2, 3] -> tensor<24x32x1x1xf32>
    %collapsed_10 = tensor.collapse_shape %17 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %expanded_11 = tensor.expand_shape %collapsed_10 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %18 = hivm.hir.vmul ins(%expanded_11, %8 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %19 = arith.truncf %cst_0 : f64 to f32
    %20 = hivm.hir.vbrc ins(%19 : f32) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %21 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %22 = hivm.hir.vmul ins(%20, %21 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %23 = hivm.hir.vadd ins(%18, %22 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %24 = hivm.hir.vrsqrt ins(%23 : tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    return %1, %3, %expanded_6, %9, %expanded_11, %24 : tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}


// -----
// CHECK: Valid
// CHECK-LABEL: @propagate_parallel
// CHECK: return
module {
  func.func @propagate_parallel(%arg0: tensor<24x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<24x32xf32>, %arg3: tensor<24x128xf32>, %arg4: tensor<24x32xf32>, %arg5: tensor<24x32xf32>) -> (tensor<24x32xf32>, tensor<24x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x128xf32>
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
    %1 = hivm.hir.vbrc ins(%expanded : tensor<1x128xf32>) outs(%0 : tensor<24x128xf32>) broadcast_dims = [0] -> tensor<24x128xf32>
    %2 = hivm.hir.vmul ins(%arg0, %1 : tensor<24x128xf32>, tensor<24x128xf32>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %expanded_1 = tensor.expand_shape %2 [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xf32> into tensor<24x32x4xf32>
    %3 = tensor.empty() : tensor<24x32xf32>
    %4 = hivm.hir.vbrc ins(%cst : f32) outs(%3 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %expanded_2 = tensor.expand_shape %arg4 [[0], [1, 2]] output_shape [24, 32, 1] : tensor<24x32xf32> into tensor<24x32x1xf32>
    %5 = hivm.hir.vreduce <sum> ins(%expanded_1 : tensor<24x32x4xf32>) outs(%expanded_2 : tensor<24x32x1xf32>) reduce_dims = [2] -> tensor<24x32x1xf32>
    %collapsed = tensor.collapse_shape %5 [[0], [1, 2]] : tensor<24x32x1xf32> into tensor<24x32xf32>
    %6 = hivm.hir.vmul ins(%collapsed, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%3 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %7 = hivm.hir.vmul ins(%arg3, %1 : tensor<24x128xf32>, tensor<24x128xf32>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %expanded_3 = tensor.expand_shape %7 [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xf32> into tensor<24x32x4xf32>
    %expanded_4 = tensor.expand_shape %4 [[0], [1, 2]] output_shape [24, 32, 1] : tensor<24x32xf32> into tensor<24x32x1xf32>
    %8 = hivm.hir.vreduce <sum> ins(%expanded_3 : tensor<24x32x4xf32>) outs(%expanded_4 : tensor<24x32x1xf32>) reduce_dims = [2] -> tensor<24x32x1xf32>
    %collapsed_5 = tensor.collapse_shape %8 [[0], [1, 2]] : tensor<24x32x1xf32> into tensor<24x32xf32>
    %9 = hivm.hir.vmul ins(%collapsed_5, %cst_0 : tensor<24x32xf32>, f32) outs(%3 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %10 = hivm.hir.vsub ins(%6, %9 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%arg5 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %collapsed, %10 : tensor<24x32xf32>, tensor<24x32xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @swap_propagate_special
// CHECK: return
func.func @swap_propagate_special(%11: tensor<24x32x1x1x8x128x96xf32>) -> tensor<24x256x1x1x128x96xf32> {
  %collapsed = tensor.collapse_shape %11 [[0], [1, 2, 3, 4], [5], [6]] : tensor<24x32x1x1x8x128x96xf32> into tensor<24x256x128x96xf32>
  %expanded_9 = tensor.expand_shape %collapsed [[0], [1, 2, 3], [4], [5]] output_shape [24, 256, 1, 1, 128, 96] : tensor<24x256x128x96xf32> into tensor<24x256x1x1x128x96xf32>
  return %expanded_9 : tensor<24x256x1x1x128x96xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @hfusion_cast_end
// CHECK: return
func.func @hfusion_cast_end(%arg0: tensor<24x512x8x6xbf16>, %arg1: tensor<24x512x8x6xbf16>, %arg2: tensor<24x512x1x1xbf16>, %arg3: tensor<24x512x1x1xbf16>, %arg4: tensor<24x512x8x6xf32>) -> (tensor<24x512xf32>, tensor<24x512xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xbf16>, tensor<24x512x1x1xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %cst_1 = arith.constant 1.000000e+00 : bf16
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<24x512x8x6xbf16>
  %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<24x512xbf16>
  %1 = tensor.empty() : tensor<24x512xf32>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<24x512xbf16>) outs(%1 : tensor<24x512xf32>) -> tensor<24x512xf32>
  %3 = tensor.empty() : tensor<24x512x8x6xf32>
  %broadcasted = linalg.broadcast ins(%2 : tensor<24x512xf32>) outs(%3 : tensor<24x512x8x6xf32>) dimensions = [2, 3]
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<24x512x8x6xbf16>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %broadcasted : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %collapsed_3 = tensor.collapse_shape %arg3 [[0], [1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<24x512xbf16>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_3 : tensor<24x512xbf16>) outs(%1 : tensor<24x512xf32>) -> tensor<24x512xf32>
  %broadcasted_4 = linalg.broadcast ins(%6 : tensor<24x512xf32>) outs(%3 : tensor<24x512x8x6xf32>) dimensions = [2, 3]
  %7 = tensor.empty() : tensor<24x512x8x6xi64>
  %8 = linalg.fill ins(%c1_i64 : i64) outs(%7 : tensor<24x512x8x6xi64>) -> tensor<24x512x8x6xi64>
  %9 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%8 : tensor<24x512x8x6xi64>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %broadcasted_4 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %cst_0 : tensor<24x512x8x6xf32>, f32) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%11 : tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%12, %cst : tensor<24x512x8x6xf32>, f32) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %14 = linalg.fill ins(%cst_1 : bf16) outs(%0 : tensor<24x512x8x6xbf16>) -> tensor<24x512x8x6xbf16>
  %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%9, %13 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<24x512x8x6xbf16>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%16, %15 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %17 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%18, %cst : tensor<24x512x8x6xf32>, f32) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%19, %13 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x512x8x6xbf16>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %20 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %broadcasted : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %arg4 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %collapsed_5 = tensor.collapse_shape %24 [[0], [1], [2, 3]] : tensor<24x512x8x6xf32> into tensor<24x512x48xf32>
  %25 = linalg.fill ins(%cst_2 : f32) outs(%1 : tensor<24x512xf32>) -> tensor<24x512xf32>
  %reduced = linalg.reduce ins(%collapsed_5 : tensor<24x512x48xf32>) outs(%25 : tensor<24x512xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %30 = arith.addf %in, %init : f32
      linalg.yield %30 : f32
    }
  %collapsed_6 = tensor.collapse_shape %23 [[0], [1], [2, 3]] : tensor<24x512x8x6xf32> into tensor<24x512x48xf32>
  %reduced_7 = linalg.reduce ins(%collapsed_6 : tensor<24x512x48xf32>) outs(%25 : tensor<24x512xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %30 = arith.addf %in, %init : f32
      linalg.yield %30 : f32
    }
  %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %4 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%3 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
  %reduced_8 = linalg.reduce ins(%26 : tensor<24x512x8x6xf32>) outs(%25 : tensor<24x512xf32>) dimensions = [2, 3]
    (%in: f32, %init: f32) {
      %30 = arith.addf %in, %init : f32
      linalg.yield %30 : f32
    }
  %expanded = tensor.expand_shape %reduced_8 [[0], [1, 2, 3]] output_shape [24, 512, 1, 1] : tensor<24x512xf32> into tensor<24x512x1x1xf32>
  %reduced_9 = linalg.reduce ins(%22 : tensor<24x512x8x6xf32>) outs(%25 : tensor<24x512xf32>) dimensions = [2, 3]
    (%in: f32, %init: f32) {
      %30 = arith.addf %in, %init : f32
      linalg.yield %30 : f32
    }
  %expanded_10 = tensor.expand_shape %reduced_9 [[0], [1, 2, 3]] output_shape [24, 512, 1, 1] : tensor<24x512xf32> into tensor<24x512x1x1xf32>
  %27 = tensor.empty() : tensor<24x512x1x1xbf16>
  %28 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<24x512x1x1xf32>) outs(%27 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
  %29 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_10 : tensor<24x512x1x1xf32>) outs(%27 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
  return %reduced, %reduced_7, %expanded, %expanded_10, %28, %29 : tensor<24x512xf32>, tensor<24x512xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xbf16>, tensor<24x512x1x1xbf16>
}

// -----

// CHECK: Valid
// CHECK-LABEL: propagate_parallel
// CHECK: tensor.extract
// CHECK-SAME: %c12, %c24, %c1
module {
  func.func @propagate_parallel(%arg0: tensor<24x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<24x32xf32>, %arg3: tensor<24x128xf32>, %arg4: tensor<24x32xf32>, %arg5: tensor<24x32xf32>) -> (tensor<24x32xf32>, tensor<24x32xf32>, f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c97 = arith.constant 97 : index
    %c12 = arith.constant 12 : index
    %c5 = arith.constant 5 : index
    %0 = tensor.empty() : tensor<24x128xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<128xf32>) outs(%0 : tensor<24x128xf32>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted : tensor<24x128xf32>, tensor<24x128xf32>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %expanded = tensor.expand_shape %1 [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xf32> into tensor<24x32x4xf32>
    %2 = tensor.empty() : tensor<24x32xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<24x32x4xf32>) outs(%arg4 : tensor<24x32xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg3, %broadcasted : tensor<24x128xf32>, tensor<24x128xf32>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %extracted = tensor.extract %5[%c12, %c97] : tensor<24x128xf32>
    %expanded_1 = tensor.expand_shape %5 [[0], [1, 2]] output_shape [24, 32, 4] : tensor<24x128xf32> into tensor<24x32x4xf32>
    %reduced_2 = linalg.reduce ins(%expanded_1 : tensor<24x32x4xf32>) outs(%3 : tensor<24x32xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced_2, %cst_0 : tensor<24x32xf32>, f32) outs(%2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%4, %6 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%arg5 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %reduced, %7, %extracted : tensor<24x32xf32>, tensor<24x32xf32>, f32
  }
}

// -----
// CHECK-LABEL: model_0_long
// CHECK: tensor.extract
// CHECK-SAME: %c12, %c12, %c2, %c5, %c40, %c0
module {
  func.func @model_0_long(%arg0: tensor<24x128x256x192xbf16>) -> (f32, tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 1.966080e+05 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %c97 = arith.constant 97 : index
    %c12 = arith.constant 12 : index
    %c1000 = arith.constant 1000 : index
    %c2 = arith.constant 2 : index
    %0 = tensor.empty() : tensor<24x128x256x192xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<24x128x256x192xbf16>) outs(%0 : tensor<24x128x256x192xf32>) -> tensor<24x128x256x192xf32>
    %2 = tensor.empty() : tensor<24x128x256x192xf16>
    %3 = hivm.hir.vcast ins(%1 : tensor<24x128x256x192xf32>) outs(%2 : tensor<24x128x256x192xf16>) round_mode = <rint> -> tensor<24x128x256x192xf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<24x128x49152xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 4, 49152] : tensor<24x128x49152xf32> into tensor<24x32x4x49152xf32>
    %4 = tensor.empty() : tensor<24x32xf32>
    %5 = hivm.hir.vbrc ins(%cst : f32) outs(%4 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %expanded_4 = tensor.expand_shape %5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %6 = hivm.hir.vreduce <sum> ins(%expanded : tensor<24x32x4x49152xf32>) outs(%expanded_4 : tensor<24x32x1x1xf32>) reduce_dims = [2, 3] -> tensor<24x32x1x1xf32>
    %collapsed_5 = tensor.collapse_shape %6 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %expanded_6 = tensor.expand_shape %collapsed_5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %7 = tensor.empty() : tensor<24x32x1x1xf32>
    %8 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %9 = hivm.hir.vmul ins(%expanded_6, %8 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %10 = tensor.empty() : tensor<24x32x4x49152xf32>
    %collapsed_7 = tensor.collapse_shape %9 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %expanded_8 = tensor.expand_shape %collapsed_7 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %11 = hivm.hir.vbrc ins(%expanded_8 : tensor<24x32x1x1xf32>) outs(%10 : tensor<24x32x4x49152xf32>) broadcast_dims = [2, 3] -> tensor<24x32x4x49152xf32>
    %12 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %13 = hivm.hir.vmul ins(%11, %12 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %14 = hivm.hir.vsub ins(%expanded, %13 : tensor<24x32x4x49152xf32>, tensor<24x32x4x49152xf32>) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %extracted = tensor.extract %14[%c12, %c12, %c2, %c1000] : tensor<24x32x4x49152xf32>
    %15 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%10 : tensor<24x32x4x49152xf32>) -> tensor<24x32x4x49152xf32>
    %expanded_9 = tensor.expand_shape %5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %17 = hivm.hir.vreduce <sum> ins(%15 : tensor<24x32x4x49152xf32>) outs(%expanded_9 : tensor<24x32x1x1xf32>) reduce_dims = [2, 3] -> tensor<24x32x1x1xf32>
    %collapsed_10 = tensor.collapse_shape %17 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %expanded_11 = tensor.expand_shape %collapsed_10 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %18 = hivm.hir.vmul ins(%expanded_11, %8 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %19 = arith.truncf %cst_0 : f64 to f32
    %20 = hivm.hir.vbrc ins(%19 : f32) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %21 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %22 = hivm.hir.vmul ins(%20, %21 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %23 = hivm.hir.vadd ins(%18, %22 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %24 = hivm.hir.vrsqrt ins(%23 : tensor<24x32x1x1xf32>) outs(%7 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    return %extracted, %1, %3, %expanded_6, %9, %expanded_11, %24 : f32, tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}


// -----
// CHECK: Valid
// CHECK-LABEL: @mlir_fused_add_div_npu_dtype_cast_pow_rsqrt_sub_sum_0
// CHECK: hivm.hir.bitcast
// CHECK-SAME: tensor<24x32x1x1x4x256x192xf32> -> tensor<24x32x1x1x4x256x192xi32>
// CHECK: return
module {
  func.func @mlir_fused_add_div_npu_dtype_cast_pow_rsqrt_sub_sum_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xbf16>, %arg2: tensor<24x128x256x192xbf16>) -> (tensor<24x128x256x192xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -2.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c31_i32 = arith.constant 31 : i32
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_3 = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant 1.000000e-05 : f64
    %cst_5 = arith.constant 1.966080e+05 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %collapsed = tensor.collapse_shape %expanded [[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x4x256x192xbf16> into tensor<768x196608xbf16>
    %expanded_6 = tensor.expand_shape %arg1 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %collapsed_7 = tensor.collapse_shape %expanded_6 [[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x4x256x192xbf16> into tensor<768x196608xbf16>
    %expanded_8 = tensor.expand_shape %arg2 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %collapsed_9 = tensor.collapse_shape %expanded_8 [[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x4x256x192xbf16> into tensor<768x196608xbf16>
    %0 = tensor.empty() : tensor<24x128x256x192xf32>
    %expanded_10 = tensor.expand_shape %0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xf32> into tensor<24x32x1x1x4x256x192xf32>
    %collapsed_11 = tensor.collapse_shape %expanded_10 [[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x4x256x192xf32> into tensor<768x196608xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_7 : tensor<768x196608xbf16>) outs(%collapsed_11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<768x196608xbf16>) outs(%collapsed_11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %1 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_9 : tensor<768x196608xbf16>) outs(%collapsed_11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %4 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %collapsed_12 = tensor.collapse_shape %5 [[0, 1]] : tensor<768x196608xf32> into tensor<150994944xf32>
    %6 = tensor.empty() : tensor<24x32xf32>
    %collapsed_13 = tensor.collapse_shape %6 [[0, 1]] : tensor<24x32xf32> into tensor<768xf32>
    %7 = linalg.fill ins(%cst_3 : f32) outs(%collapsed_13 : tensor<768xf32>) -> tensor<768xf32>
    %reduced = linalg.reduce ins(%5 : tensor<768x196608xf32>) outs(%7 : tensor<768xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %43 = arith.addf %in, %init : f32
        linalg.yield %43 : f32
      }
    %8 = tensor.empty() : tensor<24x32x1x1xf32>
    %collapsed_14 = tensor.collapse_shape %8 [[0, 1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<768xf32>
    %9 = linalg.fill ins(%cst_5 : f32) outs(%collapsed_14 : tensor<768xf32>) -> tensor<768xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %9 : tensor<768xf32>, tensor<768xf32>) outs(%collapsed_14 : tensor<768xf32>) -> tensor<768xf32>
    %11 = tensor.empty() : tensor<24x32x4x49152xf32>
    %collapsed_15 = tensor.collapse_shape %11 [[0, 1], [2, 3]] : tensor<24x32x4x49152xf32> into tensor<768x196608xf32>
    %broadcasted = linalg.broadcast ins(%10 : tensor<768xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) dimensions = [1]
    %12 = tensor.empty() : tensor<24x32x4x49152xi64>
    %collapsed_16 = tensor.collapse_shape %12 [[0, 1], [2, 3]] : tensor<24x32x4x49152xi64> into tensor<768x196608xi64>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%5, %broadcasted : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %collapsed_17 = tensor.collapse_shape %13 [[0, 1]] : tensor<768x196608xf32> into tensor<150994944xf32>
    %14 = linalg.fill ins(%c2_i64 : i64) outs(%collapsed_16 : tensor<768x196608xi64>) -> tensor<768x196608xi64>
    %15 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%14 : tensor<768x196608xi64>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %16 = hivm.hir.bitcast %collapsed_17 : tensor<150994944xf32> -> tensor<150994944xi32>
    %expanded_18 = tensor.expand_shape %16 [[0, 1]] output_shape [768, 196608] : tensor<150994944xi32> into tensor<768x196608xi32>
    %17 = tensor.empty() : tensor<24x32x4x49152xi32>
    %collapsed_19 = tensor.collapse_shape %17 [[0, 1], [2, 3]] : tensor<24x32x4x49152xi32> into tensor<768x196608xi32>
    %18 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%expanded_18, %c31_i32 : tensor<768x196608xi32>, i32) outs(%collapsed_19 : tensor<768x196608xi32>) -> tensor<768x196608xi32>
    %19 = tensor.empty() : tensor<24x32x4x49152xi1>
    %collapsed_20 = tensor.collapse_shape %19 [[0, 1], [2, 3]] : tensor<24x32x4x49152xi1> into tensor<768x196608xi1>
    %20 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%18 : tensor<768x196608xi32>) outs(%collapsed_16 : tensor<768x196608xi64>) -> tensor<768x196608xi64>
    %21 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%20, %c1_i64 : tensor<768x196608xi64>, i64) outs(%collapsed_20 : tensor<768x196608xi1>) -> tensor<768x196608xi1>
    %22 = hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%15 : tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %23 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%22, %15 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_20 : tensor<768x196608xi1>) -> tensor<768x196608xi1>
    %24 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%21, %23 : tensor<768x196608xi1>, tensor<768x196608xi1>) outs(%collapsed_20 : tensor<768x196608xi1>) -> tensor<768x196608xi1>
    %25 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%15 : tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %cst : tensor<768x196608xf32>, f32) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %27 = hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%26 : tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%27, %cst_1 : tensor<768x196608xf32>, f32) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%25, %28 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%29, %cst_0 : tensor<768x196608xf32>, f32) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%30, %cst_2 : tensor<768x196608xf32>, f32) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %32 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%13 : tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %33 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%32 : tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%33, %cst_1 : tensor<768x196608xf32>, f32) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %35 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%34 : tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%35, %31 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %37 = hfusion.select ins(%24, %36, %35 : tensor<768x196608xi1>, tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%collapsed_15 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
    %reduced_21 = linalg.reduce ins(%37 : tensor<768x196608xf32>) outs(%7 : tensor<768xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %43 = arith.addf %in, %init : f32
        linalg.yield %43 : f32
      }
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_21, %9 : tensor<768xf32>, tensor<768xf32>) outs(%collapsed_14 : tensor<768xf32>) -> tensor<768xf32>
    %39 = arith.truncf %cst_4 : f64 to f32
    %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%38, %39 : tensor<768xf32>, f32) outs(%collapsed_14 : tensor<768xf32>) -> tensor<768xf32>
    %41 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%40 : tensor<768xf32>) outs(%collapsed_14 : tensor<768xf32>) -> tensor<768xf32>
    %42 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%41 : tensor<768xf32>) outs(%collapsed_14 : tensor<768xf32>) -> tensor<768xf32>
    %expanded_22 = tensor.expand_shape %collapsed_12 [[0, 1, 2, 3]] output_shape [24, 128, 256, 192] : tensor<150994944xf32> into tensor<24x128x256x192xf32>
    %expanded_23 = tensor.expand_shape %reduced [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
    %expanded_24 = tensor.expand_shape %10 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
    %expanded_25 = tensor.expand_shape %reduced_21 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
    %expanded_26 = tensor.expand_shape %42 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
    return %expanded_22, %expanded_23, %expanded_24, %expanded_25, %expanded_26 : tensor<24x128x256x192xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}
// -----

// CHECK: Valid
// CHECK-LABEL: concat_pad_expand
// CHECK: tensor.pad
// CHECK-SAME: 0, 0, 0, 0, 0
// CHECK-SAME: 0, 0, 0, 3, 0
// CHECK: tensor.concat
// CHECK-SAME: 3x8x1x3x1xf32
// CHECK-SAME: 3x8x1x5x1xf32
// CHECK-SAME: 3x8x1x1x1xf32
// CHECK-SAME: 3x8x1x9x1xf32

module {
  func.func @concat_pad_expand(%arg0: tensor<24x3xf32>, %arg1: tensor<24x2xf32>, %arg2: tensor<24x1xf32>) -> tensor<3x8x1x9x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %out0 = tensor.empty() : tensor<24x3xf32>
    %out1 = tensor.empty() : tensor<24x2xf32>

    %a = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%arg0 : tensor<24x3xf32>)
      outs(%out0 : tensor<24x3xf32>) -> tensor<24x3xf32>

    %b = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%arg1 : tensor<24x2xf32>)
      outs(%out1 : tensor<24x2xf32>) -> tensor<24x2xf32>

    %pad = arith.constant 0.000000e+00 : f32

    %yaypad = tensor.pad %b low[0, 0] high[0, 3] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %pad : f32
    } : tensor<24x2xf32> to tensor<24x5xf32>
    %concatted = tensor.concat dim(1) %a, %yaypad, %arg2 : (tensor<24x3xf32>, tensor<24x5xf32>, tensor<24x1xf32>) -> tensor<24x9xf32>

    %expanded = tensor.expand_shape %concatted [[0, 1],[2, 3, 4]] output_shape [3, 8, 1, 9, 1] : tensor<24x9xf32> into tensor<3x8x1x9x1xf32>

    %out2 = tensor.empty() : tensor<3x8x1x9x1xf32>
    %result = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%expanded : tensor<3x8x1x9x1xf32>)
      outs(%out2 : tensor<3x8x1x9x1xf32>) -> tensor<3x8x1x9x1xf32>
    %result2 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%result : tensor<3x8x1x9x1xf32>)
      outs(%out2 : tensor<3x8x1x9x1xf32>) -> tensor<3x8x1x9x1xf32>
    return %result2 : tensor<3x8x1x9x1xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: concat_pad_collapse
// CHECK: tensor.pad
// CHECK-SAME: 0, 0, 0, 0, 0
// CHECK-SAME: 0, 0, 0, 3, 0
// CHECK: tensor.concat
// CHECK-SAME: 3x8x1x3x1xf32
// CHECK-SAME: 3x8x1x5x1xf32
// CHECK-SAME: 3x8x1x1x1xf32
// CHECK-SAME: 3x8x1x9x1xf32

module {
  func.func @concat_pad_collapse(%arg0: tensor<24x3xf32>, %arg1: tensor<3x8x1x2x1xf32>, %arg2: tensor<24x1xf32>) -> tensor<24x9xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %out0 = tensor.empty() : tensor<24x3xf32>
    %out1 = tensor.empty() : tensor<3x8x1x2x1xf32>

    %a = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%arg0 : tensor<24x3xf32>)
      outs(%out0 : tensor<24x3xf32>) -> tensor<24x3xf32>

    %b = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%arg1 : tensor<3x8x1x2x1xf32>)
      outs(%out1 : tensor<3x8x1x2x1xf32>) -> tensor<3x8x1x2x1xf32>

    %pad = arith.constant 0.000000e+00 : f32

    %collapsed = tensor.collapse_shape %b [[0, 1], [2, 3, 4]] : tensor<3x8x1x2x1xf32> into tensor<24x2xf32>
    %yaypad = tensor.pad %collapsed low[0, 0] high[0, 3] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %pad : f32
    } : tensor<24x2xf32> to tensor<24x5xf32>
    %concatted = tensor.concat dim(1) %a, %yaypad, %arg2 : (tensor<24x3xf32>, tensor<24x5xf32>, tensor<24x1xf32>) -> tensor<24x9xf32>

    %out2 = tensor.empty() : tensor<24x9xf32>
    %result = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%concatted : tensor<24x9xf32>)
      outs(%out2 : tensor<24x9xf32>) -> tensor<24x9xf32>
    %result2 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%result : tensor<24x9xf32>)
      outs(%out2 : tensor<24x9xf32>) -> tensor<24x9xf32>
    return %result2 : tensor<24x9xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_collapsedown_empty_reassoc(
func.func @main_collapsedown_empty_reassoc(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<?x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1xf32> into tensor<f32>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%2 : tensor<?xf32>) dimensions = [0]
  %expanded = tensor.expand_shape %broadcasted [[0, 1]] output_shape [%dim, 1] : tensor<?xf32> into tensor<?x1xf32>
  return %expanded : tensor<?x1xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_collapsedown_empty_reassoc_1(
func.func @main_collapsedown_empty_reassoc_1(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1xf32> into tensor<f32>
  %2 = tensor.empty() : tensor<f32>
  %broadcasted = linalg.elemwise_unary ins(%collapsed : tensor<f32>) outs(%2 : tensor<f32>) -> tensor<f32>
  return %broadcasted : tensor<f32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_collapsedown_empty_reassoc_2(
func.func @main_collapsedown_empty_reassoc_2(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1xf32> into tensor<f32>

  %2 = tensor.empty(%dim) : tensor<?xf32>
  %reduced = linalg.reduce ins(%2 : tensor<?xf32>) outs(%collapsed : tensor<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  return %reduced : tensor<f32>
}

// -----

// CHECK: Valid
// CHECK-LABEL: @main_expandup_reduce(
func.func @main_expandup_reduce(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<1xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>

  %2 = tensor.empty(%dim) : tensor<?xf32>
  %emptyres = tensor.empty() : tensor<f32>
  %reduced = linalg.reduce ins(%2 : tensor<?xf32>) outs(%emptyres : tensor<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [] output_shape [1] : tensor<f32> into tensor<1xf32>
  %lmao = tensor.empty() : tensor<1xi32>
  %castfence = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<1xf32>) outs(%lmao : tensor<1xi32>) -> tensor<1xi32>

  return %castfence : tensor<1xi32>
}

// -----

// CHECK: Valid
// CHECK-LABEL: @main_expandup_unary(
func.func @main_expandup_unary(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<1xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>

  %2 = tensor.empty() : tensor<f32>
  %emptyres = tensor.empty() : tensor<f32>
  %unary = linalg.elemwise_unary ins(%2 : tensor<f32>) outs(%emptyres : tensor<f32>) -> tensor<f32>
  %expanded = tensor.expand_shape %unary [] output_shape [1] : tensor<f32> into tensor<1xf32>
  %lmao = tensor.empty() : tensor<1xi32>
  %castfence = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<1xf32>) outs(%lmao : tensor<1xi32>) -> tensor<1xi32>

  return %castfence : tensor<1xi32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_expandup_reduce_multi(
// CHECK: tensor.expand_shape
// CHECK: %{{.*}} {{\[}}{{\[}}0, 1], [2]] output_shape [1, %{{.*}}, 3]
// CHECK: reduce
// CHECK-SAME: dimensions = [1, 2]
func.func @main_expandup_reduce_multi(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<1xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>

  %2 = tensor.empty(%dim) : tensor<?x3xf32>
  %emptyres = tensor.empty() : tensor<f32>
  %reduced = linalg.reduce ins(%2 : tensor<?x3xf32>) outs(%emptyres : tensor<f32>) dimensions = [0, 1]
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [] output_shape [1] : tensor<f32> into tensor<1xf32>
  %lmao = tensor.empty() : tensor<1xi32>
  %castfence = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<1xf32>) outs(%lmao : tensor<1xi32>) -> tensor<1xi32>

  return %castfence : tensor<1xi32>
}


// -----
// CHECK: Valid
// CHECK-LABEL: @main_expandup_reduce_multi(
// CHECK: tensor.expand_shape
// CHECK: {{\[}}{{\[}}0, 1, 2, 3], [4]] output_shape [1, 1, 1, %{{.*}}, 3]
// CHECK: reduce
// CHECK-SAME: dimensions = [3, 4]
func.func @main_expandup_reduce_multi(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<1x1x1xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty(%dim) : tensor<?x3xf32>
  %1 = tensor.empty() : tensor<f32>
  %reduced = linalg.reduce ins(%0 : tensor<?x3xf32>) outs(%1 : tensor<f32>) dimensions = [0, 1]
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [] output_shape [1, 1, 1] : tensor<f32> into tensor<1x1x1xf32>
  %2 = tensor.empty() : tensor<1x1x1xi32>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<1x1x1xf32>) outs(%2 : tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
  return %3 : tensor<1x1x1xi32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: concat_multiple_main
// CHECK: tensor.concat
// CHECK-SAME: tensor<24x16x1x1x32x8x6xbf16>
// CHECK-SAME: tensor<24x16x1x1x32x8x6xbf16>
// CHECK-SAME: tensor<24x32x1x1x32x8x6xbf16>
// CHECK: return
module {
  func.func @concat_multiple_main(%arg0: tensor<24x512x8x6xbf16>, %arg1: tensor<24x512x8x6xbf16>, %arg2: tensor<24x512x8x6xf32>, %arg3: tensor<24x512x8x6xbf16>, %arg4: tensor<24x512x8x6xbf16>) -> (tensor<24x1024x8x6xbf16>, tensor<24x1024x8x6xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 1.536000e+03 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x512x8x6xbf16>
    %1 = tensor.empty() : tensor<24x512x8x6xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %2 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<24x512x8x6xf32>) outs(%0 : tensor<24x512x8x6xbf16>) -> tensor<24x512x8x6xbf16>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg3 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %6 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg4 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%7, %8 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<24x512x8x6xf32>) outs(%0 : tensor<24x512x8x6xbf16>) -> tensor<24x512x8x6xbf16>
    %concat = tensor.concat dim(1) %5, %10 : (tensor<24x512x8x6xbf16>, tensor<24x512x8x6xbf16>) -> tensor<24x1024x8x6xbf16>
    %11 = tensor.empty() : tensor<24x1024x8x6xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%concat : tensor<24x1024x8x6xbf16>) outs(%11 : tensor<24x1024x8x6xf32>) -> tensor<24x1024x8x6xf32>
    %collapsed = tensor.collapse_shape %12 [[0], [1], [2, 3]] : tensor<24x1024x8x6xf32> into tensor<24x1024x48xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 32, 48] : tensor<24x1024x48xf32> into tensor<24x32x32x48xf32>
    %13 = tensor.empty() : tensor<24x32xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<24x32x32x48xf32>) outs(%14 : tensor<24x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %26 = arith.addf %in, %init : f32
        linalg.yield %26 : f32
      }
    %expanded_3 = tensor.expand_shape %reduced [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %15 = tensor.empty() : tensor<24x32x1x1xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_3, %16 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %18 = tensor.empty() : tensor<24x32x32x48xf32>
    %collapsed_4 = tensor.collapse_shape %17 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed_4 : tensor<24x32xf32>) outs(%18 : tensor<24x32x32x48xf32>) dimensions = [2, 3]
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%expanded, %broadcasted : tensor<24x32x32x48xf32>, tensor<24x32x32x48xf32>) outs(%18 : tensor<24x32x32x48xf32>) -> tensor<24x32x32x48xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%19, %19 : tensor<24x32x32x48xf32>, tensor<24x32x32x48xf32>) outs(%18 : tensor<24x32x32x48xf32>) -> tensor<24x32x32x48xf32>
    %reduced_5 = linalg.reduce ins(%20 : tensor<24x32x32x48xf32>) outs(%14 : tensor<24x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %26 = arith.addf %in, %init : f32
        linalg.yield %26 : f32
      }
    %expanded_6 = tensor.expand_shape %reduced_5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_6, %16 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %22 = arith.truncf %cst_0 : f64 to f32
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %22 : tensor<24x32x1x1xf32>, f32) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %24 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%23 : tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %25 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%24 : tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    return %concat, %12, %expanded_3, %17, %expanded_6, %25 : tensor<24x1024x8x6xbf16>, tensor<24x1024x8x6xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: expand_multiple_concat
// CHECK: return
module {
  func.func @expand_multiple_concat(%arg0: tensor<24x512x8x6xbf16>, %arg1: tensor<24x512x8x6xbf16>, %arg2: tensor<24x512x8x6xf32>, %arg3: tensor<24x512x8x6xbf16>, %arg4: tensor<24x512x8x6xbf16>) -> (tensor<24x1024x8x6xbf16>, tensor<24x1024x8x6xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 1.536000e+03 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x512x8x6xbf16>
    %1 = tensor.empty() : tensor<24x512x8x6xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %2 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<24x512x8x6xf32>) outs(%0 : tensor<24x512x8x6xbf16>) -> tensor<24x512x8x6xbf16>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg3 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %6 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg4 : tensor<24x512x8x6xbf16>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%7, %8 : tensor<24x512x8x6xf32>, tensor<24x512x8x6xf32>) outs(%1 : tensor<24x512x8x6xf32>) -> tensor<24x512x8x6xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<24x512x8x6xf32>) outs(%0 : tensor<24x512x8x6xbf16>) -> tensor<24x512x8x6xbf16>
    %concat = tensor.concat dim(1) %5, %10 : (tensor<24x512x8x6xbf16>, tensor<24x512x8x6xbf16>) -> tensor<24x1024x8x6xbf16>
    %11 = tensor.empty() : tensor<24x1024x8x6xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%concat : tensor<24x1024x8x6xbf16>) outs(%11 : tensor<24x1024x8x6xf32>) -> tensor<24x1024x8x6xf32>
    %collapsed = tensor.collapse_shape %12 [[0], [1], [2, 3]] : tensor<24x1024x8x6xf32> into tensor<24x1024x48xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [24, 32, 32, 48] : tensor<24x1024x48xf32> into tensor<24x32x32x48xf32>
    %13 = tensor.empty() : tensor<24x32xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<24x32x32x48xf32>) outs(%14 : tensor<24x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %26 = arith.addf %in, %init : f32
        linalg.yield %26 : f32
      }
    %expanded_3 = tensor.expand_shape %reduced [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %15 = tensor.empty() : tensor<24x32x1x1xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_3, %16 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %18 = tensor.empty() : tensor<24x32x32x48xf32>
    %collapsed_4 = tensor.collapse_shape %17 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed_4 : tensor<24x32xf32>) outs(%18 : tensor<24x32x32x48xf32>) dimensions = [2, 3]
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%expanded, %broadcasted : tensor<24x32x32x48xf32>, tensor<24x32x32x48xf32>) outs(%18 : tensor<24x32x32x48xf32>) -> tensor<24x32x32x48xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%19, %19 : tensor<24x32x32x48xf32>, tensor<24x32x32x48xf32>) outs(%18 : tensor<24x32x32x48xf32>) -> tensor<24x32x32x48xf32>
    %reduced_5 = linalg.reduce ins(%20 : tensor<24x32x32x48xf32>) outs(%14 : tensor<24x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %26 = arith.addf %in, %init : f32
        linalg.yield %26 : f32
      }
    %expanded_6 = tensor.expand_shape %reduced_5 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_6, %16 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %22 = arith.truncf %cst_0 : f64 to f32
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %22 : tensor<24x32x1x1xf32>, f32) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %24 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%23 : tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %25 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%24 : tensor<24x32x1x1xf32>) outs(%15 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    return %concat, %12, %expanded_3, %17, %expanded_6, %25 : tensor<24x1024x8x6xbf16>, tensor<24x1024x8x6xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @hivm_extract_slice(
// CHECK: return
module {
  func.func @hivm_extract_slice(%arg0: tensor<24x1024x8x6xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}) -> tensor<24x32x16x48xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %cst = arith.constant 1.000000e+00 : bf16
    hivm.hir.set_mask_norm
    %0 = tensor.empty() : tensor<24x32x16x48xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<24x1024x8x6xbf16> into tensor<24x1024x48xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0] [24, 512, 48] [1, 1, 1] : tensor<24x1024x48xbf16> to tensor<24x512x48xbf16>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2], [3]] output_shape [24, 32, 16, 48] : tensor<24x512x48xbf16> into tensor<24x32x16x48xbf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%expanded : tensor<24x32x16x48xbf16>) outs(%0 : tensor<24x32x16x48xbf16>) -> tensor<24x32x16x48xbf16>
    %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%1 : tensor<24x32x16x48xbf16>) outs(%0 : tensor<24x32x16x48xbf16>) -> tensor<24x32x16x48xbf16>
    return %2 : tensor<24x32x16x48xbf16>
  }
}


// -----
// CHECK: Valid
// CHECK-LABEL: @main_expandup_fill(
func.func @main_expandup_fill(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<1xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %cst = arith.constant 2.13909504E+9 : f32

  %2 = tensor.empty() : tensor<f32>
  %emptyres = tensor.empty() : tensor<f32>
  %unary = linalg.fill ins(%cst : f32) outs(%emptyres : tensor<f32>) -> tensor<f32>
  %expanded = tensor.expand_shape %unary [] output_shape [1] : tensor<f32> into tensor<1xf32>
  %lmao = tensor.empty() : tensor<1xi32>
  %castfence = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<1xf32>) outs(%lmao : tensor<1xi32>) -> tensor<1xi32>

  return %castfence : tensor<1xi32>
}

// -----
// cse and canonicalize will cause this to be failed
// CHECK: Failed
// CHECK-LABEL: @main_collapsedown_fill(
func.func @main_collapsedown_fill(%arg0: tensor<?x4096xf16>, %arg1: tensor<1xi64>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi64>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %collapsed = tensor.collapse_shape %1 [] : tensor<1xf32> into tensor<f32>
  %2 = tensor.empty() : tensor<f32>
  %cst = arith.constant 2.13909504E+9 : f32
  %broadcasted = linalg.fill ins(%cst : f32) outs(%collapsed : tensor<f32>) -> tensor<f32>
  return %broadcasted : tensor<f32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @hfusion_reduce_with_index_expand
func.func @hfusion_reduce_with_index_expand(%input: tensor<6x7x8xf32>) -> tensor<6x2x4xf32> {
  %unary_output = tensor.empty() : tensor<6x7x8xf32>
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
    ins(%input : tensor<6x7x8xf32>) outs(%unary_output : tensor<6x7x8xf32>)
    -> tensor<6x7x8xf32>
  %output = tensor.empty() : tensor<6x8xf32>
  %output_index = tensor.empty() : tensor<6x8xi32>
  %reduce_result:2 = hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%unary : tensor<6x7x8xf32>) outs(%output, %output_index : tensor<6x8xf32>, tensor<6x8xi32>)
    dimensions = [1]
    -> tensor<6x8xf32>, tensor<6x8xi32>
  %expanded = tensor.expand_shape %reduce_result#0 [[0], [1, 2]] output_shape [6, 2, 4] : tensor<6x8xf32> into tensor<6x2x4xf32>
  %new_unary_output = tensor.empty() : tensor<6x2x4xf32>
  %new_unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
    ins(%expanded : tensor<6x2x4xf32>) outs(%new_unary_output : tensor<6x2x4xf32>)
    -> tensor<6x2x4xf32>
  return %new_unary : tensor<6x2x4xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @hfusion_reduce_with_index_collapse
func.func @hfusion_reduce_with_index_collapse(%input: tensor<2x3x7x8xf32>) -> tensor<6x8xf32> {
  %unary_output = tensor.empty() : tensor<2x3x7x8xf32>
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
    ins(%input : tensor<2x3x7x8xf32>) outs(%unary_output : tensor<2x3x7x8xf32>)
    -> tensor<2x3x7x8xf32>
  %collapsed = tensor.collapse_shape %unary [[0, 1], [2], [3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
  %output = tensor.empty() : tensor<6x8xf32>
  %output_index = tensor.empty() : tensor<6x8xi32>
  %reduce_result:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
    ins(%collapsed : tensor<6x7x8xf32>) outs(%output, %output_index : tensor<6x8xf32>, tensor<6x8xi32>)
    dimensions = [1]
    -> tensor<6x8xf32>, tensor<6x8xi32>
  %new_unary_output = tensor.empty() : tensor<6x8xf32>
  %new_unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
    ins(%reduce_result#0 : tensor<6x8xf32>) outs(%new_unary_output : tensor<6x8xf32>)
    -> tensor<6x8xf32>
  return %new_unary : tensor<6x8xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: main_expandup_transpose
// CHECK: expand_shape
// CHECK-SAME: tensor<6x7xf32>
// CHECK-SAME: tensor<2x3x7xf32>
// CHECK: expand_shape
// CHECK-SAME: tensor<7x6xf32>
// CHECK-SAME: tensor<7x2x3xf32>
// CHECK: linalg.transpose
// CHECK-SAME: 2, 0, 1
// CHECK: return
func.func @main_expandup_transpose(%arg1: tensor<6x7xf32>) -> tensor<7x2x3xi32> {
  %emptyres = tensor.empty() : tensor<7x6xf32>
  %transposed = linalg.transpose ins(%arg1 : tensor<6x7xf32>) outs(%emptyres : tensor<7x6xf32>) permutation = [1, 0]
  %expanded = tensor.expand_shape %transposed [[0], [1, 2]] output_shape [7, 2, 3] : tensor<7x6xf32> into tensor<7x2x3xf32>
  %lmao = tensor.empty() : tensor<7x2x3xi32>
  %castfence = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<7x2x3xf32>) outs(%lmao : tensor<7x2x3xi32>) -> tensor<7x2x3xi32>
  %lnao = tensor.empty() : tensor<7x2x3xi32>
  %castfence_1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<7x2x3xf32>) outs(%lnao : tensor<7x2x3xi32>) -> tensor<7x2x3xi32>
  %loao = tensor.empty() : tensor<7x2x3xi32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%castfence, %castfence_1 : tensor<7x2x3xi32>, tensor<7x2x3xi32>) outs(%loao : tensor<7x2x3xi32>) -> tensor<7x2x3xi32>
  return %result : tensor<7x2x3xi32>
}


// -----
// CHECK: Failed
// CHECK-LABEL: main_expandup_transpose
// FUTURE-CHECK: expand_shape
// FUTURE-CHECK-SAME: tensor<6x7xf32>
// FUTURE-CHECK-SAME: tensor<2x3x7xf32>
// FUTURE-CHECK: expand_shape
// FUTURE-CHECK-SAME: tensor<7x6xf32>
// FUTURE-CHECK-SAME: tensor<7x2x3xf32>
// FUTURE-CHECK: hivm.hir.vtranspose
// FUTURE-CHECK-SAME: 2, 0, 1
// CHECK: return
func.func @main_expandup_transpose(%arg1: tensor<6x7xf32>) -> tensor<7x2x3xi32> {
  %emptyres = tensor.empty() : tensor<7x6xf32>
  %transposed = hivm.hir.vtranspose ins(%arg1 : tensor<6x7xf32>) outs(%emptyres : tensor<7x6xf32>) permutation = [1, 0] -> tensor<7x6xf32>
  %expanded = tensor.expand_shape %transposed [[0], [1, 2]] output_shape [7, 2, 3] : tensor<7x6xf32> into tensor<7x2x3xf32>
  %lmao = tensor.empty() : tensor<7x2x3xi32>
  %castfence = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<7x2x3xf32>) outs(%lmao : tensor<7x2x3xi32>) -> tensor<7x2x3xi32>
  %lnao = tensor.empty() : tensor<7x2x3xi32>
  %castfence_1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<7x2x3xf32>) outs(%lnao : tensor<7x2x3xi32>) -> tensor<7x2x3xi32>
  %loao = tensor.empty() : tensor<7x2x3xi32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%castfence, %castfence_1 : tensor<7x2x3xi32>, tensor<7x2x3xi32>) outs(%loao : tensor<7x2x3xi32>) -> tensor<7x2x3xi32>
  return %result : tensor<7x2x3xi32>
}


// -----
// CHECK: Valid
// CHECK-LABEL: collapse_transpose
// CHECK: linalg.transpose
// CHECK-SAME: 3, 0, 1, 2, 6, 4, 5
// CHECK: collapse_shape
// CHECK-SAME: tensor<5x2x3x4x8x6x7xf32>
// CHECK-SAME: tensor<5x6x4x8x42xf32>
// CHECK: return
func.func @collapse_transpose(%arg0: tensor<2x3x4x5x6x7x8xf32>) -> tensor<5x6x4x8x42xf32> {
  %0 = tensor.empty() : tensor<5x6x4x8x42xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7x8xf32>) outs(%arg0 : tensor<2x3x4x5x6x7x8xf32>) -> tensor<2x3x4x5x6x7x8xf32>
  %collapsed = tensor.collapse_shape %unary [[0, 1], [2], [3], [4, 5], [6]] : tensor<2x3x4x5x6x7x8xf32> into tensor<6x4x5x42x8xf32>
  %transposed = linalg.transpose ins(%collapsed : tensor<6x4x5x42x8xf32>) outs(%0 : tensor<5x6x4x8x42xf32>) permutation = [2, 0, 1, 4, 3] 
  return %transposed : tensor<5x6x4x8x42xf32>
}

// -----
// CHECK: Failed
// CHECK: collapse_transpose
// FUTURE-CHECK-LABEL: collapse_transpose
// FUTURE-CHECK: hivm.hir.vtranspose
// FUTURE-CHECK-SAME: 0, 1, 2, 3, 6, 4, 5
// FUTURE-CHECK: collapse_shape
// FUTURE-CHECK-SAME: tensor<2x3x4x5x8x6x7xf32>
// FUTURE-CHECK-SAME: tensor<6x4x5x8x42xf32>
// CHECK: return
func.func @collapse_transpose(%arg0: tensor<2x3x4x5x6x7x8xf32>) -> tensor<6x4x5x8x42xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7x8xf32>) outs(%arg0 : tensor<2x3x4x5x6x7x8xf32>) -> tensor<2x3x4x5x6x7x8xf32>
  %collapsed = tensor.collapse_shape %unary [[0, 1], [2], [3], [4, 5], [6]] : tensor<2x3x4x5x6x7x8xf32> into tensor<6x4x5x42x8xf32>
  %0 = tensor.empty() : tensor<6x4x5x8x42xf32>
  %transposed = hivm.hir.vtranspose ins(%collapsed : tensor<6x4x5x42x8xf32>) outs(%0 : tensor<6x4x5x8x42xf32>) permutation = [0, 1, 2, 4, 3] -> tensor<6x4x5x8x42xf32>
  return %transposed : tensor<6x4x5x8x42xf32>
}

// -----
// CHECK-LABEL: expandup_arange
// CHECK: hfusion.arange
// CHECK-SAME: tensor<6x48xi32>
func.func @expandup_arange() -> tensor<48x6xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<288xi32>
  %1 = hfusion.arange strides[%c1] outs(%0 : tensor<288xi32>) -> tensor<288xi32>
  %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [6, 48] : tensor<288xi32> into tensor<6x48xi32>
  %2 = tensor.empty() : tensor<48x6xi32>
  %transposed = linalg.transpose ins(%expanded:tensor<6x48xi32>) outs(%2:tensor<48x6xi32>) permutation = [1, 0]
  return %transposed : tensor<48x6xi32>
}

// -----
// CHECK-LABEL: expandup_arange_dyn
// CHECK: hfusion.arange
// CHECK-SAME: tensor<?x?xi32>
func.func @expandup_arange_dyn(%size : index, %size2 : index, %size3 : index) -> tensor<?x?xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %0 = tensor.empty(%size) : tensor<?xi32>
  %1 = hfusion.arange strides[%c1] outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [%size2, %size3] : tensor<?xi32> into tensor<?x?xi32>
  %2 = tensor.empty(%size3, %size2) : tensor<?x?xi32>
  %transpose = linalg.transpose ins(%expanded:tensor<?x?xi32>) outs(%2:tensor<?x?xi32>) permutation = [1, 0]
  return %transpose : tensor<?x?xi32>
}

// -----
// CHECK-LABEL: expandup_arange_multid
// CHECK: hfusion.arange
// CHECK-SAME: tensor<6x24x2xi32>
func.func @expandup_arange_multid() -> tensor<2x24x6xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %c48 = arith.constant 48 : index
  %0 = tensor.empty() : tensor<6x48xi32>
  %1 = hfusion.arange strides[%c48, %c1] outs(%0 : tensor<6x48xi32>) -> tensor<6x48xi32>
  %expanded = tensor.expand_shape %1 [[0], [1, 2]] output_shape [6, 24, 2] : tensor<6x48xi32> into tensor<6x24x2xi32>
  %2 = tensor.empty() : tensor<2x24x6xi32>
  %transpose = linalg.transpose ins(%expanded:tensor<6x24x2xi32>) outs(%2:tensor<2x24x6xi32>) permutation = [2, 1, 0]
  return %transpose : tensor<2x24x6xi32>
}

// -----
// CHECK-LABEL: expandup_arange_multi_expand
// CHECK: hfusion.arange
// CHECK-SAME: tensor<6x24x2xi32>
func.func @expandup_arange_multi_expand() -> tensor<2x24x6xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<288xi32>
  %1 = hfusion.arange strides[%c1] outs(%0 : tensor<288xi32>) -> tensor<288xi32>
  %expanded = tensor.expand_shape %1 [[0, 1, 2]] output_shape [6, 24, 2] : tensor<288xi32> into tensor<6x24x2xi32>
  %2 = tensor.empty() : tensor<2x24x6xi32>
  %transposed = linalg.transpose ins(%expanded : tensor<6x24x2xi32>) outs(%2 : tensor<2x24x6xi32>) permutation = [2, 1, 0]
  return %transposed : tensor<2x24x6xi32>
}

// -----
// CHECK-LABEL: expandup_arange_multid_multi_expand
// CHECK: hfusion.arange
// CHECK-SAME: tensor<2x3x24x2xi32>
func.func @expandup_arange_multid_multi_expand() -> tensor<2x24x3x2xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %c48 = arith.constant 48 : index
  %0 = tensor.empty() : tensor<6x48xi32>
  %1 = hfusion.arange strides[%c48, %c1] outs(%0 : tensor<6x48xi32>) -> tensor<6x48xi32>
  %expanded = tensor.expand_shape %1 [[0, 1], [2, 3]] output_shape [2, 3, 24, 2] : tensor<6x48xi32> into tensor<2x3x24x2xi32>
  %2 = tensor.empty() : tensor<2x24x3x2xi32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x3x24x2xi32>) outs(%2 : tensor<2x24x3x2xi32>) permutation = [3, 2, 1, 0]
  return %transposed : tensor<2x24x3x2xi32>
}

// -----
// CHECK-LABEL: collapse_arange
// CHECK-DAG: %[[C64:[a-zA-Z0-9]+]] = arith.constant 64
// CHECK-DAG: %[[C8:[a-zA-Z0-9]+]] = arith.constant 8
// CHECK-DAG: %[[C1:[a-zA-Z0-9]+]] = arith.constant 1
// CHECK-DAG: %[[C32:[a-zA-Z0-9]+]] = arith.constant 32
// CHECK-DAG: %[[C0:[a-zA-Z0-9]+]] = arith.constant 0
// CHECK: arange offset[%[[C0]]] strides[%[[C64]], %[[C32]], %[[C8]], %[[C1]]
// CHECK-SAME: tensor<2x2x4x8xi32>
// CHECK: collapse_shape
// CHECK-SAME: into tensor<2x8x8xi32>
func.func @collapse_arange(%arg:tensor<2x2x4x8xi32>) -> tensor<2x8x8xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %init = tensor.extract_slice %arg[0,0,0,0][2,2,4,8][1,1,1,1] : tensor<2x2x4x8xi32> to tensor<2x2x4x8xi32>
  %collapsed = tensor.collapse_shape %init [[0], [1, 2], [3]] : tensor<2x2x4x8xi32> into tensor<2x8x8xi32>
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %arange = hfusion.arange strides[%c64, %c8, %c1] outs(%collapsed : tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %arange : tensor<2x8x8xi32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @test_expand_up_interleave
func.func @test_expand_up_interleave(%arg1: tensor<30x14x13xf32>, %arg2: tensor<30x14x13xf32>) -> (tensor<6x5x7x2x13x2xf32>, tensor<30x14x52xf32>) {
  
  %a = hfusion.interleave %arg1, %arg2 : tensor<30x14x13xf32>, tensor<30x14x13xf32> -> tensor<30x14x26xf32>
  %b = tensor.expand_shape %a [[0, 1], [2, 3], [4, 5]] output_shape [6, 5, 7, 2, 13, 2] : tensor<30x14x26xf32> into tensor<6x5x7x2x13x2xf32>
  
  %c = tensor.empty() : tensor<30x14x26xf32>
  %e = hfusion.interleave %a, %c : tensor<30x14x26xf32>, tensor<30x14x26xf32> -> tensor<30x14x52xf32>
  %em = tensor.empty() : tensor<6x5x7x2x13x2xf32>
  %g = linalg.copy ins(%b : tensor<6x5x7x2x13x2xf32>) outs(%em : tensor<6x5x7x2x13x2xf32>) -> tensor<6x5x7x2x13x2xf32>
  %h = linalg.copy ins(%g : tensor<6x5x7x2x13x2xf32>) outs(%em : tensor<6x5x7x2x13x2xf32>) -> tensor<6x5x7x2x13x2xf32>
  return %h, %e : tensor<6x5x7x2x13x2xf32>, tensor<30x14x52xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @test_collapse_down_interleave
func.func @test_collapse_down_interleave(%a: tensor<2x3x5x7x11xf32>, %b: tensor<2x3x5x7x11xf32>) -> (tensor<6x35x11xf32>, tensor<6x35x11xf32>, tensor<6x35x22xf32>) {
  %em0 = tensor.empty() : tensor<2x3x5x7x11xf32>
  %A = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%a : tensor<2x3x5x7x11xf32>) outs(%em0 : tensor<2x3x5x7x11xf32>) -> tensor<2x3x5x7x11xf32>
  %em1 = tensor.empty() : tensor<2x3x5x7x11xf32>
  %B = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%b : tensor<2x3x5x7x11xf32>) outs(%em1 : tensor<2x3x5x7x11xf32>) -> tensor<2x3x5x7x11xf32>

  %collapsed_A = tensor.collapse_shape %A [[0, 1], [2, 3], [4]] : tensor<2x3x5x7x11xf32> into tensor<6x35x11xf32>
  %collapsed_B = tensor.collapse_shape %B [[0, 1], [2, 3], [4]] : tensor<2x3x5x7x11xf32> into tensor<6x35x11xf32>
  %res = hfusion.interleave %collapsed_A, %collapsed_B : tensor<6x35x11xf32>, tensor<6x35x11xf32> -> tensor<6x35x22xf32>

  %em2 = tensor.empty() : tensor<6x35x11xf32>
  %g0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%collapsed_A : tensor<6x35x11xf32>) outs(%em2 : tensor<6x35x11xf32>) -> tensor<6x35x11xf32>
  %h0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%g0 : tensor<6x35x11xf32>) outs(%em2 : tensor<6x35x11xf32>) -> tensor<6x35x11xf32>
  %em3 = tensor.empty() : tensor<6x35x11xf32>
  %g1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%collapsed_B : tensor<6x35x11xf32>) outs(%em3 : tensor<6x35x11xf32>) -> tensor<6x35x11xf32>
  %h1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%g1 : tensor<6x35x11xf32>) outs(%em3 : tensor<6x35x11xf32>) -> tensor<6x35x11xf32>
  return %h0, %h1, %res : tensor<6x35x11xf32>, tensor<6x35x11xf32>, tensor<6x35x22xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @main_expand_up_deinterleave
func.func @main_expand_up_deinterleave(%arg: tensor<30x14x52xf32>) -> (tensor<6x5x7x2x13x2xf32>, tensor<6x5x7x2x13x2xf32>, tensor<30x14x26xf32>) {
  %em = tensor.empty() : tensor<30x14x52xf32>
  %A = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg : tensor<30x14x52xf32>) outs(%em : tensor<30x14x52xf32>) -> tensor<30x14x52xf32>
  
  %a = hfusion.deinterleave %A channel<0> : tensor<30x14x52xf32> -> tensor<30x14x26xf32>
  %b = tensor.expand_shape %a [[0, 1], [2, 3], [4, 5]] output_shape [6, 5, 7, 2, 13, 2] : tensor<30x14x26xf32> into tensor<6x5x7x2x13x2xf32>

  %em0 = tensor.empty() : tensor<30x14x26xf32>
  %g0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%a : tensor<30x14x26xf32>) outs(%em0 : tensor<30x14x26xf32>) -> tensor<30x14x26xf32>
  %h0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%g0 : tensor<30x14x26xf32>) outs(%em0 : tensor<30x14x26xf32>) -> tensor<30x14x26xf32>
  %em1 = tensor.empty() : tensor<6x5x7x2x13x2xf32>
  %g1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%b : tensor<6x5x7x2x13x2xf32>) outs(%em1 : tensor<6x5x7x2x13x2xf32>) -> tensor<6x5x7x2x13x2xf32>
  %h1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%g1 : tensor<6x5x7x2x13x2xf32>) outs(%em1 : tensor<6x5x7x2x13x2xf32>) -> tensor<6x5x7x2x13x2xf32>  
  return %h1, %b, %h0 : tensor<6x5x7x2x13x2xf32>, tensor<6x5x7x2x13x2xf32>, tensor<30x14x26xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: @test_collapse_down_deinterleave
func.func @test_collapse_down_deinterleave(%arg: tensor<6x5x7x2x26x2xf32>) -> (tensor<30x14x26xf32>, tensor<30x14x26xf32>, tensor<30x14x52xf32>) {
  %em = tensor.empty() : tensor<6x5x7x2x26x2xf32>
  %A = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg : tensor<6x5x7x2x26x2xf32>) outs(%em : tensor<6x5x7x2x26x2xf32>) -> tensor<6x5x7x2x26x2xf32>
  
  %a = tensor.collapse_shape %A [[0, 1], [2, 3], [4, 5]] : tensor<6x5x7x2x26x2xf32> into tensor<30x14x52xf32>
  %b = hfusion.deinterleave %a channel<0> : tensor<30x14x52xf32> -> tensor<30x14x26xf32>
  
  %em0 = tensor.empty() : tensor<30x14x26xf32>
  %g0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%b : tensor<30x14x26xf32>) outs(%em0 : tensor<30x14x26xf32>) -> tensor<30x14x26xf32>
  %h0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%g0 : tensor<30x14x26xf32>) outs(%em0 : tensor<30x14x26xf32>) -> tensor<30x14x26xf32>
  %em1 = tensor.empty() : tensor<30x14x52xf32>
  %g1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%a : tensor<30x14x52xf32>) outs(%em1 : tensor<30x14x52xf32>) -> tensor<30x14x52xf32>
  %h1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%g1 : tensor<30x14x52xf32>) outs(%em1 : tensor<30x14x52xf32>) -> tensor<30x14x52xf32>
  return %b, %h0, %h1 : tensor<30x14x26xf32>, tensor<30x14x26xf32>, tensor<30x14x52xf32>
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @transpose_corner_error(
module {
  func.func @transpose_corner_error(%arg0: tensor<2x1x32x1x1x40x1x1x16x16xf32>, %arg1: tensor<2x16x16x1280xf32>) -> tensor<2x1280x16x16xf32> {
    %0 = tensor.empty() : tensor<2x1x32x1x1x40x1x1x16x16xf32>
    %1 = tensor.empty() : tensor<2x1280x16x16xf32>
    %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<2x1x32x1x1x40x1x1x16x16xf32>) outs(%0 : tensor<2x1x32x1x1x40x1x1x16x16xf32>) -> tensor<2x1x32x1x1x40x1x1x16x16xf32>
    %collapsed = tensor.collapse_shape %2 [[0], [1, 2, 3, 4, 5], [6, 7, 8], [9]] : tensor<2x1x32x1x1x40x1x1x16x16xf32> into tensor<2x1280x16x16xf32>
    %3 = tensor.empty() : tensor<2x1280x16x16xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<2x16x16x1280xf32>) outs(%collapsed : tensor<2x1280x16x16xf32>) permutation = [0, 3, 1, 2]  {debug_instruction_number = 12 : i32}
    %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%transposed : tensor<2x1280x16x16xf32>) outs(%3 : tensor<2x1280x16x16xf32>) -> tensor<2x1280x16x16xf32>
    return %5 : tensor<2x1280x16x16xf32>
  }
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @concat_with_dynamic(
func.func @concat_with_dynamic(%arg0: tensor<1x?x2560xbf16>, %arg1: tensor<1x3x1x2560xbf16>) -> tensor<1x?x2560xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %emp = tensor.empty() : tensor<1x3x1x2560xbf16>
  %una = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : tensor<1x3x1x2560xbf16>) outs(%emp : tensor<1x3x1x2560xbf16>) -> tensor<1x3x1x2560xbf16>
  %collapsed = tensor.collapse_shape %una [[0], [1], [2, 3]] : tensor<1x3x1x2560xbf16> into tensor<1x3x2560xbf16>
  %concat = tensor.concat dim(1) %arg0, %collapsed : (tensor<1x?x2560xbf16>, tensor<1x3x2560xbf16>) -> tensor<1x?x2560xbf16>
  return %concat : tensor<1x?x2560xbf16>
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @concat_with_dynamic_2(
func.func @concat_with_dynamic_2(%arg0: tensor<1x?x1x2560xbf16>, %arg1: tensor<1x3x2560xbf16>, %inp : index) -> tensor<1x?x2560xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %emp = tensor.empty(%inp) : tensor<1x?x1x2560xbf16>
  %una = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<1x?x1x2560xbf16>) outs(%emp : tensor<1x?x1x2560xbf16>) -> tensor<1x?x1x2560xbf16>
  %collapsed = tensor.collapse_shape %una [[0], [1], [2, 3]] : tensor<1x?x1x2560xbf16> into tensor<1x?x2560xbf16>
  %concat = tensor.concat dim(1) %collapsed, %arg1 : (tensor<1x?x2560xbf16>, tensor<1x3x2560xbf16>) -> tensor<1x?x2560xbf16>
  return %concat : tensor<1x?x2560xbf16>
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @concat_with_dynamic_3(
func.func @concat_with_dynamic_3(%arg0: tensor<1x1x?x2560xbf16>, %arg1: tensor<1x3x2560xbf16>, %inp : index) -> tensor<1x?x2560xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %emp = tensor.empty(%inp) : tensor<1x1x?x2560xbf16>
  %una = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<1x1x?x2560xbf16>) outs(%emp : tensor<1x1x?x2560xbf16>) -> tensor<1x1x?x2560xbf16>
  %collapsed = tensor.collapse_shape %una [[0], [1, 2], [3]] : tensor<1x1x?x2560xbf16> into tensor<1x?x2560xbf16>
  %concat = tensor.concat dim(1) %collapsed, %arg1 : (tensor<1x?x2560xbf16>, tensor<1x3x2560xbf16>) -> tensor<1x?x2560xbf16>
  return %concat : tensor<1x?x2560xbf16>
}

// -----
// CHECK: Valid
// CHECK-LABEL: func.func @concat_with_dynamic_4(
func.func @concat_with_dynamic_4(%arg0: tensor<1x?x1x2560xbf16>, %arg1: tensor<1x3x2560xbf16>, %inp : index) -> tensor<1x?x2560xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %emp = tensor.empty(%inp) : tensor<1x?x1x2560xbf16>
  %una = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<1x?x1x2560xbf16>) outs(%emp : tensor<1x?x1x2560xbf16>) -> tensor<1x?x1x2560xbf16>
  %collapsed = tensor.collapse_shape %una [[0], [1, 2], [3]] : tensor<1x?x1x2560xbf16> into tensor<1x?x2560xbf16>
  %concat = tensor.concat dim(1) %collapsed, %arg1 : (tensor<1x?x2560xbf16>, tensor<1x3x2560xbf16>) -> tensor<1x?x2560xbf16>
  return %concat : tensor<1x?x2560xbf16>
}
