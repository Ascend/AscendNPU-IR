// RUN: bishengir-opt -hfusion-cache-io %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_single_op
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_single_op(
  %src : tensor<6x4xbf16>, %dst : tensor<6x4xbf16>) -> tensor<6x4xbf16> {
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
    ins(%src : tensor<6x4xbf16>)
    outs(%dst : tensor<6x4xbf16>)
    -> tensor<6x4xbf16>
  return %res : tensor<6x4xbf16>
}

// -----

// CHECK-LABEL: func.func @test_reshape
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_reshape(%arg0: tensor<?x256xf32>) -> tensor<?x1x1x256xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x256xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [%dim, 1, 1, 256] : tensor<?x256xf32> into tensor<?x1x1x256xf32>
  %0 = tensor.empty(%dim) : tensor<?x1x1x256xf32>
  %1 = hfusion.elemwise_unary ins(%expanded : tensor<?x1x1x256xf32>) outs(%0 : tensor<?x1x1x256xf32>) -> tensor<?x1x1x256xf32>
  return %1 : tensor<?x1x1x256xf32>
}

// -----
// CHECK-LABEL: func.func @duplicate_return_reshape
func.func @duplicate_return_reshape(%arg0: tensor<24x6x256x192xbf16>) -> (tensor<144xf32>, tensor<24x6xf32>, tensor<24x6xf32>, tensor<24x6xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<144x49152xf32>
  %1 = tensor.empty() : tensor<144xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<144xf32>) -> tensor<144xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x6x256x192xbf16> into tensor<144x49152xbf16>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<144x49152xbf16>) outs(%0 : tensor<144x49152xf32>) -> tensor<144x49152xf32>
  %reduced = linalg.reduce ins(%3 : tensor<144x49152xf32>) outs(%2 : tensor<144xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  // CHECK: %[[store4:.*]] = hfusion.store 
  // CHECK: %[[store3:.*]] = hfusion.store 
  // CHECK: %[[store2:.*]] = hfusion.store 
  // CHECK: %[[store1:.*]] = hfusion.store
  // CHECK: %[[expand1:.*]] = tensor.expand_shape %[[store1]]
  // CHECK: %[[expand2:.*]] = tensor.expand_shape %[[store2]]
  // CHECK: %[[expand3:.*]] = tensor.expand_shape %[[store3]]
  // CHECK: %[[expand4:.*]] = tensor.expand_shape %[[store4]]
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [24, 6] : tensor<144xf32> into tensor<24x6xf32>
  // CHECK: return %[[store1]], %[[expand2]], %[[expand3]], %[[expand4]]
  return %reduced, %expanded, %expanded, %expanded : tensor<144xf32>, tensor<24x6xf32>, tensor<24x6xf32>, tensor<24x6xf32>
}

// -----

// CHECK-LABEL: func.func @duplicate_return_reshape_cast
func.func @duplicate_return_reshape_cast(%arg0: tensor<24x6x256x192xbf16>) -> (tensor<144xf32>, tensor<24x6xf32>, tensor<24x6xf32>, tensor<8x3x6xbf16>, tensor<8x3x6xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<144x49152xf32>
  %1 = tensor.empty() : tensor<144xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<144xf32>) -> tensor<144xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x6x256x192xbf16> into tensor<144x49152xbf16>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<144x49152xbf16>) outs(%0 : tensor<144x49152xf32>) -> tensor<144x49152xf32>
  %reduced = linalg.reduce ins(%3 : tensor<144x49152xf32>) outs(%2 : tensor<144xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %4 = tensor.empty() : tensor<8x3x6xbf16>
  // CHECK: %[[store3:.*]] = hfusion.store 
  // CHECK: %[[store2:.*]] = hfusion.store 
  // CHECK: %[[store1:.*]] = hfusion.store 
  // CHECK: %[[expand1:.*]] = tensor.expand_shape %[[store1]]
  // CHECK: %[[expand2:.*]] = tensor.expand_shape %[[store2]]
  // CHECK: %[[expand3:.*]] = tensor.expand_shape %[[store3]]
  // CHECK: %[[store5:.*]] = hfusion.store 
  // CHECK: %[[store4:.*]] = hfusion.store
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [24, 6] : tensor<144xf32> into tensor<24x6xf32>
  %expanded2 = tensor.expand_shape %expanded [[0, 1],[2]] output_shape [8, 3, 6] : tensor<24x6xf32> into tensor<8x3x6xf32>
  %casted = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded2 : tensor<8x3x6xf32>) outs(%4 : tensor<8x3x6xbf16>) -> tensor<8x3x6xbf16>
  // CHECK: return %[[store1]], %[[expand2]], %[[expand3]], %[[store4]], %[[store5]]
  return %reduced, %expanded, %expanded, %casted, %casted : tensor<144xf32>, tensor<24x6xf32>, tensor<24x6xf32>, tensor<8x3x6xbf16>, tensor<8x3x6xbf16>
}

// -----
// CHECK-LABEL: func.func @reshape_direct_return
// CHECK: hfusion.store
// CHECK: hfusion.store
func.func @reshape_direct_return(%arg0: tensor<24x256x1x1xbf16>) -> (tensor<1x256xf32>, tensor<256xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<24x256xf32>
  %1 = tensor.empty() : tensor<256xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<24x256x1x1xbf16> into tensor<24x256xbf16>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<24x256xbf16>) outs(%0 : tensor<24x256xf32>) -> tensor<24x256xf32>
  %reduced = linalg.reduce ins(%3 : tensor<24x256xf32>) outs(%2 : tensor<256xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 256] : tensor<256xf32> into tensor<1x256xf32>
  return %expanded, %reduced : tensor<1x256xf32>, tensor<256xf32>
}

// -----
// CHECK-LABEL: func.func @test_already_cached_io_0
// CHECK-DAG: hfusion.load
// CHECK-DAG: hfusion.store
// CHECK-DAG: hfusion.store
// CHECK-NOT: hfusion.load
// CHECK-NOT: hfusion.store
func.func @test_already_cached_io_0(%arg0: tensor<16x16xf32> {hacc.cached_io}) -> (tensor<64x4xf32>, tensor<16x16xf32> {hacc.cached_io}) {
  %0 = tensor.empty() : tensor<16x4x4xf32>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [16, 4, 4] : tensor<16x16xf32> into tensor<16x4x4xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%expanded : tensor<16x4x4xf32>) outs(%0 : tensor<16x4x4xf32>) -> tensor<16x4x4xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<16x4x4xf32> into tensor<64x4xf32>
  %2 = tensor.empty() : tensor<16x16xf32>
  %3 = hfusion.load ins(%arg0 : tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = hfusion.store ins(%3 : tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %collapsed, %4 : tensor<64x4xf32>, tensor<16x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_trace_ignore_slice
// CHECK: %[[slice:.*]] = tensor.extract_slice %{{.*}}[0, 1, 0]   
// CHECK: hfusion.store ins(%[[slice]]
// CHECK: hfusion.store
func.func @test_trace_ignore_slice(%arg0: tensor<128x6912xf16>, %arg1: tensor<128x768xf16>) -> (tensor<128x768xf16>, tensor<128x1x768xf16>) {
  %cst = arith.constant 2.000000e+00 : f16
  %0 = tensor.empty() : tensor<128x1x768xf16>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [128, 9, 768] : tensor<128x6912xf16> into tensor<128x9x768xf16>
  %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [128, 1, 768] [1, 1, 1] : tensor<128x9x768xf16> to tensor<128x1x768xf16>
  %extracted_slice_0 = tensor.extract_slice %expanded[0, 1, 0] [128, 1, 768] [1, 1, 1] : tensor<128x9x768xf16> to tensor<128x1x768xf16>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %cst : tensor<128x1x768xf16>, f16) outs(%0 : tensor<128x1x768xf16>) -> tensor<128x1x768xf16>
  %collapsed = tensor.collapse_shape %2 [[0], [1, 2]] : tensor<128x1x768xf16> into tensor<128x768xf16>
  return %collapsed, %extracted_slice_0 : tensor<128x768xf16>, tensor<128x1x768xf16>
}


// -----
// CHECK-LABEL: func.func @test_reshape_used_twice
func.func @test_reshape_used_twice(%arg0: tensor<24x6x256x192xbf16>) -> (tensor<144xf32>, tensor<24x6xf32>, tensor<144xf32>, tensor<12x2x6xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<144x49152xf32>
  %1 = tensor.empty() : tensor<144xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<144xf32>) -> tensor<144xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x6x256x192xbf16> into tensor<144x49152xbf16>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<144x49152xbf16>) outs(%0 : tensor<144x49152xf32>) -> tensor<144x49152xf32>
  %reduced = linalg.reduce ins(%3 : tensor<144x49152xf32>) outs(%2 : tensor<144xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  // CHECK: %[[store4:.*]] = hfusion.store 
  // CHECK: %[[store3:.*]] = hfusion.store 
  // CHECK: %[[store2:.*]] = hfusion.store 
  // CHECK: %[[store1:.*]] = hfusion.store
  // CHECK: %[[expand1:.*]] = tensor.expand_shape %[[store1]]
  // CHECK: %[[expand2:.*]] = tensor.expand_shape %[[store2]]
  // CHECK: %[[expand3:.*]] = tensor.expand_shape %[[store3]]
  // CHECK: %[[expand4:.*]] = tensor.expand_shape %[[store4]]
  // CHECK: %[[collapse3:.*]] = tensor.collapse_shape %[[expand3]]
  // CHECK: %[[expand4_0:.*]] = tensor.expand_shape %[[expand4]]
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [24, 6] : tensor<144xf32> into tensor<24x6xf32>
  %collapsed2 = tensor.collapse_shape %expanded [[0, 1]] : tensor<24x6xf32> into tensor<144xf32>
  %expanded2 = tensor.expand_shape %expanded [[0, 1],[2]] output_shape [12, 2, 6]: tensor<24x6xf32> into tensor<12x2x6xf32>
  // CHECK: return %[[store1]], %[[expand2]], %[[collapse3]], %[[expand4_0]]
  return %reduced, %expanded, %collapsed2, %expanded2 : tensor<144xf32>, tensor<24x6xf32>, tensor<144xf32>, tensor<12x2x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_interleave_0
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_interleave_0(%arg0: tensor<1xf16>) -> tensor<2xf16> {
  %0 = hfusion.interleave %arg0, %arg0 : tensor<1xf16>, tensor<1xf16> -> tensor<2xf16>
  return %0 : tensor<2xf16>
}

// -----

// CHECK-LABEL: func.func @test_deinterleave_0
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_deinterleave_0(%arg0: tensor<2xf16>) -> tensor<1xf16> {
  %0 = hfusion.deinterleave %arg0 channel<0> : tensor<2xf16> -> tensor<1xf16>
  return %0 : tensor<1xf16>
}