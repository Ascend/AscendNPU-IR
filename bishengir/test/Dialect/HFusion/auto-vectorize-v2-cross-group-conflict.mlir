// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --outline-vector-function

// Two independent chains with different dim0 prevent 7/9 cross-group:
//   7 ([128,256])  --expand_shape-->   8 ([32,4,256])      7 -> 8
//   9 ([32,4,256]) --collapse_shape--> 10 ([128,256])      9 -> 10
//
// plan to fuse {7,10}, {8,9}.
//
// 7 -> 8
// > {7, 10} -> {8, 9}
// 9 -> 10
// > {8, 9} -> {7, 10}
//
// circular dependency between groups

func.func @test_cross_group_conflict(%arg0: tensor<128x256xf16>, %arg1: tensor<128x256xf16>) -> (tensor<32x4x256xf16>, tensor<128x256xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  // Op 7: 2D [128, 256], leaf (user: expand1 = NonVectorizableOp)
  %empty7 = tensor.empty() : tensor<128x256xf16>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<128x256xf16>, tensor<128x256xf16>) outs(%empty7 : tensor<128x256xf16>) -> tensor<128x256xf16>

  // expand1: NonVectorizableOp, 7 -> expand1 -> 8, creates 7<->8 conflict
  %expand1 = tensor.expand_shape %7 [[0, 1], [2]] output_shape [32, 4, 256] : tensor<128x256xf16> into tensor<32x4x256xf16>

  // Op 8: 3D [32, 4, 256], leaf (user: return), uses 7 via expand1
  %empty8 = tensor.empty() : tensor<32x4x256xf16>
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expand1, %expand1 : tensor<32x4x256xf16>, tensor<32x4x256xf16>) outs(%empty8 : tensor<32x4x256xf16>) -> tensor<32x4x256xf16>

  // Op 9: 3D [32, 4, 256], leaf (user: collapse2 = NonVectorizableOp)
  %in9a = tensor.empty() : tensor<32x4x256xf16>
  %in9b = tensor.empty() : tensor<32x4x256xf16>
  %empty9 = tensor.empty() : tensor<32x4x256xf16>
  %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%in9a, %in9b : tensor<32x4x256xf16>, tensor<32x4x256xf16>) outs(%empty9 : tensor<32x4x256xf16>) -> tensor<32x4x256xf16>

  // collapse2: NonVectorizableOp, 9 -> collapse2 -> 10, creates 9<->10 conflict
  %collapse2 = tensor.collapse_shape %9 [[0, 1], [2]] : tensor<32x4x256xf16> into tensor<128x256xf16>

  // Op 10: 2D [128, 256], leaf (user: return), uses 9 via collapse2
  %empty10 = tensor.empty() : tensor<128x256xf16>
  %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapse2, %collapse2 : tensor<128x256xf16>, tensor<128x256xf16>) outs(%empty10 : tensor<128x256xf16>) -> tensor<128x256xf16>

  return %8, %10 : tensor<32x4x256xf16>, tensor<128x256xf16>
}
