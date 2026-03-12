// REQUIRES: asserts
// RUN: bishengir-opt %s --propagate-reshape="for-regbased=true" --cse --canonicalize -split-input-file | FileCheck %s



// CHECK-LABEL: @elemwise_binary_expand(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, {{.*}} : tensor<2x3x7x8xf32>, tensor<2x3x7x8xf32>) outs({{.*}} : tensor<2x3x7x8xf32>) -> tensor<2x3x7x8xf32>
func.func @elemwise_binary_expand(%arg0: tensor<6x7x8xf32>, %arg1: tensor<6x7x8xf32>) -> tensor<2x3x7x8xf32> {
  %0 = math.absf %arg0 : tensor<6x7x8xf32>
  %1 = math.absf %arg1 : tensor<6x7x8xf32>
  %2 = tensor.empty() : tensor<6x7x8xf32>
  %binary = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %1 : tensor<6x7x8xf32>, tensor<6x7x8xf32>) outs(%2 : tensor<6x7x8xf32>) -> tensor<6x7x8xf32>
  %expanded = tensor.expand_shape %binary [[0, 1], [2], [3]] output_shape [2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
  %3 = math.absf %expanded : tensor<2x3x7x8xf32>
  return %3 : tensor<2x3x7x8xf32>
}

// -----

// CHECK-LABEL: @collapse_elemwise_binary(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, {{.*}} : tensor<2x3x7x8xf32>, tensor<2x3x7x8xf32>) outs({{.*}} : tensor<2x3x7x8xf32>) -> tensor<2x3x7x8xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
func.func @collapse_elemwise_binary(%arg0: tensor<2x3x7x8xf32>, %arg1: tensor<6x7x8xf32>) -> tensor<6x7x8xf32> {
  %0 = math.absf %arg0 : tensor<2x3x7x8xf32>
  %collapse = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
  %1 = math.absf %arg1 : tensor<6x7x8xf32>
  %2 = tensor.empty() : tensor<6x7x8xf32>
  %binary = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapse, %1 : tensor<6x7x8xf32>, tensor<6x7x8xf32>) outs(%2 : tensor<6x7x8xf32>) -> tensor<6x7x8xf32>
  return %binary : tensor<6x7x8xf32>
}

// -----

// CHECK-LABEL: @elemwise_unary_expand(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins({{.*}} : tensor<2x3x7x8xf32>) outs({{.*}} : tensor<2x3x7x8xf32>) -> tensor<2x3x7x8xf32>
func.func @elemwise_unary_expand(%arg0: tensor<6x7x8xf32>) -> tensor<2x3x7x8xf32> {
  %0 = math.absf %arg0 : tensor<6x7x8xf32>
  %1 = tensor.empty() : tensor<6x7x8xf32>
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%0 : tensor<6x7x8xf32>) outs(%1 : tensor<6x7x8xf32>) -> tensor<6x7x8xf32>
  %expanded = tensor.expand_shape %unary [[0, 1], [2], [3]] output_shape [2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
  %2 = math.absf %expanded: tensor<2x3x7x8xf32>
  return %2 : tensor<2x3x7x8xf32>
}

// -----

// CHECK-LABEL: @collapse_elemwise_unary(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins({{.*}} : tensor<2x3x7x8xf32>) outs({{.*}} : tensor<2x3x7x8xf32>) -> tensor<2x3x7x8xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
func.func @collapse_elemwise_unary(%arg0: tensor<2x3x7x8xf32>) -> tensor<6x7x8xf32> {
  %0 = math.absf %arg0 : tensor<2x3x7x8xf32>
  %collapse = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
  %1 = tensor.empty() : tensor<6x7x8xf32>
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%collapse : tensor<6x7x8xf32>) outs(%1 : tensor<6x7x8xf32>) -> tensor<6x7x8xf32>
  return %unary : tensor<6x7x8xf32>
}

// -----

// CHECK-LABEL: @collapse_transpose(
// CHECK: linalg.transpose ins{{.*}} : tensor<2x3x7x8xf32>) outs({{.*}} : tensor<7x2x3x8xf32>) permutation = {{\[}}2, 0, 1, 3]
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] : tensor<7x2x3x8xf32> into tensor<7x6x8xf32>
func.func @collapse_transpose(%arg0: tensor<2x3x7x8xf32>) -> tensor<7x6x8xf32> {
  %0 = math.absf %arg0 : tensor<2x3x7x8xf32>
  %collapse = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
  %1 = tensor.empty() : tensor<7x6x8xf32>
  %transposed = linalg.transpose ins(%collapse : tensor<6x7x8xf32>) outs(%1 : tensor<7x6x8xf32>) permutation = [1, 0, 2]
  return %transposed : tensor<7x6x8xf32>
}

// -----

// CHECK-LABEL: @transpose_expand(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2], {{\[}}3, 4]] output_shape {{\[}}2, 3, 7, 2, 4] : tensor<2x3x7x8xf32> into tensor<2x3x7x2x4xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2], {{\[}}3, 4]] output_shape {{\[}}3, 2, 7, 2, 4] : tensor<3x2x7x8xf32> into tensor<3x2x7x2x4xf32>
// CHECK:  linalg.transpose ins(%expanded : tensor<2x3x7x2x4xf32>) outs(%expanded_0 : tensor<3x2x7x2x4xf32>) permutation = {{\[}}1, 0, 2, 3, 4]
func.func @transpose_expand(%arg0: tensor<2x3x7x8xf32>) -> tensor<3x2x7x2x4xf32> {
  %1 = tensor.empty() : tensor<3x2x7x8xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x7x8xf32>) outs(%1 : tensor<3x2x7x8xf32>) permutation = [1, 0, 2, 3]
  %expanded = tensor.expand_shape %transposed [[0], [1], [2], [3, 4]] output_shape [3, 2, 7, 2, 4] : tensor<3x2x7x8xf32> into tensor<3x2x7x2x4xf32>
  %2 = math.absf %expanded: tensor<3x2x7x2x4xf32>
  return %2 : tensor<3x2x7x2x4xf32>
}

// -----

// CHECK-LABEL: @collapse_broadcast_in(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3], {{\[}}4]] output_shape {{\[}}5, 2, 3, 7, 8] : tensor<5x6x7x8xf32> into tensor<5x2x3x7x8xf32>
// CHECK:  linalg.broadcast ins({{.*}}:  tensor<2x3x7x8xf32>) outs({{.*}} : tensor<5x2x3x7x8xf32>) dimensions = {{\[}}0]
// CHECK:  tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3], {{\[}}4]] : tensor<5x2x3x7x8xf32> into tensor<5x6x7x8xf32>
func.func @collapse_broadcast_in(%arg0: tensor<2x3x7x8xf32>, %arg1: tensor<5x6x7x8xf32>) -> tensor<5x6x7x8xf32> {
  %0 = math.absf %arg0 : tensor<2x3x7x8xf32>
  %1 = math.absf %arg1 : tensor<5x6x7x8xf32>
  %collapsed_in = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<2x3x7x8xf32> into tensor<6x7x8xf32>
  %broadcasted = linalg.broadcast ins(%collapsed_in : tensor<6x7x8xf32>) outs(%1 : tensor<5x6x7x8xf32>) dimensions = [0]
  return %broadcasted : tensor<5x6x7x8xf32>
}

// -----

// CHECK-LABEL: @collapse_broadcast_out(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2], {{\[}}3]] output_shape {{\[}}2, 3, 7, 8] : tensor<6x7x8xf32> into tensor<2x3x7x8xf32>
// CHECK:  linalg.broadcast ins({{.*}}:  tensor<2x3x7x8xf32>) outs({{.*}} : tensor<5x2x3x7x8xf32>) dimensions = {{\[}}0]
// CHECK:  tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3], {{\[}}4]] : tensor<5x2x3x7x8xf32> into tensor<5x6x7x8xf32>
func.func @collapse_broadcast_out(%arg0: tensor<6x7x8xf32>, %arg1: tensor<5x2x3x7x8xf32>) -> tensor<5x6x7x8xf32> {
  %0 = math.absf %arg0 : tensor<6x7x8xf32>
  %1 = math.absf %arg1 : tensor<5x2x3x7x8xf32>
  %collapsed_out = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<5x2x3x7x8xf32> into tensor<5x6x7x8xf32>
  %broadcasted = linalg.broadcast ins(%0 : tensor<6x7x8xf32>) outs(%collapsed_out : tensor<5x6x7x8xf32>) dimensions = [0]
  return %broadcasted : tensor<5x6x7x8xf32>
}
// -----

// CHECK-LABEL: @hfusion_reduce_with_index_expand(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3]] output_shape {{\[}}6, 7, 2, 4] : tensor<6x7x8xf32> into tensor<6x7x2x4xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] output_shape {{\[}}6, 2, 4] : tensor<6x8xf32> into tensor<6x2x4xf32>
// CHECK: linalg.reduce ins({{.*}}: tensor<6x7x2x4xf32>) outs({{.*}} : tensor<6x2x4xf32>) dimensions = {{\[}}1]
func.func @hfusion_reduce_with_index_expand(%arg0: tensor<6x7x8xf32>, %arg1: tensor<6x7x8xi32>) -> tensor<6x2x4xf32> {
  %0 = math.absf %arg0 : tensor<6x7x8xf32>
  %1 = math.absi %arg1 : tensor<6x7x8xi32>
  %output = tensor.empty() : tensor<6x8xf32>
  %output_index = tensor.empty() : tensor<6x8xi32>
  %reduce_result:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
      ins(%0, %1 : tensor<6x7x8xf32>, tensor<6x7x8xi32>) outs(%output, %output_index : tensor<6x8xf32>, tensor<6x8xi32>) dimensions = [1] -> tensor<6x8xf32>, tensor<6x8xi32>
  %expanded = tensor.expand_shape %reduce_result#0 [[0], [1, 2]] output_shape [6, 2, 4] : tensor<6x8xf32> into tensor<6x2x4xf32>
  %new_unary = math.absf %expanded : tensor<6x2x4xf32>
  return %new_unary : tensor<6x2x4xf32>
}

// -----

// CHECK-LABEL: @hfusion_reduce_with_index_expand_index(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3]] output_shape {{\[}}6, 7, 2, 4] : tensor<6x7x8xf32> into tensor<6x7x2x4xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] output_shape {{\[}}6, 2, 4] : tensor<6x8xf32> into tensor<6x2x4xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] output_shape {{\[}}6, 2, 4] : tensor<6x8xi32> into tensor<6x2x4xi32>
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins({{.*}}, {{.*}} : tensor<6x7x2x4xf32>, tensor<6x7x2x4xi32>) outs({{.*}}, {{.*}} : tensor<6x2x4xf32>, tensor<6x2x4xi32>) dimensions = {{\[}}1]
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] : tensor<6x2x4xi32> into tensor<6x8xi32>
func.func @hfusion_reduce_with_index_expand_index(%arg0: tensor<6x7x8xf32>, %arg1: tensor<6x7x8xi32>) -> (tensor<6x2x4xf32>, tensor<6x8xi32>) {
  %0 = math.absf %arg0 : tensor<6x7x8xf32>
  %1 = math.absi %arg1 : tensor<6x7x8xi32>
  %output = tensor.empty() : tensor<6x8xf32>
  %output_index = tensor.empty() : tensor<6x8xi32>
  %reduce_result:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
      ins(%0, %1 : tensor<6x7x8xf32>, tensor<6x7x8xi32>) outs(%output, %output_index : tensor<6x8xf32>, tensor<6x8xi32>) dimensions = [1] -> tensor<6x8xf32>, tensor<6x8xi32>
  %expanded = tensor.expand_shape %reduce_result#0 [[0], [1, 2]] output_shape [6, 2, 4] : tensor<6x8xf32> into tensor<6x2x4xf32>
  %new_unary = math.absf %expanded : tensor<6x2x4xf32>
  return %new_unary, %reduce_result#1 : tensor<6x2x4xf32>, tensor<6x8xi32>
}

// -----

// CHECK-LABEL: @reduce_expand(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2], {{\[}}3], {{\[}}4], {{\[}}5], {{\[}}6], {{\[}}7], {{\[}}8, 9]] output_shape {{\[}}2, 3, 4, 5, 6, 7, 8, 9, 2, 5] : tensor<2x3x4x5x6x7x8x9x10xf32> into tensor<2x3x4x5x6x7x8x9x2x5xf32>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3, 4], {{\[}}5, 6]] output_shape {{\[}}2, 5, 6, 7, 8, 2, 5] : tensor<2x5x336x10xf32> into tensor<2x5x6x7x8x2x5xf32>
// CHECK: linalg.reduce ins({{.*}} : tensor<2x3x4x5x6x7x8x9x2x5xf32>) outs({{.*}} : tensor<2x5x6x7x8x2x5xf32>) dimensions = {{\[}}1, 2, 7]
func.func @reduce_expand(%arg0: tensor<2x3x4x5x6x7x8x9x10xf32>) -> tensor<2x5x6x7x8x2x5xf32> {
  %init = tensor.empty() : tensor<2x5x336x10xf32>
  %expand = tensor.expand_shape %init [[0], [1], [2, 3, 4], [5]] output_shape [2, 5, 6, 7, 8, 10] : tensor<2x5x336x10xf32> into tensor<2x5x6x7x8x10xf32>

  %reduced = linalg.reduce ins(%arg0 : tensor<2x3x4x5x6x7x8x9x10xf32>)
                            outs(%expand : tensor<2x5x6x7x8x10xf32>)
                            dimensions = [1, 2, 7]
                      (%asd: f32, %asdInit: f32) {
                        %inside = arith.addf %asd, %asdInit : f32
                        linalg.yield %inside : f32
                      }
  %expand_0 = tensor.expand_shape %reduced [[0], [1], [2], [3], [4], [5, 6]] output_shape [2, 5, 6, 7, 8, 2, 5] : tensor<2x5x6x7x8x10xf32> into tensor<2x5x6x7x8x2x5xf32>
  %0 = math.absf %expand_0: tensor<2x5x6x7x8x2x5xf32>
  return %0 : tensor<2x5x6x7x8x2x5xf32>
}

// -----

// CHECK-LABEL: @collapse_reduce(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3, 4], {{\[}}5]] output_shape {{\[}}2, 5, 6, 7, 8, 10] : tensor<2x5x336x10xf32> into tensor<2x5x6x7x8x10xf32>
// CHECK: linalg.reduce ins({{.*}} : tensor<2x3x4x5x6x7x8x9x10xf32>) outs({{.*}} : tensor<2x5x6x7x8x10xf32>) dimensions = {{\[}}1, 2, 7]
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3, 4], {{\[}}5]] : tensor<2x5x6x7x8x10xf32> into tensor<2x5x336x10xf32>
func.func @collapse_reduce(%arg0: tensor<2x3x4x5x6x7x8x9x10xf32>) -> tensor<2x5x336x10xf32> {
  %init = tensor.empty() : tensor<2x5x336x10xf32>
  %0 = tensor.empty() : tensor<2x3x4x5x6x7x8x9x10xf32>
  %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7x8x9x10xf32>) outs(%0 : tensor<2x3x4x5x6x7x8x9x10xf32>) -> tensor<2x3x4x5x6x7x8x9x10xf32>

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

// CHECK-LABEL: @collapse_arg_reduce(
// CHECK-NOT: expand_shape
func.func @collapse_arg_reduce(%arg0: tensor<2x3x4x5x6x7x8x9x10xf32>) -> tensor<2x12x5x336x10xf32> {
  %init = tensor.empty() : tensor<2x12x5x336x10xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7], [8]] :
    tensor<2x3x4x5x6x7x8x9x10xf32> into tensor<2x12x5x336x9x10xf32>

  %reduced = linalg.reduce ins(%collapsed : tensor<2x12x5x336x9x10xf32>)
                            outs(%init : tensor<2x12x5x336x10xf32>)
                            dimensions = [4]
                      (%asd: f32, %asdInit: f32) {
                        %inside = arith.addf %asd, %asdInit : f32
                        linalg.yield %inside : f32
                      }
  return %reduced : tensor<2x12x5x336x10xf32>
}

// -----

// CHECK-LABEL: @reinterpret_cast_with_stride(
// CHECK: memref.reinterpret_cast
// CHECK-SAME: memref<?xi8> to memref<1x32x1x1xi8, strided<[32, 256, 1, 1], offset: ?>>
// CHECK: memref.reinterpret_cast
// CHECK-SAME: memref<?xi8> to memref<1x32x1x1xi8, strided<[32, 256, 1, 1], offset: ?>>
// CHECK: memref.reinterpret_cast
// CHECK-SAME: memref<?xi8> to memref<1x32x1x2xi8, strided<[64, 512, 2, 1], offset: ?>>

func.func @reinterpret_cast_with_stride(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c512 = arith.constant 512 : index
  %c256 = arith.constant 256 : index
  %c16384_i32 = arith.constant 16384 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.muli %arg9, %c32_i32 : i32
  %1 = arith.muli %arg10, %c2_i32 : i32
  %2 = arith.muli %arg8, %c8192_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.index_cast %0 : i32 to index
  %5 = arith.muli %4, %c256 : index
  %6 = arith.index_cast %arg10 : i32 to index
  %7 = arith.addi %3, %6 : index
  %8 = arith.addi %7, %5 : index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [1, 32, 1], strides: [32, 256, 1] : memref<?xi8> to memref<1x32x1xi8, strided<[32, 256, 1], offset: ?>>
  %alloc = memref.alloc() : memref<1x32x1xi8>
  memref.copy %reinterpret_cast, %alloc : memref<1x32x1xi8, strided<[32, 256, 1], offset: ?>> to memref<1x32x1xi8>
  %9 = bufferization.to_tensor %alloc restrict writable : memref<1x32x1xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [1, 32, 1], strides: [32, 256, 1] : memref<?xi8> to memref<1x32x1xi8, strided<[32, 256, 1], offset: ?>>
  %alloc_1 = memref.alloc() : memref<1x32x1xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x32x1xi8, strided<[32, 256, 1], offset: ?>> to memref<1x32x1xi8>
  %10 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x32x1xi8>
  %expanded = tensor.expand_shape %10 [[0], [1], [2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32x1xi8> into tensor<1x32x1x1xi8>
  %expanded_2 = tensor.expand_shape %9 [[0], [1], [2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32x1xi8> into tensor<1x32x1x1xi8>
  %11 = hfusion.interleave %expanded_2, %expanded : tensor<1x32x1x1xi8>, tensor<1x32x1x1xi8> -> tensor<1x32x1x2xi8>
  %collapsed = tensor.collapse_shape %11 [[0], [1], [2, 3]] : tensor<1x32x1x2xi8> into tensor<1x32x2xi8>
  %12 = arith.muli %arg8, %c16384_i32 : i32
  %13 = arith.index_cast %12 : i32 to index
  %14 = arith.muli %4, %c512 : index
  %15 = arith.index_cast %1 : i32 to index
  %16 = arith.addi %13, %14 : index
  %17 = arith.addi %16, %15 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [1, 32, 2], strides: [64, 512, 1] : memref<?xi8> to memref<1x32x2xi8, strided<[64, 512, 1], offset: ?>>
  bufferization.materialize_in_destination %collapsed in writable %reinterpret_cast_3 : (tensor<1x32x2xi8>, memref<1x32x2xi8, strided<[64, 512, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: @reinterpret_cast_with_dynamic_stride
// CHECK:       %[[C1024:.*]] = arith.constant 1024 : index
// CHECK:       %[[NEW_STRIDE:.*]] = arith.muli %arg2, %[[C1024]] : index
// CHECK:       %[[NEW_REINTERPRET:.*]] = memref.reinterpret_cast %arg0
// CHECK-SAME:    to offset: [%arg1], sizes: [1, 1024], strides: [%[[NEW_STRIDE]], %arg2]

func.func @reinterpret_cast_with_dynamic_stride(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> memref<1x1024xf16, strided<[?, ?], offset: ?>> {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1024], strides: [%arg2] : memref<?xf16> to memref<1024xf16, strided<[?], offset: ?>>
  %expand_shape = memref.expand_shape %reinterpret_cast [[0, 1]] output_shape [1, 1024] : memref<1024xf16, strided<[?], offset: ?>> into memref<1x1024xf16, strided<[?, ?], offset: ?>>
  return %expand_shape : memref<1x1024xf16, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: func.func @if_fill_kernel
// CHECK: scf.if
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<2x1x14x1x4xf32>)
// CHECK: } {hivm.unlikely_condition}
func.func @if_fill_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg10: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c56 = arith.constant 56 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.muli %arg17, %c2_i32 : i32
  %1 = arith.addi %0, %c2_i32 : i32
  %2 = arith.minsi %1, %arg11 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 14, 1], strides: [14, 1, 1] : memref<?xf32> to memref<1x14x1xf32, strided<[14, 1, 1]>>
  %alloc = memref.alloc() : memref<1x14x1xf32>
  memref.copy %reinterpret_cast, %alloc : memref<1x14x1xf32, strided<[14, 1, 1]>> to memref<1x14x1xf32>
  %3 = bufferization.to_tensor %alloc restrict writable : memref<1x14x1xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32>
  %alloc_1 = memref.alloc() : memref<1x1xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x1xf32> to memref<1x1xf32>
  %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32>
  %alloc_3 = memref.alloc() : memref<1x1xf32>
  memref.copy %reinterpret_cast_2, %alloc_3 : memref<1x1xf32> to memref<1x1xf32>
  %5 = bufferization.to_tensor %alloc_3 restrict writable : memref<1x1xf32>
  %6 = arith.index_cast %0 : i32 to index
  %7 = arith.muli %6, %c56 : index
  %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%7], sizes: [2, 14, 4], strides: [56, 4, 1] : memref<?xf32> to memref<2x14x4xf32, strided<[56, 4, 1], offset: ?>>
  %alloc_5 = memref.alloc() : memref<2x1x14x1x4xf32>
  %collapse_shape = memref.collapse_shape %alloc_5 [[0], [1, 2, 3], [4]] : memref<2x1x14x1x4xf32> into memref<2x14x4xf32>
  %8 = arith.addi %6, %c2 : index
  %9 = arith.index_cast %2 : i32 to index
  %10 = arith.maxsi %6, %9 : index
  %11 = arith.minsi %8, %10 : index
  %12 = arith.subi %11, %6 : index
  %13 = arith.cmpi slt, %12, %c2 : index
  scf.if %13 {
    linalg.fill ins(%cst : f32) outs(%collapse_shape : memref<2x14x4xf32>)
  } {hivm.unlikely_condition}
  %subview = memref.subview %reinterpret_cast_4[0, 0, 0] [%12, 14, 4] [1, 1, 1] : memref<2x14x4xf32, strided<[56, 4, 1], offset: ?>> to memref<?x14x4xf32, strided<[56, 4, 1], offset: ?>>
  %subview_6 = memref.subview %collapse_shape[0, 0, 0] [%12, 14, 4] [1, 1, 1] : memref<2x14x4xf32> to memref<?x14x4xf32, strided<[56, 4, 1]>>
  memref.copy %subview, %subview_6 : memref<?x14x4xf32, strided<[56, 4, 1], offset: ?>> to memref<?x14x4xf32, strided<[56, 4, 1]>>
  %14 = bufferization.to_tensor %alloc_5 restrict writable : memref<2x1x14x1x4xf32>
  %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 14, 1], strides: [14, 1, 1] : memref<?xf32> to memref<1x14x1xf32, strided<[14, 1, 1]>>
  %alloc_8 = memref.alloc() : memref<1x14x1xf32>
  return
}

// -----

// CHECK-LABEL: @saturate(
// CHECK: annotation.mark
// CHECK-SAME: tensor<24x32x1x1xf32>
module {
  func.func @saturate(%arg0: tensor<24x128x256x192xbf16>) -> tensor<24x32x1x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
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
    annotation.mark %reduced {overflow_mode = "saturate"} : tensor<24x32xf32>
    %expanded_0 = tensor.expand_shape %reduced [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %3 = math.absf %expanded_0: tensor<24x32x1x1xf32>
    return %3 : tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK-LABEL: @flip_expand(
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] output_shape {{\[}}2, 3, 4, 5] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
// CHECK: hfusion.flip {{.*}} : tensor<2x3x4x5xf32> flip_axis = 3 -> tensor<2x3x4x5xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
func.func @flip_expand(%arg0: tensor<2x12x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x12x5xf32>) {
    %0 = math.absf %arg0 : tensor<2x12x5xf32>
    %1 = hfusion.flip %0 : tensor<2x12x5xf32> flip_axis = 2 -> tensor<2x12x5xf32>
    %expand = tensor.expand_shape %1 [[0], [1, 2], [3]] output_shape [2, 3, 4, 5] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
    %2 = math.absf %expand: tensor<2x3x4x5xf32>
    return %2, %1 : tensor<2x3x4x5xf32>, tensor<2x12x5xf32>
}

// -----
// CHECK-LABEL: @collapse_flip(
// CHECK: hfusion.flip {{.*}} : tensor<2x3x4x5xf32> flip_axis = 3 -> tensor<2x3x4x5xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
func.func @collapse_flip(%arg0: tensor<2x3x4x5xf32>) -> (tensor<2x12x5xf32>) {
    %0 = math.absf %arg0 : tensor<2x3x4x5xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
    %1 = hfusion.flip %collapsed : tensor<2x12x5xf32> flip_axis = 2 -> tensor<2x12x5xf32>
    return %1 : tensor<2x12x5xf32>
}

// -----
// CHECK-LABEL: @collapse_flip_axis(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
// CHECK: hfusion.flip {{.*}} : tensor<2x12x5xf32> flip_axis = 1 -> tensor<2x12x5xf32>
func.func @collapse_flip_axis(%arg0: tensor<2x3x4x5xf32>) -> (tensor<2x12x5xf32>) {
    %0 = math.absf %arg0 : tensor<2x3x4x5xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
    %1 = hfusion.flip %collapsed : tensor<2x12x5xf32> flip_axis = 1 -> tensor<2x12x5xf32>
    return %1 : tensor<2x12x5xf32>
}

// CHECK-LABEL: triton_scope
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[COLLAPSED1:.*]], %[[COLLAPSED2:.*]] : tensor<64xf32>, tensor<64xf32>)
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL:.*]] : tensor<1x64xf32>)
module {
  func.func @triton_scope(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: memref<?xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %5 = linalg.matmul {input_precison = "ieee"} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<128x128xf32>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      scope.return
    } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, noinline}
    %2 = tensor.empty() : tensor<1x64xf32>
    %3 = tensor.empty() : tensor<64xf32>
    %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%expanded : tensor<1x64xf32>) -> tensor<1x64xf32>
    scope.scope : () -> () {
      %memspacecast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
      %5 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
      %6 = scf.for %arg3 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg4 = %4) -> (tensor<1x64xf32>)  : i32 {
        %8 = arith.index_cast %arg3 : i32 to index
        %extracted_slice = tensor.extract_slice %5[%8, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
        %collapsed_1 = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
        %collapsed_2 = tensor.collapse_shape %arg4 [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
        %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed_1, %collapsed_2 : tensor<64xf32>, tensor<64xf32>) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
        %expanded_3 = tensor.expand_shape %9 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
        scf.yield %expanded_3 : tensor<1x64xf32>
      }
      %7 = scope.scope : () -> tensor<1x64xf32> {
        %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%6 : tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
        scope.return %8 : tensor<1x64xf32>
      } {noinline, outline = true, vector_mode = "simd"}
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
      %collapsed = tensor.collapse_shape %7 [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
      bufferization.materialize_in_destination %collapsed in writable %reinterpret_cast : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
      scope.return
    } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, noinline}
    return
  }
}

// -----
// CHECK-LABEL: @collapse_user_different_region(
// CHECK: scf.for
// CHECK: %[[VAL5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[VAL5]]
// CHECK-NEXT: scf.yield %[[COLLAPSED]] : tensor<1x608xf32>
func.func @collapse_user_different_region(%arg0: tensor<38x16x16xf32>, %arg1: tensor<1x608xf32>) -> (tensor<1x608xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x608xf32>
    %1 = tensor.empty() : tensor<16x38x16xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<38x16x16xf32>) outs(%1 : tensor<16x38x16xf32>) permutation = [1, 0, 2]
    %collapsed = tensor.collapse_shape %transposed [[0], [1, 2]] : tensor<16x38x16xf32> into tensor<16x608xf32>
    %45 = scf.for %arg15 = %c1_i32 to %c16_i32 step %c1_i32 iter_args(%arg16 = %arg1) -> (tensor<1x608xf32>)  : i32 {
      %57 = arith.index_cast %arg15 : i32 to index
      %extracted_slice_11 = tensor.extract_slice %collapsed[%57, 0] [1, 608] [608, 1] : tensor<16x608xf32> to tensor<1x608xf32>
      %58 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice_11, %arg16 : tensor<1x608xf32>, tensor<1x608xf32>) outs(%0 : tensor<1x608xf32>) -> tensor<1x608xf32>
      scf.yield %58 : tensor<1x608xf32>
    }
    %46 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%45 : tensor<1x608xf32>) outs(%0 : tensor<1x608xf32>) -> tensor<1x608xf32>
    return %46 : tensor<1x608xf32>
}

// -----
// CHECK-LABEL: sum_512_tensors
// CHECK: tensor.extract_slice 
// CHECK-SAME: tensor<64x8x2x2xf32> to tensor<64x8x2x1xf32>
func.func @sum_512_tensors(%arg0: tensor<64x32xf32>) -> tensor<512xf32> {
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [64, 16, 2] : tensor<64x32xf32> into tensor<64x16x2xf32>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] : tensor<64x16x2xf32> into tensor<1024x2xf32>
  %extracted_slice = tensor.extract_slice %collapsed[0, 0] [1024, 1] [1, 2] : tensor<1024x2xf32> to tensor<1024x1xf32>
  %collapsed_0 = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1024x1xf32> into tensor<1024xf32>
  %extracted_slice_1 = tensor.extract_slice %collapsed[0, 1] [1024, 1] [1, 2] : tensor<1024x2xf32> to tensor<1024x1xf32>
  %collapsed_2 = tensor.collapse_shape %extracted_slice_1 [[0, 1]] : tensor<1024x1xf32> into tensor<1024xf32>
  %expanded_3 = tensor.expand_shape %collapsed_0 [[0, 1]] output_shape [512, 2] : tensor<1024xf32> into tensor<512x2xf32>
  %extracted_slice_4 = tensor.extract_slice %expanded_3[0, 0] [512, 1] [1, 2] : tensor<512x2xf32> to tensor<512x1xf32>
  %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1]] : tensor<512x1xf32> into tensor<512xf32>
  %extracted_slice_6 = tensor.extract_slice %expanded_3[0, 1] [512, 1] [1, 2] : tensor<512x2xf32> to tensor<512x1xf32>
  %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1]] : tensor<512x1xf32> into tensor<512xf32>
  %expanded_8 = tensor.expand_shape %collapsed_2 [[0, 1]] output_shape [512, 2] : tensor<1024xf32> into tensor<512x2xf32>
  %extracted_slice_9 = tensor.extract_slice %expanded_8[0, 0] [512, 1] [1, 2] : tensor<512x2xf32> to tensor<512x1xf32>
  %collapsed_10 = tensor.collapse_shape %extracted_slice_9 [[0, 1]] : tensor<512x1xf32> into tensor<512xf32>
  %extracted_slice_11 = tensor.extract_slice %expanded_8[0, 1] [512, 1] [1, 2] : tensor<512x2xf32> to tensor<512x1xf32>
  %collapsed_12 = tensor.collapse_shape %extracted_slice_11 [[0, 1]] : tensor<512x1xf32> into tensor<512xf32>
  %0 = tensor.empty() : tensor<512xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapsed_5, %collapsed_7 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %collapsed_10 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %collapsed_12 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  return %3 : tensor<512xf32>
}

// -----
// CHECK-LABEL: extract_from_collapsed
// CHECK: tensor.extract %
// CHECK-SAME: : tensor<64x16x1xf32>
func.func @extract_from_collapsed(%arg0: tensor<64x32xf32>) -> tensor<15xf32> {
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [64, 16, 2] : tensor<64x32xf32> into tensor<64x16x2xf32>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] : tensor<64x16x2xf32> into tensor<1024x2xf32>
  %extracted_slice = tensor.extract_slice %collapsed[0, 0] [1024, 1] [1, 2] : tensor<1024x2xf32> to tensor<1024x1xf32>
  %collapsed_target = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1024x1xf32> into tensor<1024xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %v0 = tensor.extract %collapsed_target[%c0] : tensor<1024xf32>
  %v1 = tensor.extract %collapsed_target[%c1] : tensor<1024xf32>
  %v2 = tensor.extract %collapsed_target[%c2] : tensor<1024xf32>
  %v3 = tensor.extract %collapsed_target[%c3] : tensor<1024xf32>
  %v4 = tensor.extract %collapsed_target[%c4] : tensor<1024xf32>
  %v5 = tensor.extract %collapsed_target[%c5] : tensor<1024xf32>
  %v6 = tensor.extract %collapsed_target[%c6] : tensor<1024xf32>
  %v7 = tensor.extract %collapsed_target[%c7] : tensor<1024xf32>
  %v8 = tensor.extract %collapsed_target[%c8] : tensor<1024xf32>
  %v9 = tensor.extract %collapsed_target[%c9] : tensor<1024xf32>
  %v10 = tensor.extract %collapsed_target[%c10] : tensor<1024xf32>
  %v11 = tensor.extract %collapsed_target[%c11] : tensor<1024xf32>
  %v12 = tensor.extract %collapsed_target[%c12] : tensor<1024xf32>
  %v13 = tensor.extract %collapsed_target[%c13] : tensor<1024xf32>
  %v14 = tensor.extract %collapsed_target[%c14] : tensor<1024xf32>
  %result = tensor.from_elements %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7, %v8, %v9, %v10, %v11, %v12, %v13, %v14 : tensor<15xf32>

  return %result : tensor<15xf32>
}
