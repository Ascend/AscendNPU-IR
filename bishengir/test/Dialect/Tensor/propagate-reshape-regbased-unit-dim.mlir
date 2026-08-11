// RUN: bishengir-opt %s -propagate-reshape="for-regbased=false" -split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: bishengir-opt %s -propagate-reshape="for-regbased=true max-unit-dims-for-propagation=6" -split-input-file | FileCheck %s --check-prefix=REGBASE

// DEFAULT-LABEL: func.func @scalar_reduce_unit_expand(
// DEFAULT: %[[EXPANDED_INIT:.*]] = tensor.expand_shape %arg1 [] output_shape [1]
// DEFAULT: %[[EXPANDED_SRC:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 4]
// DEFAULT: linalg.reduce ins(%[[EXPANDED_SRC]] : tensor<1x4xi32>) outs(%[[EXPANDED_INIT]] : tensor<1xi32>) dimensions = [1]
// DEFAULT-NOT: tensor.expand_shape
// REGBASE-LABEL: func.func @scalar_reduce_unit_expand(
// REGBASE: %[[REDUCED:.*]] = linalg.reduce ins(%arg0 : tensor<4xi32>) outs(%arg1 : tensor<i32>) dimensions = [0]
// REGBASE: tensor.expand_shape %[[REDUCED]] [] output_shape [1]
func.func @scalar_reduce_unit_expand(
    %src: tensor<4xi32>, %init: tensor<i32>) -> tensor<1xi32> {
  %reduced = linalg.reduce
      ins(%src : tensor<4xi32>) outs(%init : tensor<i32>) dimensions = [0]
      (%in: i32, %acc: i32) {
        %sum = arith.addi %in, %acc : i32
        linalg.yield %sum : i32
      }
  %expanded = tensor.expand_shape %reduced [] output_shape [1] :
      tensor<i32> into tensor<1xi32>
  %out = tensor.empty() : tensor<1xi32>
  %copied = linalg.copy ins(%expanded : tensor<1xi32>)
      outs(%out : tensor<1xi32>) -> tensor<1xi32>
  return %copied : tensor<1xi32>
}

// -----

// DEFAULT-LABEL: func.func @extract_slice_unit_expand(
// DEFAULT: %[[EXPANDED_SRC:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [2, 1, 4]
// DEFAULT: tensor.extract_slice %[[EXPANDED_SRC]][0, 0, 0] [1, 1, 4] [1, 1, 1]
// REGBASE-LABEL: func.func @extract_slice_unit_expand(
// REGBASE: %[[SLICE:.*]] = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1]
// REGBASE: tensor.expand_shape %[[SLICE]] {{\[\[}}0], [1, 2]] output_shape [1, 1, 4]
func.func @extract_slice_unit_expand(
    %src: tensor<2x4xi32>) -> tensor<1x1x4xi32> {
  %slice = tensor.extract_slice %src[0, 0] [1, 4] [1, 1] :
      tensor<2x4xi32> to tensor<1x4xi32>
  %expanded = tensor.expand_shape %slice [[0], [1, 2]]
      output_shape [1, 1, 4] :
      tensor<1x4xi32> into tensor<1x1x4xi32>
  %out = tensor.empty() : tensor<1x1x4xi32>
  %copied = linalg.copy ins(%expanded : tensor<1x1x4xi32>)
      outs(%out : tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
  return %copied : tensor<1x1x4xi32>
}

// -----

// DEFAULT-LABEL: func.func @collapse_unit_insert_slice(
// DEFAULT: %[[EXPANDED_DEST:.*]] = tensor.expand_shape %arg2 {{\[\[}}0, 1]] output_shape [4, 1]
// DEFAULT: %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into %[[EXPANDED_DEST]][1, 0] [2, 1] [1, 1]
// DEFAULT: tensor.collapse_shape %[[INSERTED]] {{\[\[}}0, 1]]
// REGBASE-LABEL: func.func @collapse_unit_insert_slice(
// REGBASE: %[[COLLAPSED:.*]] = tensor.collapse_shape {{.*}} {{\[\[}}0, 1]]
// REGBASE: tensor.insert_slice %[[COLLAPSED]] into %arg2[1] [2] [1]
func.func @collapse_unit_insert_slice(
    %src: tensor<2x1xi32>, %init: tensor<2x1xi32>,
    %dest: tensor<4xi32>) -> tensor<4xi32> {
  %copied = linalg.copy ins(%src : tensor<2x1xi32>)
      outs(%init : tensor<2x1xi32>) -> tensor<2x1xi32>
  %collapsed = tensor.collapse_shape %copied [[0, 1]] :
      tensor<2x1xi32> into tensor<2xi32>
  %inserted = tensor.insert_slice %collapsed into %dest[1] [2] [1] :
      tensor<2xi32> into tensor<4xi32>
  return %inserted : tensor<4xi32>
}

// -----

// DEFAULT-LABEL: func.func @triton_cumprod_4D(
// DEFAULT: %[[VIEW:.*]] = memref.reinterpret_cast %arg2
// DEFAULT-SAME: to memref<3x9x8xi8, strided<[72, 8, 1]>>
// DEFAULT: %[[ALLOC:.*]] = memref.alloc() : memref<3x9x8xi8>
// DEFAULT: memref.copy %[[VIEW]], %[[ALLOC]]
// DEFAULT: %[[EXPANDED:.*]] = memref.expand_shape %[[ALLOC]]
// DEFAULT-SAME: into memref<3x9x8x1xi8>
// DEFAULT: %[[TENSOR:.*]] = bufferization.to_tensor %[[EXPANDED]]
// DEFAULT: %[[RESULT:.*]] = hfusion.cumprod %[[TENSOR]]
// DEFAULT: %[[DEST:.*]] = memref.reinterpret_cast %arg3
// DEFAULT-SAME: to memref<3x9x8xi8, strided<[72, 8, 1]>>
// DEFAULT: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[RESULT]]
// DEFAULT: bufferization.materialize_in_destination %[[COLLAPSED]]
// DEFAULT-SAME: in writable %[[DEST]]
// REGBASE-LABEL: func.func @triton_cumprod_4D(
// REGBASE: %[[VIEW:.*]] = memref.reinterpret_cast %arg2
// REGBASE-SAME: to memref<3x9x8x1xi8, strided<[72, 8, 1, 1]>>
// REGBASE: %[[ALLOC:.*]] = memref.alloc() : memref<3x9x8x1xi8>
// REGBASE: memref.copy %[[VIEW]], %[[ALLOC]]
// REGBASE-NOT: expand_shape
// REGBASE: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]]
// REGBASE: %[[RESULT:.*]] = hfusion.cumprod %[[TENSOR]]
// REGBASE: %[[DEST:.*]] = memref.reinterpret_cast %arg3
// REGBASE-SAME: to memref<3x9x8x1xi8, strided<[72, 8, 1, 1]>>
// REGBASE-NOT: collapse_shape
// REGBASE: bufferization.materialize_in_destination %[[RESULT]]
// REGBASE-SAME: in writable %[[DEST]]
func.func @triton_cumprod_4D(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg2: memref<?xi8> {tt.divisibility = 16 : i32,
                         tt.tensor_kind = 0 : i32},
    %arg3: memref<?xi8> {tt.divisibility = 16 : i32,
                         tt.tensor_kind = 1 : i32},
    %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32)
    attributes {
      SyncBlockLockArgIdx = 0 : i64,
      WorkspaceArgIdx = 1 : i64,
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      mix_mode = "aiv",
      parallel_mode = "simd"
    } {
  %view = memref.reinterpret_cast %arg2 to offset: [0],
      sizes: [3, 9, 8], strides: [72, 8, 1] :
      memref<?xi8> to memref<3x9x8xi8, strided<[72, 8, 1]>>
  %alloc = memref.alloc() : memref<3x9x8xi8>
  memref.copy %view, %alloc :
      memref<3x9x8xi8, strided<[72, 8, 1]>> to memref<3x9x8xi8>
  %tensor = bufferization.to_tensor %alloc restrict writable :
      memref<3x9x8xi8>
  %expanded = tensor.expand_shape %tensor [[0], [1], [2, 3]]
      output_shape [3, 9, 8, 1] :
      tensor<3x9x8xi8> into tensor<3x9x8x1xi8>
  %result = hfusion.cumprod %expanded :
      tensor<3x9x8x1xi8> cum_dims = [0] reverse = false
      -> tensor<3x9x8x1xi8>
  %dest = memref.reinterpret_cast %arg3 to offset: [0],
      sizes: [3, 9, 8], strides: [72, 8, 1] :
      memref<?xi8> to memref<3x9x8xi8, strided<[72, 8, 1]>>
  %collapsed = tensor.collapse_shape %result [[0], [1], [2, 3]] :
      tensor<3x9x8x1xi8> into tensor<3x9x8xi8>
  bufferization.materialize_in_destination %collapsed in writable %dest :
      (tensor<3x9x8xi8>,
       memref<3x9x8xi8, strided<[72, 8, 1]>>) -> ()
  return
}

// -----

// REGBASE-LABEL: func.func @stop_expand_above_unit_dim_threshold(
// REGBASE: %[[FILLED:.*]] = linalg.fill {{.*}} -> tensor<4xf32>
// REGBASE: tensor.expand_shape %[[FILLED]] {{\[\[}}0, 1, 2, 3, 4, 5, 6, 7, 8]] output_shape [2, 1, 1, 1, 1, 1, 1, 1, 2]
func.func @stop_expand_above_unit_dim_threshold() -> tensor<2x1x1x1x1x1x1x1x2xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %expanded = tensor.expand_shape %filled [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
      output_shape [2, 1, 1, 1, 1, 1, 1, 1, 2]
      : tensor<4xf32> into tensor<2x1x1x1x1x1x1x1x2xf32>
  return %expanded : tensor<2x1x1x1x1x1x1x1x2xf32>
}

// -----

// REGBASE-LABEL: func.func @stop_collapse_above_unit_dim_threshold(
// REGBASE: %[[COLLAPSED:.*]] = tensor.collapse_shape {{.*}} : tensor<2x1x1x1x1x1x1x1x2xf32> into tensor<2x2xf32>
// REGBASE: linalg.elemwise_unary {{.*}} ins(%[[COLLAPSED]] : tensor<2x2xf32>)
func.func @stop_collapse_above_unit_dim_threshold() -> tensor<2x2xf32> {
  %cst = arith.constant 0.0 : f32
  %src_empty = tensor.empty() : tensor<2x1x1x1x1x1x1x1x2xf32>
  %src = linalg.fill ins(%cst : f32) outs(%src_empty : tensor<2x1x1x1x1x1x1x1x2xf32>) -> tensor<2x1x1x1x1x1x1x1x2xf32>
  %collapsed = tensor.collapse_shape %src [[0, 1, 2, 3, 4, 5, 6, 7], [8]]
      : tensor<2x1x1x1x1x1x1x1x2xf32> into tensor<2x2xf32>
  %dst = tensor.empty() : tensor<2x2xf32>
  %abs = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%collapsed : tensor<2x2xf32>) outs(%dst : tensor<2x2xf32>)
      -> tensor<2x2xf32>
  return %abs : tensor<2x2xf32>
}
