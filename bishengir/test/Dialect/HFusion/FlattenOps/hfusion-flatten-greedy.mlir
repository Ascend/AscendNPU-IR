// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=greedy" -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fill(
// CHECK-SAME:                  %[[ARG0:.*]]: f32,
// CHECK-SAME:                  %[[ARG1:.*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.fill ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : memref<224xf32>)
func.func @fill(%cst: f32, %arg: memref<32x7xf32>) {
    linalg.fill ins(%cst: f32) outs(%arg: memref<32x7xf32>)
    return
}

// -----

// CHECK-LABEL: func.func @fill_tensor(
// CHECK-SAME:                         %[[ARG0:.*]]: f32,
// CHECK-SAME:                         %[[ARG1:.*]]: tensor<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_RESULT:.*]] = linalg.fill ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : tensor<224xf32>)
// CHECK-NEXT:    %[[RESULT:.*]] = tensor.expand_shape %[[FLATTENED_RESULT]] {{\[}}[0, 1]]
func.func @fill_tensor(%cst: f32, %arg: tensor<32x7xf32>) -> tensor<32x7xf32> {
    %0 = linalg.fill ins(%cst: f32) outs(%arg: tensor<32x7xf32>) ->  tensor<32x7xf32>
    return %0 :  tensor<32x7xf32>
}

// -----

// CHECK-LABEL: func.func @map(
// CHECK-SAME:                 %[[ARG0:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG1:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG2:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.map { arith.addf } ins(%[[FLATTENED_0]], %[[FLATTENED_1]] : memref<224xf32>, memref<224xf32>) outs(%[[FLATTENED_2]] : memref<224xf32>)
func.func @map(%arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) {
    linalg.map {arith.addf} ins(%arg0, %arg1: memref<32x7xf32>, memref<32x7xf32>) outs(%arg2: memref<32x7xf32>)
    return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @generic
// CHECK-SAME:                 %[[ARG0:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG1:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG2:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[FLATTENED_0]], %[[FLATTENED_1]] : memref<224xf32>, memref<224xf32>) outs(%[[FLATTENED_2]] : memref<224xf32>)
// CHECK-NEXT:       ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32)
// CHECK-NEXT:         %[[SUM:.*]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT:         linalg.yield %[[SUM]]
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic( %arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) {
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1: memref<32x7xf32>, memref<32x7xf32>) outs(%arg2: memref<32x7xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
            %0 = arith.addf %a, %b : f32
            linalg.yield %0 : f32
    }
    return
}

// -----

// CHECK-LABEL: func.func @hfusion_elemwise(
// CHECK-SAME:                  %[[ARG0:.*]]: memref<32x7xf32>,
// CHECK-SAME:                  %[[ARG1:.*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    hfusion.elemwise_unary {{.*}} ins(%[[FLATTENED_0]] : memref<224xf32>) outs(%[[FLATTENED_1]] : memref<224xf32>)
func.func @hfusion_elemwise(%arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>) {
    hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%arg0: memref<32x7xf32>) outs(%arg1: memref<32x7xf32>)
    return
}

// -----
 
func.func @dot_scale_kernel_2D(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf8E5M2> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf8E5M2> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf8E5M2> to memref<64x64xf8E5M2, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<64x64xf8E5M2>
  memref.copy %reinterpret_cast, %alloc : memref<64x64xf8E5M2, strided<[64, 1]>> to memref<64x64xf8E5M2>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf8E5M2>
  // CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8> to memref<64x64xi8, strided<[64, 1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8> to memref<64x64xi8, strided<[64, 1]>>
  %alloc_1 = memref.alloc() : memref<64x64xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x64xi8, strided<[64, 1]>> to memref<64x64xi8>
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x64xi8>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf8E5M2> to memref<64x64xf8E5M2, strided<[64, 1]>>
  %alloc_3 = memref.alloc() : memref<64x64xf8E5M2>
  memref.copy %reinterpret_cast_2, %alloc_3 : memref<64x64xf8E5M2, strided<[64, 1]>> to memref<64x64xf8E5M2>
  %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<64x64xf8E5M2>
  // CHECK: %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8> to memref<64x64xi8, strided<[64, 1]>>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8> to memref<64x64xi8, strided<[64, 1]>>
  %alloc_5 = memref.alloc() : memref<64x64xi8>
  memref.copy %reinterpret_cast_4, %alloc_5 : memref<64x64xi8, strided<[64, 1]>> to memref<64x64xi8>
  %5 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xi8>
  %6 = hfusion.matmul_mx ins(%2, %4, %3, %5 : tensor<64x64xf8E5M2>, tensor<64x64xf8E5M2>, tensor<64x64xi8>, tensor<64x64xi8>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %reinterpret_cast_6 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
  bufferization.materialize_in_destination %6 in writable %reinterpret_cast_6 : (tensor<64x64xf32>, memref<64x64xf32, strided<[64, 1]>>) -> ()
  return
}