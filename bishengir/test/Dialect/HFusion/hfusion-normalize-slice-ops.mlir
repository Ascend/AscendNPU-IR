// RUN: bishengir-opt --hfusion-normalize-slice-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt --hfusion-normalize-slice-ops="skip-aligned-slice=true" -split-input-file %s | FileCheck %s -check-prefix=CHECK-TRITON

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_2:.*]] = hfusion.interleave %[[VAL_0]], %[[VAL_1]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.insert_slice
func.func @test_interleave() -> tensor<4x2x128xf16> {
  %0 = tensor.empty() : tensor<4x2x64xf16>
  %1 = tensor.empty() : tensor<4x2x64xf16>
  %2 = tensor.empty() : tensor<4x2x128xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [4, 2, 64] [1, 1, 2] : tensor<4x2x64xf16> into tensor<4x2x128xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 1] [4, 2, 64] [1, 1, 2] : tensor<4x2x64xf16> into tensor<4x2x128xf16>
  return %4 : tensor<4x2x128xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK-DAG: %[[EXPAND_0:.*]] = tensor.expand_shape %[[VAL_0]] {{\[}}[0], [1, 2]] output_shape [4, 2, 1] : tensor<4x2xf16> into tensor<4x2x1xf16>
// CHECK-DAG: %[[EXPAND_1:.*]] = tensor.expand_shape %[[VAL_1]] {{\[}}[0], [1, 2]] output_shape [4, 2, 1] : tensor<4x2xf16> into tensor<4x2x1xf16>
// CHECK: hfusion.interleave %[[EXPAND_0]], %[[EXPAND_1]] : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.insert_slice
func.func @test_interleave_reduced_rank() -> tensor<4x2x2xf16> {
  %0 = tensor.empty() : tensor<4x2xf16>
  %1 = tensor.empty() : tensor<4x2xf16>
  %2 = tensor.empty() : tensor<4x2x2xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [4, 2, 1] [1, 1, 2] : tensor<4x2xf16> into tensor<4x2x2xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 1] [4, 2, 1] [1, 1, 2] : tensor<4x2xf16> into tensor<4x2x2xf16>
  return %4 : tensor<4x2x2xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty(%[[DIM:.*]]) : tensor<?x2xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty(%[[DIM]]) : tensor<?x2xf16>
// CHECK-DAG: %[[EXPAND_0:.*]] = tensor.expand_shape %[[VAL_0]] {{\[}}[0], [1, 2]] output_shape [%[[DIM_0:.*]], 2, 1] : tensor<?x2xf16> into tensor<?x2x1xf16>
// CHECK-DAG: %[[EXPAND_1:.*]] = tensor.expand_shape %[[VAL_1]] {{\[}}[0], [1, 2]] output_shape [%[[DIM_1:.*]], 2, 1] : tensor<?x2xf16> into tensor<?x2x1xf16>
// CHECK: hfusion.interleave %[[EXPAND_0]], %[[EXPAND_1]] : tensor<?x2x1xf16>, tensor<?x2x1xf16> -> tensor<?x2x2xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.insert_slice
func.func @test_interleave_reduced_rank(%arg0: tensor<?x2xf16>) -> tensor<?x2x2xf16> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xf16>
  %0 = tensor.empty(%dim) : tensor<?x2xf16>
  %1 = tensor.empty(%dim) : tensor<?x2xf16>
  %2 = tensor.empty(%dim) : tensor<?x2x2xf16>
  %dim_0 = tensor.dim %0, %c0 : tensor<?x2xf16>
  %dim_1 = tensor.dim %1, %c0 : tensor<?x2xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [%dim_0, 2, 1] [1, 1, 2] : tensor<?x2xf16> into tensor<?x2x2xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 1] [%dim_1 , 2, 1] [1, 1, 2] : tensor<?x2xf16> into tensor<?x2x2xf16>
  return %4 : tensor<?x2x2xf16>
}

// -----

// CHECK-NOT: hfusion.interleave
// CHECK: tensor.insert_slice
func.func @test_not_interleave() -> tensor<4x32x64xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = tensor.empty() : tensor<4x32xf16>
  %2 = tensor.empty() : tensor<4x32x64xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [4, 32, 1] [1, 1, 2] : tensor<4x32xf16> into tensor<4x32x64xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 2] [4, 32, 1] [1, 1, 2] : tensor<4x32xf16> into tensor<4x32x64xf16>
  return %4 : tensor<4x32x64xf16>
}

// -----

// CHECK-LABEL: func.func @test_interleave_i64_non_regbase
// CHECK: hfusion.interleave
// CHECK-NOT: tensor.insert_slice
func.func @test_interleave_i64_non_regbase() -> tensor<4x2xi64> {
  %0 = tensor.empty() : tensor<4xi64>
  %1 = tensor.empty() : tensor<4xi64>
  %2 = tensor.empty() : tensor<4x2xi64>
  %3 = tensor.insert_slice %0 into %2[0, 0] [4, 1] [1, 2] : tensor<4xi64> into tensor<4x2xi64>
  %4 = tensor.insert_slice %1 into %3[0, 1] [4, 1] [1, 2] : tensor<4xi64> into tensor<4x2xi64>
  return %4 : tensor<4x2xi64>
}

// -----

// CHECK-LABEL: func.func @test_interleave_i64_regbase
// CHECK: hfusion.interleave
// CHECK-NOT: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @test_interleave_i64_regbase() -> tensor<4x2xi64> {
    %0 = tensor.empty() : tensor<4xi64>
    %1 = tensor.empty() : tensor<4xi64>
    %2 = tensor.empty() : tensor<4x2xi64>
    %3 = tensor.insert_slice %0 into %2[0, 0] [4, 1] [1, 2] : tensor<4xi64> into tensor<4x2xi64>
    %4 = tensor.insert_slice %1 into %3[0, 1] [4, 1] [1, 2] : tensor<4xi64> into tensor<4x2xi64>
    return %4 : tensor<4x2xi64>
  }
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x128xf16>
// CHECK: %[[VAL_1:.*]] = hfusion.deinterleave %[[VAL_0]] channel<0> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave() -> tensor<4x2x64xf16> {
  %0 = tensor.empty() : tensor<4x2x128xf16>
  %1 = tensor.extract_slice %0[0, 0, 0] [4, 2, 64] [1, 1, 2] : tensor<4x2x128xf16> to tensor<4x2x64xf16>
  return %1 : tensor<4x2x64xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x128xf16>
// CHECK: %[[VAL_1:.*]] = hfusion.deinterleave %[[VAL_0]] channel<1> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave() -> tensor<4x2x64xf16> {
  %0 = tensor.empty() : tensor<4x2x128xf16>
  %1 = tensor.extract_slice %0[0, 0, 1] [4, 2, 64] [1, 1, 2] : tensor<4x2x128xf16> to tensor<4x2x64xf16>
  return %1 : tensor<4x2x64xf16>
}

// -----

// CHECK-DAG: %[[INPUT:.*]] = tensor.empty() : tensor<4x2x2xf16>
// CHECK: %[[OUTPUT:.*]] = hfusion.deinterleave %[[INPUT]] channel<1> : tensor<4x2x2xf16> -> tensor<4x2x1xf16>
// CHECK: %[[COLLPASE:.*]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0], [1, 2]] : tensor<4x2x1xf16> into tensor<4x2xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave_reduced_rank() -> tensor<4x2xf16> {
  %0 = tensor.empty() : tensor<4x2x2xf16>
  %1 = tensor.extract_slice %0[0, 0, 1] [4, 2, 1] [1, 1, 2] : tensor<4x2x2xf16> to tensor<4x2xf16>
  return %1 : tensor<4x2xf16>
}

// -----

// CHECK: %[[OUTPUT:.*]] = hfusion.deinterleave %arg0 channel<1> : tensor<?x?x2xf16> -> tensor<?x?x1xf16>
// CHECK: %[[COLLPASE:.*]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0], [1, 2]] : tensor<?x?x1xf16> into tensor<?x?xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave_dynamic(%arg0: tensor<?x?x2xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x?x2xf16>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x2xf16>

  %0 = tensor.extract_slice %arg0[0, 0, 1] [%dim_0, %dim_1, 1] [1, 1, 2] : tensor<?x?x2xf16> to tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// CHECK-NOT: hfusion.deinterleave
// CHECK: tensor.extract_slice
func.func @test_not_deinterleave() -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = tensor.extract_slice %0[0, 0] [4, 1] [1, 2] : tensor<4x32xf16> to tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK: hfusion.deinterleave
// CHECK: tensor.collapse_shape
func.func @test_deinterleave_with_reduce_rank(%arg0: tensor<1x32x2xbf16>) -> tensor<1x32xbf16> {
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, 32, 1] [1, 1, 1] : tensor<1x32x2xbf16> to tensor<1x32xbf16>
  return %0 : tensor<1x32xbf16>
}


// -----

// CHECK-NOT: hfusion.deinterleave
// CHECK: tensor.extract_slice
func.func @test_not_deinterleave_with_dynamic_last_dim_offset(%arg0: tensor<1x32x2xbf16>, %offset: index) -> tensor<1x32x1xbf16> {
  %0 = tensor.extract_slice %arg0[0, 0, %offset] [1, 32, 1] [1, 1, 1] : tensor<1x32x2xbf16> to tensor<1x32x1xbf16>
  return %0 : tensor<1x32x1xbf16>
}

// -----

// CHECK-LABEL: func.func @test_deinterleave_i64_non_regbase
// CHECK: hfusion.deinterleave
// CHECK-NOT: tensor.extract_slice
func.func @test_deinterleave_i64_non_regbase(%arg0: tensor<4x2xi64>) -> tensor<4xi64> {
  %0 = tensor.extract_slice %arg0[0, 1] [4, 1] [1, 2] : tensor<4x2xi64> to tensor<4xi64>
  return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func.func @test_deinterleave_i64_regbase
// CHECK: hfusion.deinterleave
// CHECK-NOT: tensor.extract_slice
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @test_deinterleave_i64_regbase(%arg0: tensor<4x2xi64>) -> tensor<4xi64> {
    %0 = tensor.extract_slice %arg0[0, 1] [4, 1] [1, 2] : tensor<4x2xi64> to tensor<4xi64>
    return %0 : tensor<4xi64>
  }
}

// -----

// CHECK-LABEL: @test_normalize_interleave_i1
// CHECK :  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_interleave_i1(%arg0 : tensor<4x2x32xi1>, %arg1 : tensor<4x2x32xi1>) -> tensor<4x2x64xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi1>, tensor<4x2x32xi1> -> tensor<4x2x64xi1>
  return %1 : tensor<4x2x64xi1>
}

// -----

// CHECK-LABEL: func.func @normalize_insert_slice_to_concat_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<2x2xf32>, %[[arg1:.*]]: tensor<2xf32>
// CHECK: %[[expanded:.*]] = tensor.expand_shape
// CHECK: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]{{\[}}1, 0] {{\[}}1, 2] {{\[}}1, 1]
// CHECK: tensor.concat dim(0) %[[expanded]], %[[extracted]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
func.func @normalize_insert_slice_to_concat_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2xf32>) -> tensor<2x2xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 2] : tensor<2xf32> into tensor<1x2xf32>
  %inserted_slice = tensor.insert_slice %expanded into %arg0[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
  return %inserted_slice : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @normalize_insert_slice_to_concat_1(
// CHECK-SAME: %[[arg0:.*]]: tensor<2048x32x192xbf16>, %[[arg1:.*]]: tensor<2048x32x128xbf16>
// CHECK: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]{{\[}}0, 0, 128] {{\[}}2048, 32, 64] {{\[}}1, 1, 1]
// CHECK: tensor.concat dim(2) %[[arg1]], %[[extracted]] : (tensor<2048x32x128xbf16>, tensor<2048x32x64xbf16>) -> tensor<2048x32x192xbf16>
func.func @normalize_insert_slice_to_concat_1(%arg0: tensor<2048x32x192xbf16>, %arg1: tensor<2048x32x128xbf16>) -> tensor<2048x32x192xbf16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %inserted_slice = tensor.insert_slice %arg1 into %arg0[0, 0, 0] [2048, 32, 128] [1, 1, 1] : tensor<2048x32x128xbf16> into tensor<2048x32x192xbf16>
  return %inserted_slice : tensor<2048x32x192xbf16>
}

// -----

// CHECK-LABEL: func.func @normalize_insert_slice_to_concat_2
// CHECK-SAME: %[[arg0:.*]]: tensor<2048x32x192xbf16>, %[[arg1:.*]]: tensor<2048x32x64xbf16>
// CHECK: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]{{\[}}0, 0, 0] {{\[}}2048, 32, 128] {{\[}}1, 1, 1]
// CHECK: tensor.concat dim(2) %[[extracted]], %[[arg1]]
func.func @normalize_insert_slice_to_concat_2(%arg0: tensor<2048x32x192xbf16>, %arg1: tensor<2048x32x64xbf16>) -> tensor<2048x32x192xbf16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %inserted_slice = tensor.insert_slice %arg1 into %arg0[0, 0, 128] [2048, 32, 64] [1, 1, 1] : tensor<2048x32x64xbf16> into tensor<2048x32x192xbf16>
  return %inserted_slice : tensor<2048x32x192xbf16>
}

// -----
// CHECK-LABEL: func.func @normalize_dynamic_offset_insert_slice_0
// CHECK-SAME: %[[arg0:.*]]: tensor<1x37xf32>, %[[arg1:.*]]: tensor<432x37xf32>
// CHECK: %[[offset:.*]] = arith.index_cast
// CHECK: %[[extract:.*]] = tensor.extract_slice %[[arg1]][0, 0] [%[[offset]], 37]
// CHECK: %[[extract0:.*]] = tensor.extract_slice %[[arg1]]
// CHECK: tensor.concat dim(0) %[[extract]], %[[arg0]], %[[extract0]]
func.func @normalize_dynamic_offset_insert_slice_0(%arg0: tensor<1x37xf32>, %arg1: tensor<432x37xf32>, %arg2: i32) -> tensor<432x37xf32> {
  %offset = arith.index_cast %arg2 : i32 to index
  %inserted_slice = tensor.insert_slice %arg0 into %arg1[%offset, 0] [1, 37] [1, 1] : tensor<1x37xf32> into tensor<432x37xf32>
  return %inserted_slice : tensor<432x37xf32>
}

// -----

// CHECK-LABEL: func.func @not_normalize_insert_slice_to_concat_0
// CHECK-NOT: tensor.concat
func.func @not_normalize_insert_slice_to_concat_0(%arg0: tensor<8x4x64xf16>, %arg1: tensor<8x4x64xf16>) -> tensor<8x4x128xf16> {
  %0 = tensor.empty() : tensor<8x4x64xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8x4x64xf16>, tensor<8x4x64xf16>) outs(%0 : tensor<8x4x64xf16>) -> tensor<8x4x64xf16>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %1 : tensor<8x4x64xf16>, tensor<8x4x64xf16>) outs(%0 : tensor<8x4x64xf16>) -> tensor<8x4x64xf16>
  %3 = tensor.empty() : tensor<8x4x128xf16>
  %inserted_slice = tensor.insert_slice %1 into %3[0, 0, 0] [8, 4, 64] [1, 1, 2] : tensor<8x4x64xf16> into tensor<8x4x128xf16>
  %inserted_slice_0 = tensor.insert_slice %2 into %inserted_slice[0, 0, 1] [8, 4, 64] [1, 1, 2] : tensor<8x4x64xf16> into tensor<8x4x128xf16>
  return %inserted_slice_0 : tensor<8x4x128xf16>
}

// -----
// CHECK-TRITON-LABEL: triton_avoid_normalize_aligned_insert_slice_0
// CHECK-TRITON-NOT：concat
func.func @triton_avoid_normalize_aligned_insert_slice_0(
    %arg0: tensor<1xf32>, %arg1: tensor<4xf32>, %arg2: tensor<33xf32>) -> (tensor<4xf32>, tensor<33xf32>) {
  %inserted_slice_0 = tensor.insert_slice %arg0 into %arg1[0] [1] [1] : tensor<1xf32> into tensor<4xf32>
  %inserted_slice_1 = tensor.insert_slice %arg0 into %arg2[32] [1] [1] : tensor<1xf32> into tensor<33xf32>
  return %inserted_slice_0, %inserted_slice_1 : tensor<4xf32>, tensor<33xf32>
}

// -----
// CHECK-TRITON-LABEL: triton_normalize_unaligned_insert_slice_0
// CHECK-TRITON：concat
func.func @triton_normalize_unaligned_insert_slice_0(
    %arg0: tensor<1xf32>, %arg1: tensor<4xf32>, %arg2: tensor<33xf32>) -> (tensor<4xf32>, tensor<33xf32>) {
  %inserted_slice_0 = tensor.insert_slice %arg0 into %arg1[1] [1] [1] : tensor<1xf32> into tensor<4xf32>
  %inserted_slice_1 = tensor.insert_slice %arg0 into %arg2[3] [1] [1] : tensor<1xf32> into tensor<33xf32>
  return %inserted_slice_0, %inserted_slice_1 : tensor<4xf32>, tensor<33xf32>
}

// -----

// CHECK-LABEL: @triton_kernel
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @triton_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 4, 3, 9, 11, 2], strides: [2376, 594, 198, 22, 2, 1] : memref<?xi8> to memref<2x4x3x9x11x2xi8, strided<[2376, 594, 198, 22, 2, 1]>>
    %alloc = memref.alloc() : memref<2x4x3x9x11x2xi8>
    memref.copy %reinterpret_cast, %alloc : memref<2x4x3x9x11x2xi8, strided<[2376, 594, 198, 22, 2, 1]>> to memref<2x4x3x9x11x2xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2x4x3x9x11x2xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 4, 3, 9, 11], strides: [1188, 297, 99, 11, 1] : memref<?xi8> to memref<2x4x3x9x11xi8, strided<[1188, 297, 99, 11, 1]>>
    %alloc_1 = memref.alloc() : memref<2x4x3x9x11xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<2x4x3x9x11xi8, strided<[1188, 297, 99, 11, 1]>> to memref<2x4x3x9x11xi8>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x4x3x9x11xi8>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0, 0, 0, 0, 0] [2, 4, 3, 9, 11, 1] [1, 1, 1, 1, 1, 1] : tensor<2x4x3x9x11xi8> into tensor<2x4x3x9x11x2xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [2, 4, 3, 9, 11, 2], strides: [2376, 594, 198, 22, 2, 1] : memref<?xi8> to memref<2x4x3x9x11x2xi8, strided<[2376, 594, 198, 22, 2, 1]>>
    bufferization.materialize_in_destination %inserted_slice in writable %reinterpret_cast_2 : (tensor<2x4x3x9x11x2xi8>, memref<2x4x3x9x11x2xi8, strided<[2376, 594, 198, 22, 2, 1]>>) -> ()
    return
  }
}
