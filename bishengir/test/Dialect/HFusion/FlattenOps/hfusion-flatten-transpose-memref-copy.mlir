// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true" -split-input-file | FileCheck %s

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 3, 2, 4, 5, 6, 7], flatten: [[0, 1], [2], [3], [4, 5, 6, 7]]
// CHECK-LABEL: func.func @case_01_single_axis_adjacent_middle_g1_g2_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x4x5x3024xf32, strided<[60480, 3024, 12096, 1]>> to memref<6x4x5x3024xf32>
func.func @case_01_single_axis_adjacent_middle_g1_g2_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 3024, 12096, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 3024, 12096, 504, 72, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 3024, 12096, 504, 72, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 2, 3, 4, 5, 7, 6], flatten: [[0, 1, 2, 3, 4, 5], [6], [7]]
// CHECK-LABEL: func.func @case_02_single_axis_adjacent_tail_g4_g5_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<5040x8x9xf32, strided<[72, 1, 8]>> to memref<5040x8x9xf32>
func.func @case_02_single_axis_adjacent_tail_g4_g5_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 1, 8]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 1, 8], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 1, 8], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [2, 3, 0, 1, 4, 5, 6, 7], flatten: [[0, 1], [2, 3], [4, 5, 6, 7]]
// CHECK-LABEL: func.func @case_03_multi_axis_adjacent_head_g0_g1_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x20x3024xf32, strided<[3024, 18144, 1]>> to memref<6x20x3024xf32>
func.func @case_03_multi_axis_adjacent_head_g0_g1_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [9072, 3024, 90720, 18144, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[9072, 3024, 90720, 18144, 504, 72, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[9072, 3024, 90720, 18144, 504, 72, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 2, 4, 5, 3, 6, 7], flatten: [[0, 1, 2], [3], [4, 5], [6, 7]]
// CHECK-LABEL: func.func @case_04_multi_axis_adjacent_middle_g2_g3_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<24x5x42x72xf32, strided<[15120, 72, 360, 1]>> to memref<24x5x42x72xf32>
func.func @case_04_multi_axis_adjacent_middle_g2_g3_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 72, 2520, 360, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 72, 2520, 360, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 72, 2520, 360, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 7, 3, 4, 5, 6, 2], flatten: [[0, 1], [2], [3, 4, 5, 6], [7]]
// CHECK-LABEL: func.func @case_05_single_axis_cross_head_tail_g1_g5_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x4x1680x9xf32, strided<[60480, 1, 4, 6720]>> to memref<6x4x1680x9xf32>
func.func @case_05_single_axis_cross_head_tail_g1_g5_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 1, 1344, 224, 32, 4, 6720]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 1, 1344, 224, 32, 4, 6720], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 1, 1344, 224, 32, 4, 6720], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [7, 2, 3, 4, 5, 6, 0, 1], flatten: [[0, 1], [2, 3, 4, 5, 6], [7]]
// CHECK-LABEL: func.func @case_06_multi_axis_cross_head_tail_g0_g5_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x6720x9xf32, strided<[1, 6, 40320]>> to memref<6x6720x9xf32>
func.func @case_06_multi_axis_cross_head_tail_g0_g5_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [3, 1, 10080, 2016, 336, 48, 6, 40320]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[3, 1, 10080, 2016, 336, 48, 6, 40320], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[3, 1, 10080, 2016, 336, 48, 6, 40320], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 6, 3, 4, 5, 2, 7], flatten: [[0, 1], [2], [3, 4, 5], [6], [7]]
// CHECK-LABEL: func.func @case_07_single_axis_cross_middle_g1_g4_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x4x210x8x9xf32, strided<[60480, 9, 36, 7560, 1]>> to memref<6x4x210x8x9xf32>
func.func @case_07_single_axis_cross_middle_g1_g4_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 9, 1512, 252, 36, 7560, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 9, 1512, 252, 36, 7560, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 9, 1512, 252, 36, 7560, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [4, 5, 2, 3, 0, 1, 6, 7], flatten: [[0, 1], [2, 3], [4, 5], [6, 7]]
// CHECK-LABEL: func.func @case_08_multi_axis_cross_middle_g0_g3_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x20x42x72xf32, strided<[72, 432, 8640, 1]>> to memref<6x20x42x72xf32>
func.func @case_08_multi_axis_cross_middle_g0_g3_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [216, 72, 2160, 432, 60480, 8640, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[216, 72, 2160, 432, 60480, 8640, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[216, 72, 2160, 432, 60480, 8640, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [2, 3, 0, 1, 4, 5, 6, 7], flatten: [[0, 1], [2, 3], [4, 5, 6, 7]]
// CHECK-LABEL: func.func @case_09_three_cycle_g0_g1_g2_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x20x3024xf32, strided<[3024, 18144, 1]>> to memref<6x20x3024xf32>
func.func @case_09_three_cycle_g0_g1_g2_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [9072, 3024, 90720, 18144, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[9072, 3024, 90720, 18144, 504, 72, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[9072, 3024, 90720, 18144, 504, 72, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 2, 3, 6, 7, 4, 5], flatten: [[0, 1, 2, 3], [4, 5], [6, 7]]
// CHECK-LABEL: func.func @case_10_three_cycle_g3_g4_g5_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<120x42x72xf32, strided<[3024, 1, 42]>> to memref<120x42x72xf32>
func.func @case_10_three_cycle_g3_g4_g5_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 7, 1, 378, 42]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 7, 1, 378, 42], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 7, 1, 378, 42], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [7, 6, 5, 4, 3, 2, 1, 0], flatten: [[0], [1], [2], [3], [4], [5], [6], [7]]
// CHECK-LABEL: func.func @case_11_reverse_all_groups_before(
// CHECK: memref.copy %reinterpret_cast, %alloc : memref<2x3x4x5x6x7x8x9xf32, strided<[1, 2, 6, 24, 120, 720, 5040, 40320]>> to memref<2x3x4x5x6x7x8x9xf32>
func.func @case_11_reverse_all_groups_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [1, 2, 6, 24, 120, 720, 5040, 40320]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[1, 2, 6, 24, 120, 720, 5040, 40320], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[1, 2, 6, 24, 120, 720, 5040, 40320], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [4, 5, 6, 3, 0, 1, 2, 7], flatten: [[0, 1, 2], [3], [4, 5, 6], [7]]
// CHECK-LABEL: func.func @case_12_double_swap_g0_g3_g1_g4_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<24x5x336x9xf32, strided<[9, 216, 1080, 1]>> to memref<24x5x336x9xf32>
func.func @case_12_double_swap_g0_g3_g1_g4_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [108, 36, 9, 216, 60480, 8640, 1080, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[108, 36, 9, 216, 60480, 8640, 1080, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[108, 36, 9, 216, 60480, 8640, 1080, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 3, 2, 5, 4, 6, 7], flatten: [[0, 1], [2], [3], [4], [5], [6, 7]]
// CHECK-LABEL: func.func @case_13_intra_group_g3_transpose_plus_g1_g2_swap_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x4x5x6x7x72xf32, strided<[60480, 3024, 12096, 72, 432, 1]>> to memref<6x4x5x6x7x72xf32>
func.func @case_13_intra_group_g3_transpose_plus_g1_g2_swap_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 3024, 12096, 72, 432, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 3024, 12096, 72, 432, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 3024, 12096, 72, 432, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [3, 0, 1, 7, 2, 6, 4, 5], flatten: [[0, 1], [2], [3], [4, 5], [6], [7]]
// CHECK-LABEL: func.func @case_14_random_shuffle_g2_g0_g5_g1_g4_g3_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x4x5x42x8x9xf32, strided<[12096, 336, 72576, 1, 42, 1344]>> to memref<6x4x5x42x8x9xf32>
func.func @case_14_random_shuffle_g2_g0_g5_g1_g4_g3_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [36288, 12096, 336, 72576, 7, 1, 42, 1344]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[36288, 12096, 336, 72576, 7, 1, 42, 1344], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[36288, 12096, 336, 72576, 7, 1, 42, 1344], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [4, 2, 3, 5, 6, 0, 1, 7], flatten: [[0, 1], [2, 3], [4], [5, 6], [7]]
// CHECK-LABEL: func.func @case_15_complex_reorder_custom_grouping_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<6x20x6x56x9xf32, strided<[9, 3024, 60480, 54, 1]>> to memref<6x20x6x56x9xf32>
func.func @case_15_complex_reorder_custom_grouping_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [27, 9, 15120, 3024, 60480, 432, 54, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[27, 9, 15120, 3024, 60480, 432, 54, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[27, 9, 15120, 3024, 60480, 432, 54, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [5, 6, 0, 1, 2, 3, 4, 7], flatten: [[0, 1, 2, 3, 4], [5, 6], [7]]
// CHECK-LABEL: func.func @case_16_large_jump_g0_after_g4_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<720x56x9xf32, strided<[9, 6480, 1]>> to memref<720x56x9xf32>
func.func @case_16_large_jump_g0_after_g4_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [3240, 1080, 270, 54, 9, 51840, 6480, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[3240, 1080, 270, 54, 9, 51840, 6480, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[3240, 1080, 270, 54, 9, 51840, 6480, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 2, 3, 4, 5, 6, 7], flatten: [[0, 1, 2, 3, 4, 5, 6, 7]]
// CHECK-LABEL: func.func @case_17_identity_grouped_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<362880xf32, strided<[1]>> to memref<362880xf32>
func.func @case_17_identity_grouped_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}

// -----
// transpose: [0, 1, 2, 3, 4, 5, 6, 7] -> [3, 4, 5, 0, 1, 2, 6, 7], flatten: [[0, 1, 2], [3, 4, 5], [6, 7]]
// CHECK-LABEL: func.func @case_21_head_merge_then_swap_with_g2_before(
// CHECK: memref.copy %collapse_shape, %collapse_shape_0 : memref<24x210x72xf32, strided<[72, 1728, 1]>> to memref<24x210x72xf32>
func.func @case_21_head_merge_then_swap_with_g2_before(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
                %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
    attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                mix_mode = "aiv", parallel_mode = "simd"} {

  // Source view: keep the logical shape, encode transpose/reorder only through strides.
  %src = memref.reinterpret_cast %arg2
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [864, 288, 72, 72576, 12096, 1728, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[864, 288, 72, 72576, 12096, 1728, 9, 1], offset: 0>>

  // UB buffer: contiguous destination for the in-flight transpose copy.
  %ub = memref.alloc() : memref<2x3x4x5x6x7x8x9xf32>

  memref.copy %src, %ub
    : memref<2x3x4x5x6x7x8x9xf32, strided<[864, 288, 72, 72576, 12096, 1728, 9, 1], offset: 0>> to memref<2x3x4x5x6x7x8x9xf32>

  %0 = bufferization.to_tensor %ub restrict writable : memref<2x3x4x5x6x7x8x9xf32>

  // Destination view: contiguous store-out on the final logical shape.
  %dst = memref.reinterpret_cast %arg3
           to offset: [0], sizes: [2, 3, 4, 5, 6, 7, 8, 9], strides: [181440, 60480, 15120, 3024, 504, 72, 9, 1]
           : memref<?xf32> to memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>

  bufferization.materialize_in_destination %0 in writable %dst
    : (tensor<2x3x4x5x6x7x8x9xf32>, memref<2x3x4x5x6x7x8x9xf32, strided<[181440, 60480, 15120, 3024, 504, 72, 9, 1], offset: 0>>) -> ()

  return
}