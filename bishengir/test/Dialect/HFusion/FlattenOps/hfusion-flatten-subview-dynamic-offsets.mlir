// RUN: bishengir-opt %s                              \
// RUN:   -pass-pipeline="builtin.module(func.func(   \
// RUN:      hfusion-flatten-ops{flatten-mode=tidy register-based=true multi-dynamic-shape=false}), \
// RUN:      canonicalize-ext)"                       \
// RUN:   -split-input-file | FileCheck %s

// -----

// A 3-D subview on a fully dynamic reinterpret_cast source (sizes/strides
// come from runtime loads, as emitted for dynamic-shape index_select_simd)
// is collapsed as [[0, 1], [2]].  The dim-0 subview offset (4) must be
// scaled by the runtime dim-1 size and folded into the collapsed offset,
// and the collapsed stride must be the runtime dim-1 size.  Before the fix
// the dynamic sentinel (INT64_MIN) was truncated into an int accumulator,
// which silently dropped the "4 * N" contribution and left the stride at
// its unscaled value, so every gathered row read from the wrong address.

// CHECK-LABEL: func.func @dynamic_subview_offsets
// CHECK:       %[[OFF4N:.*]] = arith.muli %{{.*}}, %c4{{.*}} : index
// CHECK:       %[[OFF:.*]] = arith.addi %[[OFF4N]], %{{.*}} : index
// CHECK:       memref.subview %{{.*}}[%[[OFF]], 128] [4, 128] [%{{.*}}, 1] : memref<?x?xf32> to memref<4x128xf32, strided<[?, 1], offset: ?>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @dynamic_subview_offsets(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<4xi32>
  memref.copy %reinterpret_cast, %alloc : memref<4xi32, strided<[1]>> to memref<4xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %1 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %2 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %3 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1]>>
  %4 = arith.index_cast %1 : i32 to index
  %5 = arith.index_cast %2 : i32 to index
  %6 = arith.index_cast %3 : i32 to index
  %7 = arith.muli %5, %6 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%4, %5, %6], strides: [%7, %6, 1] : memref<?xf32> to memref<?x?x?xf32>
  %alloc_4 = memref.alloc() : memref<4x4x128xf32>
  scf.for %arg14 = %c0 to %c4 step %c1 {
    %extracted = tensor.extract %0[%arg14] : tensor<4xi32>
    %9 = arith.index_cast %extracted : i32 to index
    %subview = memref.subview %reinterpret_cast_3[4, %9, 128] [4, 1, 128] [1, 1, 1] : memref<?x?x?xf32> to memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, %arg14, 0] [4, 1, 128] [1, 1, 1] : memref<4x4x128xf32> to memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    annotation.mark %subview_6 {hfusion.stride_align_dims = array<i32: 1>, hfusion.stride_align_value_in_byte = array<i32: 32>} keys = ["write"] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>> into memref<4x128xf32, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2]] output_shape [4, 1, 128] : memref<4x128xf32, strided<[512, 1], offset: ?>> into memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
    memref.copy %subview, %expand_shape : memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>> to memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
  } {hivm.parallel_loop}
  %8 = bufferization.to_tensor %alloc_4 restrict writable {index_select_simd} : memref<4x4x128xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4, 128], strides: [512, 128, 1] : memref<?xf32> to memref<4x4x128xf32, strided<[512, 128, 1]>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_5 : (tensor<4x4x128xf32>, memref<4x4x128xf32, strided<[512, 128, 1]>>) -> ()
  return
}
}

// -----

// Exact 3-D regression case: source shape [M,N,D], offsets [4,idx,0],
// sizes [4,1,128], strides [1,1,1], collapse groups [[0,1],[2]].
// The first collapsed dimension must use N (not D): offset = 4*N + idx,
// stride = N; dim 2 remains a separate output dimension with offset 0.

// CHECK-LABEL: func.func @dynamic_subview_offsets_zero_tail
// CHECK:       %[[OFF4N_ZERO:.*]] = arith.muli %{{.*}}, %c4{{.*}} : index
// CHECK:       %[[OFF_ZERO:.*]] = arith.addi %[[OFF4N_ZERO]], %{{.*}} : index
// CHECK:       memref.subview %{{.*}}[%[[OFF_ZERO]], 0] [4, 128] [%{{.*}}, 1] : memref<?x?xf32> to memref<4x128xf32, strided<[?, 1], offset: ?>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @dynamic_subview_offsets_zero_tail(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<4xi32>
  memref.copy %reinterpret_cast, %alloc : memref<4xi32, strided<[1]>> to memref<4xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %1 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %2 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %3 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1]>>
  %4 = arith.index_cast %1 : i32 to index
  %5 = arith.index_cast %2 : i32 to index
  %6 = arith.index_cast %3 : i32 to index
  %7 = arith.muli %5, %6 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%4, %5, %6], strides: [%7, %6, 1] : memref<?xf32> to memref<?x?x?xf32>
  %alloc_4 = memref.alloc() : memref<4x4x128xf32>
  scf.for %arg14 = %c0 to %c4 step %c1 {
    %extracted = tensor.extract %0[%arg14] : tensor<4xi32>
    %9 = arith.index_cast %extracted : i32 to index
    %subview = memref.subview %reinterpret_cast_3[4, %9, 0] [4, 1, 128] [1, 1, 1] : memref<?x?x?xf32> to memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, %arg14, 0] [4, 1, 128] [1, 1, 1] : memref<4x4x128xf32> to memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    annotation.mark %subview_6 {hfusion.stride_align_dims = array<i32: 1>, hfusion.stride_align_value_in_byte = array<i32: 32>} keys = ["write"] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>> into memref<4x128xf32, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2]] output_shape [4, 1, 128] : memref<4x128xf32, strided<[512, 1], offset: ?>> into memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
    memref.copy %subview, %expand_shape : memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>> to memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
  } {hivm.parallel_loop}
  %8 = bufferization.to_tensor %alloc_4 restrict writable {index_select_simd} : memref<4x4x128xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4, 128], strides: [512, 128, 1] : memref<?xf32> to memref<4x4x128xf32, strided<[512, 128, 1]>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_5 : (tensor<4x4x128xf32>, memref<4x4x128xf32, strided<[512, 128, 1]>>) -> ()
  return
}
}

// -----

// Same pattern with a fully static source must keep folding to constants:
// offset 400 = 4 * 100 and stride 100, as before the dynamic-shape fix.

// CHECK-LABEL: func.func @static_subview_offsets
// CHECK:       memref.subview {{.*}}[%{{.*}}, 128] [4, 128] [100, 1] : memref<800x256xf32, strided<[256, 1]>> to memref<4x128xf32, strided<[25600, 1], offset: ?>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @static_subview_offsets(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<4xi32>
  memref.copy %reinterpret_cast, %alloc : memref<4xi32, strided<[1]>> to memref<4xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xi32>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 100, 256], strides: [25600, 256, 1] : memref<?xf32> to memref<8x100x256xf32, strided<[25600, 256, 1]>>
  %alloc_4 = memref.alloc() : memref<4x4x128xf32>
  scf.for %arg14 = %c0 to %c4 step %c1 {
    %extracted = tensor.extract %0[%arg14] : tensor<4xi32>
    %9 = arith.index_cast %extracted : i32 to index
    %subview = memref.subview %reinterpret_cast_3[4, %9, 128] [4, 1, 128] [1, 1, 1] : memref<8x100x256xf32, strided<[25600, 256, 1]>> to memref<4x1x128xf32, strided<[25600, 256, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, %arg14, 0] [4, 1, 128] [1, 1, 1] : memref<4x4x128xf32> to memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    annotation.mark %subview_6 {hfusion.stride_align_dims = array<i32: 1>, hfusion.stride_align_value_in_byte = array<i32: 32>} keys = ["write"] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>> into memref<4x128xf32, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2]] output_shape [4, 1, 128] : memref<4x128xf32, strided<[512, 1], offset: ?>> into memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
    memref.copy %subview, %expand_shape : memref<4x1x128xf32, strided<[25600, 256, 1], offset: ?>> to memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
  } {hivm.parallel_loop}
  %8 = bufferization.to_tensor %alloc_4 restrict writable {index_select_simd} : memref<4x4x128xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4, 128], strides: [512, 128, 1] : memref<?xf32> to memref<4x4x128xf32, strided<[512, 128, 1]>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_5 : (tensor<4x4x128xf32>, memref<4x4x128xf32, strided<[512, 128, 1]>>) -> ()
  return
}
}

// -----

// 4-D dynamic source [L, M, N, D] with the gather on dim 1, collapsed as
// [[0, 1, 2], [3]].  The dim-1 offset (idx) must be scaled by the runtime
// size of dim 2 (N), and the stride of the leading dim by M*N; asking for
// dim d of the source must never resolve through the intermediate shapes
// of a collapse chain (where dim 1 would read as N instead of M).

// CHECK-LABEL: func.func @dynamic_4d_select_dim1
// CHECK:       %[[STRIDE:.*]] = arith.muli %arg6, %arg7 : index
// CHECK:       %[[OFF:.*]] = arith.muli %{{.*}}, %arg7 : index
// CHECK:       memref.subview %{{.*}}[%[[OFF]], 0] [4, 128] [%[[STRIDE]], 1] : memref<?x?xf32> to memref<4x128xf32, strided<[?, 1], offset: ?>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @dynamic_4d_select_dim1(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %argL: index, %argM: index, %argN: index, %argD: index) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<4xi32>
  memref.copy %reinterpret_cast, %alloc : memref<4xi32, strided<[1]>> to memref<4xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xi32>
  %s2 = arith.muli %argN, %argD : index
  %s1 = arith.muli %argM, %s2 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%argL, %argM, %argN, %argD], strides: [%s1, %s2, %argD, 1] : memref<?xf32> to memref<?x?x?x?xf32>
  %alloc_4 = memref.alloc() : memref<4x4x1x128xf32>
  scf.for %arg14 = %c0 to %c4 step %c1 {
    %extracted = tensor.extract %0[%arg14] : tensor<4xi32>
    %9 = arith.index_cast %extracted : i32 to index
    %subview = memref.subview %reinterpret_cast_3[0, %9, 0, 0] [4, 1, 1, 128] [1, 1, 1, 1] : memref<?x?x?x?xf32> to memref<4x1x1x128xf32, strided<[?, ?, ?, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, %arg14, 0, 0] [4, 1, 1, 128] [1, 1, 1, 1] : memref<4x4x1x128xf32> to memref<4x1x1x128xf32, strided<[512, 128, 128, 1], offset: ?>>
    annotation.mark %subview_6 {hfusion.stride_align_dims = array<i32: 1>, hfusion.stride_align_value_in_byte = array<i32: 32>} keys = ["write"] : memref<4x1x1x128xf32, strided<[512, 128, 128, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview_6 [[0, 1, 2], [3]] : memref<4x1x1x128xf32, strided<[512, 128, 128, 1], offset: ?>> into memref<4x128xf32, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1, 2], [3]] output_shape [4, 1, 1, 128] : memref<4x128xf32, strided<[512, 1], offset: ?>> into memref<4x1x1x128xf32, strided<[512, 512, 512, 1], offset: ?>>
    memref.copy %subview, %expand_shape : memref<4x1x1x128xf32, strided<[?, ?, ?, 1], offset: ?>> to memref<4x1x1x128xf32, strided<[512, 512, 512, 1], offset: ?>>
  } {hivm.parallel_loop}
  %8 = bufferization.to_tensor %alloc_4 restrict writable {index_select_simd} : memref<4x4x1x128xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4, 1, 128], strides: [512, 128, 128, 1] : memref<?xf32> to memref<4x4x1x128xf32, strided<[512, 128, 128, 1]>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_5 : (tensor<4x4x1x128xf32>, memref<4x4x1x128xf32, strided<[512, 128, 128, 1]>>) -> ()
  return
}
}

// -----

// A dynamic slice stride must be scaled by C_p like a static one: strides
// [S,1,1] on a dynamic [M,N,D] source with sizes [4,1,128] collapsed as
// [[0,1],[2]] must yield a collapsed stride of S*N, not S.  The offsets
// [4, idx, 128] keep composing as before, and no dead trailing-size
// computation (the old N*M chain) may be emitted.

// CHECK-LABEL: func.func @dynamic_slice_stride
// CHECK:       %[[STRIDE:.*]] = arith.muli %arg14, %{{.*}} : index
// CHECK:       %[[OFF4N:.*]] = arith.muli %{{.*}}, %c4{{.*}} : index
// CHECK:       %[[OFF:.*]] = arith.addi %[[OFF4N]], %{{.*}} : index
// CHECK:       memref.subview %{{.*}}[%[[OFF]], 128] [4, 128] [%[[STRIDE]], 1] : memref<?x?xf32> to memref<4x128xf32, strided<[?, 1], offset: ?>>
// CHECK-NOT:   arith.muli %{{.*}}, %4 : index
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @dynamic_slice_stride(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: index) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<4xi32>
  memref.copy %reinterpret_cast, %alloc : memref<4xi32, strided<[1]>> to memref<4xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %1 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %2 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  %3 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1]>>
  %4 = arith.index_cast %1 : i32 to index
  %5 = arith.index_cast %2 : i32 to index
  %6 = arith.index_cast %3 : i32 to index
  %7 = arith.muli %5, %6 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%4, %5, %6], strides: [%7, %6, 1] : memref<?xf32> to memref<?x?x?xf32>
  %alloc_4 = memref.alloc() : memref<4x4x128xf32>
  scf.for %arg15 = %c0 to %c4 step %c1 {
    %extracted = tensor.extract %0[%arg15] : tensor<4xi32>
    %9 = arith.index_cast %extracted : i32 to index
    %subview = memref.subview %reinterpret_cast_3[4, %9, 128] [4, 1, 128] [%arg14, 1, 1] : memref<?x?x?xf32> to memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, %arg15, 0] [4, 1, 128] [1, 1, 1] : memref<4x4x128xf32> to memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    annotation.mark %subview_6 {hfusion.stride_align_dims = array<i32: 1>, hfusion.stride_align_value_in_byte = array<i32: 32>} keys = ["write"] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<4x1x128xf32, strided<[512, 128, 1], offset: ?>> into memref<4x128xf32, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2]] output_shape [4, 1, 128] : memref<4x128xf32, strided<[512, 1], offset: ?>> into memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
    memref.copy %subview, %expand_shape : memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>> to memref<4x1x128xf32, strided<[512, 512, 1], offset: ?>>
  } {hivm.parallel_loop}
  %8 = bufferization.to_tensor %alloc_4 restrict writable {index_select_simd} : memref<4x4x128xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4, 128], strides: [512, 128, 1] : memref<?xf32> to memref<4x4x128xf32, strided<[512, 128, 1]>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_5 : (tensor<4x4x128xf32>, memref<4x4x128xf32, strided<[512, 128, 1]>>) -> ()
  return
}
}


// -----

// A loop-carried iter_arg is rewritten in place to the collapsed (1-D)
// type while the slice inside the loop still speaks the pre-flatten rank.
// With all-zero offsets and an all-unit slice the composed offset is 0 and
// the stride is never consumed, so the pass must not try to resolve the
// runtime shapes of the already-collapsed source.

// CHECK-LABEL: func.func @loop_single_row
// CHECK:       %[[COLLAPSED:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1, 2]{{\]}} : memref<?x?x256xf32> into memref<?xf32>
// CHECK:       memref.subview %[[COLLAPSED]][0] [128] [1] : memref<?xf32> to memref<128xf32, strided<[1]>>
// CHECK-NOT:   memref.dim
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @loop_single_row(%src: memref<?x?x256xf32>) {
  %out = memref.alloc() : memref<1x1x128xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%v = %src) -> (memref<?x?x256xf32>) {
    %slice = memref.subview %v[0, 0, 0] [1, 1, 128] [1, 1, 1] : memref<?x?x256xf32> to memref<1x1x128xf32, strided<[?, 256, 1]>>
    memref.copy %slice, %out : memref<1x1x128xf32, strided<[?, 256, 1]>> to memref<1x1x128xf32>
    scf.yield %v : memref<?x?x256xf32>
  }
  return
}
}

// -----

// Non-zero leading offset with an in-place collapsed loop iter_arg: the
// offset path must resolve the original source coordinate system instead of
// aborting after the iter_arg has become 1-D.
// CHECK-LABEL: func.func @loop_nonzero_offset
// CHECK:       %[[COLLAPSED_NZ:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1, 2]{{\]}} : memref<?x?x256xf32> into memref<?xf32>
// CHECK:       %[[OFF_NZ:.*]] = arith.muli %{{.*}}, %c1024 : index
// CHECK:       memref.subview %[[COLLAPSED_NZ]][%[[OFF_NZ]]] [128] [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @loop_nonzero_offset(%src: memref<?x?x256xf32>) {
  %out = memref.alloc() : memref<1x1x128xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%v = %src) -> (memref<?x?x256xf32>) {
    %slice = memref.subview %v[4, 0, 0] [1, 1, 128] [1, 1, 1] : memref<?x?x256xf32> to memref<1x1x128xf32, strided<[?, 256, 1], offset: ?>>
    memref.copy %slice, %out : memref<1x1x128xf32, strided<[?, 256, 1], offset: ?>> to memref<1x1x128xf32>
    scf.yield %v : memref<?x?x256xf32>
  }
  return
}
}

// -----

// An arith.constant Value offset must be recognized as zero for the
// short-circuit (no dim queries, no arithmetic chains); canonicalize then
// folds the constant-0 dynamic offset of the rebuilt subview back to a
// static zero.

// CHECK-LABEL: func.func @ssa_zero_offset
// CHECK:       memref.subview %arg0[0] [64] [1] : memref<128xf32> to memref<64xf32, strided<[1]>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @ssa_zero_offset(%src: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %slice = memref.subview %src[%c0] [64] [1] : memref<128xf32> to memref<64xf32, strided<[1], offset: ?>>
  %out = memref.alloc() : memref<64xf32>
  memref.copy %slice, %out : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
  return
}
}

// -----

// A loop that yields a different shape producer than the init operand must
// not read runtime dimensions from the init: the carried shape may change
// between iterations. This test uses scf.yield %src (same value as init),
// which *is* safe to recover from init; see @loop_changing_shape for the
// unsafe variant (a different reinterpret_cast producer is yielded, so the
// flattener bails and keeps the loop untouched).

// CHECK-LABEL: func.func @loop_yield_same_value
// CHECK:       %[[COLL:.*]] = memref.collapse_shape %reinterpret_cast {{\[}}[0, 1], {{\[}}2]{{\]}} : memref<?x?x256xf32> into memref<?x256xf32>
// CHECK:       %[[OFF_Y:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:       memref.subview %[[COLL]][%[[OFF_Y]], 0] [4, 128] [%{{.*}}, 1] : memref<?x256xf32> to memref<4x128xf32, strided<[?, 1], offset: ?>>
// CHECK-NOT:   memref.dim %arg
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @loop_yield_same_value(%base: memref<?xf32>, %M: index, %N: index, %idx: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %nd = arith.muli %N, %c256 : index
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [%M, %N, 256], strides: [%nd, 256, 1] : memref<?xf32> to memref<?x?x256xf32>
  %out = memref.alloc() : memref<4x1x128xf32>
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%v = %src) -> (memref<?x?x256xf32>) {
    %slice = memref.subview %v[4, %idx, 0] [4, 1, 128] [1, 1, 1] : memref<?x?x256xf32> to memref<4x1x128xf32, strided<[?, 256, 1], offset: ?>>
    memref.copy %slice, %out : memref<4x1x128xf32, strided<[?, 256, 1], offset: ?>> to memref<4x1x128xf32>
    scf.yield %v : memref<?x?x256xf32>
  }
  return
}
}


// -----

// An all-unit slice [1,1,1] with all-zero offsets on a fully dynamic source:
// the composed offset is provably 0 (no dim queries, no arithmetic chains)
// and the result reads exactly element 0 of the collapsed buffer.  The
// consumer-side collapse is reconstructed through expand/cast so the copy
// keeps its original operand types.
// CHECK-LABEL: func.func @unit_extent
// CHECK:       %[[COLL_U:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1, 2]{{\]}} : memref<?x?x?xf32> into memref<?xf32>
// CHECK:       %[[STRIDE_U:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:       memref.subview %[[COLL_U]][0] [1] [%[[STRIDE_U]]] : memref<?xf32> to memref<1xf32, strided<[?]>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @unit_extent(%src: memref<?x?x?xf32>) {
  %out = memref.alloc() : memref<1xf32>
  %c0 = arith.constant 0 : index
  %slice = memref.subview %src[0, 0, 0] [1, 1, 1] [1, 1, 1] : memref<?x?x?xf32> to memref<1x1x1xf32, strided<[?, ?, 1]>>
  %collapsed = memref.collapse_shape %slice [[0, 1, 2]] : memref<1x1x1xf32, strided<[?, ?, 1]>> into memref<1xf32, strided<[?]>>
  memref.copy %collapsed, %out : memref<1xf32, strided<[?]>> to memref<1xf32>
  return
}
}

// -----

// A loop whose yield re-produces the carried buffer with a DIFFERENT dynamic
// shape owner (%src1 with N1 vs the init %src0 with N0): the carried shape
// may change between iterations, so resolving the collapsed offsets/strides
// through the init is unsound.  The flattener must bail: the whole chain
// (init, iter_arg, inner subview, yield) keeps its original rank and
// addressing, so every iteration reads its own producer's layout.
// CHECK-LABEL: func.func @loop_changing_shape
// CHECK:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (memref<?x?x256xf32>) {
// CHECK:       memref.subview %{{.*}}[4, %{{.*}}, 0] [4, 1, 128] [1, 1, 1] : memref<?x?x256xf32> to memref<4x1x128xf32, strided<[?, 256, 1], offset: ?>>
// CHECK:       scf.yield %{{.*}} : memref<?x?x256xf32>
// CHECK-NOT:   memref.collapse_shape
// CHECK-NOT:   memref.dim
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @loop_changing_shape(%base: memref<?xf32>, %M: index, %N0: index, %N1: index, %idx: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %nd0 = arith.muli %N0, %c256 : index
  %nd1 = arith.muli %N1, %c256 : index
  %src0 = memref.reinterpret_cast %base to offset: [0], sizes: [%M, %N0, 256], strides: [%nd0, 256, 1] : memref<?xf32> to memref<?x?x256xf32>
  %src1 = memref.reinterpret_cast %base to offset: [0], sizes: [%M, %N1, 256], strides: [%nd1, 256, 1] : memref<?xf32> to memref<?x?x256xf32>
  %out = memref.alloc() : memref<4x1x128xf32>
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%v = %src0) -> (memref<?x?x256xf32>) {
    %slice = memref.subview %v[4, %idx, 0] [4, 1, 128] [1, 1, 1] : memref<?x?x256xf32> to memref<4x1x128xf32, strided<[?, 256, 1], offset: ?>>
    memref.copy %slice, %out : memref<4x1x128xf32, strided<[?, 256, 1], offset: ?>> to memref<4x1x128xf32>
    scf.yield %src1 : memref<?x?x256xf32>
  }
  return
}
}
