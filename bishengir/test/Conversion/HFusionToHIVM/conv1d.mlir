// RUN: bishengir-opt -convert-hfusion-to-hivm %s | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s | FileCheck %s

// CHECK-LABEL: triton_conv1d_2d_kernel
func.func @triton_conv1d_2d_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 128], strides: [128, 1] : memref<?xf16> to memref<32x128xf16, strided<[128, 1]>>
  %alloc = memref.alloc() : memref<32x128xf16>
  memref.copy %reinterpret_cast, %alloc : memref<32x128xf16, strided<[128, 1]>> to memref<32x128xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<32x128xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32, 5], strides: [160, 5, 1] : memref<?xf16> to memref<32x32x5xf16, strided<[160, 5, 1]>>
  %alloc_1 = memref.alloc() : memref<32x32x5xf16>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32x5xf16, strided<[160, 5, 1]>> to memref<32x32x5xf16>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32x5xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1]>>
  %alloc_3 = memref.alloc() : memref<32xf16>
  memref.copy %reinterpret_cast_2, %alloc_3 : memref<32xf16, strided<[1]>> to memref<32xf16>
  %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<32xf16>
  %3 = tensor.empty() : tensor<32x126xf16>
  // CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128xf16>, tensor<32x32x5xf16>, i1, tensor<32xf16>) outs(%{{.*}} : tensor<32x126xf16>) -> tensor<32x126xf16>
  %4 = hfusion.conv1d {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32} ins(%0, %1, %2 : tensor<32x128xf16>, tensor<32x32x5xf16>, tensor<32xf16>) outs(%3 : tensor<32x126xf16>) -> tensor<32x126xf16>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [32, 126], strides: [126, 1] : memref<?xf16> to memref<32x126xf16, strided<[126, 1]>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_4 : (tensor<32x126xf16>, memref<32x126xf16, strided<[126, 1]>>) -> ()
  return
}


// CHECK-LABEL: triton_conv1d_3d_kernel
func.func @triton_conv1d_3d_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<?xf16> to memref<2x32x128xf16, strided<[4096, 128, 1]>>
  %alloc = memref.alloc() : memref<2x32x128xf16>
  memref.copy %reinterpret_cast, %alloc : memref<2x32x128xf16, strided<[4096, 128, 1]>> to memref<2x32x128xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x32x128xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32, 5], strides: [160, 5, 1] : memref<?xf16> to memref<32x32x5xf16, strided<[160, 5, 1]>>
  %alloc_1 = memref.alloc() : memref<32x32x5xf16>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32x5xf16, strided<[160, 5, 1]>> to memref<32x32x5xf16>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32x5xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1]>>
  %alloc_3 = memref.alloc() : memref<32xf16>
  memref.copy %reinterpret_cast_2, %alloc_3 : memref<32xf16, strided<[1]>> to memref<32xf16>
  %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<32xf16>
  %3 = tensor.empty() : tensor<2x32x126xf16>
  // CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x32x128xf16>, tensor<32x32x5xf16>, i1, tensor<32xf16>) outs(%{{.*}} : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  %4 = hfusion.conv1d {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32} ins(%0, %1, %2 : tensor<2x32x128xf16>, tensor<32x32x5xf16>, tensor<32xf16>) outs(%3 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [2, 32, 126], strides: [4032, 126, 1] : memref<?xf16> to memref<2x32x126xf16, strided<[4032, 126, 1]>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_4 : (tensor<2x32x126xf16>, memref<2x32x126xf16, strided<[4032, 126, 1]>>) -> ()
  return
}