// REQUIRES: asserts
// RUN: bishengir-opt %s --propagate-reshape --cse --canonicalize --valid-propagate --debug-only="propagate-valid-check" -split-input-file | FileCheck %s

// CHECK: Valid
// CHECK-LABEL: @test_collapse_down_memref_strided
func.func @test_collapse_down_memref_strided(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c0_i32 = arith.constant 0 : i32
  %c1048576_i32 = arith.constant 1048576 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  %c48_i32 = arith.constant 48 : i32
  %c8_i32 = arith.constant 8 : i32
  %c384_i32 = arith.constant 384 : i32
  %c256_i32 = arith.constant 256 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg7, %arg6 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg5 : i32
  hivm.hir.set_mask_norm
  %5 = arith.muli %4, %c256_i32 : i32
  scf.for %arg8 = %c0_i32 to %c384_i32 step %c1_i32  : i32 {
    %6 = arith.divsi %arg8, %c384_i32 : i32
    %7 = arith.divsi %arg8, %c8_i32 : i32
    %8 = arith.remsi %7, %c48_i32 : i32
    %9 = arith.remsi %arg8, %c8_i32 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.addi %5, %10 : i32
    %12 = arith.muli %11, %c128_i32 : i32
    %13 = arith.muli %6, %c48_i32 : i32
    %14 = arith.addi %13, %8 : i32
    %15 = arith.muli %14, %c8192_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.index_cast %11 : i32 to index
    %18 = arith.addi %16, %17 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc : memref<32xf32>)
    %19 = bufferization.to_tensor %alloc restrict writable : memref<32xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<32xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc_1 : memref<32xf32>)
    %20 = bufferization.to_tensor %alloc_1 restrict writable : memref<32xf32>
    %21 = arith.muli %14, %c1048576_i32 : i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = arith.index_cast %12 : i32 to index
    %24 = arith.addi %22, %23 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<4096xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<4096xf32, strided<[1], offset: ?>>) outs(%alloc_3 : memref<4096xf32>)
    %25 = bufferization.to_tensor %alloc_3 restrict writable : memref<4096xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<4096xf32>
    hivm.hir.load ins(%reinterpret_cast_4 : memref<4096xf32, strided<[1], offset: ?>>) outs(%alloc_5 : memref<4096xf32>)
    %26 = bufferization.to_tensor %alloc_5 restrict writable : memref<4096xf32>
    %27 = tensor.empty() : tensor<32xf32>
    %28 = hivm.hir.vexp ins(%20 : tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %29 = hivm.hir.vexp ins(%19 : tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %30 = hivm.hir.vadd ins(%28, %29 : tensor<32xf32>, tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %31 = hivm.hir.vln ins(%30 : tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %32 = hivm.hir.vsub ins(%19, %31 : tensor<32xf32>, tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %expanded = tensor.expand_shape %32 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %33 = hivm.hir.vsub ins(%20, %31 : tensor<32xf32>, tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %expanded_6 = tensor.expand_shape %33 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %34 = tensor.empty() : tensor<32x1xf32>
    %35 = hivm.hir.vexp ins(%expanded : tensor<32x1xf32>) outs(%34 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %36 = tensor.empty() : tensor<32x128xf32>
    %37 = hivm.hir.vbrc ins(%35 : tensor<32x1xf32>) outs(%36 : tensor<32x128xf32>) broadcast_dims = [1] -> tensor<32x128xf32>
    %expanded_7 = tensor.expand_shape %25 [[0, 1]] output_shape [32, 128] : tensor<4096xf32> into tensor<32x128xf32>
    %38 = hivm.hir.vmul ins(%37, %expanded_7 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%36 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %39 = hivm.hir.vexp ins(%expanded_6 : tensor<32x1xf32>) outs(%34 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %40 = hivm.hir.vbrc ins(%39 : tensor<32x1xf32>) outs(%36 : tensor<32x128xf32>) broadcast_dims = [1] -> tensor<32x128xf32>
    %expanded_8 = tensor.expand_shape %26 [[0, 1]] output_shape [32, 128] : tensor<4096xf32> into tensor<32x128xf32>
    %41 = hivm.hir.vmul ins(%40, %expanded_8 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%36 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %42 = hivm.hir.vadd ins(%38, %41 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%36 : tensor<32x128xf32>) -> tensor<32x128xf32>
    hivm.hir.store ins(%31 : tensor<32xf32>) outs(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>>)
    %collapsed = tensor.collapse_shape %42 [[0, 1]] : tensor<32x128xf32> into tensor<4096xf32>
    hivm.hir.store ins(%collapsed : tensor<4096xf32>) outs(%reinterpret_cast_2 : memref<4096xf32, strided<[1], offset: ?>>)
  }
  return
}

// -----

// CHECK: Valid
// CHECK-LABEL: @expand_after_to_tensor_strided
// CHECK:       %[[T:.*]] = bufferization.to_tensor {{.*}} : memref<32x64xf32, strided<[128, 1]>>
// CHECK:       tensor.expand_shape %[[T]]
// CHECK-NOT:   memref.expand_shape
func.func @expand_after_to_tensor_strided(%arg0: memref<32x128xf32>) -> tensor<4x8x64xf32> {
  %m = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 64], strides: [128, 1] : memref<32x128xf32> to memref<32x64xf32, strided<[128, 1]>>
  %t = bufferization.to_tensor %m restrict : memref<32x64xf32, strided<[128, 1]>>
  %e = tensor.expand_shape %t [[0, 1], [2]] output_shape [4, 8, 64] : tensor<32x64xf32> into tensor<4x8x64xf32>
  %init = tensor.empty() : tensor<4x8x64xf32>
  %add = hivm.hir.vadd ins(%e, %e : tensor<4x8x64xf32>, tensor<4x8x64xf32>) outs(%init : tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
  return %add : tensor<4x8x64xf32>
}

// -----

// CHECK: Valid
// CHECK-LABEL: @expand_after_to_tensor_identity
// CHECK:       memref.expand_shape
// CHECK:       bufferization.to_tensor
// CHECK-NOT:   tensor.expand_shape %{{.*}} {{\[}}[0, 1]
func.func @expand_after_to_tensor_identity(%arg0: memref<32x64xf32>) -> tensor<4x8x64xf32> {
  %t = bufferization.to_tensor %arg0 restrict : memref<32x64xf32>
  %e = tensor.expand_shape %t [[0, 1], [2]] output_shape [4, 8, 64] : tensor<32x64xf32> into tensor<4x8x64xf32>
  %init = tensor.empty() : tensor<4x8x64xf32>
  %add = hivm.hir.vadd ins(%e, %e : tensor<4x8x64xf32>, tensor<4x8x64xf32>) outs(%init : tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
  return %add : tensor<4x8x64xf32>
}

// -----

// CHECK: Valid
// CHECK-LABEL: @propagate_memref_expand_512_sink
// CHECK: %[[C512:.*]] = arith.constant 512 : index
// CHECK: memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1, 32, 1, 16], strides: [1024, 32, 16, 1]
// CHECK: memref.subview %reinterpret_cast[0, %arg2, 0, 0] [1, 16, 1, 16]
// CHECK: memref.collapse_shape %subview {{\[\[}}0, 1], [2, 3]]
// CHECK: memref.expand_shape %collapse_shape {{\[\[}}0, 1], [2, 3]] output_shape [1, 16, 1, 16] : memref<16x16xf16, strided<[32, 1], offset: ?>> into memref<1x16x1x16xf16, strided<[512, 32, 16, 1], offset: ?>>
// CHECK: memref.reinterpret_cast %subview to offset: [%offset], sizes: [%c1, %c16, %c1, %c16], strides: [%[[C512]], %c32, %c16, %c1]
// CHECK: memref.subview %reinterpret_cast_0[0, 0, 0, 0] [1, %arg3, 1, %arg4]
// CHECK: memref.cast
// CHECK-NOT: memref.expand_shape %subview_0
func.func @propagate_memref_expand_512_sink(%arg0: memref<?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: tensor<1x16x1x16xf32>) -> memref<1x?x1x?xf16, strided<[?, 32, ?, 1], offset: ?>> {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 16], strides: [32, 1] : memref<?xf16> to memref<32x16xf16, strided<[32, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[%arg2, 0] [16, 16] [1, 1] : memref<32x16xf16, strided<[32, 1], offset: ?>> to memref<16x16xf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %subview[0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>

  %expand_shape = memref.expand_shape %subview_0 [[0, 1], [2, 3]] output_shape [1, %arg3, 1, %arg4] : memref<?x?xf16, strided<[32, 1], offset: ?>> into memref<1x?x1x?xf16, strided<[?, 32, ?, 1], offset: ?>>
  return %expand_shape : memref<1x?x1x?xf16, strided<[?, 32, ?, 1], offset: ?>>
}

// -----

// CHECK: %[[value:.*]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] : memref<?xi8> to memref<1x1x1x1xi8>
// CHECK: %[[value1:.*]] = memref.collapse_shape %[[value]] {{\[\[}}0, 1, 2, 3{{\]\]}} : memref<1x1x1x1xi8> into memref<1xi8>
func.func @triton_min_5d_dim4(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %true = arith.constant true
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  %alloc = memref.alloc() : memref<1xi8>
  memref.copy %reinterpret_cast, %alloc : memref<1xi8, strided<[1]>> to memref<1xi8>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
  hfusion.assert "Expecting input to be integer type" %true : i1
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  bufferization.materialize_in_destination %0 in writable %reinterpret_cast_0 : (tensor<1xi8>, memref<1xi8, strided<[1]>>) -> ()
  %alloc_1 = memref.alloc() : memref<1xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<1xi8, strided<[1]>> to memref<1xi8>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<1xi8>
  %expanded = tensor.expand_shape %1 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi8> into tensor<1x1x1x1xi8>
  hfusion.print " \0A: " {hex = false} %expanded : tensor<1x1x1x1xi8>
  return
}