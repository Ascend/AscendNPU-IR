// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2 -canonicalize -cse -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @triton_unk_fused_cat_2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant -1.000000e+00 : f32
  %c10 = arith.constant 10 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c2560 = arith.constant 2560 : index
  %c32 = arith.constant 32 : index
  %c640_i32 = arith.constant 640 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = arith.muli %arg12, %c4_i32 : i32
  %1 = arith.addi %0, %c4_i32 : i32
  %2 = arith.minsi %1, %arg5 : i32
  scf.for %arg15 = %c0_i32 to %arg6 step %c1_i32  : i32 {
    %3 = arith.cmpi slt, %arg15, %arg6 : i32
    %4 = arith.muli %arg15, %c64_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.addi %5, %c32 : index
    %7 = arith.index_cast %0 : i32 to index
    %8 = arith.muli %7, %c2560 : index
    %9 = arith.addi %6, %8 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [4, 10, 64], strides: [2560, 256, 1] : memref<?xf32> to memref<4x10x64xf32, strided<[2560, 256, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x10x64xf32>
    %10 = arith.addi %7, %c4 : index
    %11 = arith.index_cast %2 : i32 to index
    %12 = arith.maxsi %7, %11 : index
    %13 = arith.minsi %10, %12 : index
    %14 = arith.subi %13, %7 : index
    %15 = arith.index_castui %3 : i1 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.minsi %14, %16 : index
    %18 = arith.minsi %15, %c1 : index
    %19 = arith.minsi %17, %c4 : index
    %20 = arith.minsi %18, %c1 : index
    annotation.mark %alloc keys = ["pad_const"] values = [%cst_0 : f32] : memref<4x10x64xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0 : f32) outs(%alloc : memref<4x10x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %21 = arith.muli %20, %c10 : index
    %subview = memref.subview %reinterpret_cast[0, 0, 0] [%19, %21, 32] [1, 1, 1] : memref<4x10x64xf32, strided<[2560, 256, 1], offset: ?>> to memref<?x?x32xf32, strided<[2560, 256, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0, 0] [%19, %21, 32] [1, 1, 1] : memref<4x10x64xf32> to memref<?x?x32xf32, strided<[640, 64, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?x32xf32, strided<[2560, 256, 1], offset: ?>> to memref<?x?x32xf32, strided<[640, 64, 1]>>
    %22 = bufferization.to_tensor %alloc restrict writable : memref<4x10x64xf32>
    %23 = arith.addi %5, %8 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [4, 10, 64], strides: [2560, 256, 1] : memref<?xf32> to memref<4x10x64xf32, strided<[2560, 256, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<4x10x64xf32>
    annotation.mark %alloc_3 keys = ["pad_const"] values = [%cst_0 : f32] : memref<4x10x64xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0 : f32) outs(%alloc_3 : memref<4x10x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_4 = memref.subview %reinterpret_cast_2[0, 0, 0] [%19, %21, 32] [1, 1, 1] : memref<4x10x64xf32, strided<[2560, 256, 1], offset: ?>> to memref<?x?x32xf32, strided<[2560, 256, 1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0, 0, 0] [%19, %21, 32] [1, 1, 1] : memref<4x10x64xf32> to memref<?x?x32xf32, strided<[640, 64, 1]>>
    memref.copy %subview_4, %subview_5 : memref<?x?x32xf32, strided<[2560, 256, 1], offset: ?>> to memref<?x?x32xf32, strided<[640, 64, 1]>>
    %24 = bufferization.to_tensor %alloc_3 restrict writable : memref<4x10x64xf32>
    %25 = tensor.empty() : tensor<4x10x64xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<4x10x64xf32>) outs(%25 : tensor<4x10x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %36 = arith.mulf %in, %cst : f32
      linalg.yield %36 : f32
    } -> tensor<4x10x64xf32>
    %27 = arith.muli %arg15, %c640_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29 = arith.addi %28, %8 : index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%29], sizes: [4, 10, 64], strides: [2560, 64, 1] : memref<?xf32> to memref<4x10x64xf32, strided<[2560, 64, 1], offset: ?>>
    %30 = arith.minsi %14, %c4 : index
    %31 = arith.muli %15, %c64 : index
    %32 = arith.minsi %30, %16 : index
    %33 = arith.minsi %31, %c32 : index
    %34 = arith.muli %18, %c10 : index
    %extracted_slice = tensor.extract_slice %26[0, 0, 0] [%32, %34, %33] [1, 1, 1] : tensor<4x10x64xf32> to tensor<?x?x?xf32>
    %subview_7 = memref.subview %reinterpret_cast_6[0, 0, 0] [%32, %34, %33] [1, 1, 1] : memref<4x10x64xf32, strided<[2560, 64, 1], offset: ?>> to memref<?x?x?xf32, strided<[2560, 64, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?x?x?xf32>, memref<?x?x?xf32, strided<[2560, 64, 1], offset: ?>>) -> ()
    %35 = arith.addi %29, %c32 : index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [4, 10, 64], strides: [2560, 64, 1] : memref<?xf32> to memref<4x10x64xf32, strided<[2560, 64, 1], offset: ?>>
    %extracted_slice_9 = tensor.extract_slice %24[0, 0, 0] [%32, %34, %33] [1, 1, 1] : tensor<4x10x64xf32> to tensor<?x?x?xf32>
    %subview_10 = memref.subview %reinterpret_cast_8[0, 0, 0] [%32, %34, %33] [1, 1, 1] : memref<4x10x64xf32, strided<[2560, 64, 1], offset: ?>> to memref<?x?x?xf32, strided<[2560, 64, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_9 in writable %subview_10 : (tensor<?x?x?xf32>, memref<?x?x?xf32, strided<[2560, 64, 1], offset: ?>>) -> ()
  }
  return
}

// CHECK: func.func @triton_unk_fused_cat_2_outlined_vf_0
// CHECK: func.func @triton_unk_fused_cat_2_outlined_vf_1
// CHECK: func.func @triton_unk_fused_cat_2_outlined_vf_2