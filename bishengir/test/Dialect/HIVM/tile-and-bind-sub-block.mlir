// RUN: bishengir-opt %s -hivm-bind-sub-block -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @mm_01_mix_aiv(
func.func @mm_01_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  hivm.hir.set_ffts_base_addr %arg0
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg8, %arg7 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg6 : i32
  hivm.hir.set_mask_norm
  %5 = arith.muli %4, %c16_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = arith.muli %6, %c32 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%7], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf16> to memref<32x16xf16, strided<[16, 1]>>
  %8 = arith.muli %6, %c16 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%8], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
  %alloc = memref.alloc() : memref<16x32xf16>
  %9 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf16>
  %alloc_3 = memref.alloc() : memref<32x16xf16>
  %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<32x16xf16>
  %alloc_4 = memref.alloc() : memref<16x16xf16>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<16x16xf16, strided<[16, 1], offset: ?>>) outs(%alloc_4 : memref<16x16xf16>)
  %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x16xf16>
  %12 = tensor.empty() : tensor<16x16xf32>
  %13 = tensor.empty() : tensor<16x16xf16>
  %view = memref.view %arg1[%c0][] : memref<?xi8> to memref<48x16x16xf16>
  %14 = hivm.hir.get_block_idx -> i64
  %15 = arith.index_cast %14 : i64 to index
  %subview = memref.subview %view[%15, 0, 0] [1, 16, 16] [1, 1, 1] : memref<48x16x16xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
  %16 = bufferization.to_tensor %subview restrict writable : memref<16x16xf16, strided<[16, 1], offset: ?>>
  %17 = tensor.empty() : tensor<16x16xf16>
  %18 = hivm.hir.load ins(%16 : tensor<16x16xf16>) outs(%17 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %19 = hivm.hir.vadd ins(%18, %11 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%13 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %20 = arith.addi %6, %c16 : index
  %21 = arith.maxsi %6, %c16 : index
  %22 = arith.minsi %20, %21 : index
  %23 = arith.subi %22, %6 : index
  %24 = arith.minsi %23, %c16 : index
  %extracted_slice = tensor.extract_slice %19[0, 0] [%24, 16] [1, 1] : tensor<16x16xf16> to tensor<?x16xf16>
  %subview_5 = memref.subview %reinterpret_cast_2[0, 0] [%24, 16] [1, 1] : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
  // CHECK: scf.if
  // CHECK: hivm.hir.store
  // CHECK: limit_sub_block_id0
  hivm.hir.store ins(%extracted_slice : tensor<?x16xf16>) outs(%subview_5 : memref<?x16xf16, strided<[16, 1], offset: ?>>)
  return
}

// -----

// CHECK-LABEL:   func.func @_attn_fwd_mix_aiv(
// CHECK:           %[[VAL_24:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_26:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_27:.*]] = %[[VAL_24]] to %[[VAL_26]] step %[[VAL_25]] {
// CHECK:               %[[VAL_68:.*]] = hivm.hir.load ins(%[[VAL_66:.*]] : tensor<32x64xf32>)
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
#map = affine_map<(d0)[s0] -> (d0 * 28672 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c28672 = arith.constant 28672 : index
    return %c28672 : index
  }
  func.func @_attn_fwd_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %true = arith.constant true
    %cst = arith.constant 1.44269502 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 0.72134751 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %cst_4 = arith.constant 0.693147182 : f32
    %cst_5 = arith.constant 2.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c8192 = arith.constant 8192 : index
    %c12288 = arith.constant 12288 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg10 : i32
    %5 = arith.remsi %4, %arg9 : i32
    %6 = arith.muli %arg10, %arg9 : i32
    %7 = arith.divsi %3, %6 : i32
    %8 = arith.remsi %7, %arg8 : i32
    %9 = tensor.empty() : tensor<1xf32>
    %10 = tensor.empty() : tensor<64xf32>
    %11 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
    %12 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
    %13 = tensor.empty() : tensor<64x32xf32>
    %14 = tensor.empty() : tensor<64x64xf32>
    %15 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %16 = arith.divsi %5, %c2_i32 : i32
    %17 = arith.remsi %5, %c2_i32 : i32
    %18 = arith.extsi %16 : i32 to i64
    %19 = arith.muli %18, %c131072_i64 : i64
    %20 = arith.extsi %17 : i32 to i64
    %21 = arith.muli %20, %c65536_i64 : i64
    %22 = arith.addi %19, %21 : i64
    %23 = arith.index_cast %22 : i64 to index
    %24 = arith.muli %8, %c64_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = arith.muli %25, %c64 : index
    %27 = arith.addi %26, %23 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%27], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %28 = tensor.empty() : tensor<1xf32>
    %29 = hivm.hir.vbrc ins(%arg7 : f32) outs(%28 : tensor<1xf32>) -> tensor<1xf32>
    %30 = hivm.hir.vmul ins(%29, %cst : tensor<1xf32>, f32) outs(%9 : tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %30[%c0] : tensor<1xf32>
    %alloc = memref.alloc() : memref<64x64xf16>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
    %31 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast_7 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    %cast_9 = memref.cast %reinterpret_cast_8 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %32:9 = scf.for %arg11 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg12 = %11, %arg13 = %15, %arg14 = %12, %arg15 = %cast, %arg16 = %cast_9, %arg17 = %23, %arg18 = %c0, %arg19 = %23, %arg20 = %c0) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<32x64xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_11 = memref.alloc() : memref<32x64xf16>
      %46 = bufferization.to_tensor %alloc_11 restrict writable : memref<32x64xf16>
      %47 = tensor.empty() : tensor<64x32xf16>
      %48 = tensor.empty() : tensor<64x32xf32>
      %49 = hivm.hir.get_block_idx -> i64
      %50 = arith.index_cast %49 : i64 to index
      %51 = affine.apply #map(%50)[%c0]
      %view = memref.view %arg1[%51][] : memref<?xi8> to memref<64x32xf32>
      %52 = bufferization.to_tensor %view restrict writable : memref<64x32xf32>
      %53 = tensor.empty() : tensor<64x32xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
      %54 = hivm.hir.load ins(%52 : tensor<64x32xf32>) outs(%53 : tensor<64x32xf32>) init_out_buffer = false -> tensor<64x32xf32>
      %55 = tensor.empty() : tensor<64x32xf32>
      %56 = hivm.hir.load ins(%52 : tensor<64x32xf32>) outs(%55 : tensor<64x32xf32>) init_out_buffer = false -> tensor<64x32xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
      %expanded_12 = tensor.expand_shape %12 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %57 = hivm.hir.vreduce <max> ins(%54 : tensor<64x32xf32>) outs(%expanded_12 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
      %collapsed = tensor.collapse_shape %57 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %58 = hivm.hir.vmul ins(%collapsed, %extracted : tensor<64xf32>, f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %59 = hivm.hir.vmax ins(%arg14, %58 : tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %60 = hivm.hir.vmul ins(%56, %extracted : tensor<64x32xf32>, f32) outs(%13 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %expanded_13 = tensor.expand_shape %59 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %61 = hivm.hir.vbrc ins(%expanded_13 : tensor<64x1xf32>) outs(%13 : tensor<64x32xf32>) broadcast_dims = [1] -> tensor<64x32xf32>
      %62 = hivm.hir.vsub ins(%60, %61 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%13 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %63 = hivm.hir.vmul ins(%62, %cst_4 : tensor<64x32xf32>, f32) outs(%13 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %64 = hivm.hir.vexp ins(%63 : tensor<64x32xf32>) outs(%13 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %65 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %expanded_14 = tensor.expand_shape %65 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %66 = hivm.hir.vreduce <sum> ins(%64 : tensor<64x32xf32>) outs(%expanded_14 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
      %collapsed_15 = tensor.collapse_shape %66 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %67 = hivm.hir.vsub ins(%arg14, %59 : tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %68 = hivm.hir.vmul ins(%67, %cst_4 : tensor<64xf32>, f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %69 = hivm.hir.vexp ins(%68 : tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %70 = hivm.hir.vmul ins(%arg12, %69 : tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %71 = hivm.hir.vadd ins(%70, %collapsed_15 : tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %expanded_16 = tensor.expand_shape %69 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %72 = hivm.hir.vbrc ins(%expanded_16 : tensor<64x1xf32>) outs(%14 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
      %73 = hivm.hir.vmul ins(%arg13, %72 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %alloc_17 = memref.alloc() : memref<32x64xf16>
      %74 = bufferization.to_tensor %alloc_17 restrict writable : memref<32x64xf16>
      %75 = hivm.hir.vcast ins(%64 : tensor<64x32xf32>) outs(%47 : tensor<64x32xf16>) -> tensor<64x32xf16>
      %76 = hivm.hir.get_block_idx -> i64
      %77 = arith.index_cast %76 : i64 to index
      %78 = affine.apply #map(%77)[%c8192]
      %view_18 = memref.view %arg1[%78][] : memref<?xi8> to memref<64x32xf16>
      %79 = bufferization.to_tensor %view_18 restrict writable : memref<64x32xf16>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
      %80 = hivm.hir.store ins(%75 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
      annotation.mark %80 : tensor<64x32xf16>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
      %81 = tensor.empty() : tensor<64x32xf16>
      %82 = tensor.empty() : tensor<64x64xf32>
      %83 = hivm.hir.get_block_idx -> i64
      %84 = arith.index_cast %83 : i64 to index
      %85 = affine.apply #map(%84)[%c12288]
      %view_19 = memref.view %arg1[%85][] : memref<?xi8> to memref<64x64xf32>
      %86 = bufferization.to_tensor %view_19 restrict writable : memref<64x64xf32>
      %87 = tensor.empty() : tensor<64x64xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
      %88 = hivm.hir.load ins(%86 : tensor<64x64xf32>) outs(%87 : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
      %89 = tensor.empty() : tensor<64x64xf32>
      %90 = hivm.hir.vadd ins(%88, %73 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%89 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %91 = hivm.hir.vmul ins(%59, %extracted : tensor<64xf32>, f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %92 = hivm.hir.vdiv ins(%91, %cst_3 : tensor<64xf32>, f32) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
      %93 = arith.addi %arg17, %c2048 : index
      %94 = arith.addi %93, %arg18 : index
      %reinterpret_cast_20 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %cast_21 = memref.cast %reinterpret_cast_20 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
      %95 = arith.addi %arg19, %c2048 : index
      %96 = arith.addi %95, %arg20 : index
      %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%96], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %cast_23 = memref.cast %reinterpret_cast_22 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %71, %90, %92, %cast_21, %cast_23, %94, %c0, %96, %c0 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<32x64xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %33 = hivm.hir.vln ins(%32#0 : tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
    %34 = tensor.empty() : tensor<64xf32>
    %35 = hivm.hir.vbrc ins(%cst_5 : f32) outs(%34 : tensor<64xf32>) -> tensor<64xf32>
    %36 = hivm.hir.vln ins(%35 : tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
    %37 = hivm.hir.vdiv ins(%33, %36 : tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
    %38 = hivm.hir.vadd ins(%32#2, %37 : tensor<64xf32>, tensor<64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
    %expanded = tensor.expand_shape %32#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %39 = hivm.hir.vbrc ins(%expanded : tensor<64x1xf32>) outs(%14 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
    %40 = hivm.hir.vdiv ins(%32#1, %39 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %41 = arith.muli %5, %c1024_i32 : i32
    %42 = arith.index_cast %41 : i32 to index
    %43 = arith.addi %42, %25 : index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%43], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%38 : tensor<64xf32>) outs(%reinterpret_cast_10 : memref<64xf32, strided<[1], offset: ?>>)
    %44 = tensor.empty() : tensor<64x64xf16>
    %45 = hivm.hir.vcast ins(%40 : tensor<64x64xf32>) outs(%44 : tensor<64x64xf16>) -> tensor<64x64xf16>
    hivm.hir.store ins(%45 : tensor<64x64xf16>) outs(%reinterpret_cast_6 : memref<64x64xf16, strided<[64, 1], offset: ?>>)
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
    return
  }
}


// -----

// CHECK-LABEL:   func.func @_attn_fwd_mix_aiv_plain(
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:             %[[VAL_61:.*]] = hivm.hir.load ins(%[[VAL_59:.*]] : tensor<32x64xf32>) outs(%[[VAL_60:.*]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:             %[[VAL_103:.*]] = hivm.hir.load ins(%[[VAL_102:.*]] : tensor<32x64xf32>) outs(%[[VAL_60:.*]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
func.func @_attn_fwd_mix_aiv_plain(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %true = arith.constant true
  %cst = arith.constant 1.44269502 : f32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 0xFF800000 : f32
  %cst_2 = arith.constant 0.72134751 : f32
  %c64_i32 = arith.constant 64 : i32
  %c4096_i64 = arith.constant 4096 : i64
  %c131072_i64 = arith.constant 131072 : i64
  %c32_i32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : index
  %cst_3 = arith.constant 0.693147182 : f32
  %cst_4 = arith.constant 2.000000e+00 : f32
  %cst_5 = arith.constant -1.000000e+00 : f32
  %c16384 = arith.constant 16384 : index
  %c24576 = arith.constant 24576 : index
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.divsi %1, %arg10 : i32
  %3 = arith.remsi %2, %arg9 : i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.divsi %1, %4 : i32
  %6 = arith.remsi %5, %arg8 : i32
  %7 = tensor.empty() : tensor<1xf32>
  %8 = tensor.empty() : tensor<64x1xf32>
  %9 = tensor.empty() : tensor<64xf32>
  %expanded = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_6 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_7 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_8 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_9 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_10 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_11 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_12 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_13 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_14 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_15 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_16 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_17 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_18 = tensor.expand_shape %9 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %10 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%expanded : tensor<64x1xf32>) -> tensor<64x1xf32>
  %11 = tensor.empty() : tensor<64x64xf32>
  %12 = arith.divsi %3, %c32_i32 : i32
  %13 = arith.remsi %3, %c32_i32 : i32
  %14 = arith.extsi %12 : i32 to i64
  %15 = arith.muli %14, %c131072_i64 : i64
  %16 = arith.extsi %13 : i32 to i64
  %17 = arith.muli %16, %c4096_i64 : i64
  %18 = arith.addi %15, %17 : i64
  %19 = arith.index_cast %18 : i64 to index
  %20 = arith.muli %6, %c64_i32 : i32
  %21 = arith.index_cast %20 : i32 to index
  %22 = arith.muli %21, %c64 : index
  %23 = arith.addi %22, %19 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
  %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%19], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
  %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
  %reinterpret_cast_21 = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
  %24 = tensor.empty() : tensor<1xf32>
  %25 = hivm.hir.vbrc ins(%arg7 : f32) outs(%24 : tensor<1xf32>) -> tensor<1xf32>
  %26 = hivm.hir.vmul ins(%25, %cst : tensor<1xf32>, f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %26[%c0] : tensor<1xf32>
  %alloc = memref.alloc() : memref<64x64xf16>
  %27 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
  %alloc_22 = memref.alloc() : memref<64x64xf16>
  %28 = bufferization.to_tensor %alloc_22 restrict writable : memref<64x64xf16>
  %29 = tensor.empty() : tensor<64x64xf16>
  %30 = tensor.empty() : tensor<64x64xf32>
  %31 = hivm.hir.get_block_idx -> i64
  %32 = arith.index_cast %31 : i64 to index
  %33 = affine.apply affine_map<(d0)[s0] -> (d0 * 40960 + s0)>(%32)[%c0]
  %view = memref.view %arg1[%33][] : memref<?xi8> to memref<64x64xf32>
  %34 = bufferization.to_tensor %view restrict writable : memref<64x64xf32>
  %35 = tensor.empty() : tensor<64x64xf32>
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
  %36 = hivm.hir.load ins(%34 : tensor<64x64xf32>) outs(%35 : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
  %37 = tensor.empty() : tensor<64x64xf32>
  %38 = hivm.hir.load ins(%34 : tensor<64x64xf32>) outs(%37 : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
  %39 = hivm.hir.vreduce <max> ins(%36 : tensor<64x64xf32>) outs(%10 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
  %40 = hivm.hir.vmul ins(%39, %extracted : tensor<64x1xf32>, f32) outs(%expanded_12 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %41 = hivm.hir.vmul ins(%38, %extracted : tensor<64x64xf32>, f32) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %42 = hivm.hir.vbrc ins(%40 : tensor<64x1xf32>) outs(%11 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
  %43 = hivm.hir.vsub ins(%41, %42 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %44 = hivm.hir.vmul ins(%43, %cst_3 : tensor<64x64xf32>, f32) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %45 = hivm.hir.vexp ins(%44 : tensor<64x64xf32>) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %46 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%expanded_6 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %47 = hivm.hir.vreduce <sum> ins(%45 : tensor<64x64xf32>) outs(%46 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
  %48 = hivm.hir.vmul ins(%40, %cst_5 : tensor<64x1xf32>, f32) outs(%expanded_11 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %49 = hivm.hir.vadd ins(%48, %cst_1 : tensor<64x1xf32>, f32) outs(%expanded_10 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %50 = hivm.hir.vmul ins(%49, %cst_3 : tensor<64x1xf32>, f32) outs(%expanded_9 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %51 = hivm.hir.vexp ins(%50 : tensor<64x1xf32>) outs(%expanded_8 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %52 = hivm.hir.vadd ins(%51, %47 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%expanded_18 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %53 = hivm.hir.vmul ins(%51, %cst_0 : tensor<64x1xf32>, f32) outs(%8 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %54 = hivm.hir.vbrc ins(%53 : tensor<64x1xf32>) outs(%11 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
  %alloc_23 = memref.alloc() : memref<64x64xf16>
  %55 = bufferization.to_tensor %alloc_23 restrict writable : memref<64x64xf16>
  %56 = hivm.hir.vcast ins(%45 : tensor<64x64xf32>) outs(%29 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %57 = hivm.hir.get_block_idx -> i64
  %58 = arith.index_cast %57 : i64 to index
  %59 = affine.apply affine_map<(d0)[s0] -> (d0 * 40960 + s0)>(%58)[%c16384]
  %view_24 = memref.view %arg1[%59][] : memref<?xi8> to memref<64x64xf16>
  %60 = bufferization.to_tensor %view_24 restrict writable : memref<64x64xf16>
  %61 = hivm.hir.store ins(%56 : tensor<64x64xf16>) outs(%60 : tensor<64x64xf16>) -> tensor<64x64xf16>
  annotation.mark %61 : tensor<64x64xf16>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 0
  %62 = tensor.empty() : tensor<64x64xf16>
  %63 = tensor.empty() : tensor<64x64xf32>
  %64 = hivm.hir.get_block_idx -> i64
  %65 = arith.index_cast %64 : i64 to index
  %66 = affine.apply affine_map<(d0)[s0] -> (d0 * 40960 + s0)>(%65)[%c24576]
  %view_25 = memref.view %arg1[%66][] : memref<?xi8> to memref<64x64xf32>
  %67 = bufferization.to_tensor %view_25 restrict writable : memref<64x64xf32>
  %68 = tensor.empty() : tensor<64x64xf32>
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
  %69 = hivm.hir.load ins(%67 : tensor<64x64xf32>) outs(%68 : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
  %70 = tensor.empty() : tensor<64x64xf32>
  %71 = hivm.hir.vadd ins(%69, %54 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%70 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %72 = hivm.hir.vmul ins(%40, %extracted : tensor<64x1xf32>, f32) outs(%expanded_13 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %73 = hivm.hir.vdiv ins(%72, %cst_2 : tensor<64x1xf32>, f32) outs(%expanded_14 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %74 = hivm.hir.vln ins(%52 : tensor<64x1xf32>) outs(%expanded_17 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %75 = tensor.empty() : tensor<64xf32>
  %expanded_26 = tensor.expand_shape %75 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %76 = hivm.hir.vbrc ins(%cst_4 : f32) outs(%expanded_26 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %77 = hivm.hir.vln ins(%76 : tensor<64x1xf32>) outs(%expanded_7 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %78 = hivm.hir.vdiv ins(%74, %77 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%expanded_16 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %79 = hivm.hir.vadd ins(%73, %78 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%expanded_15 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %80 = hivm.hir.vbrc ins(%52 : tensor<64x1xf32>) outs(%11 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
  %81 = hivm.hir.vdiv ins(%71, %80 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %82 = arith.muli %3, %c64_i32 : i32
  %83 = arith.index_cast %82 : i32 to index
  %84 = arith.addi %83, %21 : index
  %reinterpret_cast_27 = memref.reinterpret_cast %arg5 to offset: [%84], sizes: [64, 1], strides: [1, 1] : memref<?xf32> to memref<64x1xf32, strided<[1, 1], offset: ?>>
  hivm.hir.store ins(%79 : tensor<64x1xf32>) outs(%reinterpret_cast_27 : memref<64x1xf32, strided<[1, 1], offset: ?>>)
  %85 = hivm.hir.vcast ins(%81 : tensor<64x64xf32>) outs(%29 : tensor<64x64xf16>) -> tensor<64x64xf16>
  hivm.hir.store ins(%85 : tensor<64x64xf16>) outs(%reinterpret_cast_21 : memref<64x64xf16, strided<[64, 1], offset: ?>>)
  return
}

// -----

// CHECK-LABEL:   func.func @matmul_x_w_bias_down_up_fused_layer_1_kernel_mix_aiv(
#map = affine_map<(d0)[s0] -> (d0 * 3072 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @matmul_x_w_bias_down_up_fused_layer_1_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<20xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.divsi %1, %arg19 : i32
    %3 = arith.remsi %2, %arg18 : i32
    %4 = arith.muli %arg19, %arg18 : i32
    %5 = arith.divsi %1, %4 : i32
    %6 = arith.remsi %5, %arg17 : i32
    %7 = tensor.empty() : tensor<16x16xf32>
    %8 = arith.muli %6, %c16_i32 : i32
    %9 = arith.muli %3, %c16_i32 : i32
    %10 = arith.index_cast %8 : i32 to index
    %11 = arith.index_cast %arg12 : i32 to index
    %12 = arith.muli %10, %11 : index
    %13 = arith.index_cast %arg13 : i32 to index
    %14 = arith.index_cast %9 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1], offset: ?>>
    %15 = arith.index_cast %arg14 : i32 to index
    %16 = arith.index_cast %arg15 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [%14], sizes: [32, 16], strides: [%16, 1] : memref<?xf16> to memref<32x16xf16, strided<[?, 1], offset: ?>>
    %17 = arith.index_cast %arg16 : i32 to index
    %18 = arith.muli %10, %17 : index
    %19 = arith.addi %18, %14 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [%19], sizes: [16, 16], strides: [%17, 1] : memref<?xf32> to memref<16x16xf32, strided<[?, 1], offset: ?>>
    %20 = arith.addi %arg11, %c15_i32 : i32
    %21 = arith.divsi %20, %c16_i32 : i32
    %22 = arith.muli %arg13, %c16_i32 : i32
    %23 = arith.muli %arg14, %c16_i32 : i32
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [16, 16], strides: [%11, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast_2 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [16, 16], strides: [%13, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
    %cast_4 = memref.cast %reinterpret_cast_3 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 32], strides: [%15, 1] : memref<?xf16> to memref<16x32xf16, strided<[?, 1]>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<16x32xf16, strided<[?, 1]>> to memref<16x32xf16, strided<[?, ?], offset: ?>>
    %24 = tensor.empty() : tensor<16x16xf32>
    %25 = tensor.empty() : tensor<16x32xf32>
    %26:11 = scf.for %arg20 = %c0_i32 to %21 step %c1_i32 iter_args(%arg21 = %24, %arg22 = %25, %arg23 = %cast, %arg24 = %cast_4, %arg25 = %cast_6, %arg26 = %12, %arg27 = %c0, %arg28 = %14, %arg29 = %c0, %arg30 = %c0, %arg31 = %c0) -> (tensor<16x16xf32>, tensor<16x32xf32>, memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index, index, index)  : i32 {
      %alloc_10 = memref.alloc() : memref<16x16xf16>
      %53 = bufferization.to_tensor %alloc_10 restrict writable : memref<16x16xf16>
      %alloc_11 = memref.alloc() : memref<16x16xf16>
      %54 = bufferization.to_tensor %alloc_11 restrict writable : memref<16x16xf16>
      %alloc_12 = memref.alloc() : memref<16x32xf16>
      %55 = bufferization.to_tensor %alloc_12 restrict writable : memref<16x32xf16>
      %56 = arith.cmpi eq, %arg20, %c0_i32 : i32
      %57 = arith.cmpi eq, %arg20, %c0_i32 : i32
      %58 = arith.addi %arg26, %c16 : index
      %59 = arith.addi %58, %arg27 : index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [16, 16], strides: [%11, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %cast_14 = memref.cast %reinterpret_cast_13 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
      %60 = arith.index_cast %22 : i32 to index
      %61 = arith.addi %arg28, %60 : index
      %62 = arith.addi %61, %arg29 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg4 to offset: [%62], sizes: [16, 16], strides: [%13, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %cast_16 = memref.cast %reinterpret_cast_15 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
      %63 = arith.index_cast %23 : i32 to index
      %64 = arith.addi %arg30, %63 : index
      %65 = arith.addi %64, %arg31 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [16, 32], strides: [%15, 1] : memref<?xf16> to memref<16x32xf16, strided<[?, 1], offset: ?>>
      %cast_18 = memref.cast %reinterpret_cast_17 : memref<16x32xf16, strided<[?, 1], offset: ?>> to memref<16x32xf16, strided<[?, ?], offset: ?>>
      scf.yield %arg21, %arg22, %cast_14, %cast_16, %cast_18, %59, %c0, %62, %c0, %65, %c0 : tensor<16x16xf32>, tensor<16x32xf32>, memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index, index, index
    }
    %27 = hivm.hir.get_block_idx -> i64
    %28 = arith.index_cast %27 : i64 to index
    %29 = affine.apply #map(%28)[%c0]
    %view = memref.view %arg2[%29][] : memref<?xi8> to memref<16x16xf32>
    %30 = bufferization.to_tensor %view restrict writable : memref<16x16xf32>
    %31 = tensor.empty() : tensor<16x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %32 = hivm.hir.load ins(%30 : tensor<16x16xf32>) outs(%31 : tensor<16x16xf32>) init_out_buffer = false -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16xf16, strided<[1], offset: ?>>) outs(%alloc : memref<16xf16>) init_out_buffer = false
    // CHECK:           %[[VAL_23:.*]] = bufferization.to_tensor %alloc
    // CHECK-NOT:       %[[VAL_24:.*]] = tensor.extract_slice %[[VAL_23]]
    %33 = bufferization.to_tensor %alloc restrict writable : memref<16xf16>
    %34 = tensor.empty() : tensor<16xf32>
    %35 = hivm.hir.vcast ins(%33 : tensor<16xf16>) outs(%34 : tensor<16xf32>) -> tensor<16xf32>
    %expanded = tensor.expand_shape %35 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
    %36 = hivm.hir.vbrc ins(%expanded : tensor<1x16xf32>) outs(%7 : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
    %37 = hivm.hir.vadd ins(%32, %36 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_7 = memref.alloc() : memref<32x16xf16>
    %38 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x16xf16>
    %39 = hivm.hir.get_block_idx -> i64
    %40 = arith.index_cast %39 : i64 to index
    %41 = affine.apply #map(%40)[%c1024]
    %view_8 = memref.view %arg2[%41][] : memref<?xi8> to memref<16x32xf16>
    %42 = bufferization.to_tensor %view_8 restrict writable : memref<16x32xf16>
    %43 = tensor.empty() : tensor<16x32xf16>
    %44 = tensor.empty() : tensor<16x16xf32>
    %45 = hivm.hir.get_block_idx -> i64
    %46 = arith.index_cast %45 : i64 to index
    %47 = affine.apply #map(%46)[%c2048]
    %view_9 = memref.view %arg2[%47][] : memref<?xi8> to memref<16x16xf32>
    %48 = bufferization.to_tensor %view_9 restrict writable : memref<16x16xf32>
    %49 = tensor.empty() : tensor<16x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %50 = hivm.hir.load ins(%48 : tensor<16x16xf32>) outs(%49 : tensor<16x16xf32>) init_out_buffer = false -> tensor<16x16xf32>
    %51 = tensor.empty() : tensor<16x16xf32>
    %52 = hivm.hir.vadd ins(%50, %37 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%51 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%52 : tensor<16x16xf32>) outs(%reinterpret_cast_1 : memref<16x16xf32, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @triton_dot_2_mix_aic(


module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @triton_dot_2_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc : memref<16x16xf16>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf16>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_3 = memref.alloc() : memref<16x16xf32>
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = hivm.hir.mmadL1 ins(%2, %3, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_4 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_4 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins({{.*}} : tensor<16x16xf32>) outs({{.*}} : memref<8x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6 : tensor<16x16xf32>) outs(%alloc_4 : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %7 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %8 = tensor.empty() : tensor<16x16xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    return
  }
  // CHECK-LABEL:   func.func @triton_dot_2_mix_aiv(
  func.func @triton_dot_2_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xf16>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_3 = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_3 : memref<16x16xf32>)
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    %alloc_4 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_4 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %6 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %7 = tensor.empty() : tensor<16x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %8 = hivm.hir.vadd ins(%6, %4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    // CHECK: hivm.hir.store ins({{.*}} : tensor<8x16xf32>) outs({{.*}} : memref<8x16xf32, strided<[16, 1], offset: ?>>) {tiled_op}
    hivm.hir.store ins(%8 : tensor<16x16xf32>) outs(%reinterpret_cast_5 : memref<16x16xf32, strided<[16, 1]>>)
    // CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}

// -----

// CHECK-LABEL:   func.func @_hstu_attn_fwd_mix_aiv(
// CHECK:        %[[VAL_39:.*]] = bufferization.to_tensor %[[memspacecast_5:.*]] restrict writable : memref<1x1x16x16xf16>
// CHECK:        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
// CHECK:        hivm.hir.copy ins(%[[VAL_38:.*]] : tensor<1x1x16x16xf16>) outs(%[[memspacecast_5:.*]] : memref<1x1x16x16xf16>)
// CHECK:        annotation.mark %[[VAL_39:.*]] : tensor<1x1x16x16xf16>
func.func @_hstu_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: f32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32, %arg30: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<31xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, parallel_mode = "simd"} {
  %cst = arith.constant -1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 1.000000e+00 : f32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg28, %arg29 : i32
  %1 = arith.muli %0, %arg30 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg30 : i32
  %5 = arith.remsi %4, %arg29 : i32
  %6 = arith.muli %arg30, %arg29 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg28 : i32
  %9 = tensor.empty() : tensor<1xf32>
  %10 = tensor.empty() : tensor<16x16xf32>
  %11 = arith.divsi %5, %arg20 : i32
  %12 = arith.index_cast %11 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
  %13 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %14 = arith.addi %12, %c1 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %15 = memref.load %reinterpret_cast_1[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %16 = arith.subi %15, %13 : i64
  %17 = arith.trunci %16 : i64 to i32
  %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %18 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %19 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %20 = arith.subi %19, %18 : i64
  %21 = arith.trunci %20 : i64 to i32
  %22 = arith.muli %8, %c16_i32 : i32
  %23 = arith.cmpi slt, %22, %17 : i32
  scf.if %23 {
    %24 = arith.sitofp %arg21 : i32 to f32
    %inserted = tensor.insert %24 into %9[%c0] : tensor<1xf32>
    %25 = tensor.empty() : tensor<1xf32>
    %26 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%25 : tensor<1xf32>) -> tensor<1xf32>
    %27 = hivm.hir.vdiv ins(%26, %inserted : tensor<1xf32>, tensor<1xf32>) outs(%9 : tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %27[%c0] : tensor<1xf32>
    %28:3 = scf.for %arg31 = %c0_i32 to %21 step %c16_i32 iter_args(%arg32 = %c0_i32, %arg33 = %c0_i32, %arg34 = %c0_i32) -> (i32, i32, i32)  : i32 {
      %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %29 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %30 = hivm.hir.vmul ins(%29, %arg16 : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " qk: ", tcoretype = #hivm.tcore_type<VECTOR>} %30 : tensor<16x16xf32>
      %31 = hivm.hir.vmul ins(%30, %cst : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %32 = hivm.hir.vexp ins(%31 : tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %33 = hivm.hir.vadd ins(%32, %cst_0 : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %34 = hivm.hir.vdiv ins(%30, %33 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %35 = hivm.hir.vmul ins(%34, %extracted : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %36 = tensor.empty() : tensor<16x16xf16>
      %37 = hivm.hir.vcast ins(%35 : tensor<16x16xf32>) outs(%36 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %expanded = tensor.expand_shape %37 [[0, 1], [2, 3]] output_shape [1, 16, 1, 16] : tensor<16x16xf16> into tensor<1x16x1x16xf16>
      %38 = tensor.empty() : tensor<1x1x16x16xf16>
      %39 = hivm.hir.vtranspose ins(%expanded : tensor<1x16x1x16xf16>) outs(%38 : tensor<1x1x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<1x1x16x16xf16>
      %alloc_4 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %memspacecast_5 = memref.memory_space_cast %alloc_4 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
      %40 = bufferization.to_tensor %memspacecast_5 restrict writable : memref<1x1x16x16xf16>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
      hivm.hir.copy ins(%39 : tensor<1x1x16x16xf16>) outs(%memspacecast_5 : memref<1x1x16x16xf16>)
      annotation.mark %40 : tensor<1x1x16x16xf16>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %42 = arith.addi %arg32, %c16_i32 : i32
      %43 = arith.addi %arg33, %c16_i32 : i32
      %44 = arith.addi %arg34, %c16_i32 : i32
      scf.yield %42, %43, %44 : i32, i32, i32
    }
  }
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
  return
}

// -----

// CHECK-LABEL:   func.func @_hstu_attn_fwd_mix_aiv_with_result(
// CHECK:        %[[VAL_41:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:        %[[VAL_42:.*]] = arith.index_cast %[[VAL_41:.*]] : i64 to index
// CHECK:        %[[VAL_43:.*]] = arith.cmpi eq, %[[VAL_42:.*]], %[[VAL_c0:.*]] : index
// CHECK:        %[[VAL_44:.*]] = scf.if %[[VAL_43:.*]] -> (tensor<1x1x16x16xf16>) {
// CHECK:          %[[VAL_48:.*]] = hivm.hir.copy ins(%[[VAL_39:.*]] : tensor<1x1x16x16xf16>) outs(%[[VAL_40:.*]] : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK:          scf.yield %[[VAL_48:.*]] : tensor<1x1x16x16xf16>
// CHECK:        } else {
// CHECK:          scf.yield %[[VAL_40:.*]] : tensor<1x1x16x16xf16>
// CHECK:        } {limit_sub_block_id0}
func.func @_hstu_attn_fwd_mix_aiv_with_result(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: f32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32, %arg30: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<31xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, parallel_mode = "simd"} {
  %cst = arith.constant -1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 1.000000e+00 : f32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg28, %arg29 : i32
  %1 = arith.muli %0, %arg30 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg30 : i32
  %5 = arith.remsi %4, %arg29 : i32
  %6 = arith.muli %arg30, %arg29 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg28 : i32
  %9 = tensor.empty() : tensor<1xf32>
  %10 = tensor.empty() : tensor<16x16xf32>
  %11 = arith.divsi %5, %arg20 : i32
  %12 = arith.index_cast %11 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
  %13 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %14 = arith.addi %12, %c1 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %15 = memref.load %reinterpret_cast_1[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %16 = arith.subi %15, %13 : i64
  %17 = arith.trunci %16 : i64 to i32
  %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %18 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %19 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %20 = arith.subi %19, %18 : i64
  %21 = arith.trunci %20 : i64 to i32
  %22 = arith.muli %8, %c16_i32 : i32
  %23 = arith.cmpi slt, %22, %17 : i32
  scf.if %23 {
    %24 = arith.sitofp %arg21 : i32 to f32
    %inserted = tensor.insert %24 into %9[%c0] : tensor<1xf32>
    %25 = tensor.empty() : tensor<1xf32>
    %26 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%25 : tensor<1xf32>) -> tensor<1xf32>
    %27 = hivm.hir.vdiv ins(%26, %inserted : tensor<1xf32>, tensor<1xf32>) outs(%9 : tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %27[%c0] : tensor<1xf32>
    %28:3 = scf.for %arg31 = %c0_i32 to %21 step %c16_i32 iter_args(%arg32 = %c0_i32, %arg33 = %c0_i32, %arg34 = %c0_i32) -> (i32, i32, i32)  : i32 {
      %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %29 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %30 = hivm.hir.vmul ins(%29, %arg16 : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " qk: ", tcoretype = #hivm.tcore_type<VECTOR>} %30 : tensor<16x16xf32>
      %31 = hivm.hir.vmul ins(%30, %cst : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %32 = hivm.hir.vexp ins(%31 : tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %33 = hivm.hir.vadd ins(%32, %cst_0 : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %34 = hivm.hir.vdiv ins(%30, %33 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %35 = hivm.hir.vmul ins(%34, %extracted : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %36 = tensor.empty() : tensor<16x16xf16>
      %37 = hivm.hir.vcast ins(%35 : tensor<16x16xf32>) outs(%36 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %expanded = tensor.expand_shape %37 [[0, 1], [2, 3]] output_shape [1, 16, 1, 16] : tensor<16x16xf16> into tensor<1x16x1x16xf16>
      %38 = tensor.empty() : tensor<1x1x16x16xf16>
      %39 = hivm.hir.vtranspose ins(%expanded : tensor<1x16x1x16xf16>) outs(%38 : tensor<1x1x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<1x1x16x16xf16>
      %alloc_4 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %memspacecast_5 = memref.memory_space_cast %alloc_4 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
      %40 = bufferization.to_tensor %memspacecast_5 restrict writable : memref<1x1x16x16xf16>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
      %41 = hivm.hir.copy ins(%39 : tensor<1x1x16x16xf16>) outs(%40 : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
      annotation.mark %41 : tensor<1x1x16x16xf16>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %42 = arith.addi %arg32, %c16_i32 : i32
      %43 = arith.addi %arg33, %c16_i32 : i32
      %44 = arith.addi %arg34, %c16_i32 : i32
      scf.yield %42, %43, %44 : i32, i32, i32
    }
  }
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
  return
}

// -----

// CHECK-LABEL:   func.func @fa_attn_fwd_mix_aiv(
// CHECK:      %[[VAL_38:.*]] = tensor.empty() : tensor<32x64xf16>
// CHECK:      %[[VAL_39:.*]] = hivm.hir.vcast ins(%[[VAL_37:.*]] : tensor<32x64xf32>) outs(%[[VAL_38:.*]] : tensor<32x64xf16>) -> tensor<32x64xf16>
// CHECK:      %[[VAL_40:.*]] = memref.subview %[[VAL_REINTERPRET_36:.*]][%1, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16, strided<[64, 1], offset: ?>>
// CHECK:      hivm.hir.store ins(%[[VAL_39:.*]] : tensor<32x64xf16>) outs(%[[VAL_40:.*]] : memref<32x64xf16, strided<[64, 1], offset: ?>>) {tiled_op}
func.func @fa_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, parallel_mode = "simd"} {
  %c64 = arith.constant 64 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 5.000000e-01 : f32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %c4194304_i64 = arith.constant 4194304 : i64
  %c524288_i64 = arith.constant 524288 : i64
  %c64_i32 = arith.constant 64 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c131072_i32 = arith.constant 131072 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst_2 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg7 : i32
  %7 = tensor.empty() : tensor<64x64xf32>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 3
  %8 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %9 = tensor.empty() : tensor<64xf32>
  %10 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
  %11 = hivm.hir.vbrc ins(%cst : f32) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
  scf.for %arg10 = %6 to %c131072_i32 step %c32_i32  : i32 {
    %12 = arith.divsi %arg10, %c128_i32 : i32
    %13 = arith.remsi %arg10, %c128_i32 : i32
    %14 = arith.divsi %12, %c8_i32 : i32
    %15 = arith.remsi %12, %c8_i32 : i32
    %16 = arith.extsi %14 : i32 to i64
    %17 = arith.muli %16, %c4194304_i64 : i64
    %18 = arith.extsi %15 : i32 to i64
    %19 = arith.muli %18, %c524288_i64 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = arith.index_cast %20 : i64 to index
    %22 = arith.muli %13, %c64_i32 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = arith.muli %23, %c64 : index
    %25 = arith.addi %24, %21 : index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %26:5 = scf.for %arg11 = %c0_i32 to %c8192_i32 step %c64_i32 iter_args(%arg12 = %11, %arg13 = %8, %arg14 = %10, %arg15 = %c0_i32, %arg16 = %c0_i32) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32, i32)  : i32 {
      %35 = tensor.empty() : tensor<64x64xf16>
      %alloc = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %36 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %37 = hivm.hir.vmul ins(%36, %cst_1 : tensor<64x64xf32>, f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      %38 = tensor.empty() : tensor<64x1xf32>
      %39 = hivm.hir.vreduce <max> ins(%37 : tensor<64x64xf32>) outs(%38 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
      %collapsed = tensor.collapse_shape %39 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %40 = hivm.hir.vmax ins(%arg14, %collapsed : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
      %expanded_4 = tensor.expand_shape %40 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %41 = hivm.hir.vsub ins(%37, %expanded_4 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%7 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
      %42 = hivm.hir.vexp ins(%41 : tensor<64x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %43 = hivm.hir.vcast ins(%42 : tensor<64x64xf32>) outs(%35 : tensor<64x64xf16>) -> tensor<64x64xf16>
      %44 = tensor.empty() : tensor<64x1xf32>
      %45 = hivm.hir.vreduce <sum> ins(%42 : tensor<64x64xf32>) outs(%44 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
      %collapsed_5 = tensor.collapse_shape %45 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %46 = hivm.hir.vsub ins(%arg14, %40 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
      %47 = hivm.hir.vexp ins(%46 : tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
      %48 = hivm.hir.vmul ins(%arg12, %47 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
      %49 = hivm.hir.vadd ins(%48, %collapsed_5 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
      %expanded_6 = tensor.expand_shape %47 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %50 = hivm.hir.vmul ins(%arg13, %expanded_6 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%7 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
      %expanded_7 = tensor.expand_shape %43 [[0, 1], [2, 3]] output_shape [4, 16, 4, 16] : tensor<64x64xf16> into tensor<4x16x4x16xf16>
      %51 = tensor.empty() : tensor<4x4x16x16xf16>
      %52 = hivm.hir.vtranspose ins(%expanded_7 : tensor<4x16x4x16xf16>) outs(%51 : tensor<4x4x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<4x4x16x16xf16>
      %alloc_8 = memref.alloc() : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %memspacecast_9 = memref.memory_space_cast %alloc_8 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16>
      %53 = bufferization.to_tensor %memspacecast_9 restrict writable : memref<4x4x16x16xf16>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
      %54 = hivm.hir.copy ins(%52 : tensor<4x4x16x16xf16>) outs(%53 : tensor<4x4x16x16xf16>) -> tensor<4x4x16x16xf16>
      annotation.mark %54 : tensor<4x4x16x16xf16>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %alloc_10 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast_11 = memref.memory_space_cast %alloc_10 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %55 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<64x64xf32>
      %56 = tensor.empty() : tensor<64x64xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %57 = hivm.hir.vadd ins(%55, %50 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%56 : tensor<64x64xf32>) -> tensor<64x64xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 3
      %58 = arith.addi %arg15, %c64_i32 : i32
      %59 = arith.addi %arg16, %c64_i32 : i32
      scf.yield %49, %57, %40, %58, %59 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32, i32
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %27 = hivm.hir.vln ins(%26#0 : tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %28 = hivm.hir.vadd ins(%26#2, %27 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %expanded = tensor.expand_shape %26#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %29 = hivm.hir.vdiv ins(%26#1, %expanded : tensor<64x64xf32>, tensor<64x1xf32>) outs(%7 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
    %30 = arith.muli %12, %c8192_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = arith.addi %31, %23 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%32], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%28 : tensor<64xf32>) outs(%reinterpret_cast_3 : memref<64xf32, strided<[1], offset: ?>>)
    %33 = tensor.empty() : tensor<64x64xf16>
    %34 = hivm.hir.vcast ins(%29 : tensor<64x64xf32>) outs(%33 : tensor<64x64xf16>) -> tensor<64x64xf16>
    hivm.hir.store ins(%34 : tensor<64x64xf16>) outs(%reinterpret_cast : memref<64x64xf16, strided<[64, 1], offset: ?>>)
  }
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
  return
}

// -----

// CHECK-LABEL:   func.func @_hstu_attn_fwd_mix_aiv_ub2ub(
// CHECK:      %[[VAL_alloc_4:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<ub>>
// CHECK:      annotation.mark %[[VAL_alloc_4:.*]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<1x1x16x16xf16, #hivm.address_space<ub>>
// CHECK:      %[[VAL_memspacecast_5:.*]] = memref.memory_space_cast %[[VAL_alloc_4:.*]] : memref<1x1x16x16xf16, #hivm.address_space<ub>> to memref<1x1x16x16xf16>
// CHECK:      %[[VAL_40:.*]] = bufferization.to_tensor %[[VAL_memspacecast_5:.*]] restrict writable : memref<1x1x16x16xf16>
// CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
// CHECK:      %[[VAL_41:.*]] = hivm.hir.copy ins(%39 : tensor<1x1x16x16xf16>) outs(%[[VAL_40:.*]] : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK:      annotation.mark %[[VAL_41:.*]] : tensor<1x1x16x16xf16>
func.func @_hstu_attn_fwd_mix_aiv_ub2ub(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: f32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32, %arg30: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<31xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, parallel_mode = "simd"} {
  %cst = arith.constant -1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 1.000000e+00 : f32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg28, %arg29 : i32
  %1 = arith.muli %0, %arg30 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg30 : i32
  %5 = arith.remsi %4, %arg29 : i32
  %6 = arith.muli %arg30, %arg29 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg28 : i32
  %9 = tensor.empty() : tensor<1xf32>
  %10 = tensor.empty() : tensor<16x16xf32>
  %11 = arith.divsi %5, %arg20 : i32
  %12 = arith.index_cast %11 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
  %13 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %14 = arith.addi %12, %c1 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %15 = memref.load %reinterpret_cast_1[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %16 = arith.subi %15, %13 : i64
  %17 = arith.trunci %16 : i64 to i32
  %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %18 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %19 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
  %20 = arith.subi %19, %18 : i64
  %21 = arith.trunci %20 : i64 to i32
  %22 = arith.muli %8, %c16_i32 : i32
  %23 = arith.cmpi slt, %22, %17 : i32
  scf.if %23 {
    %24 = arith.sitofp %arg21 : i32 to f32
    %inserted = tensor.insert %24 into %9[%c0] : tensor<1xf32>
    %25 = tensor.empty() : tensor<1xf32>
    %26 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%25 : tensor<1xf32>) -> tensor<1xf32>
    %27 = hivm.hir.vdiv ins(%26, %inserted : tensor<1xf32>, tensor<1xf32>) outs(%9 : tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %27[%c0] : tensor<1xf32>
    %28:3 = scf.for %arg31 = %c0_i32 to %21 step %c16_i32 iter_args(%arg32 = %c0_i32, %arg33 = %c0_i32, %arg34 = %c0_i32) -> (i32, i32, i32)  : i32 {
      %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %29 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %30 = hivm.hir.vmul ins(%29, %arg16 : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " qk: ", tcoretype = #hivm.tcore_type<VECTOR>} %30 : tensor<16x16xf32>
      %31 = hivm.hir.vmul ins(%30, %cst : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %32 = hivm.hir.vexp ins(%31 : tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %33 = hivm.hir.vadd ins(%32, %cst_0 : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %34 = hivm.hir.vdiv ins(%30, %33 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %35 = hivm.hir.vmul ins(%34, %extracted : tensor<16x16xf32>, f32) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %36 = tensor.empty() : tensor<16x16xf16>
      %37 = hivm.hir.vcast ins(%35 : tensor<16x16xf32>) outs(%36 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %expanded = tensor.expand_shape %37 [[0, 1], [2, 3]] output_shape [1, 16, 1, 16] : tensor<16x16xf16> into tensor<1x16x1x16xf16>
      %38 = tensor.empty() : tensor<1x1x16x16xf16>
      %39 = hivm.hir.vtranspose ins(%expanded : tensor<1x16x1x16xf16>) outs(%38 : tensor<1x1x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<1x1x16x16xf16>
      %alloc_4 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<1x1x16x16xf16, #hivm.address_space<ub>>
      %memspacecast_5 = memref.memory_space_cast %alloc_4 : memref<1x1x16x16xf16, #hivm.address_space<ub>> to memref<1x1x16x16xf16>
      %40 = bufferization.to_tensor %memspacecast_5 restrict writable : memref<1x1x16x16xf16>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
      %41 = hivm.hir.copy ins(%39 : tensor<1x1x16x16xf16>) outs(%40 : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
      annotation.mark %41 : tensor<1x1x16x16xf16>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %42 = arith.addi %arg32, %c16_i32 : i32
      %43 = arith.addi %arg33, %c16_i32 : i32
      %44 = arith.addi %arg34, %c16_i32 : i32
      scf.yield %42, %43, %44 : i32, i32, i32
    }
  }
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
  return
}