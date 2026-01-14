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
      %57 = hivm.hir.vreduce <max> ins(%54 : tensor<64x32xf32>) outs(%expanded_12 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
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
      %66 = hivm.hir.vreduce <sum> ins(%64 : tensor<64x32xf32>) outs(%expanded_14 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
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
// CHECK:             %[[VAL_61:.*]] = hivm.hir.load ins(%[[VAL_59:.*]] : tensor<32x64xf32>) outs(%[[VAL_60:.*]] : tensor<32x64xf32>) init_out_buffer = false -> tensor<32x64xf32>
// CHECK:             %[[VAL_103:.*]] = hivm.hir.load ins(%[[VAL_102:.*]] : tensor<32x64xf32>) outs(%[[VAL_60:.*]] : tensor<32x64xf32>) init_out_buffer = false -> tensor<32x64xf32>
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
  %39 = hivm.hir.vreduce <max> ins(%36 : tensor<64x64xf32>) outs(%10 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
  %40 = hivm.hir.vmul ins(%39, %extracted : tensor<64x1xf32>, f32) outs(%expanded_12 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %41 = hivm.hir.vmul ins(%38, %extracted : tensor<64x64xf32>, f32) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %42 = hivm.hir.vbrc ins(%40 : tensor<64x1xf32>) outs(%11 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
  %43 = hivm.hir.vsub ins(%41, %42 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %44 = hivm.hir.vmul ins(%43, %cst_3 : tensor<64x64xf32>, f32) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %45 = hivm.hir.vexp ins(%44 : tensor<64x64xf32>) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %46 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%expanded_6 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %47 = hivm.hir.vreduce <sum> ins(%45 : tensor<64x64xf32>) outs(%46 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
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

// CHECK-LABEL:   func.func @fa_with_cvPipeline
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:             %[[VAL_61:.*]] = hivm.hir.load ins(%[[VAL_59:.*]] : tensor<64x256xf32>) outs(%[[VAL_60:.*]] : tensor<64x256xf32>) init_out_buffer = false -> tensor<64x256xf32>
// CHECK:             %[[VAL_103:.*]] = hivm.hir.load ins(%[[VAL_102:.*]] : tensor<64x64xf32>) outs(%[[VAL_60:.*]] : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
#map = affine_map<(d0)[s0] -> (d0 * 458752 + s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0 + d1, 4096)>
#map2 = affine_map<(d0, d1)[s0] -> ((d0 - d1) floordiv s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @fa_with_cvPipeline(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c16384 = arith.constant 16384 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c512_i32 = arith.constant 512 : i32
    %c196608 = arith.constant 196608 : index
    %c131072 = arith.constant 131072 : index
    %c4096 = arith.constant 4096 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg11 : i32
    %5 = arith.remsi %4, %arg10 : i32
    %6 = arith.muli %arg11, %arg10 : i32
    %7 = arith.divsi %3, %6 : i32
    %8 = arith.remsi %7, %arg9 : i32
    %9 = tensor.empty() : tensor<128xf32>
    %10 = hivm.hir.vbrc ins(%cst : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %11 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %12 = tensor.empty() : tensor<128x256xf32>
    %13 = tensor.empty() : tensor<128x64xf32>
    %14 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%13 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %15 = arith.divsi %5, %c32_i32 : i32
    %16 = arith.remsi %5, %c32_i32 : i32
    %17 = arith.extsi %15 : i32 to i64
    %18 = arith.muli %17, %c8388608_i64 : i64
    %19 = arith.extsi %16 : i32 to i64
    %20 = arith.muli %19, %c262144_i64 : i64
    %21 = arith.addi %18, %20 : i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = arith.muli %8, %c128_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = arith.muli %24, %c64 : index
    %26 = arith.addi %25, %22 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x64xf16>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 6
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 7
    %27 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast_3 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
    %cast_5 = memref.cast %reinterpret_cast_4 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
    %28:9 = scf.for %arg12 = %c0_i32 to %c4096_i32 step %c512_i32 iter_args(%arg13 = %10, %arg14 = %14, %arg15 = %11, %arg16 = %cast, %arg17 = %cast_5, %arg18 = %22, %arg19 = %c0, %arg20 = %22, %arg21 = %c0) -> (tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, memref<256x64xf16, strided<[?, ?], offset: ?>>, memref<256x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %38 = hivm.hir.get_block_idx -> i64
      %39 = arith.index_cast %38 : i64 to index
      %40 = affine.apply #map(%39)[%c196608]
      %view = memref.view %arg2[%40][] : memref<?xi8> to memref<2x128x256xf32>
      %41 = hivm.hir.get_block_idx -> i64
      %42 = arith.index_cast %41 : i64 to index
      %43 = affine.apply #map(%42)[%c131072]
      %view_7 = memref.view %arg2[%43][] : memref<?xi8> to memref<2x128x64xf32>
      %44 = hivm.hir.get_block_idx -> i64
      %45 = arith.index_cast %44 : i64 to index
      %46 = affine.apply #map(%45)[%c0]
      %view_8 = memref.view %arg2[%46][] : memref<?xi8> to memref<2x128x256xf16>
      %47 = arith.index_cast %arg12 : i32 to index
      %48 = affine.min #map1(%47, %c512, %c4096)
      %49 = affine.apply #map2(%48, %47)[%c256]
      annotation.mark %view : memref<2x128x256xf32>
      annotation.mark %view_8 : memref<2x128x256xf16>
      annotation.mark %view_7 : memref<2x128x64xf32>
      %50:2 = scf.for %arg22 = %c0 to %c0 step %c1 iter_args(%arg23 = %arg17, %arg24 = %arg20) -> (memref<256x64xf16, strided<[?, ?], offset: ?>>, index) {
        %alloc_9 = memref.alloc() : memref<256x64xf16>
        %58 = bufferization.to_tensor %alloc_9 restrict writable : memref<256x64xf16>
        %59 = tensor.empty() : tensor<128x256xf32>
        %subview = memref.subview %view[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<1x128x256xf32, strided<[32768, 256, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x256xf32, strided<[32768, 256, 1], offset: ?>> into memref<128x256xf32, strided<[256, 1], offset: ?>>
        %60 = arith.index_cast %arg22 : index to i64
        %61 = arith.addi %arg22, %c2 : index
        %62 = arith.index_cast %61 : index to i64
        %63 = arith.addi %arg24, %c16384 : index
        %64 = arith.addi %63, %arg21 : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %cast_11 = memref.cast %reinterpret_cast_10 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
        scf.yield %cast_11, %64 : memref<256x64xf16, strided<[?, ?], offset: ?>>, index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %51 = bufferization.to_tensor %view restrict : memref<2x128x256xf32>
      %52 = tensor.empty() : tensor<2x128x64xf32>
      %53:3 = scf.for %arg22 = %c0 to %49 step %c1 iter_args(%arg23 = %arg15, %arg24 = %arg13, %arg25 = %52) -> (tensor<128xf32>, tensor<128xf32>, tensor<2x128x64xf32>) {
        %58 = tensor.empty() : tensor<128x256xf32>
        %extracted_slice = tensor.extract_slice %51[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf32> to tensor<128x256xf32>
        %59 = arith.addi %arg22, %c2 : index
        %60 = arith.index_cast %59 : index to i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = %60
        %61 = hivm.hir.load ins(%extracted_slice : tensor<128x256xf32>) outs(%58 : tensor<128x256xf32>) init_out_buffer = false -> tensor<128x256xf32>
        %62 = tensor.empty() : tensor<128x256xf32>
        %extracted_slice_9 = tensor.extract_slice %51[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf32> to tensor<128x256xf32>
        %63 = arith.index_cast %arg22 : index to i64
        %64 = hivm.hir.load ins(%extracted_slice_9 : tensor<128x256xf32>) outs(%62 : tensor<128x256xf32>) init_out_buffer = false -> tensor<128x256xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = %63
        %expanded_10 = tensor.expand_shape %11 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %65 = hivm.hir.vreduce <max> ins(%61 : tensor<128x256xf32>) outs(%expanded_10 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed = tensor.collapse_shape %65 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %66 = hivm.hir.vmul ins(%collapsed, %arg8 : tensor<128xf32>, f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %67 = hivm.hir.vmax ins(%arg23, %66 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %68 = hivm.hir.vmul ins(%64, %arg8 : tensor<128x256xf32>, f32) outs(%12 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %expanded_11 = tensor.expand_shape %67 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %69 = hivm.hir.vbrc ins(%expanded_11 : tensor<128x1xf32>) outs(%12 : tensor<128x256xf32>) broadcast_dims = [1] -> tensor<128x256xf32>
        %70 = hivm.hir.vsub ins(%68, %69 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%12 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %71 = hivm.hir.vexp ins(%70 : tensor<128x256xf32>) outs(%12 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %72 = tensor.empty() : tensor<128x256xf16>
        %73 = hivm.hir.vcast ins(%71 : tensor<128x256xf32>) outs(%72 : tensor<128x256xf16>) -> tensor<128x256xf16>
        %subview = memref.subview %view_8[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<1x128x256xf16, strided<[32768, 256, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x256xf16, strided<[32768, 256, 1], offset: ?>> into memref<128x256xf16, strided<[256, 1], offset: ?>>
        %74 = arith.addi %arg22, %c4 : index
        %75 = arith.index_cast %74 : index to i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = %75
        %76 = arith.addi %arg22, %c2 : index
        %77 = arith.index_cast %76 : index to i64
        hivm.hir.store ins(%73 : tensor<128x256xf16>) outs(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = %77
        %78 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_12 = tensor.expand_shape %78 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %79 = hivm.hir.vreduce <sum> ins(%71 : tensor<128x256xf32>) outs(%expanded_12 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed_13 = tensor.collapse_shape %79 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %80 = hivm.hir.vsub ins(%arg23, %67 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %81 = hivm.hir.vexp ins(%80 : tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %82 = hivm.hir.vmul ins(%arg24, %81 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %83 = hivm.hir.vadd ins(%82, %collapsed_13 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_14 = tensor.expand_shape %81 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %extracted_slice_15 = tensor.extract_slice %arg25[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<2x128x64xf32> to tensor<128x64xf32>
        %84 = hivm.hir.vbrc ins(%expanded_14 : tensor<128x1xf32>) outs(%extracted_slice_15 : tensor<128x64xf32>) broadcast_dims = [1] -> tensor<128x64xf32>
        %inserted_slice = tensor.insert_slice %84 into %arg25[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<128x64xf32> into tensor<2x128x64xf32>
        scf.yield %67, %83, %inserted_slice : tensor<128xf32>, tensor<128xf32>, tensor<2x128x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      %54 = bufferization.to_tensor %view_8 restrict : memref<2x128x256xf16>
      %55:2 = scf.for %arg22 = %c0 to %c0 step %c1 iter_args(%arg23 = %arg16, %arg24 = %arg18) -> (memref<256x64xf16, strided<[?, ?], offset: ?>>, index) {
        %58 = tensor.empty() : tensor<128x256xf16>
        %extracted_slice = tensor.extract_slice %54[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf16> to tensor<128x256xf16>
        %59 = arith.addi %arg22, %c2 : index
        %60 = arith.index_cast %59 : index to i64
        %61 = arith.addi %arg22, %c4 : index
        %62 = arith.index_cast %61 : index to i64
        %alloc_9 = memref.alloc() : memref<256x64xf16>
        %63 = bufferization.to_tensor %alloc_9 restrict writable : memref<256x64xf16>
        %64 = tensor.empty() : tensor<128x64xf32>
        %subview = memref.subview %view_7[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : memref<2x128x64xf32> to memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>> into memref<128x64xf32, strided<[64, 1], offset: ?>>
        %65 = arith.addi %arg22, %c6 : index
        %66 = arith.index_cast %65 : index to i64
        %67 = arith.addi %arg22, %c2 : index
        %68 = arith.index_cast %67 : index to i64
        %69 = arith.addi %arg24, %c16384 : index
        %70 = arith.addi %69, %arg19 : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%70], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %cast_11 = memref.cast %reinterpret_cast_10 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
        scf.yield %cast_11, %70 : memref<256x64xf16, strided<[?, ?], offset: ?>>, index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %56 = bufferization.to_tensor %view_7 restrict : memref<2x128x64xf32>
      %57 = scf.for %arg22 = %c0 to %49 step %c1 iter_args(%arg23 = %arg14) -> (tensor<128x64xf32>) {
        %extracted_slice = tensor.extract_slice %53#2[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<2x128x64xf32> to tensor<128x64xf32>
        %58 = hivm.hir.vmul ins(%arg23, %extracted_slice : tensor<128x64xf32>, tensor<128x64xf32>) outs(%13 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %59 = tensor.empty() : tensor<128x64xf32>
        %extracted_slice_9 = tensor.extract_slice %56[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<2x128x64xf32> to tensor<128x64xf32>
        %60 = arith.addi %arg22, %c2 : index
        %61 = arith.index_cast %60 : index to i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = %61
        %62 = arith.addi %arg22, %c6 : index
        %63 = arith.index_cast %62 : index to i64
        %64 = hivm.hir.load ins(%extracted_slice_9 : tensor<128x64xf32>) outs(%59 : tensor<128x64xf32>) init_out_buffer = false -> tensor<128x64xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = %63
        %65 = tensor.empty() : tensor<128x64xf32>
        %66 = hivm.hir.vadd ins(%64, %58 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%65 : tensor<128x64xf32>) -> tensor<128x64xf32>
        scf.yield %66 : tensor<128x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      scf.yield %53#1, %57, %53#0, %55#0, %50#0, %55#1, %c0, %50#1, %c0 : tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, memref<256x64xf16, strided<[?, ?], offset: ?>>, memref<256x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %29 = hivm.hir.vln ins(%28#0 : tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %30 = hivm.hir.vadd ins(%28#2, %29 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %expanded = tensor.expand_shape %28#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %31 = hivm.hir.vbrc ins(%expanded : tensor<128x1xf32>) outs(%13 : tensor<128x64xf32>) broadcast_dims = [1] -> tensor<128x64xf32>
    %32 = hivm.hir.vdiv ins(%28#1, %31 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%13 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %33 = arith.muli %5, %c4096_i32 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = arith.addi %34, %24 : index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%35], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%30 : tensor<128xf32>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>>)
    %36 = tensor.empty() : tensor<128x64xf16>
    %37 = hivm.hir.vcast ins(%32 : tensor<128x64xf32>) outs(%36 : tensor<128x64xf16>) -> tensor<128x64xf16>
    hivm.hir.store ins(%37 : tensor<128x64xf16>) outs(%reinterpret_cast_2 : memref<128x64xf16, strided<[64, 1], offset: ?>>)
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 4
    return
  }
}

// -----

// CHECK-LABEL:   func.func @fa_after_cv_tile_nested_loop
// CHECK-DAG:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_25:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:                     %[[VAL_102:.*]] = hivm.hir.load ins(%[[VAL_100:.*]] : tensor<16x256xf32>) outs(%[[VAL_101:.*]] : tensor<16x256xf32>
// CHECK:                   %[[VAL_144:.*]] = hivm.hir.load ins(%[[VAL_143:.*]] : tensor<64x64xf32>) outs(%[[VAL_43:.*]] : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
#map = affine_map<(d0)[s0] -> (d0 * 458752 + s0)>
#map1 = affine_map<(d0) -> (2048, d0 + 512)>
#map2 = affine_map<(d0, d1) -> ((d0 - d1) ceildiv 256)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @fa_after_cv_tile_nested_loop(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %c20_i32 = arith.constant 20 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c4194304_i64 = arith.constant 4194304 : i64
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c16384 = arith.constant 16384 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c512_i32 = arith.constant 512 : i32
    %c32 = arith.constant 32 : index
    %c196608 = arith.constant 196608 : index
    %c65536 = arith.constant 65536 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg10, %arg9 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg8 : i32
    %7 = tensor.empty() : tensor<128xf32>
    %8 = hivm.hir.vbrc ins(%cst : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %9 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %10 = tensor.empty() : tensor<128x256xf32>
    %11 = tensor.empty() : tensor<128x64xf32>
    %12 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%11 : tensor<128x64xf32>) -> tensor<128x64xf32>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 6
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 7
    scf.for %arg11 = %6 to %c2048_i32 step %c20_i32  : i32 {
      %13 = arith.divsi %arg11, %c16_i32 : i32
      %14 = arith.remsi %arg11, %c16_i32 : i32
      %15 = arith.divsi %13, %c32_i32 : i32
      %16 = arith.remsi %13, %c32_i32 : i32
      %17 = arith.extsi %15 : i32 to i64
      %18 = arith.muli %17, %c4194304_i64 : i64
      %19 = arith.extsi %16 : i32 to i64
      %20 = arith.muli %19, %c131072_i64 : i64
      %21 = arith.addi %18, %20 : i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = arith.muli %14, %c128_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.muli %24, %c64 : index
      %26 = arith.addi %25, %22 : index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x64xf16>
      %27 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<128x64xf16, strided<[64, 1], offset: ?>>
      %28 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf16>
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
      %cast = memref.cast %reinterpret_cast_4 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
      %cast_6 = memref.cast %reinterpret_cast_5 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
      %29:9 = scf.for %arg12 = %c0_i32 to %c2048_i32 step %c512_i32 iter_args(%arg13 = %8, %arg14 = %12, %arg15 = %9, %arg16 = %cast, %arg17 = %cast_6, %arg18 = %22, %arg19 = %c0, %arg20 = %22, %arg21 = %c0) -> (tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, memref<256x64xf16, strided<[?, ?], offset: ?>>, memref<256x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
        %38 = hivm.hir.get_block_idx -> i64
        %39 = arith.index_cast %38 : i64 to index
        %40 = affine.apply #map(%39)[%c0]
        %view = memref.view %arg2[%40][] : memref<?xi8> to memref<2x128x64xf32>
        %41 = hivm.hir.get_block_idx -> i64
        %42 = arith.index_cast %41 : i64 to index
        %43 = affine.apply #map(%42)[%c196608]
        %view_8 = memref.view %arg2[%43][] : memref<?xi8> to memref<2x128x256xf32>
        %44 = hivm.hir.get_block_idx -> i64
        %45 = arith.index_cast %44 : i64 to index
        %46 = affine.apply #map(%45)[%c65536]
        %view_9 = memref.view %arg2[%46][] : memref<?xi8> to memref<2x128x256xf16>
        %47 = arith.index_cast %arg12 : i32 to index
        %48 = affine.min #map1(%47)
        %49 = affine.apply #map2(%48, %47)
        annotation.mark %view_8 : memref<2x128x256xf32>
        annotation.mark %view_9 : memref<2x128x256xf16>
        annotation.mark %view : memref<2x128x64xf32>
        %50:2 = scf.for %arg22 = %c0 to %c0 step %c1 iter_args(%arg23 = %arg17, %arg24 = %arg20) -> (memref<256x64xf16, strided<[?, ?], offset: ?>>, index) {
          %alloc_10 = memref.alloc() : memref<256x64xf16>
          %58 = bufferization.to_tensor %arg23 restrict writable : memref<256x64xf16, strided<[?, ?], offset: ?>>
          %59 = bufferization.to_tensor %alloc_10 restrict writable : memref<256x64xf16>
          %subview = memref.subview %view_8[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<1x128x256xf32, strided<[32768, 256, 1], offset: ?>>
          %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x256xf32, strided<[32768, 256, 1], offset: ?>> into memref<128x256xf32, strided<[256, 1], offset: ?>>
          %60 = arith.index_cast %arg22 : index to i64
          %61 = arith.addi %arg22, %c2 : index
          %62 = arith.index_cast %61 : index to i64
          %63 = arith.addi %arg24, %c16384 : index
          %64 = arith.addi %63, %arg21 : index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
          %cast_12 = memref.cast %reinterpret_cast_11 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
          scf.yield %cast_12, %64 : memref<256x64xf16, strided<[?, ?], offset: ?>>, index
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
        %51 = bufferization.to_tensor %view_8 restrict : memref<2x128x256xf32>
        %52 = tensor.empty() : tensor<2x128xf32>
        %53:3 = scf.for %arg22 = %c0 to %49 step %c1 iter_args(%arg23 = %arg15, %arg24 = %arg13, %arg25 = %52) -> (tensor<128xf32>, tensor<128xf32>, tensor<2x128xf32>) {
          %extracted_slice = tensor.extract_slice %51[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf32> to tensor<128x256xf32>
          %58 = tensor.empty() : tensor<128x1xf32>
          %59 = tensor.empty() : tensor<128x256xf16>
          %subview = memref.subview %view_9[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<1x128x256xf16, strided<[32768, 256, 1], offset: ?>>
          %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x256xf16, strided<[32768, 256, 1], offset: ?>> into memref<128x256xf16, strided<[256, 1], offset: ?>>
          %extracted_slice_10 = tensor.extract_slice %arg25[%arg22, 0] [1, 128] [1, 1] : tensor<2x128xf32> to tensor<128xf32>
          %60 = arith.addi %arg22, %c2 : index
          %61 = arith.index_cast %60 : index to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = %61
          %62 = arith.addi %arg22, %c4 : index
          %63 = arith.index_cast %62 : index to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = %63
          %64 = arith.index_cast %arg22 : index to i64
          %65 = arith.addi %arg22, %c2 : index
          %66 = arith.index_cast %65 : index to i64
          %67:3 = scf.for %arg26 = %c0 to %c128 step %c32 iter_args(%arg27 = %7, %arg28 = %7, %arg29 = %extracted_slice_10) -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
            %extracted_slice_11 = tensor.extract_slice %arg23[%arg26] [32] [1] : tensor<128xf32> to tensor<32xf32>
            %extracted_slice_12 = tensor.extract_slice %extracted_slice[%arg26, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
            %extracted_slice_13 = tensor.extract_slice %10[%arg26, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
            %68 = hivm.hir.load ins(%extracted_slice_12 : tensor<32x256xf32>) outs(%extracted_slice_13 : tensor<32x256xf32>) {vector_producer_to_fuse_0} init_out_buffer = false -> tensor<32x256xf32>
            %extracted_slice_14 = tensor.extract_slice %10[%arg26, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
            %69 = hivm.hir.vmul {vector_producer_to_fuse_0} ins(%68, %cst_2 : tensor<32x256xf32>, f32) outs(%extracted_slice_14 : tensor<32x256xf32>) -> tensor<32x256xf32>
            %extracted_slice_15 = tensor.extract_slice %58[%arg26, 0] [32, 1] [1, 1] : tensor<128x1xf32> to tensor<32x1xf32>
            %70 = hivm.hir.vreduce {vector_producer_to_fuse_0} <max> ins(%69 : tensor<32x256xf32>) outs(%extracted_slice_15 : tensor<32x1xf32>) reduce_dims = [1] -> tensor<32x1xf32>
            %collapsed = tensor.collapse_shape %70 [[0, 1]] {vector_producer_to_fuse_0} : tensor<32x1xf32> into tensor<32xf32>
            %extracted_slice_16 = tensor.extract_slice %7[%arg26] [32] [1] : tensor<128xf32> to tensor<32xf32>
            %71 = hivm.hir.vmax {vector_producer_to_fuse_0} ins(%extracted_slice_11, %collapsed : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_16 : tensor<32xf32>) -> tensor<32xf32>
            %inserted_slice_17 = tensor.insert_slice %71 into %arg27[%arg26] [32] [1] : tensor<32xf32> into tensor<128xf32>
            %extracted_slice_18 = tensor.extract_slice %arg24[%arg26] [32] [1] : tensor<128xf32> to tensor<32xf32>
            %72 = hivm.hir.vsub {vector_producer_to_fuse_0} ins(%extracted_slice_11, %71 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_16 : tensor<32xf32>) -> tensor<32xf32>
            %extracted_slice_19 = tensor.extract_slice %extracted_slice_10[%arg26] [32] [1] : tensor<128xf32> to tensor<32xf32>
            %73 = hivm.hir.vexp {vector_producer_to_fuse_0} ins(%72 : tensor<32xf32>) outs(%extracted_slice_19 : tensor<32xf32>) -> tensor<32xf32>
            %74 = hivm.hir.vmul {vector_producer_to_fuse_0} ins(%extracted_slice_18, %73 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_16 : tensor<32xf32>) -> tensor<32xf32>
            %expanded_20 = tensor.expand_shape %71 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
            %75 = hivm.hir.vsub {vector_producer_to_fuse_0} ins(%69, %expanded_20 : tensor<32x256xf32>, tensor<32x1xf32>) outs(%extracted_slice_14 : tensor<32x256xf32>) broadcast = [1] -> tensor<32x256xf32>
            %76 = hivm.hir.vexp {vector_producer_to_fuse_0} ins(%75 : tensor<32x256xf32>) outs(%extracted_slice_14 : tensor<32x256xf32>) -> tensor<32x256xf32>
            %77 = hivm.hir.vreduce {vector_producer_to_fuse_0} <sum> ins(%76 : tensor<32x256xf32>) outs(%extracted_slice_15 : tensor<32x1xf32>) reduce_dims = [1] -> tensor<32x1xf32>
            %collapsed_21 = tensor.collapse_shape %77 [[0, 1]] {vector_producer_to_fuse_0} : tensor<32x1xf32> into tensor<32xf32>
            %78 = hivm.hir.vadd {vector_producer_to_fuse_0} ins(%74, %collapsed_21 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_16 : tensor<32xf32>) -> tensor<32xf32>
            %inserted_slice_22 = tensor.insert_slice %78 into %arg28[%arg26] [32] [1] : tensor<32xf32> into tensor<128xf32>
            %inserted_slice_23 = tensor.insert_slice %73 into %arg29[%arg26] [32] [1] : tensor<32xf32> into tensor<128xf32>
            %extracted_slice_24 = tensor.extract_slice %59[%arg26, 0] [32, 256] [1, 1] : tensor<128x256xf16> to tensor<32x256xf16>
            %79 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%76 : tensor<32x256xf32>) outs(%extracted_slice_24 : tensor<32x256xf16>) -> tensor<32x256xf16>
            %subview_25 = memref.subview %collapse_shape[%arg26, 0] [32, 256] [1, 1] : memref<128x256xf16, strided<[256, 1], offset: ?>> to memref<32x256xf16, strided<[256, 1], offset: ?>>
            hivm.hir.store ins(%79 : tensor<32x256xf16>) outs(%subview_25 : memref<32x256xf16, strided<[256, 1], offset: ?>>) {op_to_tile_0_0}
            scf.yield %inserted_slice_17, %inserted_slice_22, %inserted_slice_23 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
          }
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = %66
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = %64
          %inserted_slice = tensor.insert_slice %67#2 into %arg25[%arg22, 0] [1, 128] [1, 1] : tensor<128xf32> into tensor<2x128xf32>
          scf.yield %67#0, %67#1, %inserted_slice : tensor<128xf32>, tensor<128xf32>, tensor<2x128xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
        %54 = bufferization.to_tensor %view_9 restrict : memref<2x128x256xf16>
        %55:2 = scf.for %arg22 = %c0 to %c0 step %c1 iter_args(%arg23 = %arg16, %arg24 = %arg18) -> (memref<256x64xf16, strided<[?, ?], offset: ?>>, index) {
          %58 = tensor.empty() : tensor<128x256xf16>
          %extracted_slice = tensor.extract_slice %54[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf16> to tensor<128x256xf16>
          %59 = arith.addi %arg22, %c2 : index
          %60 = arith.index_cast %59 : index to i64
          %61 = arith.addi %arg22, %c4 : index
          %62 = arith.index_cast %61 : index to i64
          %alloc_10 = memref.alloc() : memref<256x64xf16>
          %63 = bufferization.to_tensor %arg23 restrict writable : memref<256x64xf16, strided<[?, ?], offset: ?>>
          %64 = bufferization.to_tensor %alloc_10 restrict writable : memref<256x64xf16>
          %subview = memref.subview %view[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : memref<2x128x64xf32> to memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>>
          %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>> into memref<128x64xf32, strided<[64, 1], offset: ?>>
          %65 = arith.addi %arg22, %c6 : index
          %66 = arith.index_cast %65 : index to i64
          %67 = arith.addi %arg22, %c2 : index
          %68 = arith.index_cast %67 : index to i64
          %69 = arith.addi %arg24, %c16384 : index
          %70 = arith.addi %69, %arg19 : index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg5 to offset: [%70], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
          %cast_12 = memref.cast %reinterpret_cast_11 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
          scf.yield %cast_12, %70 : memref<256x64xf16, strided<[?, ?], offset: ?>>, index
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
        %56 = bufferization.to_tensor %view restrict : memref<2x128x64xf32>
        %57 = scf.for %arg22 = %c0 to %49 step %c1 iter_args(%arg23 = %arg14) -> (tensor<128x64xf32>) {
          %extracted_slice = tensor.extract_slice %53#2[%arg22, 0] [1, 128] [1, 1] : tensor<2x128xf32> to tensor<128xf32>
          %expanded_10 = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %58 = hivm.hir.vmul ins(%arg23, %expanded_10 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%11 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
          %extracted_slice_11 = tensor.extract_slice %56[%arg22, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<2x128x64xf32> to tensor<128x64xf32>
          %59 = arith.addi %arg22, %c2 : index
          %60 = arith.index_cast %59 : index to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = %60
          %61 = arith.addi %arg22, %c6 : index
          %62 = arith.index_cast %61 : index to i64
          %63 = hivm.hir.load ins(%extracted_slice_11 : tensor<128x64xf32>) outs(%11 : tensor<128x64xf32>) init_out_buffer = false -> tensor<128x64xf32>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = %62
          %64 = hivm.hir.vadd ins(%63, %58 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%11 : tensor<128x64xf32>) -> tensor<128x64xf32>
          scf.yield %64 : tensor<128x64xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
        scf.yield %53#1, %57, %53#0, %55#0, %50#0, %55#1, %c0, %50#1, %c0 : tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, memref<256x64xf16, strided<[?, ?], offset: ?>>, memref<256x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
      }
      %30 = hivm.hir.vln ins(%29#0 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %31 = hivm.hir.vadd ins(%29#2, %30 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %expanded = tensor.expand_shape %29#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %32 = hivm.hir.vdiv ins(%29#1, %expanded : tensor<128x64xf32>, tensor<128x1xf32>) outs(%11 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
      %33 = arith.muli %13, %c2048_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.addi %34, %24 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%35], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%31 : tensor<128xf32>) outs(%reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>>)
      %36 = tensor.empty() : tensor<128x64xf16>
      %37 = hivm.hir.vcast ins(%32 : tensor<128x64xf32>) outs(%36 : tensor<128x64xf16>) -> tensor<128x64xf16>
      hivm.hir.store ins(%37 : tensor<128x64xf16>) outs(%reinterpret_cast_3 : memref<128x64xf16, strided<[64, 1], offset: ?>>)
    }
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 4
    return
  }
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @simple_testcase_after_tiling(
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:             %[[VAL_7:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]]]
// CHECK:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]]]
// CHECK:             %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_2]] step %[[VAL_4]] iter_args(%[[VAL_11:.*]] = %{{.*}}) -> (tensor<512xf32>) {
// CHECK:               %[[VAL_12:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_13:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_8]]] [256] [1] {to_be_bubbled_slice} : tensor<512xf32> to tensor<256xf32>
// CHECK:               %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_14]]{{\[}}%[[VAL_13]]] [32] [1] : tensor<256xf32> to tensor<32xf32>
// CHECK:               hivm.hir.store ins(%{{.*}} : tensor<32xf32>) outs(%{{.*}} : memref<32xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:               scf.yield %{{.*}} : tensor<512xf32>
// CHECK:             }
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
#map = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @simple_testcase_after_tiling(%arg0: tensor<512xf32>, %arg1: memref<512xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %0 = tensor.empty() : tensor<64xf32>
    %1 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %arg0) -> (tensor<512xf32>) {
      %2 = affine.apply #map()[%arg2]
      %extracted_slice = tensor.extract_slice %arg3[%2] [64] [1] : tensor<512xf32> to tensor<64xf32>
      %3 = hivm.hir.vln ins(%extracted_slice : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %subview = memref.subview %arg1[%2] [64] [1] : memref<512xf32> to memref<64xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%3 : tensor<64xf32>) outs(%subview : memref<64xf32, strided<[1], offset: ?>>)
      scf.yield %arg0 : tensor<512xf32>
    }
    return
  }
}

// -----
// CHECK-LABEL:   func.func @simple_testcase_unaligned(
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<63xf32>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vln ins(%[[random1:.*]] : tensor<63xf32>) outs(%[[VAL_3]] : tensor<63xf32>) -> tensor<63xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_2]] : index
// CHECK:           scf.if %[[VAL_7]] {
// CHECK:             hivm.hir.store ins(%[[VAL_4]] : tensor<63xf32>) outs(%[[random2:.*]] : memref<63xf32>)
// CHECK:           } {limit_sub_block_id0}
// CHECK:           return
// CHECK:         }
module {
  func.func @simple_testcase_unaligned(%arg0: tensor<63xf32>, %arg1: memref<63xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = tensor.empty() : tensor<63xf32>
    %3 = hivm.hir.vln ins(%arg0 : tensor<63xf32>) outs(%0 : tensor<63xf32>) -> tensor<63xf32>
    hivm.hir.store ins(%3 : tensor<63xf32>) outs(%arg1 : memref<63xf32>)
    return
  }
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @simple_testcase_slicingUB(
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_6:.*]] = affine.apply #[[$ATTR_0]]()[%[[VAL_5]]]
// CHECK:             %[[VAL_7:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:             %[[VAL_8:.*]] = hivm.hir.vln ins(%[[VAL_7]] : tensor<32xf32>) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:             %[[VAL_9:.*]] = memref.subview %{{.*}}[%[[VAL_6]]] [32] [1] {to_be_bubbled_slice} : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:             hivm.hir.store ins(%[[VAL_8]] : tensor<32xf32>) outs(%[[VAL_9]] : memref<32xf32, strided<[1], offset: ?>>)
// CHECK:           }
// CHECK:           return
// CHECK:         }
module {
  func.func @simple_testcase_slicingUB(%arg0: tensor<64xf32>, %arg1: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
        %alloc_4 = memref.alloc() : memref<64xf32>
    %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<64xf32>
    %0 = tensor.empty() : tensor<64xf32>
    %3 = hivm.hir.vln ins(%11 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    hivm.hir.store ins(%3 : tensor<64xf32>) outs(%arg1 : memref<64xf32>)
    return
  }
}

// -----
// CHECK-LABEL:   func.func @simple_testcase_dynamic(
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vln ins(%[[random1:.*]] : tensor<?xf32>) outs(%[[random2:.*]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_3]] : index
// CHECK:           scf.if %[[VAL_7]] {
// CHECK:             hivm.hir.store ins(%[[VAL_4]] : tensor<?xf32>) outs(%[[random3:.*]] : memref<?xf32>)
// CHECK:           } {limit_sub_block_id0}
// CHECK:           return
// CHECK:         }
module {
  func.func @simple_testcase_dynamic(%arg0: tensor<?xf32>, %arg1: memref<?xf32>, %arg2: tensor<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %3 = hivm.hir.vln ins(%arg0 : tensor<?xf32>) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
    hivm.hir.store ins(%3 : tensor<?xf32>) outs(%arg1 : memref<?xf32>)
    return
  }
}

// -----
// CHECK-LABEL:   func.func @simple_testcase_safely_revert(
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vreduce <sum> ins(%[[random1:.*]] : tensor<6xf32>) outs(%[[random2:.*]] : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_3]] : index
// CHECK:           scf.if %[[VAL_8]] {
// CHECK:             hivm.hir.store ins(%[[VAL_5]] : tensor<1xf32>) outs(%[[VAL_4:.*]] : memref<1xf32>)
// CHECK:           } {limit_sub_block_id0}
// CHECK:           return
// CHECK:         }
module {
  func.func @simple_testcase_safely_revert(%arg0: tensor<6xf32>, %arg1: memref<1xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = tensor.empty() : tensor<1xf32>
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<6xf32>) outs(%0 : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    hivm.hir.store ins(%1 : tensor<1xf32>) outs(%arg1 : memref<1xf32>)
    return
  }
}

// -----
// CHECK-LABEL:   func.func @simple_testcase_store_with_result(
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = scf.for %[[VAL_7:.*]] = %[[VAL_3:.*]] to %[[VAL_2:.*]] step %[[VAL_4:.*]] iter_args(%[[VAL_8:.*]] = %[[VAL_0:.*]]) -> (tensor<64xf32>) {
// CHECK:             %[[VAL_9:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_5]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:             %[[VAL_10:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:             %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i64 to index
// CHECK:             %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_3]] : index
// CHECK:             %[[VAL_13:.*]] = scf.if %[[VAL_12]] -> (tensor<64xf32>) {
// CHECK:               %[[VAL_14:.*]] = hivm.hir.store ins(%[[VAL_9]] : tensor<64xf32>) outs(%[[VAL_8]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:               scf.yield %[[VAL_14]] : tensor<64xf32>
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_8]] : tensor<64xf32>
// CHECK:             } {limit_sub_block_id0}
// CHECK:             scf.yield %[[VAL_13]] : tensor<64xf32>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @simple_testcase_store_with_result(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = tensor.empty() : tensor<64xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    // For now, store with result and result is used afterwards is not supported yet, but
    // this pass will revert to 1:1 and still succeed the pass, but limit to unique subblock
    // to store.
    %10 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg01 = %arg0) -> tensor<64xf32> {
      %1 = affine.apply #map1()[%arg2]
      hivm.hir.load ins(%arg01 : tensor<64xf32>) outs(%arg0 : tensor<64xf32>) init_out_buffer = false -> tensor<64xf32>
      %3 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %4 = hivm.hir.store ins(%3 : tensor<64xf32>) outs(%arg01 :  tensor<64xf32>) -> tensor<64xf32>
      annotation.mark %4 : tensor<64xf32>
      scf.yield %4 : tensor<64xf32>
    }
    return
  }
}
