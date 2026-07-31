// RUN: bishengir-opt %s -hivm-bind-sub-block -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @mm_01_mix_aiv(
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_14:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_12]] {
// CHECK:             %[[VAL_43:.*]] = tensor.extract_slice %[[VAL_37:.*]][0, 0] {{\[}}%[[VAL_42:.*]], 16] [1, 1] : tensor<8x16xf16> to tensor<?x16xf16>
// CHECK:             %[[VAL_44:.*]] = memref.subview %[[VAL_26:.*]][0, 0] {{\[}}%[[VAL_42]], 16] [1, 1] : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
// CHECK:             hivm.hir.store ins(%[[VAL_43]] : tensor<?x16xf16>) outs(%[[VAL_44]] : memref<?x16xf16, strided<[16, 1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
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
      %54 = hivm.hir.load ins(%52 : tensor<64x32xf32>) outs(%53 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %55 = tensor.empty() : tensor<64x32xf32>
      %56 = hivm.hir.load ins(%52 : tensor<64x32xf32>) outs(%55 : tensor<64x32xf32>) -> tensor<64x32xf32>
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
      %88 = hivm.hir.load ins(%86 : tensor<64x64xf32>) outs(%87 : tensor<64x64xf32>) -> tensor<64x64xf32>
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
  %36 = hivm.hir.load ins(%34 : tensor<64x64xf32>) outs(%35 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %37 = tensor.empty() : tensor<64x64xf32>
  %38 = hivm.hir.load ins(%34 : tensor<64x64xf32>) outs(%37 : tensor<64x64xf32>) -> tensor<64x64xf32>
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
  %69 = hivm.hir.load ins(%67 : tensor<64x64xf32>) outs(%68 : tensor<64x64xf32>) -> tensor<64x64xf32>
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
    %32 = hivm.hir.load ins(%30 : tensor<16x16xf32>) outs(%31 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16xf16, strided<[1], offset: ?>>) outs(%alloc : memref<16xf16>)
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
    %50 = hivm.hir.load ins(%48 : tensor<16x16xf32>) outs(%49 : tensor<16x16xf32>) -> tensor<16x16xf32>
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
// CHECK:             %[[VAL_61:.*]] = hivm.hir.load ins(%[[VAL_59:.*]] : tensor<64x256xf32>) outs(%[[VAL_60:.*]] : tensor<64x256xf32>) -> tensor<64x256xf32>
// CHECK:             %[[VAL_103:.*]] = hivm.hir.load ins(%[[VAL_102:.*]] : tensor<64x64xf32>) outs(%[[VAL_60:.*]] : tensor<64x64xf32>) -> tensor<64x64xf32>
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
        %61 = hivm.hir.load ins(%extracted_slice : tensor<128x256xf32>) outs(%58 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %62 = tensor.empty() : tensor<128x256xf32>
        %extracted_slice_9 = tensor.extract_slice %51[%arg22, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf32> to tensor<128x256xf32>
        %63 = arith.index_cast %arg22 : index to i64
        %64 = hivm.hir.load ins(%extracted_slice_9 : tensor<128x256xf32>) outs(%62 : tensor<128x256xf32>) -> tensor<128x256xf32>
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
        %64 = hivm.hir.load ins(%extracted_slice_9 : tensor<128x64xf32>) outs(%59 : tensor<128x64xf32>) -> tensor<128x64xf32>
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
// CHECK:                   %[[VAL_144:.*]] = hivm.hir.load ins(%[[VAL_143:.*]] : tensor<64x64xf32>) outs(%[[VAL_43:.*]] : tensor<64x64xf32>) -> tensor<64x64xf32>
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
            %68 = hivm.hir.load ins(%extracted_slice_12 : tensor<32x256xf32>) outs(%extracted_slice_13 : tensor<32x256xf32>) {vector_producer_to_fuse_0} -> tensor<32x256xf32>
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
          %63 = hivm.hir.load ins(%extracted_slice_11 : tensor<128x64xf32>) outs(%11 : tensor<128x64xf32>) -> tensor<128x64xf32>
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
// CHECK:             %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_2]] step %[[VAL_4]] iter_args(%[[VAL_11:.*]] = %{{.*}}) -> (tensor<256xf32>) {
// CHECK:               %[[VAL_12:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_13:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_13]]] [32] [1] : tensor<256xf32> to tensor<32xf32>
// CHECK:               hivm.hir.store ins(%{{.*}} : tensor<32xf32>) outs(%{{.*}} : memref<32xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:               scf.yield %{{.*}} : tensor<256xf32>
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
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vln ins(%[[SRC:.*]] : tensor<63xf32>) outs(%[[VAL_3]] : tensor<63xf32>) -> tensor<63xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_2]] : index
// CHECK:           scf.if %[[VAL_7]] {
// CHECK:             hivm.hir.store ins(%[[VAL_4]] : tensor<63xf32>) outs(%[[DST:.*]] : memref<63xf32>)
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
// CHECK:             %[[VAL_7:.*]] = memref.subview %{{.*}}[%[[VAL_6]]] [32] [1] {to_be_bubbled_slice} : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_8:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:             %[[VAL_9:.*]] = hivm.hir.vln ins(%[[VAL_8]] : tensor<32xf32>) outs(%[[VAL_8]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:             hivm.hir.store ins(%[[VAL_9]] : tensor<32xf32>) outs(%[[VAL_7]] : memref<32xf32, strided<[1], offset: ?>>)
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
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @simple_testcase_store_with_result(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: tensor<64xf32>)
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_6:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_5]]]
// CHECK:             %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_6]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_10:.*]] = %[[VAL_7]]) -> (tensor<32xf32>) {
// CHECK:               %[[VAL_11:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:               %[[VAL_12:.*]] = hivm.hir.vln ins(%[[VAL_7]] : tensor<32xf32>) outs(%[[VAL_11]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:               %[[VAL_13:.*]] = hivm.hir.store ins(%[[VAL_12]] : tensor<32xf32>) outs(%[[VAL_10]] : tensor<32xf32>) {tiled_op} -> tensor<32xf32>
// CHECK:               annotation.mark %[[VAL_13]] : tensor<32xf32>
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_13]]{{\[}}%[[VAL_6]]] [32] [1] {to_be_bubbled_slice} : tensor<32xf32> to tensor<32xf32>
// CHECK:               scf.yield %[[VAL_14]] : tensor<32xf32>
// CHECK:             }
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @simple_testcase_store_with_result(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = tensor.empty() : tensor<64xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %10 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg01 = %arg0) -> tensor<64xf32> {
      %1 = affine.apply #map1()[%arg2]
      hivm.hir.load ins(%arg01 : tensor<64xf32>) outs(%arg0 : tensor<64xf32>) -> tensor<64xf32>
      %3 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %4 = hivm.hir.store ins(%3 : tensor<64xf32>) outs(%arg01 :  tensor<64xf32>) -> tensor<64xf32>
      annotation.mark %4 : tensor<64xf32>
      scf.yield %4 : tensor<64xf32>
    }
    return
  }
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @store_with_nonzero_offset(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: memref<64xf32>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: index,
// CHECK-SAME:                                         %[[VAL_3:.*]]: index)
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:             %[[VAL_9:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_8]]]
// CHECK:             %[[VAL_10:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_9]]] [32] [1] {to_be_bubbled_slice} : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_11:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_9]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_12:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:             %[[VAL_13:.*]] = hivm.hir.vln ins(%[[VAL_11]] : tensor<32xf32>) outs(%[[VAL_12]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[VAL_4]] : index
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : index
// CHECK:             %[[VAL_16:.*]] = arith.maxsi %[[VAL_2]], %[[VAL_9]] : index
// CHECK:             %[[VAL_17:.*]] = arith.minsi %[[VAL_15]], %[[VAL_14]] : index
// CHECK:             %[[VAL_18:.*]] = arith.maxsi %[[VAL_16]], %[[VAL_17]] : index
// CHECK:             %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_16]] : index
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_16]], %[[VAL_9]] : index
// CHECK:             %[[VAL_21:.*]] = tensor.extract_slice %[[VAL_13]]{{\[}}%[[VAL_20]]] {{\[}}%[[VAL_19]]] [1] : tensor<32xf32> to tensor<?xf32>
// CHECK:             %[[VAL_22:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_20]]] {{\[}}%[[VAL_19]]] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:             hivm.hir.store ins(%[[VAL_21]] : tensor<?xf32>) outs(%[[VAL_22]] : memref<?xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
func.func @store_with_nonzero_offset(%arg0: tensor<64xf32>, %arg1: memref<64xf32>, %arg2: index, %arg3: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %extracted_slice = tensor.extract_slice %1[%arg2] [%arg3] [1] : tensor<64xf32> to tensor<?xf32>
  %subview = memref.subview %arg1[%arg2] [%arg3] [1] : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
  hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview : memref<?xf32, strided<[1], offset: ?>>)
  return
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @broadcast_two_different_dims(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: memref<64x64xf32>)
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_6:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_5]]]
// CHECK:             %[[VAL_7:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_6]], 0] [32, 64] [1, 1] {to_be_bubbled_slice} : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_8:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:             %[[VAL_9:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_8]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:             %[[VAL_10:.*]] = tensor.expand_shape %[[VAL_9]] {{\[\[}}0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
// CHECK:             %[[VAL_11:.*]] = tensor.extract_slice %[[VAL_9]]{{\[}}%[[VAL_6]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_12:.*]] = tensor.expand_shape %[[VAL_11]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:             %[[VAL_13:.*]] = tensor.empty() : tensor<32x64xf32>
// CHECK:             %[[VAL_14:.*]] = hivm.hir.vadd ins(%[[VAL_12]], %[[VAL_10]] : tensor<32x1xf32>, tensor<1x64xf32>) outs(%[[VAL_13]] : tensor<32x64xf32>) broadcast = [0, 1] -> tensor<32x64xf32>
// CHECK:             hivm.hir.store ins(%[[VAL_14]] : tensor<32x64xf32>) outs(%[[VAL_7]] : memref<32x64xf32, strided<[64, 1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
// expected-remark @+1{{Selected tiling dim might have broadcast two different axis. Automatically disables strict mode.}}
func.func @broadcast_two_different_dims(%arg0: tensor<64xf32>, %arg1: memref<64x64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = tensor.empty() : tensor<64x64xf32>
  %2 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
// expected-warning @+1{{Extract slice is not fully bubbled up}}
  %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_0 = tensor.expand_shape %2 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
  %3 = hivm.hir.vadd ins(%expanded, %expanded_0 : tensor<64x1xf32>, tensor<1x64xf32>) outs(%1 : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
  hivm.hir.store ins(%3 : tensor<64x64xf32>) outs(%arg1 : memref<64x64xf32>)
  return
}

// -----
// CHECK-LABEL:   func.func @load_and_store_same_GM(
// CHECK:           limit_sub_block_id0
func.func @load_and_store_same_GM(%arg0: tensor<64xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c64 = arith.constant 64 : index
  %alloc = memref.alloc() : memref<64xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32>
  %0 = arith.minsi %arg5, %c64 : index
  %subview = memref.subview %reinterpret_cast[0] [%0] [1] : memref<64xf32> to memref<?xf32>
  %subview_0 = memref.subview %alloc[0] [%0] [1] : memref<64xf32> to memref<?xf32>
  hivm.hir.load ins(%subview : memref<?xf32>) outs(%subview_0 : memref<?xf32>)
  %1 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %3 = hivm.hir.vadd ins(%arg0, %1 : tensor<64xf32>, tensor<64xf32>) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
  %extracted_slice = tensor.extract_slice %3[0] [%0] [1] : tensor<64xf32> to tensor<?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview : memref<?xf32>)
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32>
  %subview_1 = memref.subview %reinterpret_cast_1[0] [%0] [1] : memref<64xf32> to memref<?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview_1 : memref<?xf32>)
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32>
  %subview_2 = memref.subview %reinterpret_cast_2[0] [%0] [1] : memref<64xf32> to memref<?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview : memref<?xf32>)
  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32>
  %subview_3 = memref.subview %reinterpret_cast_3[0] [%0] [1] : memref<64xf32> to memref<?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview : memref<?xf32>)
  return
}

// -----
// CHECK-LABEL:   func.func @unstructure_store(
// CHECK:           scf.if %[[VAL_14:.*]] {
// CHECK:             hivm.hir.store ins(%[[VAL_10:.*]] : tensor<1xf32>) outs(%[[VAL_11:.*]] : memref<1xf32, strided<[1], offset: ?>>)
// CHECK:           } {limit_sub_block_id0}
// CHECK:           scf.if %[[VAL_17:.*]] {
// CHECK:             hivm.hir.store ins(%[[VAL_2:.*]] : tensor<64xf32>) outs(%[[VAL_3:.*]] : memref<64xf32>)
// CHECK:           } {limit_sub_block_id0}
func.func @unstructure_store(%arg0: tensor<64xf32>, %arg1: memref<64xf32>, %arg2: tensor<64xf32>, %arg3: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  scf.for %arg4 = %c0 to %c64 step %c1 {
    %extracted_slice = tensor.extract_slice %1[%arg4] [1] [1] : tensor<64xf32> to tensor<1xf32>
    %subview = memref.subview %arg1[%arg4] [1] [1] : memref<64xf32> to memref<1xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<1xf32>) outs(%subview : memref<1xf32, strided<[1], offset: ?>>)
  } {ExtractedLoadOrStore}
  hivm.hir.store ins(%arg2 : tensor<64xf32>) outs(%arg3 : memref<64xf32>)
  return
}

// -----
// CHECK-LABEL:   func.func @store_with_static_mask(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: memref<64xf32>)
// CHECK:           %[[VAL_2:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:             %[[VAL_7:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]]]
// CHECK:             %[[VAL_8:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_7]]] [32] [1] {to_be_bubbled_slice} : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_7]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_10:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:             %[[VAL_11:.*]] = hivm.hir.vln ins(%[[VAL_9]] : tensor<32xf32>) outs(%[[VAL_10]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:             %[[VAL_12:.*]] = arith.minsi %[[VAL_7]], %[[VAL_4]] : index
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_4]], %[[VAL_12]] : index
// CHECK:             %[[VAL_14:.*]] = arith.minsi %[[VAL_13]], %[[VAL_2]] : index
// CHECK:             %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_11]][0] {{\[}}%[[VAL_14]]] [1] : tensor<32xf32> to tensor<?xf32>
// CHECK:             %[[VAL_16:.*]] = memref.subview %[[VAL_8]][0] {{\[}}%[[VAL_14]]] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:             hivm.hir.store ins(%[[VAL_15]] : tensor<?xf32>) outs(%[[VAL_16]] : memref<?xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
func.func @store_with_static_mask(%arg0: tensor<64xf32>, %arg1: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %extracted_slice = tensor.extract_slice %1[0] [1] [1] : tensor<64xf32> to tensor<1xf32>
  %subview = memref.subview %arg1[0] [1] [1] : memref<64xf32> to memref<1xf32>
  hivm.hir.store ins(%extracted_slice : tensor<1xf32>) outs(%subview : memref<1xf32>)
  return
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @tile_and_bind_while(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: memref<64xf32>)
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_7]]]
// CHECK:             %[[VAL_9:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_8]]] [32] [1] {to_be_bubbled_slice} : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_10:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_8]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_11:.*]]:2 = scf.while (%[[VAL_12:.*]] = %[[VAL_10]], %[[VAL_13:.*]] = %[[VAL_3]]) : (tensor<32xf32>, i32) -> (tensor<32xf32>, i32) {
// CHECK:               %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_2]] : i32
// CHECK:               scf.condition(%[[VAL_14]]) %[[VAL_12]], %[[VAL_13]] : tensor<32xf32>, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_15:.*]]: tensor<32xf32>, %[[VAL_16:.*]]: i32):
// CHECK:               %[[VAL_17:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:               %[[VAL_18:.*]] = hivm.hir.vln ins(%[[VAL_15]] : tensor<32xf32>) outs(%[[VAL_17]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_16]], %[[VAL_2]] : i32
// CHECK:               scf.yield %[[VAL_18]], %[[VAL_19]] : tensor<32xf32>, i32
// CHECK:             }
// CHECK:             hivm.hir.store ins(%[[VAL_11]]#0 : tensor<32xf32>) outs(%[[VAL_9]] : memref<32xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
func.func @tile_and_bind_while(%arg0: tensor<64xf32>, %arg1: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<64xf32>
  %1:2 = scf.while (%arg2 = %arg0, %arg3 = %c0_i32) : (tensor<64xf32>, i32) -> (tensor<64xf32>, i32) {
    %2 = arith.cmpi slt, %arg3, %c1_i32 : i32
    scf.condition(%2) %arg2, %arg3 : tensor<64xf32>, i32
  } do {
  ^bb0(%arg2: tensor<64xf32>, %arg3: i32):
    %2 = hivm.hir.vln ins(%arg2 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %3 = arith.addi %arg3, %c1_i32 : i32
    scf.yield %2, %3 : tensor<64xf32>, i32
  }
  hivm.hir.store ins(%1#0 : tensor<64xf32>) outs(%arg1 : memref<64xf32>)
  return
}

// -----
// A 3-D while result split on its middle axis must not be bound to two
// subblocks when one row of the trailing axis is smaller than a DMA block.
// 3x4x3 makes dimension 1 the only divisible tiling candidate.
// CHECK-LABEL:   func.func @tile_and_bind_while_3d_middle_unaligned(
// CHECK:           %[[SUB_BLOCK_ID:.*]] = hivm.hir.get_sub_block_idx
// CHECK:           scf.if %{{.*}} {
// CHECK:             hivm.hir.store ins(%{{.*}} : tensor<3x4x3xf32>) outs(%{{.*}} : memref<3x4x3xf32>)
// CHECK:           }
func.func @tile_and_bind_while_3d_middle_unaligned(%arg0: tensor<3x4x3xf32>, %arg1: memref<3x4x3xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<3x4x3xf32>
  %1:2 = scf.while (%arg2 = %arg0, %arg3 = %c0_i32) : (tensor<3x4x3xf32>, i32) -> (tensor<3x4x3xf32>, i32) {
    %2 = arith.cmpi slt, %arg3, %c1_i32 : i32
    scf.condition(%2) %arg2, %arg3 : tensor<3x4x3xf32>, i32
  } do {
  ^bb0(%arg2: tensor<3x4x3xf32>, %arg3: i32):
    %2 = hivm.hir.vln ins(%arg2 : tensor<3x4x3xf32>) outs(%0 : tensor<3x4x3xf32>) -> tensor<3x4x3xf32>
    %3 = arith.addi %arg3, %c1_i32 : i32
    scf.yield %2, %3 : tensor<3x4x3xf32>, i32
  }
  hivm.hir.store ins(%1#0 : tensor<3x4x3xf32>) outs(%arg1 : memref<3x4x3xf32>)
  return
}

// -----
// Keep the aligned counterpart enabled: the same middle-axis 1:2 split is
// safe once the trailing row is 32 bytes.
// CHECK-LABEL:   func.func @tile_and_bind_while_3d_middle_aligned(
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             tensor.extract_slice %{{.*}}[0, %{{.*}}, 0] [3, 2, 8] [1, 1, 1] {to_be_bubbled_slice} : tensor<3x4x8xf32> to tensor<3x2x8xf32>
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK-NOT:       limit_sub_block_id0
func.func @tile_and_bind_while_3d_middle_aligned(%arg0: tensor<3x4x8xf32>, %arg1: memref<3x4x8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<3x4x8xf32>
  %1:2 = scf.while (%arg2 = %arg0, %arg3 = %c0_i32) : (tensor<3x4x8xf32>, i32) -> (tensor<3x4x8xf32>, i32) {
    %2 = arith.cmpi slt, %arg3, %c1_i32 : i32
    scf.condition(%2) %arg2, %arg3 : tensor<3x4x8xf32>, i32
  } do {
  ^bb0(%arg2: tensor<3x4x8xf32>, %arg3: i32):
    %2 = hivm.hir.vln ins(%arg2 : tensor<3x4x8xf32>) outs(%0 : tensor<3x4x8xf32>) -> tensor<3x4x8xf32>
    %3 = arith.addi %arg3, %c1_i32 : i32
    scf.yield %2, %3 : tensor<3x4x8xf32>, i32
  }
  hivm.hir.store ins(%1#0 : tensor<3x4x8xf32>) outs(%arg1 : memref<3x4x8xf32>)
  return
}

// -----
// CHECK-LABEL:   func.func @dynamic_shape_insert_slice
// CHECK:         {limit_sub_block_id0}
func.func @dynamic_shape_insert_slice(%arg0: tensor<64x?xf32>, %arg1: memref<64x64xf32>, %arg2: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %0 = tensor.empty() : tensor<64x64xf32>
  %inserted_slice = tensor.insert_slice %arg0 into %0[0, 0] [64, %arg2] [1, 1] : tensor<64x?xf32> into tensor<64x64xf32>
  hivm.hir.store ins(%inserted_slice : tensor<64x64xf32>) outs(%arg1 : memref<64x64xf32>)
  return
}


// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @tile_and_bind_scope(
// CHECK-SAME:                                   %[[VAL_0:.*]]: memref<?xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<64xf32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: tensor<64xf32>)
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:             %[[VAL_7:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]]]
// CHECK:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_6]]]
// CHECK:             %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_8]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_10:.*]] = tensor.extract_slice %[[VAL_2]]{{\[}}%[[VAL_8]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_11:.*]] = scope.scope : () -> tensor<32xf32> {
// CHECK:               %[[VAL_12:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:               %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (tensor<32xf32>) {
// CHECK:                 %[[VAL_16:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_14]]]
// CHECK:                 %[[VAL_17:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_14]]]
// CHECK:                 %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_17]]], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_20:.*]] = memref.subview %[[VAL_18]]{{\[}}%[[VAL_7]]] [16] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<16xf32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_19:.*]] = memref.alloc() : memref<16xf32>
// CHECK:                 hivm.hir.load ins(%[[VAL_20]] : memref<16xf32, strided<[1], offset: ?>>) outs(%[[VAL_19]] : memref<16xf32>)
// CHECK:                 %[[VAL_21:.*]] = bufferization.to_tensor %[[VAL_19]] restrict writable : memref<16xf32>
// CHECK:                 %[[VAL_22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK:                 %[[VAL_23:.*]] = hivm.hir.vln ins(%[[VAL_21]] : tensor<16xf32>) outs(%[[VAL_22]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK:                 %[[VAL_24:.*]] = tensor.insert_slice %[[VAL_23]] into %[[VAL_15]]{{\[}}%[[VAL_16]]] [16] [1] : tensor<16xf32> into tensor<32xf32>
// CHECK:                 scf.yield %[[VAL_24]] : tensor<32xf32>
// CHECK:               }
// CHECK:               %[[VAL_25:.*]] = hivm.hir.store ins(%[[VAL_13]] : tensor<32xf32>) outs(%[[VAL_9]] : tensor<32xf32>) {tiled_op} -> tensor<32xf32>
// CHECK:               annotation.mark %[[VAL_25]] : tensor<32xf32>
// CHECK:               scope.return %[[VAL_13]] : tensor<32xf32>
// CHECK:             }
// CHECK:             %[[VAL_26:.*]] = hivm.hir.store ins(%[[VAL_11]] : tensor<32xf32>) outs(%[[VAL_10]] : tensor<32xf32>) {tiled_op} -> tensor<32xf32>
// CHECK:             annotation.mark %[[VAL_26]] : tensor<32xf32>
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
#map = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @tile_and_bind_scope(%arg0: memref<?xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = tensor.empty() : tensor<32xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %2 = scope.scope : () -> tensor<64xf32> {
      %4 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %0) -> (tensor<64xf32>) {
        %6 = affine.apply #map()[%arg3]
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %alloc = memref.alloc() : memref<32xf32>
        hivm.hir.load ins(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc : memref<32xf32>)
        %7 = bufferization.to_tensor %alloc restrict writable : memref<32xf32>
        %8 = hivm.hir.vln ins(%7 : tensor<32xf32>) outs(%1 : tensor<32xf32>) -> tensor<32xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg4[%6] [32] [1] : tensor<32xf32> into tensor<64xf32>
        scf.yield %inserted_slice : tensor<64xf32>
      }
      %5 = hivm.hir.store ins(%4 : tensor<64xf32>) outs(%arg1 : tensor<64xf32>) -> tensor<64xf32>
      annotation.mark %5 : tensor<64xf32>
      scope.return %4 : tensor<64xf32>
    }
    %3 = hivm.hir.store ins(%2 : tensor<64xf32>) outs(%arg2 : tensor<64xf32>) -> tensor<64xf32>
    annotation.mark %3 : tensor<64xf32>
    return
  }
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @store_with_nonzero_offset_dynamic_mask(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: memref<64xf32>,
// CHECK-SAME:                                                      %[[VAL_2:.*]]: index,
// CHECK-SAME:                                                      %[[VAL_3:.*]]: index)
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:             %[[VAL_9:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_8]]]
// CHECK:             %[[VAL_10:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_9]]] [32] [1] {to_be_bubbled_slice} : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_11:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_9]]] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:             %[[VAL_12:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:             %[[VAL_13:.*]] = hivm.hir.vln ins(%[[VAL_11]] : tensor<32xf32>) outs(%[[VAL_12]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[VAL_4]] : index
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : index
// CHECK:             %[[VAL_16:.*]] = arith.maxsi %[[VAL_2]], %[[VAL_9]] : index
// CHECK:             %[[VAL_17:.*]] = arith.minsi %[[VAL_15]], %[[VAL_14]] : index
// CHECK:             %[[VAL_18:.*]] = arith.maxsi %[[VAL_16]], %[[VAL_17]] : index
// CHECK:             %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_16]] : index
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_16]], %[[VAL_9]] : index
// CHECK:             %[[VAL_21:.*]] = tensor.extract_slice %[[VAL_13]]{{\[}}%[[VAL_20]]] {{\[}}%[[VAL_19]]] [1] : tensor<32xf32> to tensor<?xf32>
// CHECK:             %[[VAL_22:.*]] = arith.minsi %[[VAL_9]], %[[VAL_3]] : index
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_3]], %[[VAL_22]] : index
// CHECK:             %[[VAL_24:.*]] = arith.minsi %[[VAL_23]], %[[VAL_4]] : index
// CHECK:             %[[VAL_25:.*]] = memref.subview %[[VAL_10]][0] {{\[}}%[[VAL_24]]] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:             hivm.hir.store ins(%[[VAL_21]] : tensor<?xf32>) outs(%[[VAL_25]] : memref<?xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
func.func @store_with_nonzero_offset_dynamic_mask(%arg0: tensor<64xf32>, %arg1: memref<64xf32>, %offset: index, %size: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %extracted_slice = tensor.extract_slice %1[%offset] [%size] [1] : tensor<64xf32> to tensor<?xf32>
  %subview = memref.subview %arg1[0] [%size] [1] : memref<64xf32> to memref<?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview : memref<?xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @check_split_indirect_store
// CHECK: scf.for
// CHECK: hivm.hir.indirect_store ins(%{{.*}} : tensor<8x64xf16>, %{{.*}} : tensor<8x64xi64>, %{{.*}} : tensor<8x64xi1>) outs(%arg0 : memref<?xf16>)
// CHECK-NOT: limit_sub_block_id0
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_split_indirect_store(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0_i64 = arith.constant 0 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<16x64xf16>
    %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<16x64xf16>) -> tensor<16x64xf16>
    %2 = tensor.empty() : tensor<16x64xi64>
    %3 = hivm.hir.vbrc ins(%c0_i64 : i64) outs(%2 : tensor<16x64xi64>) -> tensor<16x64xi64>
    %4 = tensor.empty() : tensor<16x64xi1>
    %5 = hivm.hir.vbrc ins(%true : i1) outs(%4 : tensor<16x64xi1>) -> tensor<16x64xi1>
    hivm.hir.indirect_store ins(%1 : tensor<16x64xf16>, %3 : tensor<16x64xi64>, %5 : tensor<16x64xi1>) outs(%arg0 : memref<?xf16>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @brc_two_dim_with_reduction_dim
// CHECK-NOT: scf.if
// CHECK: hivm.hir.vreduce <sum> ins(%[[VAL_24:.*]] : tensor<16x8xf32>) outs(%[[VAL_26:.*]] : tensor<1x8xf32>) reduce_dims = [0] -> tensor<1x8xf32>
// CHECK: hivm.hir.store
// CHECK-NOT: limit_sub_block_id0
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // expected-remark @+1{{Selected tiling dim might have broadcast two different axis. Automatically disables strict mode.}}
  func.func @brc_two_dim_with_reduction_dim(%arg0: tensor<16xf32>, %arg1: memref<?xf32>, %arg2: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = tensor.empty() : tensor<16xf32>
    %2 = tensor.empty() : tensor<16xi32>
    %3 = hivm.hir.varange offset[%c0] strides[%c1] outs(%2 : tensor<16xi32>) -> tensor<16xi32>
    %4 = tensor.empty() : tensor<16x16xi32>
    %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
    %5 = hivm.hir.vbrc ins(%expanded : tensor<16x1xi32>) outs(%4 : tensor<16x16xi32>) broadcast_dims = [1] -> tensor<16x16xi32>
    // expected-warning @+1{{Extract slice is not fully bubbled up}}
    %expanded_0 = tensor.expand_shape %3 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %6 = hivm.hir.vbrc ins(%expanded_0 : tensor<1x16xi32>) outs(%4 : tensor<16x16xi32>) broadcast_dims = [0] -> tensor<16x16xi32>
    %7 = tensor.empty() : tensor<16x16xi1>
    %8 = hivm.hir.vcmp ins(%5, %6 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%7 : tensor<16x16xi1>) -> tensor<16x16xi1>
    %9 = hivm.hir.vcast ins(%8 : tensor<16x16xi1>) outs(%0 : tensor<16x16xf32>) cast = <cast_unsigned> -> tensor<16x16xf32>
    %expanded_1 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
    %10 = hivm.hir.vreduce <sum> ins(%9 : tensor<16x16xf32>) outs(%expanded_1 : tensor<1x16xf32>) reduce_dims = [0] -> tensor<1x16xf32>
    %collapsed = tensor.collapse_shape %10 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    %11 = hivm.hir.vadd ins(%collapsed, %cst : tensor<16xf32>, f32) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %expanded_2 = tensor.expand_shape %11 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
    %12 = hivm.hir.vbrc ins(%expanded_2 : tensor<1x16xf32>) outs(%0 : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
    %extracted_slice = tensor.extract_slice %12[0, 0] [%arg2, 16] [1, 1] : tensor<16x16xf32> to tensor<?x16xf32>
    %13 = hivm.hir.vbrc ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %14 = hivm.hir.vadd ins(%9, %13 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %14[0, 0] [%arg2, 16] [1, 1] : tensor<?x16xf32> into tensor<16x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32>
    hivm.hir.store ins(%inserted_slice : tensor<16x16xf32>) outs(%reinterpret_cast : memref<16x16xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @triton_dot_2_mix_aiv_workspace
// CHECK: scf.if %{{.*}} -> (tensor<8x16xf32>) {
// CHECK: scf.yield %{{.*}} : tensor<8x16xf32>
// CHECK: } else {
// CHECK: %[[WS_STORE:.*]] = hivm.hir.store ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {tiled_op} -> tensor<8x16xf32>
// CHECK: %[[WS_SLICE:.*]] = tensor.extract_slice %[[WS_STORE]]{{\[}}%{{.*}}, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<8x16xf32> to tensor<8x16xf32>
// CHECK: scf.yield %[[WS_SLICE]] : tensor<8x16xf32>
// CHECK: }
// CHECK: hivm.hir.store ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : memref<8x16xf32, strided<[16, 1], offset: ?>>) {tiled_op}
// CHECK-NOT: limit_sub_block_id0
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @triton_dot_2_mix_aiv_workspace(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i8, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = arith.trunci %arg7 : i8 to i1
    %3 = tensor.empty() : tensor<16x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc : memref<16x16xf32>) may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf32>) may_implicit_transpose_with_last_axis = false
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_3 = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_3 : memref<16x16xf32>) may_implicit_transpose_with_last_axis = false
    %6 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf32>
    %7 = scf.if %2 -> (tensor<16x16xf32>) {
      %10 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<16x16xf32>
      %11 = bufferization.to_tensor %10 restrict writable : memref<16x16xf32>
      scf.yield %11 : tensor<16x16xf32>
    } else {
      %10 = hivm.hir.vadd ins(%4, %5 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %11 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8> to memref<16x16xf32>
      %12 = bufferization.to_tensor %11 restrict writable : memref<16x16xf32>
      %13 = hivm.hir.store ins(%10 : tensor<16x16xf32>) outs(%12 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %13 : tensor<16x16xf32>
    }
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %8 = hivm.hir.load ins(%7 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) may_implicit_transpose_with_last_axis = false -> tensor<16x16xf32>
    annotation.mark %7 {"InsertLoadStoreForMixCV::markToAvoidDCE" = 1 : i32} : tensor<16x16xf32>
    %9 = hivm.hir.vadd ins(%8, %6 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    hivm.hir.store ins(%9 : tensor<16x16xf32>) outs(%reinterpret_cast_4 : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @prepare_wy_repr_fwd_kernel_chunk64_mix_aiv(

#map = affine_map<()[s0, s1] -> (s0 + s1 * 1024)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 1024 + 32)>
#map3 = affine_map<()[s0] -> (s0 + 1)>
// expected-remark@+1 {{Selected tiling dim might have broadcast two different axis. Automatically disables strict mode.}}
func.func @prepare_wy_repr_fwd_kernel_chunk64_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %c-32_i32 = arith.constant -32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg5 : i32
  %5 = arith.divsi %3, %arg5 : i32
  %6 = arith.remsi %5, %arg6 : i32
  %7 = tensor.empty() : tensor<32x32xf32>
  %8 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %9 = arith.divsi %6, %c16_i32 : i32
  %10 = arith.remsi %6, %c16_i32 : i32
  %11 = arith.muli %9, %arg4 : i32
  %12 = arith.muli %11, %c16_i32 : i32
  %13 = arith.addi %12, %10 : i32
  %14 = arith.muli %13, %c64_i32 : i32
  %15 = arith.index_cast %14 : i32 to index
  %16 = arith.muli %4, %c64_i32 : i32
  %17 = arith.maxsi %16, %c0_i32 : i32
  %18 = arith.index_cast %17 : i32 to index
  %19 = affine.apply #map()[%15, %18]
  %20 = arith.index_cast %arg4 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [1024, 1] : memref<?xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  %21 = arith.addi %16, %c32_i32 : i32
  %22 = arith.maxsi %21, %c0_i32 : i32
  %23 = arith.index_cast %22 : i32 to index
  %24 = affine.apply #map2()[%15, %23]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [32, 32], strides: [1024, 1] : memref<?xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [32, 32], strides: [1024, 1] : memref<?xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [32, 32], strides: [1024, 1] : memref<?xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  %25 = affine.apply #map2()[%15, %18]
  %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [32, 32], strides: [1024, 1] : memref<?xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  %alloc = memref.alloc() : memref<32x32xf32>
  %26 = affine.apply #map1()[%20, %18]
  %27 = arith.maxsi %26, %c0 : index
  %28 = arith.minsi %27, %c32 : index
  %29 = arith.subi %c0_i32, %16 : i32
  %30 = arith.maxsi %29, %c0_i32 : i32
  %31 = arith.index_cast %30 : i32 to index
  %32 = arith.minsi %31, %28 : index
  %33 = affine.apply #map1()[%28, %32]
  %34 = arith.cmpi slt, %33, %c32 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%33, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<?x32xf32, strided<[1024, 1], offset: ?>>
  %subview_5 = memref.subview %alloc[%32, 0] [%33, 32] [1, 1] : memref<32x32xf32> to memref<?x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%subview : memref<?x32xf32, strided<[1024, 1], offset: ?>>) outs(%subview_5 : memref<?x32xf32, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %34 : i1 eviction_policy = <EvictFirst>
  %35 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32>
  %alloc_6 = memref.alloc() : memref<32x32xf32>
  %36 = affine.apply #map1()[%20, %23]
  %37 = arith.maxsi %36, %c0 : index
  %38 = arith.minsi %37, %c32 : index
  %39 = arith.subi %c-32_i32, %16 : i32
  %40 = arith.maxsi %39, %c0_i32 : i32
  %41 = arith.index_cast %40 : i32 to index
  %42 = arith.minsi %41, %38 : index
  %43 = affine.apply #map1()[%38, %42]
  %44 = arith.cmpi slt, %43, %c32 : index
  %subview_7 = memref.subview %reinterpret_cast_1[0, 0] [%43, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<?x32xf32, strided<[1024, 1], offset: ?>>
  %subview_8 = memref.subview %alloc_6[%42, 0] [%43, 32] [1, 1] : memref<32x32xf32> to memref<?x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%subview_7 : memref<?x32xf32, strided<[1024, 1], offset: ?>>) outs(%subview_8 : memref<?x32xf32, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %44 : i1 eviction_policy = <EvictFirst>
  %45 = bufferization.to_tensor %alloc_6 restrict writable : memref<32x32xf32>
  %46 = tensor.empty() : tensor<32xi32>
  %47 = hivm.hir.varange offset[%c0] strides[%c1] outs(%46 : tensor<32xi32>) -> tensor<32xi32>
  %48 = tensor.empty() : tensor<32x32xi32>
  %expanded = tensor.expand_shape %47 [[0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
  %49 = hivm.hir.vbrc ins(%expanded : tensor<32x1xi32>) outs(%48 : tensor<32x32xi32>) broadcast_dims = [1] -> tensor<32x32xi32>
  // expected-warning@+1 {{Extract slice is not fully bubbled up}}
  %expanded_9 = tensor.expand_shape %47 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
  %50 = hivm.hir.vbrc ins(%expanded_9 : tensor<1x32xi32>) outs(%48 : tensor<32x32xi32>) broadcast_dims = [0] -> tensor<32x32xi32>
  %51 = tensor.empty() : tensor<32x32xi1>
  %52 = hivm.hir.vcmp ins(%49, %50 : tensor<32x32xi32>, tensor<32x32xi32>) outs(%51 : tensor<32x32xi1>) compare_mode = <gt> -> tensor<32x32xi1>
  %53 = hivm.hir.vsel ins(%52, %35, %cst_0 : tensor<32x32xi1>, tensor<32x32xf32>, f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %54 = hivm.hir.vsel ins(%52, %45, %cst_0 : tensor<32x32xi1>, tensor<32x32xf32>, f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %55:2 = scf.for %arg8 = %c1_i32 to %c32_i32 step %c1_i32 iter_args(%arg9 = %53, %arg10 = %54) -> (tensor<32x32xf32>, tensor<32x32xf32>)  : i32 {
    %65 = arith.trunci %arg8 : i32 to i16
    %66 = tensor.empty() : tensor<1x32xf32>
    %67 = scf.for %arg11 = %c0 to %c32 step %c1 iter_args(%arg12 = %66) -> (tensor<1x32xf32>) {
      %91 = arith.index_cast %65 : i16 to index
      %extracted = tensor.extract %arg9[%91, %arg11] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<32x32xf32>
      %inserted = tensor.insert %extracted into %arg12[%c0, %arg11] : tensor<1x32xf32>
      scf.yield %inserted : tensor<1x32xf32>
    }
    %68 = tensor.empty() : tensor<32xf32>
    %69 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%68 : tensor<32xf32>) -> tensor<32xf32>
    %collapsed = tensor.collapse_shape %67 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    %70 = scf.for %arg11 = %c0 to %c32 step %c1 iter_args(%arg12 = %66) -> (tensor<1x32xf32>) {
      %91 = arith.index_cast %65 : i16 to index
      %extracted = tensor.extract %arg10[%91, %arg11] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<32x32xf32>
      %inserted = tensor.insert %extracted into %arg12[%c0, %arg11] : tensor<1x32xf32>
      scf.yield %inserted : tensor<1x32xf32>
    }
    %collapsed_28 = tensor.collapse_shape %70 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    %expanded_29 = tensor.expand_shape %collapsed [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %71 = hivm.hir.vmul ins(%expanded_29, %arg9 : tensor<32x1xf32>, tensor<32x32xf32>) outs(%7 : tensor<32x32xf32>) broadcast = [1] -> tensor<32x32xf32>
    %expanded_30 = tensor.expand_shape %69 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
    %72 = hivm.hir.vreduce <sum> ins(%71 : tensor<32x32xf32>) outs(%expanded_30 : tensor<1x32xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x32xf32>
    %collapsed_31 = tensor.collapse_shape %72 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    %73 = tensor.empty() : tensor<32xi1>
    %74 = hivm.hir.vcmp ins(%47, %arg8 : tensor<32xi32>, i32) outs(%73 : tensor<32xi1>) compare_mode = <lt> -> tensor<32xi1>
    %75 = hivm.hir.vsel ins(%74, %cst, %cst_0 : tensor<32xi1>, f32, f32) outs(%68 : tensor<32xf32>) -> tensor<32xf32>
    %76 = hivm.hir.vmul ins(%collapsed_31, %75 : tensor<32xf32>, tensor<32xf32>) outs(%68 : tensor<32xf32>) -> tensor<32xf32>
    %77 = hivm.hir.vadd ins(%collapsed, %76 : tensor<32xf32>, tensor<32xf32>) outs(%68 : tensor<32xf32>) -> tensor<32xf32>
    %expanded_32 = tensor.expand_shape %collapsed_28 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %78 = hivm.hir.vmul ins(%expanded_32, %arg10 : tensor<32x1xf32>, tensor<32x32xf32>) outs(%7 : tensor<32x32xf32>) broadcast = [1] -> tensor<32x32xf32>
    %79 = hivm.hir.vreduce <sum> ins(%78 : tensor<32x32xf32>) outs(%expanded_30 : tensor<1x32xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x32xf32>
    %collapsed_33 = tensor.collapse_shape %79 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    %80 = hivm.hir.vmul ins(%collapsed_33, %75 : tensor<32xf32>, tensor<32xf32>) outs(%68 : tensor<32xf32>) -> tensor<32xf32>
    %81 = hivm.hir.vadd ins(%collapsed_28, %80 : tensor<32xf32>, tensor<32xf32>) outs(%68 : tensor<32xf32>) -> tensor<32xf32>
    %expanded_34 = tensor.expand_shape %77 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
    %82 = hivm.hir.vbrc ins(%expanded_34 : tensor<1x32xf32>) outs(%7 : tensor<32x32xf32>) broadcast_dims = [0] -> tensor<32x32xf32>
    scf.yield %82, %78 : tensor<32x32xf32>, tensor<32x32xf32>
  }
  %subview_23 = memref.subview %reinterpret_cast_2[0, 0] [32, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  // CHECK:           } {limit_sub_block_id0}
  hivm.hir.store ins(%55#0 : tensor<32x32xf32>) outs(%subview_23 : memref<32x32xf32, strided<[1024, 1], offset: ?>>)
  %subview_27 = memref.subview %reinterpret_cast_4[0, 0] [32, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
  hivm.hir.store ins(%55#0 : tensor<32x32xf32>) outs(%subview_27 : memref<32x32xf32, strided<[1024, 1], offset: ?>>)
  return
}

// -----

// CHECK-LABEL: func.func @ub_alloc_vreduce_dim0_aic(
// CHECK: memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK-LABEL: func.func @ub_alloc_vreduce_dim0_aiv(
// CHECK-NOT: hivm.hir.vreduce {tiled_op}
// CHECK: hivm.hir.vreduce <sum> ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<1x16xf32>) reduce_dims = [0]
// CHECK: scf.if
// CHECK: hivm.hir.store
// CHECK: } {limit_sub_block_id0}
// CHECK-NOT: map_for_to_forall
module attributes {hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @ub_alloc_vreduce_dim0_aic() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = tensor.empty() : tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    return
  }
  func.func @ub_alloc_vreduce_dim0_aiv(%arg0: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %0 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %1 = tensor.empty() : tensor<16xf32>
    %2 = hivm.hir.vbrc ins(%cst : f32) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %3 = hivm.hir.vreduce <sum> ins(%0 : tensor<16x16xf32>) outs(%expanded : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1]>>
    hivm.hir.store ins(%collapsed : tensor<16xf32>) outs(%reinterpret_cast : memref<16xf32, strided<[1]>>)
    return
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-LABEL: func.func @calc_cube_vector_mix_aiv(
  // CHECK:       scf.for
  // CHECK:       annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<67x8xf32, #hivm.address_space<ub>>
  // CHECK:       hivm.hir.store ins(%{{.*}} : tensor<67x8xf32>) outs(%{{.*}} : memref<67x8xf32, strided<[8, 1], offset: ?>>) {tiled_op}
  // CHECK-NOT:   limit_sub_block_id0
  // CHECK:       scf.if
  // CHECK:         hivm.hir.store ins(%{{.*}} : tensor<1x1x1x1x1x1x1xi64>) outs(%{{.*}} : memref<1x1x1x1x1x1x1xi64, strided<[1, 1, 1, 1, 1, 1, 1]>>)
  // CHECK:       } {limit_sub_block_id0}
  // CHECK:       } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  func.func @calc_cube_vector_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [134, 8], strides: [8, 1] : memref<?xf32> to memref<134x8xf32, strided<[8, 1]>>
    %alloc = memref.alloc() : memref<134x8xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<134x8xf32, strided<[8, 1]>>) outs(%alloc : memref<134x8xf32>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<134x8xf32>
    %alloc_0 = memref.alloc() : memref<134x8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<134x8xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_0 : memref<134x8xf32, #hivm.address_space<ub>> to memref<134x8xf32>
    %3 = bufferization.to_tensor %memspacecast restrict writable : memref<134x8xf32>
    %4 = tensor.empty() : tensor<134x8xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %5 = hivm.hir.vadd ins(%3, %2 : tensor<134x8xf32>, tensor<134x8xf32>) outs(%4 : tensor<134x8xf32>) -> tensor<134x8xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [134, 8], strides: [8, 1] : memref<?xf32> to memref<134x8xf32, strided<[8, 1]>>
    hivm.hir.store ins(%5 : tensor<134x8xf32>) outs(%reinterpret_cast_1 : memref<134x8xf32, strided<[8, 1]>>)
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 1, 1, 1, 1, 1, 1], strides: [1, 1, 1, 1, 1, 1, 1] : memref<?xi64> to memref<1x1x1x1x1x1x1xi64, strided<[1, 1, 1, 1, 1, 1, 1]>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1, 1, 1, 1, 1, 1, 1], strides: [1, 1, 1, 1, 1, 1, 1] : memref<?xi64> to memref<1x1x1x1x1x1x1xi64, strided<[1, 1, 1, 1, 1, 1, 1]>>
    %alloc_4 = memref.alloc() : memref<1x1x1x1x1x1x1xi64>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<1x1x1x1x1x1x1xi64, strided<[1, 1, 1, 1, 1, 1, 1]>>) outs(%alloc_4 : memref<1x1x1x1x1x1x1xi64>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %6 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x1x1x1x1x1x1xi64>
    %alloc_5 = memref.alloc() : memref<1x1x1x1x1x1x1xi64>
    %7 = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock lock_var(%7 : memref<1xi64>)
    hivm.hir.load ins(%reinterpret_cast_3 : memref<1x1x1x1x1x1x1xi64, strided<[1, 1, 1, 1, 1, 1, 1]>>) outs(%alloc_5 : memref<1x1x1x1x1x1x1xi64>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %8 = bufferization.to_tensor %alloc_5 restrict writable : memref<1x1x1x1x1x1x1xi64>
    %9 = hivm.hir.vadd ins(%8, %6 : tensor<1x1x1x1x1x1x1xi64>, tensor<1x1x1x1x1x1x1xi64>) outs(%8 : tensor<1x1x1x1x1x1x1xi64>) -> tensor<1x1x1x1x1x1x1xi64>
    hivm.hir.store ins(%9 : tensor<1x1x1x1x1x1x1xi64>) outs(%reinterpret_cast_3 : memref<1x1x1x1x1x1x1xi64, strided<[1, 1, 1, 1, 1, 1, 1]>>)
    hivm.hir.sync_block_unlock lock_var(%7 : memref<1xi64>)
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}

// -----

// CHECK-LABEL:   func.func @fixpipe_subview_layout_recompute_mix_aic(
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<2x32x128xf32, #hivm.address_space<ub>>
// CHECK:           memref.subview %[[ALLOC]]
// CHECK-SAME:        memref<2x32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK:           hivm.hir.fixpipe
// CHECK-SAME:        outs({{.*}} : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @fixpipe_subview_layout_recompute_mix_aic(%arg0: i32, %arg1: i32, %arg2: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %0 = arith.muli %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg2 : i32
    annotation.mark %1 {logical_block_num} : i32
    scf.for %i = %c0 to %c2 step %c1 {
      %alloc = memref.alloc() : memref<2x64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x128xf32, #hivm.address_space<ub>>
      %subview = memref.subview %alloc[%i, 0, 0] [1, 64, 128] [1, 1, 1] : memref<2x64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<8x4x16x16xf32>
      %mmad = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%empty, %empty, %true, %c64, %c128, %c128 : tensor<8x4x16x16xf32>, tensor<8x4x16x16xf32>, i1, index, index, index) outs(%empty : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<8x4x16x16xf32>) outs(%subview : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>)
    }
    return
  }

  func.func @fixpipe_subview_layout_recompute_mix_aiv(%arg0: memref<64x128xf32>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg1, %arg2 : i32
    %1 = arith.muli %0, %arg3 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<2x64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x128xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc : memref<2x64x128xf32, #hivm.address_space<ub>> to memref<2x64x128xf32>
    %tensor = bufferization.to_tensor %memspacecast restrict writable : memref<2x64x128xf32>
    scf.for %i = %c0 to %c2 step %c1 {
      %slice = tensor.extract_slice %tensor[%i, 0, 0] [1, 64, 128] [1, 1, 1] : tensor<2x64x128xf32> to tensor<64x128xf32>
      hivm.hir.store ins(%slice : tensor<64x128xf32>) outs(%arg0 : memref<64x128xf32>)
    }
    return
  }
}

// -----

// Column split (tiling last dim of NZ2ND): after bubble-up the compact alloc is
// 2x64x32 and the rank-reduced subview must use strided<[32,1]>, not the
// pre-tile [64,1]. AIV mirrors the store-after-reduce pattern so analyzer picks
// tiling_dim = 2.
// CHECK-LABEL:   func.func @fixpipe_subview_column_split_layout_mix_aic(
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<2x64x32xf32, #hivm.address_space<ub>>
// CHECK:           memref.subview %[[ALLOC]]
// CHECK-SAME:        memref<2x64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK:           hivm.hir.fixpipe
// CHECK-SAME:        outs({{.*}} : memref<64x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @fixpipe_subview_column_split_layout_mix_aic(%arg0: i32, %arg1: i32, %arg2: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %0 = arith.muli %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg2 : i32
    annotation.mark %1 {logical_block_num} : i32
    scf.for %i = %c0 to %c2 step %c1 {
      %alloc = memref.alloc() : memref<2x64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x64xf32, #hivm.address_space<ub>>
      %subview = memref.subview %alloc[%i, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<4x4x16x16xf32>
      %mmad = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%empty, %empty, %true, %c64, %c64, %c64 : tensor<4x4x16x16xf32>, tensor<4x4x16x16xf32>, i1, index, index, index) outs(%empty : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<4x4x16x16xf32>) outs(%subview : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>)
    }
    return
  }

  func.func @fixpipe_subview_column_split_layout_mix_aiv(%arg0: memref<64xf32>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg1, %arg2 : i32
    %1 = arith.muli %0, %arg3 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<2x64x64xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x64xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc : memref<2x64x64xf32, #hivm.address_space<ub>> to memref<2x64x64xf32>
    %tensor = bufferization.to_tensor %memspacecast restrict writable : memref<2x64x64xf32>
    scf.for %i = %c0 to %c2 step %c1 {
      %slice = tensor.extract_slice %tensor[%i, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<2x64x64xf32> to tensor<64x64xf32>
      %empty_out = tensor.empty() : tensor<1x64xf32>
      %reduced = hivm.hir.vreduce <sum> ins(%slice : tensor<64x64xf32>) outs(%empty_out : tensor<1x64xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x64xf32>
      %collapsed = tensor.collapse_shape %reduced [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
      hivm.hir.store ins(%collapsed : tensor<64xf32>) outs(%arg0 : memref<64xf32>)
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @copy_last_dim_width_unaligned_aiv(
// CHECK:           hivm.hir.copy
// CHECK-NOT:       tiled_op
// CHECK:           scf.if
// CHECK:             hivm.hir.store
// CHECK:           } {limit_sub_block_id0}
// CHECK-NOT:       map_for_to_forall
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @copy_last_dim_width_unaligned_aiv(%arg0: tensor<1x1x1x8xf32>, %arg1: memref<1x1x1x8xf32>, %arg2: memref<1x1x1x8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.copy ins(%arg0 : tensor<1x1x1x8xf32>) outs(%arg1 : memref<1x1x1x8xf32>) {"inserted-copy"}
    hivm.hir.store ins(%arg0 : tensor<1x1x1x8xf32>) outs(%arg2 : memref<1x1x1x8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @reduce_dim_subblock_aiv(
// CHECK: hivm.hir.vreduce {tiled_op} <sum> ins(%{{.*}} : tensor<2048xf32>)
// CHECK: memref_ext.alloc_workspace() : memref<2xf32>
// CHECK: hivm.hir.sync_block[<ALL_SUB_VECTOR>] tvector_pipe = <PIPE_ALL>
// CHECK: %[[FINAL:.*]] = hivm.hir.vreduce <sum> ins(%{{.*}} : tensor<2xf32>)
// CHECK: scope.return %[[FINAL]] : tensor<1xf32>
// CHECK: annotation.mark
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9589">,
  hivm.module_core_type = #hivm.module_core_type<MIX>
} {
  func.func @reduce_dim_subblock_aiv(
      %arg0: tensor<4096xf32>,
      %arg1: memref<4096xf32>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.part_of_mix,
        mix_mode = "mix"
      } {
    hivm.hir.store ins(%arg0 : tensor<4096xf32>)
        outs(%arg1 : memref<4096xf32>)
    %empty = tensor.empty() : tensor<1xf32>
    %reduced = hivm.hir.vreduce <sum>
        ins(%arg0 : tensor<4096xf32>)
        outs(%empty : tensor<1xf32>)
        reduce_dims = [0] -> tensor<1xf32>
    annotation.mark %reduced : tensor<1xf32>
    return
  }
}

// -----
// Test that 1:2 tiling succeeds when cbuf tightly-coupled buffers are present
// alongside UB buffers. Cbuf marks must not leak into
// tightlyCoupledBufferToTilingDim and cause a false "UB not tiled" failure.
// CHECK-LABEL:   func.func @cbuf_filter_in_prune_mix_aiv(
// CHECK:         scf.for
// CHECK:         map_for_to_forall
// CHECK:         mapping = [#hivm.sub_block<x>]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cbuf_filter_in_prune_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<1x1x16x16xf16>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc : memref<1x1x16x16xf16>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1x1x16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1x16x16xf16>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc_1 : memref<1x1x16x16xf16>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x16x16xf16>
    %4 = tensor.empty() : tensor<1x1x16x16xf32>
    %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%2, %3, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%4 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<1x1x16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
  func.func @cbuf_filter_in_prune_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16 = arith.constant 16 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf32> to memref<32x16xf32, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<32x16xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<32x16xf32, strided<[16, 1]>>) outs(%alloc : memref<32x16xf32>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32>
    %alloc_1 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_1 {effects = ["write", "read"], hivm.multi_buffer = 2 : i32, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<32x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_1 : memref<32x16xf32, #hivm.address_space<ub>> to memref<32x16xf32>
    %3 = bufferization.to_tensor %memspacecast restrict writable : memref<32x16xf32>
    %4 = tensor.empty() : tensor<32x16xf32>
    %5 = hivm.hir.vadd ins(%3, %2 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%4 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %alloc_2 = memref.alloc() : memref<32x16xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.multi_buffer = 2 : i32, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<32x16xf32, #hivm.address_space<cbuf>>
    %memspacecast_0 = memref.memory_space_cast %alloc_2 : memref<32x16xf32, #hivm.address_space<cbuf>> to memref<32x16xf32>
    hivm.hir.copy ins(%5 : tensor<32x16xf32>) outs(%memspacecast_0 : memref<32x16xf32>) {"inserted-copy"}
    %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf32> to memref<32x16xf32, strided<[16, 1]>>
    hivm.hir.store ins(%5 : tensor<32x16xf32>) outs(%reinterpret_cast_0 : memref<32x16xf32, strided<[16, 1]>>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @copy_last_dim_width_unaligned_aiv(
// CHECK:           hivm.hir.copy
// CHECK-NOT:       tiled_op
// CHECK:           scf.if
// CHECK:             hivm.hir.store
// CHECK:           } {limit_sub_block_id0}
// CHECK-NOT:       map_for_to_forall
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @copy_last_dim_width_unaligned_aiv(%arg0: tensor<1x1x1x8xf32>, %arg1: memref<1x1x1x8xf32>, %arg2: memref<1x1x1x8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.copy ins(%arg0 : tensor<1x1x1x8xf32>) outs(%arg1 : memref<1x1x1x8xf32>) {"inserted-copy"}
    hivm.hir.store ins(%arg0 : tensor<1x1x1x8xf32>) outs(%arg2 : memref<1x1x1x8xf32>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @trace_def_ops_fixpipe_readview_mix_aic(
// CHECK:           memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins({{.*}} : tensor<8x4x16x16xf32>) outs({{.*}} : memref<32x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
// CHECK-NOT:       memref.memory_space_cast
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:           annotation.mark %[[EMPTY]] {matmul_at_least_once} : tensor<64x128xf32>
// CHECK-LABEL:   func.func @trace_def_ops_fixpipe_readview_mix_aiv(
// CHECK:           hivm.hir.store ins({{.*}} : tensor<32x128xf32>) outs({{.*}} : memref<32x128xf32, strided<[128, 1], offset: ?>>) {tiled_op}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @trace_def_ops_fixpipe_readview_mix_aic(%cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %src = tensor.empty() : tensor<8x4x16x16xf32>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x128xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
    %tensor = bufferization.to_tensor %cast restrict writable : memref<64x128xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4x16x16xf32>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
    %selected = scf.if %cond -> (tensor<64x128xf32>) {
      %empty = tensor.empty() : tensor<64x128xf32>
      scf.yield %empty : tensor<64x128xf32>
    } else {
      scf.yield %tensor : tensor<64x128xf32>
    }
    annotation.mark %selected {matmul_at_least_once} : tensor<64x128xf32>
    return
  }

  func.func @trace_def_ops_fixpipe_readview_mix_aiv(%arg0: memref<64x128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x128xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
    %tensor = bufferization.to_tensor %cast restrict writable : memref<64x128xf32>
    %empty = tensor.empty() : tensor<64x128xf32>
    %result = hivm.hir.vadd ins(%tensor, %tensor : tensor<64x128xf32>, tensor<64x128xf32>) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
    hivm.hir.store ins(%result : tensor<64x128xf32>) outs(%arg0 : memref<64x128xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @for_two_iter_args_shared_yield_aiv(
// CHECK: hivm.hir.load
// CHECK: scf.for
// CHECK: hivm.hir.store{{.*}} {tiled_op}
// CHECK: hivm.hir.store{{.*}} {tiled_op}
// CHECK-NOT: limit_sub_block_id0
// CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @for_two_iter_args_shared_yield_aiv(%arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    %reinterpret_cast1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    %reinterpret_cast2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    %reinterpret_cast3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    %alloc0 = memref.alloc() : memref<32x32xf32>
    hivm.hir.load ins(%reinterpret_cast0 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc0 : memref<32x32xf32>) eviction_policy = <EvictFirst>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x32xf32>
    %alloc1 = memref.alloc() : memref<32x32xf32>
    hivm.hir.load ins(%reinterpret_cast1 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc1 : memref<32x32xf32>) eviction_policy = <EvictFirst>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x32xf32>
    %results:2 = scf.for %iv = %c1_i32 to %c4_i32 step %c1_i32 iter_args(%a = %t0, %b = %t1) -> (tensor<32x32xf32>, tensor<32x32xf32>) : i32 {
      hivm.hir.store ins(%a : tensor<32x32xf32>) outs(%reinterpret_cast2 : memref<32x32xf32, strided<[32, 1]>>)
      hivm.hir.store ins(%b : tensor<32x32xf32>) outs(%reinterpret_cast3 : memref<32x32xf32, strided<[32, 1]>>)
      scf.yield %a, %a : tensor<32x32xf32>, tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_mix_aiv
// CHECK:           tiled_op
// CHECK:           map_for_to_forall
// CHECK-NOT:       limit_sub_block_id0
#map = affine_map<(d0, d1) -> ((d0 - d1) ceildiv 28)>
#map1 = affine_map<(d0)[s0] -> (d0 * 28 + s0)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 * 1024 + 64)>
#map4 = affine_map<()[s0, s1, s2] -> (s0 * 8 + s1 + s2)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1 * 1024)>
#map6 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>
#map7 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024)>
#map8 = affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1)>
#map10 = affine_map<()[s0] -> (s0 + 64)>
#map11 = affine_map<()[s0] -> (-(s0 floordiv 128) + 128)>
#map12 = affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 128)>
#map13 = affine_map<()[s0, s1] -> (s0 + s1 + 8192)>
#map14 = affine_map<()[s0] -> (-(s0 floordiv 128) + 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_957c">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 0.693147182 : f32
    %c7_i32 = arith.constant 7 : i32
    %c2_i32 = arith.constant 2 : i32
    %c6_i32 = arith.constant 6 : i32
    %c36_i64 = arith.constant 36 : i64
    %c12_i64 = arith.constant 12 : i64
    %c28_i64 = arith.constant 28 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %c44_i64 = arith.constant 44 : i64
    %c4_i64 = arith.constant 4 : i64
    %c20_i64 = arith.constant 20 : i64
    %c40_i64 = arith.constant 40 : i64
    %c24_i64 = arith.constant 24 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i32 = arith.constant 64 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c16384_i64 = arith.constant 16384 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i64 = arith.constant 128 : i64
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c56_i32 = arith.constant 56 : i32
    hivm.hir.anchor {id = 0 : i64}
    %0 = arith.muli %arg12, %arg13 : i32
    %1 = arith.muli %0, %arg14 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg15 = %3 to %1 step %c56_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      %4 = arith.index_cast %1 : i32 to index
      %5 = arith.index_cast %arg15 : i32 to index
      %6 = affine.apply #map(%4, %5)
      %7 = arith.minui %6, %c2 : index
      hivm.hir.anchor {id = 3 : i64}
      scf.for %arg16 = %c0 to %7 step %c1 {
        hivm.hir.anchor {id = 4 : i64}
        %8 = affine.apply #map1(%arg16)[%5]
        %9 = arith.index_cast %8 : index to i32
        %10 = arith.remsi %9, %arg12 : i32
        %11 = arith.divsi %9, %arg12 : i32
        %12 = arith.remsi %11, %arg13 : i32
        hivm.hir.anchor {id = 5 : i64}
        %13 = hivm.hir.get_sub_block_idx -> i64
        %14 = arith.muli %13, %c1024_i64 : i64
        %15 = llvm.inttoptr %14 : i64 to !llvm.ptr<11>
        %16 = arith.addi %14, %c24_i64 : i64
        %17 = llvm.inttoptr %16 : i64 to !llvm.ptr<11>
        %18 = arith.addi %14, %c40_i64 : i64
        %19 = llvm.inttoptr %18 : i64 to !llvm.ptr<11>
        %20 = arith.addi %14, %c20_i64 : i64
        %21 = llvm.inttoptr %20 : i64 to !llvm.ptr<11>
        %22 = arith.addi %14, %c4_i64 : i64
        %23 = llvm.inttoptr %22 : i64 to !llvm.ptr<11>
        %24 = arith.addi %14, %c44_i64 : i64
        %25 = llvm.inttoptr %24 : i64 to !llvm.ptr<11>
        %26 = arith.addi %14, %c32_i64 : i64
        %27 = llvm.inttoptr %26 : i64 to !llvm.ptr<11>
        %28 = arith.addi %14, %c16_i64 : i64
        %29 = llvm.inttoptr %28 : i64 to !llvm.ptr<11>
        %30 = arith.addi %14, %c8_i64 : i64
        %31 = llvm.inttoptr %30 : i64 to !llvm.ptr<11>
        %32 = arith.addi %14, %c28_i64 : i64
        %33 = llvm.inttoptr %32 : i64 to !llvm.ptr<11>
        %34 = arith.addi %14, %c12_i64 : i64
        %35 = llvm.inttoptr %34 : i64 to !llvm.ptr<11>
        %36 = arith.addi %14, %c36_i64 : i64
        %37 = llvm.inttoptr %36 : i64 to !llvm.ptr<11>
        %38 = arith.addi %arg11, %c63_i32 : i32
        %39 = arith.divsi %38, %c64_i32 : i32
        %40 = arith.subi %39, %c1_i32 : i32
        %41 = tensor.empty() : tensor<64x32xf32>
        %42 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %43 = arith.divsi %12, %c8_i32 : i32
        %44 = arith.remsi %12, %c8_i32 : i32
        %45 = arith.muli %43, %39 : i32
        %46 = arith.muli %45, %c8_i32 : i32
        %47 = arith.addi %46, %44 : i32
        %48 = arith.extsi %47 : i32 to i64
        %49 = arith.muli %48, %c16384_i64 : i64
        %50 = tensor.empty() : tensor<64xf32>
        %51 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
        %52 = arith.index_cast %44 : i32 to index
        %53 = arith.muli %43, %arg11 : i32
        %54 = arith.muli %53, %c8_i32 : i32
        %55 = arith.addi %54, %44 : i32
        %56 = arith.extsi %55 : i32 to i64
        %57 = arith.muli %56, %c128_i64 : i64
        %58 = arith.index_cast %57 : i64 to index
        %59 = arith.muli %10, %c32_i32 : i32
        %60 = arith.index_cast %54 : i32 to index
        %alloc = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        %alloc_2 = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        %alloc_3 = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %alloc_4 = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        %alloc_6 = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        %alloc_7 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>} : memref<64x32xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
        %alloc_8 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>} : memref<64x32xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
        %alloc_9 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>} : memref<64x32xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
        %alloc_10 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>} : memref<64x32xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 10
        %alloc_11 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>} : memref<64x32xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 11
        %alloc_12 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>} : memref<64xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_12 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
        %alloc_13 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>} : memref<64xf32, #hivm.address_space<ub>>
        %memspacecast_14 = memref.memory_space_cast %alloc_13 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
        %alloc_15 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_15 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>} : memref<64x64xf32, #hivm.address_space<ub>>
        %memspacecast_16 = memref.memory_space_cast %alloc_15 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %alloc_17 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>} : memref<64x64xf32, #hivm.address_space<ub>>
        %memspacecast_18 = memref.memory_space_cast %alloc_17 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %alloc_19 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_19 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<15>} : memref<64x32xf32, #hivm.address_space<ub>>
        %memspacecast_20 = memref.memory_space_cast %alloc_19 {ssbuffer.intraDeps = [2 : i32, 1 : i32]} : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %alloc_21 = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_21 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<16>} : memref<64x32xf32, #hivm.address_space<ub>>
        %memspacecast_22 = memref.memory_space_cast %alloc_21 {ssbuffer.intraDeps = [2 : i32, 1 : i32]} : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
        %61 = arith.muli %39, %c6_i32 : i32
        %62 = arith.addi %61, %c7_i32 : i32
        %63:14 = scf.for %arg17 = %c-1_i32 to %62 step %c1_i32 iter_args(%arg18 = %42, %arg19 = %42, %arg20 = %c-1_i32, %arg21 = %c-1_i32, %arg22 = %c-1_i32, %arg23 = %c-1_i32, %arg24 = %c-1_i32, %arg25 = %c-1_i32, %arg26 = %c-1_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c1_i32, %arg31 = %c1_i32) -> (tensor<64x32xf32>, tensor<64x32xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          %64 = arith.cmpi slt, %arg20, %40 : i32
          %65 = scf.if %64 -> (i32) {
            %134 = arith.addi %arg20, %c1_i32 : i32
            scf.yield %134 : i32
          } else {
            scf.yield %arg20 : i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 25 : i32}
          %66 = llvm.load volatile %37 : !llvm.ptr<11> -> i32
          %67 = arith.cmpi slt, %66, %c1_i32 : i32
          %68 = arith.cmpi eq, %arg31, %c1_i32 : i32
          %69 = arith.cmpi slt, %arg21, %40 : i32
          %70 = arith.andi %67, %68 : i1
          %71 = arith.andi %70, %69 : i1
          %72:2 = scf.if %71 -> (i32, i32) {
            %expanded = tensor.expand_shape %arg19 [[0], [1, 2]] output_shape [64, 4, 8] : tensor<64x32xf32> into tensor<64x4x8xf32>
            %134 = tensor.empty() : tensor<4x64x8xf32>
            %135 = hivm.hir.vtranspose ins(%expanded : tensor<64x4x8xf32>) outs(%134 : tensor<4x64x8xf32>) permutation = [1, 0, 2] -> tensor<4x64x8xf32>
            %expanded_23 = tensor.expand_shape %135 [[0], [1, 2], [3]] output_shape [4, 4, 16, 8] : tensor<4x64x8xf32> into tensor<4x4x16x8xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
            hivm.hir.copy ins(%expanded_23 : tensor<4x4x16x8xf32>) outs(%alloc_2 : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
            %136 = llvm.load volatile %37 : !llvm.ptr<11> -> i32
            %137 = arith.addi %136, %c1_i32 : i32
            llvm.store volatile %137, %37 : i32, !llvm.ptr<11>
            %138 = arith.subi %arg31, %c1_i32 : i32
            %139 = arith.addi %arg21, %c1_i32 : i32
            scf.yield %138, %139 : i32, i32
          } else {
            scf.yield %arg31, %arg21 : i32, i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 36 : i32}
          %73 = llvm.load volatile %33 : !llvm.ptr<11> -> i32
          %74 = arith.cmpi slt, %73, %c1_i32 : i32
          %75 = llvm.load volatile %35 : !llvm.ptr<11> -> i32
          %76 = arith.cmpi slt, %75, %c1_i32 : i32
          %77 = arith.andi %74, %76 : i1
          %78 = arith.cmpi eq, %arg30, %c1_i32 : i32
          %79 = arith.cmpi slt, %arg22, %40 : i32
          %80 = arith.andi %77, %78 : i1
          %81 = arith.andi %80, %79 : i1
          %82:2 = scf.if %81 -> (i32, i32) {
            %expanded = tensor.expand_shape %arg18 [[0], [1, 2]] output_shape [64, 4, 8] : tensor<64x32xf32> into tensor<64x4x8xf32>
            %134 = tensor.empty() : tensor<4x64x8xf32>
            %135 = hivm.hir.vtranspose ins(%expanded : tensor<64x4x8xf32>) outs(%134 : tensor<4x64x8xf32>) permutation = [1, 0, 2] -> tensor<4x64x8xf32>
            %expanded_23 = tensor.expand_shape %135 [[0], [1, 2], [3]] output_shape [4, 4, 16, 8] : tensor<4x64x8xf32> into tensor<4x4x16x8xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
            hivm.hir.copy ins(%expanded_23 : tensor<4x4x16x8xf32>) outs(%alloc : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
            %136 = llvm.load volatile %33 : !llvm.ptr<11> -> i32
            %137 = arith.addi %136, %c1_i32 : i32
            llvm.store volatile %137, %33 : i32, !llvm.ptr<11>
            %138 = llvm.load volatile %35 : !llvm.ptr<11> -> i32
            %139 = arith.addi %138, %c1_i32 : i32
            llvm.store volatile %139, %35 : i32, !llvm.ptr<11>
            %140 = arith.subi %arg30, %c1_i32 : i32
            %141 = arith.addi %arg22, %c1_i32 : i32
            scf.yield %140, %141 : i32, i32
          } else {
            scf.yield %arg30, %arg22 : i32, i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 35 : i32}
          %83 = llvm.load volatile %31 : !llvm.ptr<11> -> i32
          %84 = arith.cmpi slt, %83, %c1_i32 : i32
          %85 = arith.cmpi slt, %arg27, %c2_i32 : i32
          %86 = arith.cmpi slt, %arg28, %c2_i32 : i32
          %87 = arith.andi %85, %86 : i1
          %88 = arith.cmpi slt, %arg23, %40 : i32
          %89 = arith.andi %84, %87 : i1
          %90 = arith.andi %89, %88 : i1
          %91:3 = scf.if %90 -> (i32, i32, i32) {
            %134 = arith.subi %40, %arg23 {ssbuffer.dep_mark = [1 : i32]} : i32
            %135 = arith.addi %134, %c-1_i32 {ssbuffer.dep_mark = [22 : i32]} : i32
            %136 = arith.muli %135, %c64_i32 {ssbuffer.dep_mark = [3 : i32]} : i32
            %137 = arith.maxsi %136, %c0_i32 : i32
            %138 = arith.index_cast %137 : i32 to index
            %139 = arith.index_cast %arg11 : i32 to index
            %140 = affine.apply #map2()[%139, %138]
            %141 = arith.maxsi %140, %c0 : index
            %142 = arith.minsi %141, %c64 : index
            %143 = arith.subi %c0_i32, %136 : i32
            %144 = arith.maxsi %143, %c0_i32 : i32
            %145 = arith.index_cast %144 : i32 to index
            %146 = arith.minsi %145, %142 {ssbuffer.dep_mark = [10 : i32]} : index
            %147 = affine.apply #map2()[%142, %146]
            %148 = arith.cmpi slt, %147, %c64 : index
            %149 = affine.apply #map3()[%58, %138]
            %150 = arith.minsi %145, %142 {ssbuffer.dep_mark = [14 : i32]} : index
            %151 = affine.apply #map2()[%142, %150]
            %152 = arith.cmpi slt, %151, %c64 : index
            %alloc_23 = memref.alloc() : memref<64xf32>
            %alloc_24 = memref.alloc() : memref<64x64xf32>
            %153 = affine.apply #map4()[%138, %60, %52]
            %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%153], sizes: [64], strides: [8] : memref<?xf32> to memref<64xf32, strided<[8], offset: ?>>
            %subview = memref.subview %reinterpret_cast[0] [%147] [1] {ssbuffer.dep_mark = [9 : i32]} : memref<64xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[8], offset: ?>>
            %subview_25 = memref.subview %alloc_23[%146] [%147] [1] {ssbuffer.dep_mark = [9 : i32, 10 : i32]} : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
            %154 = arith.remui %146, %c8 : index
            hivm.hir.load ins(%subview : memref<?xf32, strided<[8], offset: ?>>) outs(%subview_25 : memref<?xf32, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %154 : index init_out_buffer = true init_condition = %148 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
            %155 = bufferization.to_tensor %alloc_23 restrict writable : memref<64xf32>
            %156 = arith.subi %arg23, %c-1_i32 : i32
            %157 = arith.remsi %156, %c2_i32 : i32
            %158 = arith.cmpi eq, %157, %c0_i32 : i32
            scf.if %158 {
              hivm.hir.copy ins(%155 : tensor<64xf32>) outs(%memspacecast : memref<64xf32>)
            } else {
              hivm.hir.copy ins(%155 : tensor<64xf32>) outs(%memspacecast_14 : memref<64xf32>)
            }
            %159 = hivm.hir.vmul ins(%155, %cst_0 : tensor<64xf32>, f32) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
            %160 = hivm.hir.vexp ins(%159 : tensor<64xf32>) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
            %expanded = tensor.expand_shape %160 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
            %161 = tensor.empty() {ssbuffer.dep_mark = [21 : i32]} : tensor<64x64xf32>
            %162 = hivm.hir.vbrc ins(%expanded : tensor<1x64xf32>) outs(%161 : tensor<64x64xf32>) broadcast_dims = [0] -> tensor<64x64xf32>
            scf.if %158 {
              hivm.hir.copy ins(%162 : tensor<64x64xf32>) outs(%memspacecast_16 : memref<64x64xf32>)
            } else {
              hivm.hir.copy ins(%162 : tensor<64x64xf32>) outs(%memspacecast_18 : memref<64x64xf32>)
            }
            %reinterpret_cast_26 = memref.reinterpret_cast %arg2 to offset: [%149], sizes: [64, 64], strides: [1024, 1] {ssbuffer.dep_mark = [11 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[1024, 1], offset: ?>>
            %subview_27 = memref.subview %reinterpret_cast_26[0, 0] [%151, 64] [1, 1] : memref<64x64xf32, strided<[1024, 1], offset: ?>> to memref<?x64xf32, strided<[1024, 1], offset: ?>>
            %subview_28 = memref.subview %alloc_24[%150, 0] [%151, 64] [1, 1] : memref<64x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
            hivm.hir.load ins(%subview_27 : memref<?x64xf32, strided<[1024, 1], offset: ?>>) outs(%subview_28 : memref<?x64xf32, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %152 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
            %163 = bufferization.to_tensor %alloc_24 restrict writable : memref<64x64xf32>
            %164 = hivm.hir.vtranspose ins(%163 : tensor<64x64xf32>) outs(%161 : tensor<64x64xf32>) permutation = [1, 0] -> tensor<64x64xf32>
            %165 = tensor.empty() : tensor<64x64xf32>
            %166 = hivm.hir.vmul ins(%164, %expanded : tensor<64x64xf32>, tensor<1x64xf32>) outs(%165 : tensor<64x64xf32>) broadcast = [0] -> tensor<64x64xf32>
            %expanded_29 = tensor.expand_shape %166 [[0], [1, 2]] output_shape [64, 8, 8] : tensor<64x64xf32> into tensor<64x8x8xf32>
            %167 = tensor.empty() : tensor<8x64x8xf32>
            %168 = hivm.hir.vtranspose ins(%expanded_29 : tensor<64x8x8xf32>) outs(%167 : tensor<8x64x8xf32>) permutation = [1, 0, 2] -> tensor<8x64x8xf32>
            %expanded_30 = tensor.expand_shape %168 [[0], [1, 2], [3]] output_shape [8, 4, 16, 8] : tensor<8x64x8xf32> into tensor<8x4x16x8xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
            hivm.hir.copy ins(%expanded_30 : tensor<8x4x16x8xf32>) outs(%alloc_4 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
            %169 = llvm.load volatile %31 : !llvm.ptr<11> -> i32
            %170 = arith.addi %169, %c1_i32 : i32
            llvm.store volatile %170, %31 : i32, !llvm.ptr<11>
            %171 = arith.addi %arg28, %c1_i32 : i32
            %172 = arith.addi %arg27, %c1_i32 : i32
            %173 = arith.addi %arg23, %c1_i32 : i32
            scf.yield %171, %172, %173 : i32, i32, i32
          } else {
            scf.yield %arg28, %arg27, %arg23 : i32, i32, i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 26 : i32}
          %92 = llvm.load volatile %29 : !llvm.ptr<11> -> i32
          %93 = arith.cmpi slt, %92, %c1_i32 : i32
          %94 = arith.cmpi sgt, %91#0, %c0_i32 : i32
          %95 = arith.cmpi slt, %arg24, %40 : i32
          %96 = arith.andi %93, %94 : i1
          %97 = arith.andi %96, %95 : i1
          %98:2 = scf.if %97 -> (i32, i32) {
            %134 = arith.subi %40, %arg24 {ssbuffer.dep_mark = [1 : i32]} : i32
            %135 = arith.addi %134, %c-1_i32 {ssbuffer.dep_mark = [22 : i32]} : i32
            %136 = arith.muli %135, %c64_i32 {ssbuffer.dep_mark = [3 : i32]} : i32
            %137 = arith.maxsi %136, %c0_i32 : i32
            %138 = arith.index_cast %137 : i32 to index
            %139 = arith.index_cast %arg11 : i32 to index
            %140 = arith.subi %c0_i32, %136 : i32
            %141 = arith.maxsi %140, %c0_i32 : i32
            %142 = arith.index_cast %141 : i32 to index
            %143 = affine.apply #map5()[%58, %138]
            %144 = affine.apply #map2()[%139, %138]
            %145 = arith.maxsi %144, %c0 : index
            %146 = arith.minsi %145, %c64 : index
            %147 = arith.minsi %142, %146 {ssbuffer.dep_mark = [19 : i32]} : index
            %148 = affine.apply #map2()[%146, %147]
            %149 = arith.cmpi slt, %148, %c64 : index
            %alloc_23 = memref.alloc() : memref<64x64xf32>
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%143], sizes: [64, 64], strides: [1024, 1] {ssbuffer.dep_mark = [16 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[1024, 1], offset: ?>>
            %subview = memref.subview %reinterpret_cast[0, 0] [%148, 64] [1, 1] : memref<64x64xf32, strided<[1024, 1], offset: ?>> to memref<?x64xf32, strided<[1024, 1], offset: ?>>
            %subview_24 = memref.subview %alloc_23[%147, 0] [%148, 64] [1, 1] : memref<64x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
            hivm.hir.load ins(%subview : memref<?x64xf32, strided<[1024, 1], offset: ?>>) outs(%subview_24 : memref<?x64xf32, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %149 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
            %150 = bufferization.to_tensor %alloc_23 restrict writable : memref<64x64xf32>
            %151 = tensor.empty() {ssbuffer.dep_mark = [21 : i32]} : tensor<64x64xf32>
            %152 = hivm.hir.vtranspose ins(%150 : tensor<64x64xf32>) outs(%151 : tensor<64x64xf32>) permutation = [1, 0] -> tensor<64x64xf32>
            %153 = arith.subi %arg24, %c-1_i32 : i32
            %154 = arith.remsi %153, %c2_i32 : i32
            %155 = arith.cmpi eq, %154, %c0_i32 : i32
            %156 = scf.if %155 -> (tensor<64x64xf32>) {
              %165 = bufferization.to_tensor %memspacecast_16 restrict writable : memref<64x64xf32>
              scf.yield %165 : tensor<64x64xf32>
            } else {
              %165 = bufferization.to_tensor %memspacecast_18 restrict writable : memref<64x64xf32>
              scf.yield %165 : tensor<64x64xf32>
            } {ssbuffer.intraDeps = [1 : i32, 0 : i32]}
            %157 = tensor.empty() : tensor<64x64xf32>
            %158 = hivm.hir.vmul ins(%152, %156 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%157 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %expanded = tensor.expand_shape %158 [[0], [1, 2]] output_shape [64, 8, 8] : tensor<64x64xf32> into tensor<64x8x8xf32>
            %159 = tensor.empty() : tensor<8x64x8xf32>
            %160 = hivm.hir.vtranspose ins(%expanded : tensor<64x8x8xf32>) outs(%159 : tensor<8x64x8xf32>) permutation = [1, 0, 2] -> tensor<8x64x8xf32>
            %expanded_25 = tensor.expand_shape %160 [[0], [1, 2], [3]] output_shape [8, 4, 16, 8] : tensor<8x64x8xf32> into tensor<8x4x16x8xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
            hivm.hir.copy ins(%expanded_25 : tensor<8x4x16x8xf32>) outs(%alloc_3 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
            %161 = llvm.load volatile %29 : !llvm.ptr<11> -> i32
            %162 = arith.addi %161, %c1_i32 : i32
            llvm.store volatile %162, %29 : i32, !llvm.ptr<11>
            %163 = arith.subi %91#0, %c1_i32 : i32
            %164 = arith.addi %arg24, %c1_i32 : i32
            scf.yield %163, %164 : i32, i32
          } else {
            scf.yield %91#0, %arg24 : i32, i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 27 : i32}
          %99 = llvm.load volatile %23 : !llvm.ptr<11> -> i32
          %100 = arith.cmpi sgt, %99, %c0_i32 : i32
          %101 = llvm.load volatile %25 : !llvm.ptr<11> -> i32
          %102 = arith.cmpi slt, %101, %c1_i32 : i32
          %103 = arith.andi %100, %102 : i1
          %104 = llvm.load volatile %27 : !llvm.ptr<11> -> i32
          %105 = arith.cmpi slt, %104, %c1_i32 : i32
          %106 = arith.andi %103, %105 : i1
          %107 = arith.cmpi sgt, %91#1, %c0_i32 : i32
          %108 = arith.cmpi slt, %arg29, %c2_i32 : i32
          %109 = arith.andi %107, %108 : i1
          %110 = arith.cmpi slt, %arg25, %40 : i32
          %111 = arith.andi %106, %109 : i1
          %112 = arith.andi %111, %110 : i1
          %113:3 = scf.if %112 -> (i32, i32, i32) {
            %134 = arith.subi %40, %arg25 {ssbuffer.dep_mark = [1 : i32]} : i32
            %135 = arith.addi %134, %c-1_i32 {ssbuffer.dep_mark = [22 : i32]} : i32
            %136 = arith.maxsi %59, %c0_i32 : i32
            %137 = arith.index_cast %136 {ssbuffer.dep_mark = [23 : i32]} : i32 to index
            %138 = arith.subi %c0_i32, %59 : i32
            %139 = arith.maxsi %138, %c0_i32 : i32
            %140 = arith.index_cast %139 {ssbuffer.dep_mark = [24 : i32]} : i32 to index
            %141 = arith.muli %135, %c64_i32 {ssbuffer.dep_mark = [3 : i32]} : i32
            %142 = arith.maxsi %141, %c0_i32 : i32
            %143 = arith.index_cast %142 : i32 to index
            %144 = arith.index_cast %arg11 : i32 to index
            %145 = arith.subi %c0_i32, %141 : i32
            %146 = arith.maxsi %145, %c0_i32 : i32
            %147 = arith.index_cast %146 : i32 to index
            %148 = affine.apply #map6()[%137, %58, %143]
            %149 = affine.apply #map7()[%144, %137, %143]
            %150 = arith.maxsi %149, %c0 : index
            %151 = arith.minsi %150, %c64 : index
            %152 = affine.apply #map8()[%137]
            %153 = arith.maxsi %152, %c0 : index
            %154 = arith.minsi %153, %c32 : index
            %155 = arith.minsi %147, %151 {ssbuffer.dep_mark = [6 : i32, 27 : i32]} : index
            %156 = affine.apply #map2()[%151, %155]
            %157 = arith.minsi %140, %154 {ssbuffer.dep_mark = [7 : i32, 28 : i32]} : index
            %158 = affine.apply #map2()[%154, %157]
            %159 = arith.cmpi slt, %156, %c64 : index
            %160 = arith.cmpi slt, %158, %c32 : index
            %161 = arith.ori %159, %160 : i1
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
            %memspacecast_23 = memref.memory_space_cast %alloc_7 : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
            %162 = bufferization.to_tensor %memspacecast_23 restrict writable : memref<64x32xf32>
            %alloc_24 = memref.alloc() : memref<64x32xf32>
            %163 = arith.muli %134, %c64_i32 {ssbuffer.dep_mark = [1 : i32]} : i32
            %164 = arith.minsi %163, %arg11 : i32
            %165 = arith.subi %164, %c1_i32 : i32
            %166 = arith.addi %53, %165 : i32
            %167 = arith.muli %166, %c8_i32 : i32
            %168 = arith.index_cast %167 : i32 to index
            %169 = affine.apply #map9()[%168, %52]
            %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%169], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %170 = memref.load %reinterpret_cast[%c0] {ssbuffer.dep_mark = [25 : i32]} : memref<1xf32, strided<[1], offset: ?>>
            %reinterpret_cast_25 = memref.reinterpret_cast %arg8 to offset: [%148], sizes: [64, 32], strides: [1024, 1] {ssbuffer.dep_mark = [2 : i32, 26 : i32]} : memref<?xf32> to memref<64x32xf32, strided<[1024, 1], offset: ?>>
            %171 = arith.subi %arg25, %c-1_i32 : i32
            %172 = arith.remsi %171, %c2_i32 : i32
            %173 = arith.cmpi eq, %172, %c0_i32 : i32
            %174 = scf.if %173 -> (tensor<64xf32>) {
              %199 = bufferization.to_tensor %memspacecast restrict writable : memref<64xf32>
              scf.yield %199 : tensor<64xf32>
            } else {
              %199 = bufferization.to_tensor %memspacecast_14 restrict writable : memref<64xf32>
              scf.yield %199 : tensor<64xf32>
            } {ssbuffer.intraDeps = [0 : i32, 0 : i32]}
            %175 = hivm.hir.vmul ins(%174, %cst : tensor<64xf32>, f32) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
            %176 = hivm.hir.vadd ins(%175, %170 : tensor<64xf32>, f32) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
            %177 = hivm.hir.vmul ins(%176, %cst_0 : tensor<64xf32>, f32) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
            %178 = hivm.hir.vexp ins(%177 : tensor<64xf32>) outs(%50 : tensor<64xf32>) -> tensor<64xf32>
            %179 = arith.index_cast %141 {ssbuffer.dep_mark = [3 : i32]} : i32 to index
            %180 = affine.apply #map10()[%179]
            %181 = arith.maxsi %179, %144 : index
            %182 = arith.minsi %180, %181 : index
            %183 = affine.apply #map2()[%182, %179]
            %extracted_slice = tensor.extract_slice %178[0] [%183] [1] : tensor<64xf32> to tensor<?xf32>
            %inserted_slice = tensor.insert_slice %extracted_slice into %51[0] [%183] [1] : tensor<?xf32> into tensor<64xf32>
            %expanded = tensor.expand_shape %inserted_slice [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
            %184 = hivm.hir.vmul ins(%162, %expanded : tensor<64x32xf32>, tensor<64x1xf32>) outs(%41 : tensor<64x32xf32>) broadcast = [1] -> tensor<64x32xf32>
            %subview = memref.subview %reinterpret_cast_25[0, 0] [%156, %158] [1, 1] {ssbuffer.dep_mark = [4 : i32, 5 : i32, 29 : i32, 30 : i32]} : memref<64x32xf32, strided<[1024, 1], offset: ?>> to memref<?x?xf32, strided<[1024, 1], offset: ?>>
            %subview_26 = memref.subview %alloc_24[%155, %157] [%156, %158] [1, 1] {ssbuffer.dep_mark = [4 : i32, 5 : i32, 6 : i32, 7 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32]} : memref<64x32xf32> to memref<?x?xf32, strided<[32, 1], offset: ?>>
            %185 = arith.remui %157, %c8 : index
            hivm.hir.load ins(%subview : memref<?x?xf32, strided<[1024, 1], offset: ?>>) outs(%subview_26 : memref<?x?xf32, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %185 : index init_out_buffer = true init_condition = %161 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
            %186 = bufferization.to_tensor %alloc_24 restrict writable : memref<64x32xf32>
            %187 = hivm.hir.vadd ins(%184, %186 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            scf.if %173 {
              hivm.hir.copy ins(%187 : tensor<64x32xf32>) outs(%memspacecast_20 : memref<64x32xf32>)
            } else {
              hivm.hir.copy ins(%187 : tensor<64x32xf32>) outs(%memspacecast_22 : memref<64x32xf32>)
            }
            %expanded_27 = tensor.expand_shape %187 [[0], [1, 2]] output_shape [64, 4, 8] : tensor<64x32xf32> into tensor<64x4x8xf32>
            %188 = tensor.empty() : tensor<4x64x8xf32>
            %189 = hivm.hir.vtranspose ins(%expanded_27 : tensor<64x4x8xf32>) outs(%188 : tensor<4x64x8xf32>) permutation = [1, 0, 2] -> tensor<4x64x8xf32>
            %expanded_28 = tensor.expand_shape %189 [[0], [1, 2], [3]] output_shape [4, 4, 16, 8] : tensor<4x64x8xf32> into tensor<4x4x16x8xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
            hivm.hir.copy ins(%expanded_28 : tensor<4x4x16x8xf32>) outs(%alloc_5 : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
            hivm.hir.copy ins(%expanded_28 : tensor<4x4x16x8xf32>) outs(%alloc_6 : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
            %190 = llvm.load volatile %23 : !llvm.ptr<11> -> i32
            %191 = arith.subi %190, %c1_i32 : i32
            llvm.store volatile %191, %23 : i32, !llvm.ptr<11>
            %192 = llvm.load volatile %25 : !llvm.ptr<11> -> i32
            %193 = arith.addi %192, %c1_i32 : i32
            llvm.store volatile %193, %25 : i32, !llvm.ptr<11>
            %194 = llvm.load volatile %27 : !llvm.ptr<11> -> i32
            %195 = arith.addi %194, %c1_i32 : i32
            llvm.store volatile %195, %27 : i32, !llvm.ptr<11>
            %196 = arith.subi %91#1, %c1_i32 : i32
            %197 = arith.addi %arg29, %c1_i32 : i32
            %198 = arith.addi %arg25, %c1_i32 : i32
            scf.yield %196, %197, %198 : i32, i32, i32
          } else {
            scf.yield %91#1, %arg29, %arg25 : i32, i32, i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 28 : i32}
          %114 = llvm.load volatile %15 : !llvm.ptr<11> -> i32
          %115 = arith.cmpi sgt, %114, %c0_i32 : i32
          %116 = llvm.load volatile %17 : !llvm.ptr<11> -> i32
          %117 = arith.cmpi sgt, %116, %c0_i32 : i32
          %118 = arith.andi %115, %117 : i1
          %119 = llvm.load volatile %19 : !llvm.ptr<11> -> i32
          %120 = arith.cmpi sgt, %119, %c0_i32 : i32
          %121 = arith.andi %118, %120 : i1
          %122 = llvm.load volatile %21 : !llvm.ptr<11> -> i32
          %123 = arith.cmpi sgt, %122, %c0_i32 : i32
          %124 = arith.andi %121, %123 : i1
          %125 = arith.cmpi sgt, %113#1, %c0_i32 : i32
          %126 = arith.cmpi eq, %72#0, %c0_i32 : i32
          %127 = arith.cmpi eq, %82#0, %c0_i32 : i32
          %128 = arith.andi %125, %126 : i1
          %129 = arith.andi %128, %127 : i1
          %130 = arith.cmpi slt, %arg26, %40 : i32
          %131 = arith.andi %124, %129 : i1
          %132 = arith.andi %131, %130 : i1
          %133:6 = scf.if %132 -> (tensor<64x32xf32>, tensor<64x32xf32>, i32, i32, i32, i32) {
            %134 = arith.subi %40, %arg26 {ssbuffer.dep_mark = [1 : i32]} : i32
            %135 = arith.addi %134, %c-1_i32 {ssbuffer.dep_mark = [22 : i32]} : i32
            %136 = arith.maxsi %59, %c0_i32 : i32
            %137 = arith.index_cast %136 {ssbuffer.dep_mark = [23 : i32]} : i32 to index
            %138 = arith.subi %c0_i32, %59 : i32
            %139 = arith.maxsi %138, %c0_i32 : i32
            %140 = arith.index_cast %139 {ssbuffer.dep_mark = [24 : i32]} : i32 to index
            %141 = arith.muli %135, %c64_i32 {ssbuffer.dep_mark = [3 : i32]} : i32
            %142 = arith.maxsi %141, %c0_i32 : i32
            %143 = arith.index_cast %142 : i32 to index
            %144 = arith.index_cast %arg11 : i32 to index
            %145 = arith.subi %c0_i32, %141 : i32
            %146 = arith.maxsi %145, %c0_i32 : i32
            %147 = arith.index_cast %146 : i32 to index
            %148 = affine.apply #map6()[%137, %58, %143]
            %149 = affine.apply #map7()[%144, %137, %143]
            %150 = arith.maxsi %149, %c0 : index
            %151 = arith.minsi %150, %c64 : index
            %152 = affine.apply #map8()[%137]
            %153 = arith.maxsi %152, %c0 : index
            %154 = arith.minsi %153, %c32 : index
            %155 = arith.minsi %147, %151 {ssbuffer.dep_mark = [6 : i32, 27 : i32]} : index
            %156 = affine.apply #map2()[%151, %155]
            %157 = arith.minsi %140, %154 {ssbuffer.dep_mark = [7 : i32, 28 : i32]} : index
            %158 = affine.apply #map2()[%154, %157]
            %159 = arith.muli %134, %c64_i32 {ssbuffer.dep_mark = [1 : i32]} : i32
            %160 = arith.minsi %159, %arg11 : i32
            %161 = arith.subi %160, %c1_i32 : i32
            %162 = arith.addi %53, %161 : i32
            %163 = arith.muli %162, %c8_i32 : i32
            %164 = arith.index_cast %163 : i32 to index
            %165 = affine.apply #map9()[%164, %52]
            %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%165], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %166 = memref.load %reinterpret_cast[%c0] {ssbuffer.dep_mark = [25 : i32]} : memref<1xf32, strided<[1], offset: ?>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 11
            %memspacecast_23 = memref.memory_space_cast %alloc_11 : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
            %167 = bufferization.to_tensor %memspacecast_23 restrict writable : memref<64x32xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 10
            %memspacecast_24 = memref.memory_space_cast %alloc_10 : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
            %168 = bufferization.to_tensor %memspacecast_24 restrict writable : memref<64x32xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 9
            %memspacecast_25 = memref.memory_space_cast %alloc_9 : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
            %169 = bufferization.to_tensor %memspacecast_25 restrict writable : memref<64x32xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 8
            %memspacecast_26 = memref.memory_space_cast %alloc_8 : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
            %170 = bufferization.to_tensor %memspacecast_26 restrict writable : memref<64x32xf32>
            %171 = arith.extsi %135 {ssbuffer.dep_mark = [22 : i32]} : i32 to i64
            %172 = arith.muli %171, %c131072_i64 : i64
            %173 = arith.addi %49, %172 : i64
            %174 = arith.index_cast %173 : i64 to index
            %175 = affine.apply #map9()[%174, %137]
            %reinterpret_cast_27 = memref.reinterpret_cast %arg7 to offset: [%175], sizes: [64, 32], strides: [128, 1] : memref<?xf32> to memref<64x32xf32, strided<[128, 1], offset: ?>>
            %176 = affine.apply #map11()[%137]
            %177 = arith.maxsi %176, %c0 : index
            %178 = arith.minsi %177, %c64 : index
            %179 = affine.apply #map12()[%137]
            %180 = arith.maxsi %179, %c0 : index
            %181 = arith.minsi %180, %c32 : index
            %182 = arith.minsi %178, %c0 : index
            %183 = affine.apply #map2()[%178, %182]
            %184 = arith.minsi %140, %181 {ssbuffer.dep_mark = [24 : i32]} : index
            %185 = affine.apply #map2()[%181, %184]
            %extracted_slice = tensor.extract_slice %arg18[%182, %184] [%183, %185] [1, 1] : tensor<64x32xf32> to tensor<?x?xf32>
            %subview = memref.subview %reinterpret_cast_27[0, 0] [%183, %185] [1, 1] : memref<64x32xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
            hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[128, 1], offset: ?>>)
            %186 = affine.apply #map13()[%137, %174]
            %reinterpret_cast_28 = memref.reinterpret_cast %arg7 to offset: [%186], sizes: [64, 32], strides: [128, 1] : memref<?xf32> to memref<64x32xf32, strided<[128, 1], offset: ?>>
            %187 = affine.apply #map14()[%137]
            %188 = arith.maxsi %187, %c0 : index
            %189 = arith.minsi %188, %c64 : index
            %190 = arith.minsi %189, %c0 : index
            %191 = affine.apply #map2()[%189, %190]
            %extracted_slice_29 = tensor.extract_slice %arg19[%190, %184] [%191, %185] [1, 1] : tensor<64x32xf32> to tensor<?x?xf32>
            %subview_30 = memref.subview %reinterpret_cast_28[0, 0] [%191, %185] [1, 1] : memref<64x32xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
            hivm.hir.store ins(%extracted_slice_29 : tensor<?x?xf32>) outs(%subview_30 : memref<?x?xf32, strided<[128, 1], offset: ?>>)
            %192 = tensor.empty() : tensor<1xf32>
            %inserted = tensor.insert %166 into %192[%c0] {ssbuffer.dep_mark = [25 : i32]} : tensor<1xf32>
            %193 = hivm.hir.vmul ins(%inserted, %cst_0 : tensor<1xf32>, f32) outs(%192 : tensor<1xf32>) -> tensor<1xf32>
            %194 = hivm.hir.vexp ins(%193 : tensor<1xf32>) outs(%192 : tensor<1xf32>) -> tensor<1xf32>
            %extracted = tensor.extract %194[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
            %reinterpret_cast_31 = memref.reinterpret_cast %arg9 to offset: [%148], sizes: [64, 32], strides: [1024, 1] {ssbuffer.dep_mark = [2 : i32, 26 : i32]} : memref<?xf32> to memref<64x32xf32, strided<[1024, 1], offset: ?>>
            %195 = arith.subi %arg26, %c-1_i32 : i32
            %196 = arith.remsi %195, %c2_i32 : i32
            %197 = arith.cmpi eq, %196, %c0_i32 : i32
            %198 = scf.if %197 -> (tensor<64x32xf32>) {
              %219 = bufferization.to_tensor %memspacecast_20 restrict writable : memref<64x32xf32>
              scf.yield %219 : tensor<64x32xf32>
            } else {
              %219 = bufferization.to_tensor %memspacecast_22 restrict writable : memref<64x32xf32>
              scf.yield %219 : tensor<64x32xf32>
            } {ssbuffer.intraDeps = [2 : i32, 0 : i32]}
            %extracted_slice_32 = tensor.extract_slice %198[%155, %157] [%156, %158] [1, 1] {ssbuffer.dep_mark = [4 : i32, 5 : i32, 6 : i32, 7 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32]} : tensor<64x32xf32> to tensor<?x?xf32>
            %subview_33 = memref.subview %reinterpret_cast_31[0, 0] [%156, %158] [1, 1] {ssbuffer.dep_mark = [4 : i32, 5 : i32, 29 : i32, 30 : i32]} : memref<64x32xf32, strided<[1024, 1], offset: ?>> to memref<?x?xf32, strided<[1024, 1], offset: ?>>
            hivm.hir.store ins(%extracted_slice_32 : tensor<?x?xf32>) outs(%subview_33 : memref<?x?xf32, strided<[1024, 1], offset: ?>>)
            %199 = hivm.hir.vmul ins(%arg18, %extracted : tensor<64x32xf32>, f32) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %200 = hivm.hir.vmul ins(%170, %arg10 : tensor<64x32xf32>, f32) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %201 = hivm.hir.vsub ins(%200, %169 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %202 = hivm.hir.vadd ins(%199, %201 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %203 = hivm.hir.vmul ins(%arg19, %extracted : tensor<64x32xf32>, f32) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %204 = hivm.hir.vmul ins(%168, %arg10 : tensor<64x32xf32>, f32) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %205 = hivm.hir.vsub ins(%204, %167 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            %206 = hivm.hir.vadd ins(%203, %205 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%41 : tensor<64x32xf32>) -> tensor<64x32xf32>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 10
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 11
            %207 = llvm.load volatile %15 : !llvm.ptr<11> -> i32
            %208 = arith.subi %207, %c1_i32 : i32
            llvm.store volatile %208, %15 : i32, !llvm.ptr<11>
            %209 = llvm.load volatile %17 : !llvm.ptr<11> -> i32
            %210 = arith.subi %209, %c1_i32 : i32
            llvm.store volatile %210, %17 : i32, !llvm.ptr<11>
            %211 = llvm.load volatile %19 : !llvm.ptr<11> -> i32
            %212 = arith.subi %211, %c1_i32 : i32
            llvm.store volatile %212, %19 : i32, !llvm.ptr<11>
            %213 = llvm.load volatile %21 : !llvm.ptr<11> -> i32
            %214 = arith.subi %213, %c1_i32 : i32
            llvm.store volatile %214, %21 : i32, !llvm.ptr<11>
            %215 = arith.addi %72#0, %c1_i32 : i32
            %216 = arith.addi %82#0, %c1_i32 : i32
            %217 = arith.subi %113#1, %c1_i32 : i32
            %218 = arith.addi %arg26, %c1_i32 : i32
            scf.yield %202, %206, %215, %216, %217, %218 : tensor<64x32xf32>, tensor<64x32xf32>, i32, i32, i32, i32
          } else {
            scf.yield %arg18, %arg19, %72#0, %82#0, %113#1, %arg26 : tensor<64x32xf32>, tensor<64x32xf32>, i32, i32, i32, i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 29 : i32}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          scf.yield %133#0, %133#1, %65, %72#1, %82#1, %91#2, %98#1, %113#2, %133#5, %113#0, %98#0, %133#4, %133#3, %133#2 : tensor<64x32xf32>, tensor<64x32xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
        }
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.anchor {id = 6 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      hivm.hir.anchor {id = 7 : i64}
      scf.for %arg16 = %c0 to %7 step %c1 {
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.anchor {id = 9 : i64}
        hivm.hir.anchor {id = 10 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      hivm.hir.anchor {id = 11 : i64}
    } {cv_unrolled_loop}
    hivm.hir.anchor {id = 12 : i64}
    return
  }
}

// -----
// Test that 1:2 tiling succeeds when cbuf tightly-coupled buffers are present
// alongside UB buffers. Cbuf marks must not leak into
// tightlyCoupledBufferToTilingDim and cause a false "UB not tiled" failure.
// CHECK-LABEL:   func.func @cbuf_filter_in_prune_mix_aiv(
// CHECK:         scf.for
// CHECK:         map_for_to_forall
// CHECK:         mapping = [#hivm.sub_block<x>]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cbuf_filter_in_prune_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<1x1x16x16xf16>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc : memref<1x1x16x16xf16>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1x1x16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1x16x16xf16>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc_1 : memref<1x1x16x16xf16>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x16x16xf16>
    %4 = tensor.empty() : tensor<1x1x16x16xf32>
    %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%2, %3, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%4 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<1x1x16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
  func.func @cbuf_filter_in_prune_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16 = arith.constant 16 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf32> to memref<32x16xf32, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<32x16xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<32x16xf32, strided<[16, 1]>>) outs(%alloc : memref<32x16xf32>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32>
    %alloc_1 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_1 {effects = ["write", "read"], hivm.multi_buffer = 2 : i32, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<32x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_1 : memref<32x16xf32, #hivm.address_space<ub>> to memref<32x16xf32>
    %3 = bufferization.to_tensor %memspacecast restrict writable : memref<32x16xf32>
    %4 = tensor.empty() : tensor<32x16xf32>
    %5 = hivm.hir.vadd ins(%3, %2 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%4 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %alloc_2 = memref.alloc() : memref<32x16xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.multi_buffer = 2 : i32, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<32x16xf32, #hivm.address_space<cbuf>>
    %memspacecast_0 = memref.memory_space_cast %alloc_2 : memref<32x16xf32, #hivm.address_space<cbuf>> to memref<32x16xf32>
    hivm.hir.copy ins(%5 : tensor<32x16xf32>) outs(%memspacecast_0 : memref<32x16xf32>) {"inserted-copy"}
    %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf32> to memref<32x16xf32, strided<[16, 1]>>
    hivm.hir.store ins(%5 : tensor<32x16xf32>) outs(%reinterpret_cast_0 : memref<32x16xf32, strided<[16, 1]>>)
    return
  }
}

// -----
 	 
// CHECK-LABEL:   func.func @indirect_load_dual_store_mix_aiv(
// CHECK:           scf.for
// CHECK:             hivm.hir.indirect_load ins(%{{.*}} : memref<?xf32>, %{{.*}} : tensor<8xi64>, %{{.*}} : tensor<8xi8>, %{{.*}} : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) {hivm.vf_mode = #hivm.vf_mode<SIMT>}
// CHECK:             hivm.hir.indirect_load ins(%{{.*}} : memref<?xf32>, %{{.*}} : tensor<8xi64>, %{{.*}} : tensor<8xi8>, %{{.*}} : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) {hivm.vf_mode = #hivm.vf_mode<SIMT>}
// CHECK:             hivm.hir.store ins(%{{.*}} : tensor<8xf32>) outs(%{{.*}} : memref<8xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:             hivm.hir.store ins(%{{.*}} : tensor<8xf32>) outs(%{{.*}} : memref<8xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
func.func @indirect_load_dual_store_mix_aiv(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xi64> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMT>, mix_mode = "mix", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16_i64 = arith.constant 16 : i64
  %c16_i32 = arith.constant 16 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg5 : i32
  %5 = arith.muli %4, %c16_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [16], strides: [1] : memref<?xi64> to memref<16xi64, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<16xi64>
  hivm.hir.load ins(%reinterpret_cast : memref<16xi64, strided<[1], offset: ?>>) outs(%alloc : memref<16xi64>)
  %7 = bufferization.to_tensor %alloc restrict writable : memref<16xi64>
  %8 = tensor.empty() : tensor<16xi1>
  %9 = hivm.hir.vcmp ins(%7, %c16_i64 : tensor<16xi64>, i64) outs(%8 : tensor<16xi1>) compare_mode = <lt> -> tensor<16xi1>
  %10 = tensor.empty() : tensor<16xi8>
  %11 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%9 : tensor<16xi1>) outs(%10 : tensor<16xi8>) -> tensor<16xi8>
  %12 = tensor.empty() : tensor<16xf32>
  %13 = hivm.hir.vbrc ins(%cst : f32) outs(%12 : tensor<16xf32>) -> tensor<16xf32>
  %14 = tensor.empty() : tensor<16xf32>
  %15 = hivm.hir.indirect_load ins(%arg0 : memref<?xf32>, %7 : tensor<16xi64>, %11 : tensor<16xi8>, %13 : tensor<16xf32>) outs(%14 : tensor<16xf32>) {hivm.vf_mode = #hivm.vf_mode<SIMT>} -> tensor<16xf32>
  %16 = tensor.empty() : tensor<16xf32>
  %17 = hivm.hir.indirect_load ins(%arg2 : memref<?xf32>, %7 : tensor<16xi64>, %11 : tensor<16xi8>, %13 : tensor<16xf32>) outs(%16 : tensor<16xf32>) {hivm.vf_mode = #hivm.vf_mode<SIMT>} -> tensor<16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%6], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
  hivm.hir.store ins(%15 : tensor<16xf32>) outs(%reinterpret_cast_0 : memref<16xf32, strided<[1], offset: ?>>)
  hivm.hir.store ins(%17 : tensor<16xf32>) outs(%reinterpret_cast_1 : memref<16xf32, strided<[1], offset: ?>>)
  return
}
