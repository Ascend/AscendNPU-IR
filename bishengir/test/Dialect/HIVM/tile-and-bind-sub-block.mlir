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
  // CHECK-NOT: scf.if
  // CHECK: hivm.hir.store
  // CHECK-NOT: limit_sub_block_id0
  hivm.hir.store ins(%extracted_slice : tensor<?x16xf16>) outs(%subview_5 : memref<?x16xf16, strided<[16, 1], offset: ?>>)
  return
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
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index} : memref<16x16xf32, #hivm.address_space<ub>>
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
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:           %[[VAL_30:.*]] = hivm.hir.vmul
// CHECK:           hivm.hir.debug {debugtype = "print", hex = false, prefix = " qk: ", tcoretype = #hivm.tcore_type<VECTOR>, tiled_op} %[[VAL_30:.*]] : tensor<8x16xf32>
// CHECK:           %[[VAL_40:.*]] = bufferization.to_tensor %[[memspacecast_5:.*]] restrict writable : memref<1x1x16x16xf16>
// CHECK:           hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
// CHECK:           %[[subview:.*]] = memref.subview %[[memspacecast_5:.*]][0, 0, %0, 0] [1, 1, 8, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<1x1x16x16xf16> to memref<1x1x8x16xf16, strided<[256, 256, 16, 1], offset: ?>>
// CHECK:           hivm.hir.copy ins(%[[VAL_39:.*]] : tensor<1x1x8x16xf16>) outs(%[[subview:.*]] : memref<1x1x8x16xf16, strided<[256, 256, 16, 1], offset: ?>>) {tiled_op}
// CHECK:           annotation.mark %[[VAL_40:.*]] : tensor<1x1x16x16xf16>
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
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:           hivm.hir.debug {debugtype = "print", hex = false, prefix = " qk: ", tcoretype = #hivm.tcore_type<VECTOR>, tiled_op} %[[VAL_30:.*]] : tensor<8x16xf32>
// CHECK:           %[[VAL_48:.*]] = hivm.hir.copy ins(%[[VAL_39:.*]] : tensor<1x1x8x16xf16>) outs(%[[VAL_40:.*]] : tensor<1x1x8x16xf16>) {tiled_op} -> tensor<1x1x8x16xf16>
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
// CHECK:      %[[VAL_45:.*]] = tensor.empty() : tensor<64x128xbf16>
// CHECK:      %[[VAL_46:.*]] = hivm.hir.vcast ins(%[[VAL_40:.*]] : tensor<64x128xf32>) outs(%[[VAL_45:.*]] : tensor<64x128xbf16>) -> tensor<64x128xbf16>
// CHECK:      %[[VAL_SUBVIEW_7:.*]] = memref.subview %[[VAL_REINTERPRET_36:.*]][{{.*}}, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<64x128xbf16, strided<[128, 1], offset: ?>>
// CHECK:      hivm.hir.store ins(%[[VAL_46:.*]] : tensor<64x128xbf16>) outs(%[[VAL_SUBVIEW_7:.*]] : memref<64x128xbf16, strided<[128, 1], offset: ?>>) {tiled_op}
func.func @fa_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %c8192 = arith.constant 8192 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant -1.000000e+04 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8388608_i64 = arith.constant 8388608 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst_3 = arith.constant 0.000000e+00 : f32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg11, %arg10 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg9 : i32
    %7 = tensor.empty() : tensor<128x128xf32>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 4
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 6
    %8 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %9 = tensor.empty() : tensor<128xf32>
    %10 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %11 = hivm.hir.vbrc ins(%cst : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %12 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %13 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %14 = arith.addi %12, %c1 : index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    scf.for %arg12 = %13 to %15 step %c1_i32  : i32 {
      %16 = arith.divsi %arg12, %c64_i32 : i32
      %17 = arith.remsi %arg12, %c64_i32 : i32
      %18 = arith.divsi %16, %c8_i32 : i32
      %19 = arith.remsi %16, %c8_i32 : i32
      %20 = arith.extsi %18 : i32 to i64
      %21 = arith.muli %20, %c8388608_i64 : i64
      %22 = arith.extsi %19 : i32 to i64
      %23 = arith.muli %22, %c1048576_i64 : i64
      %24 = arith.addi %21, %23 : i64
      %25 = arith.index_cast %24 : i64 to index
      %26 = arith.muli %17, %c128_i32 : i32
      %27 = arith.maxsi %26, %c0_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.muli %28, %c128 : index
      %30 = arith.addi %29, %25 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%30], sizes: [128, 128], strides: [128, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
      %31:5 = scf.for %arg13 = %c0_i32 to %26 step %c128_i32 iter_args(%arg14 = %11, %arg15 = %8, %arg16 = %10, %arg17 = %c0_i32, %arg18 = %c0_i32) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32)  : i32 {
        %45 = tensor.empty() : tensor<128x128xbf16>
        %alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %46 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %47 = hivm.hir.vmul ins(%46, %cst_2 : tensor<128x128xf32>, f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
        %expanded_7 = tensor.expand_shape %10 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %48 = hivm.hir.vreduce <max> ins(%47 : tensor<128x128xf32>) outs(%expanded_7 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed = tensor.collapse_shape %48 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %49 = hivm.hir.vmax ins(%arg16, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_8 = tensor.expand_shape %49 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %50 = hivm.hir.vsub ins(%47, %expanded_8 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %51 = hivm.hir.vexp ins(%50 : tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %52 = hivm.hir.vcast ins(%51 : tensor<128x128xf32>) outs(%45 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %53 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_9 = tensor.expand_shape %53 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %54 = hivm.hir.vreduce <sum> ins(%51 : tensor<128x128xf32>) outs(%expanded_9 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed_10 = tensor.collapse_shape %54 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %55 = hivm.hir.vsub ins(%arg16, %49 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %56 = hivm.hir.vexp ins(%55 : tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %57 = hivm.hir.vmul ins(%arg14, %56 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %58 = hivm.hir.vadd ins(%57, %collapsed_10 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_11 = tensor.expand_shape %56 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %59 = hivm.hir.vmul ins(%arg15, %expanded_11 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %expanded_12 = tensor.expand_shape %52 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
        %60 = tensor.empty() : tensor<8x128x16xbf16>
        %61 = hivm.hir.vtranspose ins(%expanded_12 : tensor<128x8x16xbf16>) outs(%60 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
        %expanded_13 = tensor.expand_shape %61 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
        annotation.mark %expanded_13 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xbf16>
        %alloc_14 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %memspacecast_15 = memref.memory_space_cast %alloc_14 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x8x16x16xbf16>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
        hivm.hir.copy ins(%expanded_13 : tensor<8x8x16x16xbf16>) outs(%memspacecast_15 : memref<8x8x16x16xbf16>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
        %alloc_16 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast_17 = memref.memory_space_cast %alloc_16 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %62 = bufferization.to_tensor %memspacecast_17 restrict writable : memref<128x128xf32>
        %63 = tensor.empty() : tensor<128x128xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %64 = hivm.hir.vadd ins(%62, %59 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%63 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 3
        %65 = arith.addi %arg17, %c128_i32 : i32
        %66 = arith.addi %arg18, %c128_i32 : i32
        scf.yield %58, %64, %49, %65, %66 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %32 = arith.muli %17, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %33 = arith.addi %17, %c1_i32 : i32
      %34 = arith.muli %33, %c128_i32 : i32
      %35:6 = scf.for %arg13 = %32 to %34 step %c128_i32 iter_args(%arg14 = %32, %arg15 = %31#0, %arg16 = %31#1, %arg17 = %31#2, %arg18 = %32, %arg19 = %32) -> (i32, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32)  : i32 {
        %45 = arith.maxsi %arg14, %c0_i32 : i32
        %46 = arith.index_cast %45 : i32 to index
        %47 = arith.muli %28, %c8192 : index
        %48 = arith.addi %47, %46 : index
        %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%48], sizes: [128, 128], strides: [8192, 1] : memref<?xf32> to memref<128x128xf32, strided<[8192, 1], offset: ?>>
        %49 = tensor.empty() : tensor<128x128xbf16>
        %alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %50 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
        %alloc_8 = memref.alloc() : memref<128x128xf32>
        hivm.hir.load ins(%reinterpret_cast_7 : memref<128x128xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_8 : memref<128x128xf32>) eviction_policy = <EvictFirst>
        %51 = bufferization.to_tensor %alloc_8 restrict writable : memref<128x128xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %52 = hivm.hir.vmul ins(%50, %cst_2 : tensor<128x128xf32>, f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 4
        %53 = tensor.empty() : tensor<128x128xi1>
        %54 = hivm.hir.vcmp ins(%51, %8 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%53 : tensor<128x128xi1>) compare_mode = <ne> -> tensor<128x128xi1>
        %55 = hivm.hir.vsel ins(%54, %cst_1, %cst_3 : tensor<128x128xi1>, f32, f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %56 = hivm.hir.vadd ins(%52, %55 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %expanded_9 = tensor.expand_shape %10 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %57 = hivm.hir.vreduce <max> ins(%56 : tensor<128x128xf32>) outs(%expanded_9 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed = tensor.collapse_shape %57 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %58 = hivm.hir.vmax ins(%arg17, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_10 = tensor.expand_shape %58 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %59 = hivm.hir.vsub ins(%56, %expanded_10 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %60 = arith.addi %arg14, %c128_i32 : i32
        %61 = hivm.hir.vexp ins(%59 : tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %62 = hivm.hir.vcast ins(%61 : tensor<128x128xf32>) outs(%49 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %63 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_11 = tensor.expand_shape %63 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %64 = hivm.hir.vreduce <sum> ins(%61 : tensor<128x128xf32>) outs(%expanded_11 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed_12 = tensor.collapse_shape %64 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %65 = hivm.hir.vsub ins(%arg17, %58 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %66 = hivm.hir.vexp ins(%65 : tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %67 = hivm.hir.vmul ins(%arg15, %66 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %68 = hivm.hir.vadd ins(%67, %collapsed_12 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_13 = tensor.expand_shape %66 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %69 = hivm.hir.vmul ins(%arg16, %expanded_13 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %expanded_14 = tensor.expand_shape %62 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
        %70 = tensor.empty() : tensor<8x128x16xbf16>
        %71 = hivm.hir.vtranspose ins(%expanded_14 : tensor<128x8x16xbf16>) outs(%70 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
        %expanded_15 = tensor.expand_shape %71 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
        annotation.mark %expanded_15 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xbf16>
        %alloc_16 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %memspacecast_17 = memref.memory_space_cast %alloc_16 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x8x16x16xbf16>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 5
        hivm.hir.copy ins(%expanded_15 : tensor<8x8x16x16xbf16>) outs(%memspacecast_17 : memref<8x8x16x16xbf16>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
        %alloc_18 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<128x128xf32, #hivm.address_space<ub>>
        %memspacecast_19 = memref.memory_space_cast %alloc_18 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %72 = bufferization.to_tensor %memspacecast_19 restrict writable : memref<128x128xf32>
        %73 = tensor.empty() : tensor<128x128xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %74 = hivm.hir.vadd ins(%72, %69 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%73 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 6
        %75 = arith.addi %arg18, %c128_i32 : i32
        %76 = arith.addi %arg19, %c128_i32 : i32
        scf.yield %60, %68, %74, %58, %75, %76 : i32, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %36 = hivm.hir.vln ins(%35#1 : tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
      %37 = hivm.hir.vadd ins(%35#3, %36 : tensor<128xf32>, tensor<128xf32>) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
      %expanded = tensor.expand_shape %35#1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %38 = hivm.hir.vdiv ins(%35#2, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
      %39 = arith.muli %16, %c8192_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.index_cast %26 : i32 to index
      %42 = arith.addi %40, %41 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%42], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%37 : tensor<128xf32>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>>)
      %43 = tensor.empty() : tensor<128x128xbf16>
      %44 = hivm.hir.vcast ins(%38 : tensor<128x128xf32>) outs(%43 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
      hivm.hir.store ins(%44 : tensor<128x128xbf16>) outs(%reinterpret_cast_5 : memref<128x128xbf16, strided<[128, 1], offset: ?>>)
    }
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 5
    return
  }

// -----

// CHECK-LABEL:   func.func @_hstu_attn_fwd_mix_aiv_ub2ub(
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:           hivm.hir.debug {debugtype = "print", hex = false, prefix = " qk: ", tcoretype = #hivm.tcore_type<VECTOR>, tiled_op} %[[VAL_29:.*]] : tensor<8x16xf32>
// CHECK:           %[[VAL_alloc_4:.*]] = memref.alloc() : memref<1x1x8x16xf16, #hivm.address_space<ub>>
// CHECK:           annotation.mark %[[VAL_alloc_4:.*]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 2 : index} : memref<1x1x8x16xf16, #hivm.address_space<ub>>
// CHECK:           %[[VAL_memspacecast_5:.*]] = memref.memory_space_cast %[[VAL_alloc_4:.*]] : memref<1x1x8x16xf16, #hivm.address_space<ub>> to memref<1x1x8x16xf16>
// CHECK:           %[[VAL_40:.*]] = bufferization.to_tensor %[[VAL_memspacecast_5:.*]] restrict writable : memref<1x1x8x16xf16>
// CHECK:           hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
// CHECK:           %[[VAL_41:.*]] = hivm.hir.copy ins(%38 : tensor<1x1x8x16xf16>) outs(%[[VAL_40:.*]] : tensor<1x1x8x16xf16>) {tiled_op} -> tensor<1x1x8x16xf16>
// CHECK:           annotation.mark %[[VAL_41:.*]] : tensor<1x1x8x16xf16>
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

// -----
 
// CHECK-LABEL: func.func @_hstu_attn_fwd_mix_aiv
// CHECK: %[[VAL_37:.*]] = memref.alloc() : memref<72x64xf32>
// CHECK: %[[VAL_38:.*]] = memref.subview %[[VAL_37:.*]][0, 0] [{{.*}}, 64] [1, 1] : memref<72x64xf32> to memref<?x64xf32, strided<[64, 1]>>
// CHECK: hivm.hir.load ins({{.*}} : memref<?x64xf32, strided<[256, 1], offset: ?>>) outs(%[[VAL_38:.*]] : memref<?x64xf32, strided<[64, 1]>>)
// CHECK: %[[VAL_39:.*]] = bufferization.to_tensor %[[VAL_37:.*]] restrict writable : memref<72x64xf32>
// CHECK: %[[VAL_45:.*]] = tensor.empty() : tensor<72x64xf16>
// CHECK: %[[VAL_46:.*]] = hivm.hir.vcast ins(%[[VAL_39:.*]] : tensor<72x64xf32>) outs(%[[VAL_45:.*]] : tensor<72x64xf16>) -> tensor<72x64xf16>
// CHECK-NOT: scf.if
// CHECK: hivm.hir.store
// CHECK-NOT: limit_sub_block_id0
func.func @_hstu_attn_fwd_mix_aiv(%arg0: memref<?xf32>, %arg1: memref<?xf16>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, parallel_mode = "simd"} {
  %c144 = arith.constant 144 : index
  %0 = arith.muli %arg2, %arg3 : i32
  %1 = arith.muli %0, %arg4 : i32
  %2 = arith.index_cast %1 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [144, 64], strides: [256, 1] : memref<?xf32> to memref<144x64xf32, strided<[256, 1], offset: ?>>
  %alloc = memref.alloc() : memref<144x64xf32>
  %3 = arith.addi %2, %c144 : index
  %4 = arith.minsi %3, %2 : index
  %5 = arith.subi %4, %2 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%5, 64] [1, 1] : memref<144x64xf32, strided<[256, 1], offset: ?>> to memref<?x64xf32, strided<[256, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%5, 64] [1, 1] : memref<144x64xf32> to memref<?x64xf32, strided<[64, 1]>>
  hivm.hir.load ins(%subview : memref<?x64xf32, strided<[256, 1], offset: ?>>) outs(%subview_0 : memref<?x64xf32, strided<[64, 1]>>)
  %6 = bufferization.to_tensor %alloc restrict writable : memref<144x64xf32>
  %7 = tensor.empty() : tensor<144x64xf16>
  %8 = hivm.hir.vcast ins(%6 : tensor<144x64xf32>) outs(%7 : tensor<144x64xf16>) -> tensor<144x64xf16>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [144, 64], strides: [256, 1] : memref<?xf16> to memref<144x64xf16, strided<[256, 1], offset: ?>>
  %extracted_slice = tensor.extract_slice %8[0, 0] [%5, 64] [1, 1] : tensor<144x64xf16> to tensor<?x64xf16>
  %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%5, 64] [1, 1] : memref<144x64xf16, strided<[256, 1], offset: ?>> to memref<?x64xf16, strided<[256, 1], offset: ?>>
  hivm.hir.store ins(%extracted_slice : tensor<?x64xf16>) outs(%subview_2 : memref<?x64xf16, strided<[256, 1], offset: ?>>)
  return
}

// ----- 
 
// CHECK-LABEL: func.func @check_while
// CHECK: scf.if
// CHECK: hivm.hir.store
// CHECK: limit_sub_block_id0
module {
  func.func @check_while(%arg0: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {func_dyn_memref_args = dense<true> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.divsi %1, %arg4 : i32
    %3 = arith.remsi %2, %arg3 : i32
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = hivm.hir.vbrc ins(%cst : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = arith.divsi %3, %c4_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %8 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %9 = arith.addi %7, %c1 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %10 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %11 = arith.subi %10, %8 : i32
    %12 = arith.minsi %11, %c32_i32 : i32
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [32, 32], strides: [%c0, 1] : memref<?xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
    %13:2 = scf.while (%arg5 = %5, %arg6 = %c0_i32) : (tensor<32x32xf32>, i32) -> (tensor<32x32xf32>, i32) {
      %14 = arith.cmpi slt, %arg6, %12 : i32
      scf.condition(%14) %arg5, %arg6 : tensor<32x32xf32>, i32
    } do {
    ^bb0(%arg5: tensor<32x32xf32>, %arg6: i32):
      %alloc = memref.alloc() : memref<32x32xf32>
      %14 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32>
      %15 = tensor.empty() : tensor<32x32xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %16 = hivm.hir.vadd ins(%14, %arg5 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 8
      %17 = arith.addi %arg6, %c32_i32 : i32
      scf.yield %16, %17 : tensor<32x32xf32>, i32
    }
    %extracted_slice = tensor.extract_slice %13#0[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
    %subview = memref.subview %reinterpret_cast_1[0, 0] [1, 32] [1, 1] : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<1x32xf32, strided<[?, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<1x32xf32>) outs(%subview : memref<1x32xf32, strided<[?, 1], offset: ?>>)
    return
  }
}


// ----- 
 
// CHECK-LABEL: func.func @check_column_split
// CHECK: scf.if
// CHECK: hivm.hir.store
// CHECK: limit_sub_block_id0
module attributes {hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} { 
  func.func @check_column_split_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8>, %arg2: memref<?xi32>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<16x16xi32>
    %alloc = memref.alloc() : memref<16x16xi32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<16x16xi32>) outs(%alloc : memref<16x16xi32, #hivm.address_space<ub>>)
    return
  }
  func.func @check_column_split_aiv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %alloc = memref.alloc() : memref<16x16xi32>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xi32>
    %1 = tensor.empty() : tensor<16xi32>
    %2 = hivm.hir.vbrc ins(%c-2147483648_i32 : i32) outs(%1 : tensor<16xi32>) -> tensor<16xi32>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %3 = hivm.hir.vreduce <max> ins(%0 : tensor<16x16xi32>) outs(%expanded : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1]>>
    hivm.hir.store ins(%collapsed : tensor<16xi32>) outs(%reinterpret_cast : memref<16xi32, strided<[1]>>)
    return
  }
}


// ----- 
 
// CHECK-LABEL: func.func @check_split_nz2dn
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%{{.*}} : tensor<32x128xf32>) outs(%{{.*}} : memref<128x32xf32, #hivm.address_space<ub>>)
// CHECK: scf.if
// CHECK: hivm.hir.store
// CHECK: limit_sub_block_id0
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_split_nz2dn_aic() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %alloc = memref.alloc() : memref<128x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"]} : memref<128x32xf32, #hivm.address_space<ub>>
    %0 = tensor.empty() : tensor<32x128xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%0 : tensor<32x128xf32>) outs(%alloc : memref<128x32xf32, #hivm.address_space<ub>>)
    return
  }
  func.func @check_split_nz2dn_aiv(%arg0: memref<?xf32>, %arg1: i32, %arg2: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg2, %arg1 : i32
    %3 = arith.divsi %1, %2 : i32
    %alloc = memref.alloc() : memref<128x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x32xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc : memref<128x32xf32, #hivm.address_space<ub>> to memref<128x32xf32>
    %4 = bufferization.to_tensor %memspacecast restrict writable : memref<128x32xf32>
    %5 = arith.index_cast %2 : i32 to index
    %6 = arith.muli %5, %c128 : index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.addi %6, %7 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [128, 32], strides: [256, 1] : memref<?xf32> to memref<128x32xf32, strided<[256, 1], offset: ?>>
    %9 = arith.minsi %5, %c128 : index
    %10 = arith.minsi %5, %c32 : index
    %extracted_slice = tensor.extract_slice %4[0, 0] [%9, %10] [1, 1] : tensor<128x32xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%9, %10] [1, 1] : memref<128x32xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[256, 1], offset: ?>>)
    return
  }
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

// CHECK-LABEL: func.func @check_indirect_store_with_batch_matmul_mix_aiv
// CHECK: %[[SUB_BLOCK:.*]] = hivm.hir.get_sub_block_idx
// CHECK: %[[SUB_BLOCK_IDX:.*]] = arith.index_cast %[[SUB_BLOCK]] : i64 to index
// CHECK: %[[IS_SUB_BLOCK_0:.*]] = arith.cmpi eq, %[[SUB_BLOCK_IDX]], %{{.*}} : index
// CHECK: scf.if %[[IS_SUB_BLOCK_0]] {
// CHECK:   hivm.hir.indirect_store
// CHECK: } {limit_sub_block_id0}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_indirect_store_with_batch_matmul_mix_aic() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = tensor.empty() : tensor<16x16xf16>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = hivm.hir.mmadL1 {batch_matmul} ins(%0, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }

  func.func @check_indirect_store_with_batch_matmul_mix_aiv(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0_i64 = arith.constant 0 : i64
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x64xf16>
    %1 = tensor.empty() : tensor<16x64xi64>
    %2 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<16x64xi64>) -> tensor<16x64xi64>
    %3 = tensor.empty() : tensor<16x64xi1>
    %4 = linalg.fill ins(%true : i1) outs(%3 : tensor<16x64xi1>) -> tensor<16x64xi1>
    hivm.hir.indirect_store ins(%0 : tensor<16x64xf16>, %2 : tensor<16x64xi64>, %4 : tensor<16x64xi1>) outs(%arg0 : memref<?xf16>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @check_indirect_store_in_dynamic_shape_mix_aiv
// CHECK: %[[SUB_BLOCK:.*]] = hivm.hir.get_sub_block_idx
// CHECK: %[[SUB_BLOCK_IDX:.*]] = arith.index_cast %[[SUB_BLOCK]] : i64 to index
// CHECK: %[[IS_SUB_BLOCK_0:.*]] = arith.cmpi eq, %[[SUB_BLOCK_IDX]], %{{.*}} : index
// CHECK: scf.if %[[IS_SUB_BLOCK_0]] {
// CHECK:   hivm.hir.indirect_store
// CHECK: } {limit_sub_block_id0}
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_indirect_store_in_dynamic_shape_mix_aiv(%arg0: memref<?xf16>, %arg1: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0_i64 = arith.constant 0 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty(%arg1) : tensor<?x64xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x64xf16>) -> tensor<?x64xf16>
    %2 = tensor.empty(%arg1) : tensor<?x64xi64>
    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<?x64xi64>) -> tensor<?x64xi64>
    %4 = tensor.empty(%arg1) : tensor<?x64xi1>
    %5 = linalg.fill ins(%true : i1) outs(%4 : tensor<?x64xi1>) -> tensor<?x64xi1>
    hivm.hir.indirect_store ins(%1 : tensor<?x64xf16>, %3 : tensor<?x64xi64>, %5 : tensor<?x64xi1>) outs(%arg0 : memref<?xf16>)
    return
  }
}


// ----- 
 
// CHECK-LABEL: func.func @chunk_gated_delta_rule_fwd_kernel_h_blockdim64_mix_aiv
// CHECK-NOT: scf.if
// CHECK: hivm.hir.store
// CHECK-NOT: limit_sub_block_id0
#map = affine_map<()[s0, s1] -> (s0 + s1 * 512)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
#map3 = affine_map<()[s0] -> (s0 * 16)>
#map4 = affine_map<()[s0] -> (s0 * 2)>
#map5 = affine_map<()[s0] -> (-s0 + (s0 floordiv 512) * 512 + 64)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map7 = affine_map<()[s0] -> (-s0 + 64)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
#map9 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 512)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 512)>
#map11 = affine_map<()[s0] -> (s0 + 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @chunk_gated_delta_rule_fwd_kernel_h_blockdim64_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg11, %arg12 : i32
    %1 = arith.muli %0, %arg13 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg13 : i32
    %5 = arith.remsi %4, %arg12 : i32
    %6 = tensor.empty() : tensor<64x32xf32>
    %7 = arith.divsi %5, %c8_i32 : i32
    %8 = arith.remsi %5, %c8_i32 : i32
    %9 = arith.muli %7, %arg10 : i32
    %10 = arith.addi %arg10, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32 : i32
    %12 = arith.muli %9, %c8_i32 : i32
    %13 = arith.addi %12, %8 : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.muli %14, %c64_i64 : i64
    %16 = arith.index_cast %15 : i64 to index
    scf.for %arg14 = %c0_i32 to %11 step %c1_i32  : i32 {
      %17 = arith.muli %arg14, %c64_i32 : i32
      %18 = arith.maxsi %17, %c0_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = affine.apply #map()[%16, %19]
      %21 = arith.index_cast %arg10 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [64, 64], strides: [512, 1] : memref<?xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf16>
      %22 = affine.apply #map1()[%21, %19]
      %23 = arith.maxsi %22, %c0 : index
      %24 = arith.minsi %23, %c64 : index
      %25 = arith.subi %c0_i32, %17 : i32
      %26 = arith.maxsi %25, %c0_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.minsi %27, %24 : index
      %29 = affine.apply #map1()[%24, %28]
      %30 = arith.cmpi slt, %29, %c64 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%29, 64] [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<?x64xf16, strided<[512, 1], offset: ?>>
      %subview_0 = memref.subview %alloc[%28, 0] [%29, 64] [1, 1] : memref<64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      hivm.hir.load ins(%subview : memref<?x64xf16, strided<[512, 1], offset: ?>>) outs(%subview_0 : memref<?x64xf16, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %30 : i1 eviction_policy = <EvictFirst>
      %31 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
      %alloc_1 = memref.alloc() : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      %memspacecast = memref.memory_space_cast %alloc_1 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x4x16x16xf16>
      %32 = bufferization.to_tensor %memspacecast restrict writable : memref<2x4x16x16xf16>
      %33 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%31, %32, %true, %c64, %c64, %c32 : tensor<64x64xf16>, tensor<2x4x16x16xf16>, i1, index, index, index) outs(%6 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %alloc_2 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index} : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%33 : tensor<64x32xf32>) outs(%alloc_2 : memref<32x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [512, 1] : memref<?xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
      %alloc_4 = memref.alloc() : memref<64x64xf16>
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%29, 64] [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<?x64xf16, strided<[512, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[%28, 0] [%29, 64] [1, 1] : memref<64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      hivm.hir.load ins(%subview_5 : memref<?x64xf16, strided<[512, 1], offset: ?>>) outs(%subview_6 : memref<?x64xf16, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %30 : i1 eviction_policy = <EvictFirst>
      %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xf16>
      %alloc_7 = memref.alloc() : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      %memspacecast_8 = memref.memory_space_cast %alloc_7 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x4x16x16xf16>
      %35 = bufferization.to_tensor %memspacecast_8 restrict writable : memref<2x4x16x16xf16>
      %36 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_already_inserted = true} ins(%34, %35, %true, %c64, %c64, %c32 : tensor<64x64xf16>, tensor<2x4x16x16xf16>, i1, index, index, index) outs(%6 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %alloc_9 = memref.alloc() : memref<64x16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 1 : index} : memref<64x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%36 : tensor<64x32xf32>) outs(%alloc_9 : memref<64x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
    }
    return
  }
  func.func @chunk_gated_delta_rule_fwd_kernel_h_blockdim64_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64 = arith.constant 64 : index
    %c63_i32 = arith.constant 63 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c32 = arith.constant 32 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %cst_1 = arith.constant -1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg14 = %c0 to %c2 step %c1 {
      %0 = affine.apply #map2()[%arg14]
      %1 = affine.apply #map3()[%arg14]
      %2 = affine.apply #map4()[%arg14]
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %3 = arith.muli %arg11, %arg12 : i32
      %4 = arith.muli %3, %arg13 : i32
      annotation.mark %4 {logical_block_num} : i32
      %5 = hivm.hir.get_block_idx -> i64
      %6 = arith.trunci %5 : i64 to i32
      %7 = arith.divsi %6, %arg13 : i32
      %8 = arith.remsi %7, %arg12 : i32
      %9 = arith.muli %arg13, %arg12 : i32
      %10 = arith.divsi %6, %9 : i32
      %11 = arith.remsi %10, %arg11 : i32
      %12 = tensor.empty() : tensor<32x32xf32>
      %13 = tensor.empty() : tensor<32xf32>
      %14 = hivm.hir.vbrc ins(%cst : f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
      %15 = tensor.empty() : tensor<16x64xf32>
      %16 = arith.divsi %8, %c8_i32 : i32
      %17 = arith.remsi %8, %c8_i32 : i32
      %18 = arith.muli %16, %arg10 : i32
      %19 = arith.addi %arg10, %c63_i32 : i32
      %20 = arith.divsi %19, %c64_i32 : i32
      %21 = arith.muli %16, %20 : i32
      %22 = arith.muli %21, %c8_i32 : i32
      %23 = arith.addi %22, %17 : i32
      %24 = arith.extsi %23 : i32 to i64
      %25 = arith.muli %24, %c4096_i64 : i64
      %26 = arith.muli %18, %c8_i32 : i32
      %27 = arith.addi %26, %17 : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = arith.muli %28, %c64_i64 : i64
      %30 = arith.index_cast %29 : i64 to index
      %31 = arith.muli %8, %c4096_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.muli %11, %c32_i32 : i32
      %34 = arith.maxsi %33, %c0_i32 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = affine.apply #map5()[%35]
      %37 = affine.apply #map6()[%32, %35]
      %38 = affine.apply #map7()[%35]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%37], sizes: [32, 64], strides: [64, 1] : memref<?xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x64xf32>
      %39 = arith.maxsi %38, %c0 : index
      %40 = arith.minsi %39, %c32 : index
      %41 = arith.subi %c0_i32, %33 : i32
      %42 = arith.maxsi %41, %c0_i32 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.minsi %43, %40 : index
      %45 = affine.apply #map1()[%40, %44]
      %46 = arith.minsi %1, %45 : index
      %47 = arith.subi %45, %46 : index
      %48 = arith.minsi %47, %c16 : index
      %49 = arith.cmpi slt, %45, %c32 : index
      %subview = memref.subview %reinterpret_cast[%1, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %subview_2 = memref.subview %subview[0, 0] [%48, 64] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1], offset: ?>>
      %subview_3 = memref.subview %alloc[%44, 0] [%48, 64] [1, 1] : memref<16x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.load ins(%subview_2 : memref<?x64xf32, strided<[64, 1], offset: ?>>) outs(%subview_3 : memref<?x64xf32, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %49 : i1 eviction_policy = <EvictFirst>
      %50 = arith.index_cast %27 : i32 to index
      %51 = bufferization.to_tensor %alloc restrict writable : memref<16x64xf32>
      %52 = scf.for %arg15 = %c0_i32 to %20 step %c1_i32 iter_args(%arg16 = %51) -> (tensor<16x64xf32>)  : i32 {
        %53 = arith.extsi %arg15 : i32 to i64
        %54 = arith.muli %53, %c32768_i64 : i64
        %55 = arith.addi %25, %54 : i64
        %56 = arith.index_cast %55 : i64 to index
        %57 = affine.apply #map6()[%56, %35]
        %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%57], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %subview_8 = memref.subview %reinterpret_cast_7[%1, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<16x64xf16, strided<[64, 1], offset: ?>>
        %58 = tensor.empty() : tensor<16x64xf16>
        %59 = hivm.hir.vcast ins(%arg16 : tensor<16x64xf32>) outs(%58 : tensor<16x64xf16>) -> tensor<16x64xf16>
        %extracted_slice_9 = tensor.extract_slice %59[%44, 0] [%48, 64] [1, 1] : tensor<16x64xf16> to tensor<?x64xf16>
        %subview_10 = memref.subview %subview_8[0, 0] [%48, 64] [1, 1] : memref<16x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
        hivm.hir.store ins(%extracted_slice_9 : tensor<?x64xf16>) outs(%subview_10 : memref<?x64xf16, strided<[64, 1], offset: ?>>) {tiled_op}
        %60 = arith.muli %arg15, %c64_i32 : i32
        %61 = arith.maxsi %60, %c0_i32 : i32
        %62 = arith.index_cast %61 : i32 to index
        %63 = affine.apply #map8()[%50, %62]
        %64 = affine.apply #map9()[%35, %30, %62]
        %65 = arith.index_cast %arg10 : i32 to index
        %66 = affine.apply #map1()[%65, %62]
        %67 = affine.apply #map10()[%65, %35, %62]
        %68 = arith.maxsi %66, %c0 : index
        %69 = arith.minsi %68, %c64 : index
        %70 = arith.subi %c0_i32, %60 : i32
        %71 = arith.maxsi %70, %c0_i32 : i32
        %72 = arith.index_cast %71 : i32 to index
        %73 = arith.minsi %72, %69 : index
        %74 = affine.apply #map1()[%69, %73]
        %75 = arith.minsi %0, %74 : index
        %76 = arith.subi %74, %75 : index
        %77 = arith.minsi %76, %c32 : index
        %78 = arith.cmpi slt, %74, %c64 : index
        %79 = tensor.empty() : tensor<64x16xf32>
        %80 = hivm.hir.vtranspose ins(%arg16 : tensor<16x64xf32>) outs(%79 : tensor<64x16xf32>) permutation = [1, 0] -> tensor<64x16xf32>
        %81 = tensor.empty() : tensor<32x32xf16>
        %82 = tensor.empty() : tensor<64x16xf16>
        %83 = hivm.hir.vcast ins(%80 : tensor<64x16xf32>) outs(%82 : tensor<64x16xf16>) -> tensor<64x16xf16>
        %84 = tensor.empty() : tensor<1x64x16xf16>
        %expanded = tensor.expand_shape %83 [[0], [1, 2]] output_shape [64, 1, 16] : tensor<64x16xf16> into tensor<64x1x16xf16>
        %85 = hivm.hir.vtranspose ins(%expanded : tensor<64x1x16xf16>) outs(%84 : tensor<1x64x16xf16>) permutation = [1, 0, 2] -> tensor<1x64x16xf16>
        %alloc_11 = memref.alloc() : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %alloc_11 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x4x16x16xf16>
        %expanded_12 = tensor.expand_shape %85 [[0], [1, 2], [3]] output_shape [1, 4, 16, 16] : tensor<1x64x16xf16> into tensor<1x4x16x16xf16>
        %subview_13 = memref.subview %memspacecast[%arg14, 0, 0, 0] [1, 4, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<2x4x16x16xf16> to memref<1x4x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>>
        hivm.hir.copy ins(%expanded_12 : tensor<1x4x16x16xf16>) outs(%subview_13 : memref<1x4x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>>) {tiled_op}
        %alloc_14 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index} : memref<32x32xf32, #hivm.address_space<ub>>
        %memspacecast_15 = memref.memory_space_cast %alloc_14 : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
        %86 = bufferization.to_tensor %memspacecast_15 restrict writable : memref<32x32xf32>
        %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%64], sizes: [64, 32], strides: [512, 1] : memref<?xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
        %alloc_17 = memref.alloc() : memref<32x32xf16>
        %87 = arith.maxsi %67, %c0 : index
        %88 = arith.minsi %87, %c64 : index
        %89 = arith.maxsi %36, %c0 : index
        %90 = arith.minsi %89, %c32 : index
        %91 = arith.minsi %72, %88 : index
        %92 = affine.apply #map1()[%88, %91]
        %93 = arith.minsi %0, %92 : index
        %94 = arith.subi %92, %93 : index
        %95 = arith.minsi %94, %c32 : index
        %96 = arith.minsi %43, %90 : index
        %97 = affine.apply #map1()[%90, %96]
        %98 = arith.cmpi slt, %92, %c64 : index
        %99 = arith.cmpi slt, %97, %c32 : index
        %100 = arith.ori %98, %99 : i1
        %subview_18 = memref.subview %reinterpret_cast_16[%0, 0] [32, 32] [1, 1] : memref<64x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16, strided<[512, 1], offset: ?>>
        %subview_19 = memref.subview %subview_18[0, 0] [%95, %97] [1, 1] : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %subview_20 = memref.subview %alloc_17[%91, %96] [%95, %97] [1, 1] : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        %101 = arith.remui %96, %c16 : index
        hivm.hir.load ins(%subview_19 : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%subview_20 : memref<?x?xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_0 : f16 left_padding_num = %101 : index init_out_buffer = true init_condition = %100 : i1 eviction_policy = <EvictFirst>
        %102 = bufferization.to_tensor %alloc_17 restrict writable : memref<32x32xf16>
        %103 = hivm.hir.vcast ins(%102 : tensor<32x32xf16>) outs(%12 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %104 = hivm.hir.vsub ins(%103, %86 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%12 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %reinterpret_cast_21 = memref.reinterpret_cast %arg5 to offset: [%64], sizes: [64, 32], strides: [512, 1] : memref<?xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
        %subview_22 = memref.subview %reinterpret_cast_21[%0, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : memref<64x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16, strided<[512, 1], offset: ?>>
        %105 = hivm.hir.vcast ins(%104 : tensor<32x32xf32>) outs(%81 : tensor<32x32xf16>) -> tensor<32x32xf16>
        %extracted_slice_23 = tensor.extract_slice %105[%91, %96] [%95, %97] [1, 1] : tensor<32x32xf16> to tensor<?x?xf16>
        %subview_24 = memref.subview %subview_22[0, 0] [%95, %97] [1, 1] : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        hivm.hir.store ins(%extracted_slice_23 : tensor<?x?xf16>) outs(%subview_24 : memref<?x?xf16, strided<[512, 1], offset: ?>>) {tiled_op}
        %106 = arith.addi %arg15, %c1_i32 : i32
        %107 = arith.muli %106, %c64_i32 : i32
        %108 = arith.minsi %107, %arg10 : i32
        %109 = arith.subi %108, %c1_i32 : i32
        %110 = arith.muli %109, %c8_i32 : i32
        %111 = arith.addi %26, %110 : i32
        %112 = arith.addi %111, %17 : i32
        %113 = arith.index_cast %112 : i32 to index
        %reinterpret_cast_25 = memref.reinterpret_cast %arg6 to offset: [%113], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %114 = memref.load %reinterpret_cast_25[%c0] : memref<1xf32, strided<[1], offset: ?>>
        %reinterpret_cast_26 = memref.reinterpret_cast %arg6 to offset: [%63], sizes: [64], strides: [8] : memref<?xf32> to memref<64xf32, strided<[8], offset: ?>>
        %alloc_27 = memref.alloc() : memref<32xf32>
        %subview_28 = memref.subview %reinterpret_cast_26[%0] [32] [1] : memref<64xf32, strided<[8], offset: ?>> to memref<32xf32, strided<[8], offset: ?>>
        %subview_29 = memref.subview %subview_28[0] [%77] [1] : memref<32xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[8], offset: ?>>
        %subview_30 = memref.subview %alloc_27[%73] [%77] [1] : memref<32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %115 = arith.remui %73, %c8 : index
        hivm.hir.load ins(%subview_29 : memref<?xf32, strided<[8], offset: ?>>) outs(%subview_30 : memref<?xf32, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %115 : index init_out_buffer = true init_condition = %78 : i1 eviction_policy = <EvictFirst>
        %116 = bufferization.to_tensor %alloc_27 restrict writable : memref<32xf32>
        %117 = hivm.hir.vmul ins(%116, %cst_1 : tensor<32xf32>, f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
        %118 = hivm.hir.vadd ins(%117, %114 : tensor<32xf32>, f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
        %119 = hivm.hir.vexp ins(%118 : tensor<32xf32>) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
        %120 = arith.index_cast %60 : i32 to index
        %121 = affine.apply #map11()[%120]
        %122 = arith.maxsi %120, %65 : index
        %123 = arith.minsi %121, %122 : index
        %124 = affine.apply #map1()[%123, %120]
        %125 = arith.minsi %0, %124 : index
        %126 = arith.subi %124, %125 : index
        %127 = arith.minsi %126, %c32 : index
        %extracted_slice_31 = tensor.extract_slice %119[0] [%127] [1] : tensor<32xf32> to tensor<?xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice_31 into %14[0] [%127] [1] : tensor<?xf32> into tensor<32xf32>
        %expanded_32 = tensor.expand_shape %inserted_slice [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %128 = hivm.hir.vmul ins(%104, %expanded_32 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%12 : tensor<32x32xf32>) broadcast = [1] -> tensor<32x32xf32>
        %129 = tensor.empty() : tensor<1xf32>
        %inserted = tensor.insert %114 into %129[%c0] : tensor<1xf32>
        %130 = hivm.hir.vexp ins(%inserted : tensor<1xf32>) outs(%129 : tensor<1xf32>) -> tensor<1xf32>
        %extracted = tensor.extract %130[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
        %131 = hivm.hir.vmul ins(%arg16, %extracted : tensor<16x64xf32>, f32) outs(%15 : tensor<16x64xf32>) -> tensor<16x64xf32>
        %132 = hivm.hir.vcast ins(%128 : tensor<32x32xf32>) outs(%81 : tensor<32x32xf16>) -> tensor<32x32xf16>
        %133 = tensor.empty() : tensor<2x32x16xf16>
        %expanded_33 = tensor.expand_shape %132 [[0], [1, 2]] output_shape [32, 2, 16] : tensor<32x32xf16> into tensor<32x2x16xf16>
        %134 = hivm.hir.vtranspose ins(%expanded_33 : tensor<32x2x16xf16>) outs(%133 : tensor<2x32x16xf16>) permutation = [1, 0, 2] -> tensor<2x32x16xf16>
        %alloc_34 = memref.alloc() : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_34 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
        %memspacecast_35 = memref.memory_space_cast %alloc_34 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x4x16x16xf16>
        %expanded_36 = tensor.expand_shape %134 [[0], [1, 2], [3]] output_shape [2, 2, 16, 16] : tensor<2x32x16xf16> into tensor<2x2x16x16xf16>
        %subview_37 = memref.subview %memspacecast_35[0, %2, 0, 0] [2, 2, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<2x4x16x16xf16> to memref<2x2x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>>
        hivm.hir.copy ins(%expanded_36 : tensor<2x2x16x16xf16>) outs(%subview_37 : memref<2x2x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>>) {tiled_op}
        %alloc_38 = memref.alloc() : memref<64x16xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_38 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 1 : index} : memref<64x16xf32, #hivm.address_space<ub>>
        %memspacecast_39 = memref.memory_space_cast %alloc_38 : memref<64x16xf32, #hivm.address_space<ub>> to memref<64x16xf32>
        %135 = bufferization.to_tensor %memspacecast_39 restrict writable : memref<64x16xf32>
        %136 = hivm.hir.vtranspose ins(%135 : tensor<64x16xf32>) outs(%15 : tensor<16x64xf32>) permutation = [1, 0] -> tensor<16x64xf32>
        %137 = hivm.hir.vadd ins(%131, %136 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%15 : tensor<16x64xf32>) -> tensor<16x64xf32>
        scf.yield %137 : tensor<16x64xf32>
      }
      %reinterpret_cast_4 = memref.reinterpret_cast %arg9 to offset: [%37], sizes: [32, 64], strides: [64, 1] : memref<?xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_5 = memref.subview %reinterpret_cast_4[%1, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %52[%44, 0] [%48, 64] [1, 1] : tensor<16x64xf32> to tensor<?x64xf32>
      %subview_6 = memref.subview %subview_5[0, 0] [%48, 64] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?x64xf32>) outs(%subview_6 : memref<?x64xf32, strided<[64, 1], offset: ?>>) {tiled_op}
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}
