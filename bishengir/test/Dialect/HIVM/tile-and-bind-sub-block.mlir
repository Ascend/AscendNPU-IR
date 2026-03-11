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

// CHECK-LABEL:   func.func @_attn_fwd_mix_aiv(
// CHECK:                   %{{.*}} = hivm.hir.vtranspose ins(%{{.*}} : tensor<4x16x8x16xf16>) outs(%{{.*}} : tensor<8x4x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<8x4x16x16xf16>
// CHECK:                   hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
// CHECK:                   hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 12
// CHECK:                   %{{.*}} = memref.memory_space_cast %{{.*}} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<8x8x16x16xf16>
// CHECK:                   %{{.*}} = memref.subview %{{.*}}[0, %{{.*}}, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xf16> to memref<8x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>>
// CHECK:                   %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref<8x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>>
// CHECK:                   %{{.*}} = hivm.hir.copy ins(%{{.*}} : tensor<8x4x16x16xf16>) outs(%{{.*}} : tensor<8x4x16x16xf16>) {tiled_op} -> tensor<8x4x16x16xf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %c8192 = arith.constant 8192 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %c2_i32 = arith.constant 2 : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(96 : i64) : i64
    %7 = llvm.mlir.constant(64 : i64) : i64
    %8 = llvm.mlir.constant(32 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8388608_i64 = arith.constant 8388608 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -1.000000e+04 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0.000000e+00 : f32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %10 = arith.muli %arg9, %arg10 : i32
    %11 = arith.muli %10, %arg11 : i32
    annotation.mark %11 {logical_block_num} : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg11, %arg10 : i32
    %15 = arith.divsi %13, %14 : i32
    %16 = arith.remsi %15, %arg9 : i32
    %17 = tensor.empty() : tensor<128xf32>
    %18 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
    %19 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
    %20 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
    %21 = tensor.empty() : tensor<128x128xf32>
    %22 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_4 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<128x128xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<128x128xf32, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_7 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<128x128xf32, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<128x128xf32, #hivm.address_space<ub>>
    %23 = llvm.inttoptr %9 : i64 to !llvm.ptr<11>
    %24 = llvm.inttoptr %8 : i64 to !llvm.ptr<11>
    %25 = llvm.inttoptr %7 : i64 to !llvm.ptr<11>
    %26 = llvm.inttoptr %6 : i64 to !llvm.ptr<11>
    llvm.store %5, %23 : i32, !llvm.ptr<11>
    llvm.store %5, %24 : i32, !llvm.ptr<11>
    llvm.store %5, %25 : i32, !llvm.ptr<11>
    llvm.store %5, %26 : i32, !llvm.ptr<11>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 12
    %27 = hivm.hir.get_sub_block_idx -> i64
    %28 = arith.muli %27, %8 : i64
    %29 = arith.addi %28, %8 : i64
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 10
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
    %30 = arith.index_cast %16 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%30], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %31 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %32 = arith.addi %30, %c1 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%32], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %33 = memref.load %reinterpret_cast_9[%c0] : memref<1xi32, strided<[1], offset: ?>>
    scf.for %arg12 = %31 to %33 step %c1_i32  : i32 {
      %34 = arith.divsi %arg12, %c64_i32 : i32
      %35 = arith.remsi %arg12, %c64_i32 : i32
      %36 = arith.divsi %34, %c8_i32 : i32
      %37 = arith.remsi %34, %c8_i32 : i32
      %38 = arith.extsi %36 : i32 to i64
      %39 = arith.muli %38, %c8388608_i64 : i64
      %40 = arith.extsi %37 : i32 to i64
      %41 = arith.muli %40, %c1048576_i64 : i64
      %42 = arith.addi %39, %41 : i64
      %43 = arith.muli %35, %c128_i32 : i32
      %44 = arith.index_cast %42 : i64 to index
      %45 = arith.maxsi %43, %c0_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = arith.muli %46, %c128 : index
      %48 = arith.addi %47, %44 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%48], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %49 = arith.muli %35, %c256_i32 : i32
      %50 = arith.divsi %49, %c128_i32 : i32
      %51 = arith.divsi %50, %3 : i32
      %52:11 = scf.for %arg13 = %c0_i32 to %49 step %c128_i32 iter_args(%arg14 = %19, %arg15 = %22, %arg16 = %20, %arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %18, %arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %18, %arg23 = %c0_i32, %arg24 = %c0_i32) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32)  : i32 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 13
        %64 = llvm.inttoptr %29 : i64 to !llvm.ptr<11>
        %65 = llvm.load %64 : !llvm.ptr<11> -> i32
        %66 = arith.andi %65, %4 : i32
        %67 = arith.cmpi eq, %66, %4 : i32
        %68 = arith.andi %65, %3 : i32
        %69 = arith.cmpi eq, %68, %c0_i32 : i32
        %70 = arith.andi %65, %2 : i32
        %71 = arith.cmpi eq, %70, %2 : i32
        %72 = arith.cmpi slt, %arg20, %51 : i32
        %73 = arith.andi %67, %69 : i1
        %74 = arith.andi %73, %72 : i1
        %75 = arith.cmpi slt, %arg21, %51 : i32
        %76 = arith.andi %71, %75 : i1
        %77:6 = scf.if %74 -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, i32, tensor<128xf32>, i32) {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %memspacecast = memref.memory_space_cast %alloc_7 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %81 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
          %82 = hivm.hir.vmul ins(%81, %cst : tensor<128x128xf32>, f32) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %expanded_11 = tensor.expand_shape %20 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %83 = hivm.hir.vreduce <max> ins(%82 : tensor<128x128xf32>) outs(%expanded_11 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
          %collapsed = tensor.collapse_shape %83 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
          %84 = hivm.hir.vmax ins(%arg16, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %expanded_12 = tensor.expand_shape %84 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %85 = hivm.hir.vsub ins(%82, %expanded_12 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%21 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
          %86 = hivm.hir.vexp ins(%85 : tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %87 = tensor.empty() : tensor<128x128xf16>
          %88 = hivm.hir.vcast ins(%86 : tensor<128x128xf32>) outs(%87 : tensor<128x128xf16>) -> tensor<128x128xf16>
          %expanded_13 = tensor.expand_shape %88 [[0, 1], [2, 3]] output_shape [8, 16, 8, 16] : tensor<128x128xf16> into tensor<8x16x8x16xf16>
          %89 = tensor.empty() : tensor<8x8x16x16xf16>
          %90 = hivm.hir.vtranspose ins(%expanded_13 : tensor<8x16x8x16xf16>) outs(%89 : tensor<8x8x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<8x8x16x16xf16>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 11
          %91 = bufferization.to_memref %90 : memref<8x8x16x16xf16, #hivm.address_space<ub>>
          hivm.hir.copy ins(%91 : memref<8x8x16x16xf16, #hivm.address_space<ub>>) outs(%alloc_6 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %expanded_14 = tensor.expand_shape %18 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %92 = hivm.hir.vreduce <sum> ins(%86 : tensor<128x128xf32>) outs(%expanded_14 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
          %collapsed_15 = tensor.collapse_shape %92 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
          %93 = hivm.hir.vsub ins(%arg16, %84 : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %94 = hivm.hir.vexp ins(%93 : tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %95 = arith.remsi %arg23, %c2_i32 : i32
          %96 = arith.cmpi eq, %95, %c0_i32 : i32
          %97 = arith.select %96, %94, %arg19 : tensor<128xf32>
          %98 = arith.select %96, %arg22, %94 : tensor<128xf32>
          %99 = arith.addi %arg23, %c1_i32 : i32
          %100 = hivm.hir.vmul ins(%arg14, %94 : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %101 = hivm.hir.vadd ins(%100, %collapsed_15 : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %102 = llvm.load %64 : !llvm.ptr<11> -> i32
          %103 = arith.andi %102, %1 : i32
          %104 = arith.ori %103, %3 : i32
          llvm.store %104, %64 : i32, !llvm.ptr<11>
          %105 = arith.addi %arg20, %4 : i32
          scf.yield %84, %97, %101, %105, %98, %99 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, i32, tensor<128xf32>, i32
        } else {
          scf.yield %arg16, %arg19, %arg14, %arg20, %arg22, %arg23 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, i32, tensor<128xf32>, i32
        }
        %78:4 = scf.if %76 -> (tensor<128x128xf32>, i32, i32, i32) {
          %81 = arith.remsi %arg24, %c2_i32 : i32
          %82 = arith.cmpi eq, %81, %c0_i32 : i32
          %83 = arith.select %82, %77#1, %77#4 : tensor<128xf32>
          %84 = arith.addi %arg24, %c1_i32 : i32
          %expanded_11 = tensor.expand_shape %83 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %85 = hivm.hir.vmul ins(%arg15, %expanded_11 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%21 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast = memref.memory_space_cast %alloc_5 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %86 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
          %87 = hivm.hir.vadd ins(%86, %85 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %88 = arith.addi %arg18, %c128_i32 : i32
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
          %89 = llvm.load %64 : !llvm.ptr<11> -> i32
          %90 = arith.andi %89, %0 : i32
          llvm.store %90, %64 : i32, !llvm.ptr<11>
          %91 = arith.addi %arg21, %4 : i32
          scf.yield %87, %88, %91, %84 : tensor<128x128xf32>, i32, i32, i32
        } else {
          scf.yield %arg15, %arg18, %arg21, %arg24 : tensor<128x128xf32>, i32, i32, i32
        }
        %79 = arith.select %76, %c128_i32, %c0_i32 : i32
        %80 = arith.addi %arg17, %79 : i32
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 12
        scf.yield %77#2, %78#0, %77#0, %80, %78#1, %77#1, %77#3, %78#2, %77#4, %77#5, %78#3 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32
      }
      %53 = arith.muli %35, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %54 = arith.addi %35, %c1_i32 : i32
      %55 = arith.muli %54, %c256_i32 : i32
      %56 = arith.subi %55, %53 : i32
      %57 = arith.subi %56, %53 : i32
      %58 = arith.divsi %57, %c128_i32 : i32
      %59 = arith.divsi %58, %3 : i32
      %60:12 = scf.for %arg13 = %53 to %56 step %c128_i32 iter_args(%arg14 = %53, %arg15 = %52#0, %arg16 = %52#1, %arg17 = %52#2, %arg18 = %53, %arg19 = %53, %arg20 = %18, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %18, %arg24 = %c0_i32, %arg25 = %c0_i32) -> (i32, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32)  : i32 {
        %64 = arith.maxsi %arg14, %c0_i32 : i32
        %65 = arith.index_cast %64 : i32 to index
        %66 = arith.muli %46, %c8192 : index
        %67 = arith.addi %66, %65 : index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg5 to offset: [%67], sizes: [128, 128], strides: [8192, 1] : memref<?xf32> to memref<128x128xf32, strided<[8192, 1], offset: ?>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 13
        %68 = llvm.inttoptr %29 : i64 to !llvm.ptr<11>
        %69 = llvm.load %68 : !llvm.ptr<11> -> i32
        %70 = arith.andi %69, %4 : i32
        %71 = arith.cmpi eq, %70, %4 : i32
        %72 = arith.andi %69, %3 : i32
        %73 = arith.cmpi eq, %72, %c0_i32 : i32
        %74 = arith.andi %69, %2 : i32
        %75 = arith.cmpi eq, %74, %2 : i32
        %76 = arith.cmpi slt, %arg21, %59 : i32
        %77 = arith.andi %71, %73 : i1
        %78 = arith.andi %77, %76 : i1
        %79 = arith.cmpi slt, %arg22, %59 : i32
        %80 = arith.andi %75, %79 : i1
        %alloc_12 = memref.alloc() : memref<128x128xf32>
        hivm.hir.load ins(%reinterpret_cast_11 : memref<128x128xf32, strided<[8192, 1], offset: ?>>) outs(%alloc_12 : memref<128x128xf32>) eviction_policy = <EvictFirst>
        %81 = bufferization.to_tensor %alloc_12 restrict writable : memref<128x128xf32>
        %82:6 = scf.if %78 -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, i32, tensor<128xf32>, i32) {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast = memref.memory_space_cast %alloc_4 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %88 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
          %89 = hivm.hir.vmul ins(%88, %cst : tensor<128x128xf32>, f32) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %90 = tensor.empty() : tensor<128x128xi1>
          %91 = hivm.hir.vcmp ins(%81, %cst_3 : tensor<128x128xf32>, f32) outs(%90 : tensor<128x128xi1>) compare_mode = <ne> -> tensor<128x128xi1>
          %92 = hivm.hir.vsel ins(%91, %cst_0, %cst_3 : tensor<128x128xi1>, f32, f32) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %93 = hivm.hir.vadd ins(%89, %92 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %expanded_13 = tensor.expand_shape %20 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %94 = hivm.hir.vreduce <max> ins(%93 : tensor<128x128xf32>) outs(%expanded_13 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
          %collapsed = tensor.collapse_shape %94 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
          %95 = hivm.hir.vmax ins(%arg17, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %expanded_14 = tensor.expand_shape %95 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %96 = hivm.hir.vsub ins(%93, %expanded_14 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%21 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
          %97 = hivm.hir.vexp ins(%96 : tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %98 = tensor.empty() : tensor<128x128xf16>
          %99 = hivm.hir.vcast ins(%97 : tensor<128x128xf32>) outs(%98 : tensor<128x128xf16>) -> tensor<128x128xf16>
          %expanded_15 = tensor.expand_shape %99 [[0, 1], [2, 3]] output_shape [8, 16, 8, 16] : tensor<128x128xf16> into tensor<8x16x8x16xf16>
          %100 = tensor.empty() : tensor<8x8x16x16xf16>
          %101 = hivm.hir.vtranspose ins(%expanded_15 : tensor<8x16x8x16xf16>) outs(%100 : tensor<8x8x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<8x8x16x16xf16>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 12
          %102 = bufferization.to_memref %101 : memref<8x8x16x16xf16, #hivm.address_space<ub>>
          hivm.hir.copy ins(%102 : memref<8x8x16x16xf16, #hivm.address_space<ub>>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
          %expanded_16 = tensor.expand_shape %18 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %103 = hivm.hir.vreduce <sum> ins(%97 : tensor<128x128xf32>) outs(%expanded_16 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
          %collapsed_17 = tensor.collapse_shape %103 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
          %104 = hivm.hir.vsub ins(%arg17, %95 : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %105 = hivm.hir.vexp ins(%104 : tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %106 = arith.remsi %arg24, %c2_i32 : i32
          %107 = arith.cmpi eq, %106, %c0_i32 : i32
          %108 = arith.select %107, %105, %arg20 : tensor<128xf32>
          %109 = arith.select %107, %arg23, %105 : tensor<128xf32>
          %110 = arith.addi %arg24, %c1_i32 : i32
          %111 = hivm.hir.vmul ins(%arg15, %105 : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %112 = hivm.hir.vadd ins(%111, %collapsed_17 : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
          %113 = llvm.load %68 : !llvm.ptr<11> -> i32
          %114 = arith.andi %113, %1 : i32
          %115 = arith.ori %114, %3 : i32
          llvm.store %115, %68 : i32, !llvm.ptr<11>
          %116 = arith.addi %arg21, %4 : i32
          scf.yield %95, %108, %112, %116, %109, %110 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, i32, tensor<128xf32>, i32
        } else {
          scf.yield %arg17, %arg20, %arg15, %arg21, %arg23, %arg24 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, i32, tensor<128xf32>, i32
        }
        %83 = arith.select %78, %c128_i32, %c0_i32 : i32
        %84 = arith.addi %arg14, %83 : i32
        %85:4 = scf.if %80 -> (tensor<128x128xf32>, i32, i32, i32) {
          %88 = arith.remsi %arg25, %c2_i32 : i32
          %89 = arith.cmpi eq, %88, %c0_i32 : i32
          %90 = arith.select %89, %82#1, %82#4 : tensor<128xf32>
          %91 = arith.addi %arg25, %c1_i32 : i32
          %expanded_13 = tensor.expand_shape %90 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %92 = hivm.hir.vmul ins(%arg16, %expanded_13 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%21 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %memspacecast = memref.memory_space_cast %alloc_8 : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %93 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
          %94 = hivm.hir.vadd ins(%93, %92 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %95 = arith.addi %arg19, %c128_i32 : i32
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 10
          %96 = llvm.load %68 : !llvm.ptr<11> -> i32
          %97 = arith.andi %96, %0 : i32
          llvm.store %97, %68 : i32, !llvm.ptr<11>
          %98 = arith.addi %arg22, %4 : i32
          scf.yield %94, %95, %98, %91 : tensor<128x128xf32>, i32, i32, i32
        } else {
          scf.yield %arg16, %arg19, %arg22, %arg25 : tensor<128x128xf32>, i32, i32, i32
        }
        %86 = arith.select %80, %c128_i32, %c0_i32 : i32
        %87 = arith.addi %arg18, %86 : i32
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 12
        scf.yield %84, %82#2, %85#0, %82#0, %87, %85#1, %82#1, %82#3, %85#2, %82#4, %82#5, %85#3 : i32, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32, tensor<128xf32>, i32, i32
      }
      %expanded = tensor.expand_shape %60#1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %61 = hivm.hir.vdiv ins(%60#2, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%21 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
      %62 = tensor.empty() : tensor<128x128xf16>
      %63 = hivm.hir.vcast ins(%61 : tensor<128x128xf32>) outs(%62 : tensor<128x128xf16>) -> tensor<128x128xf16>
      hivm.hir.store ins(%63 : tensor<128x128xf16>) outs(%reinterpret_cast_10 : memref<128x128xf16, strided<[128, 1], offset: ?>>)
    }
    return
  }
}