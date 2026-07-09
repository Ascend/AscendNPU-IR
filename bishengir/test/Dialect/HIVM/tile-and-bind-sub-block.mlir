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

#off_scalar_ub = affine_map<()[s0] -> (s0 * 128)>
module {
  // The scalar UB alloc mirrors input_fa_bwd's memref<f32, ub> tightly-coupled
  // buffers: it is allowed to stay untiled while other ops still bind sub-blocks.
  // CHECK-LABEL:   func.func @scalar_ub_tightly_coupled_buffer_allows_bind(
  // CHECK:           annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = -1 : index} : memref<f32, #hivm.address_space<ub>>
  // CHECK:           hivm.hir.store{{.*}} {tiled_op}
  // CHECK:         } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  func.func @scalar_ub_tightly_coupled_buffer_allows_bind(%arg0: tensor<256xf32>, %arg1: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<f32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<f32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c2 step %c1 {
      %offset = affine.apply #off_scalar_ub()[%i]
      %slice = tensor.extract_slice %arg0[%offset] [128] [1] {to_be_bubbled_slice} : tensor<256xf32> to tensor<128xf32>
      hivm.hir.store ins(%slice : tensor<128xf32>) outs(%arg1 : memref<128xf32>)
    }
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
// CHECK:           %[[subview:.*]] = memref.subview %[[memspacecast_5:.*]][0, 0, %0, 0] [1, 1, 8, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<1x1x16x16xf16> to memref<1x1x8x16xf16, strided<[256, 256, 16, 1], offset: ?>>
// CHECK:           %[[VAL_40:.*]] = bufferization.to_tensor %[[memspacecast_5:.*]] restrict writable : memref<1x1x16x16xf16>
// CHECK:           hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
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
// CHECK:      hivm.hir.store ins(%[[VAL_46:.*]] : tensor<64x128xbf16>) outs(%{{.*}} : memref<64x128xbf16, strided<[128, 1], offset: ?>>) {tiled_op}
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
// CHECK:           annotation.mark %[[VAL_alloc_4:.*]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 2 : index, tiledAlloc} : memref<1x1x8x16xf16, #hivm.address_space<ub>>
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
// CHECK: scf.for
// CHECK: scf.while
// CHECK: hivm.hir.store{{.*}} {tiled_op}
// CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
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

// CHECK-LABEL: func.func @check_column_split_aic(
// CHECK: tensor.empty() : tensor<16x16xi32>
// CHECK: hivm.hir.fixpipe
// CHECK-LABEL: func.func @check_column_split_aiv(
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
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%{{.*}} : tensor<32x128xf32>) outs(%{{.*}} : memref<64x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
// CHECK: hivm.hir.store
// CHECK: map_for_to_forall
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

// CHECK-LABEL:  func.func @_attn_bwd_mix_aiv(
// CHECK: %[[VAL_137:.*]] = hivm.hir.vreduce {tiled_op} <max> ins(%[[VAL_135:.*]] : tensor<2048xf32>) outs(%[[VAL_136:.*]] : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
// CHECK: %[[VAL_138:.*]] = memref_ext.alloc_workspace()
// CHECK: %[[VAL_139:.*]] = memref.subview %[[VAL_138]]{{\[}}%[[VAL_38:.*]]] [1] [1] : memref<2xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK: hivm.hir.store ins(%[[VAL_137]] : tensor<1xf32>) outs(%[[VAL_139]] : memref<1xf32, strided<[1], offset: ?>>) {tiled_op}
// CHECK: hivm.hir.sync_block[<ALL_SUB_VECTOR>] tvector_pipe = <PIPE_ALL>
// CHECK: %[[VAL_142:.*]] = hivm.hir.vreduce <max> ins(%[[VAL_141:.*]] : tensor<2xf32>) outs(%[[VAL_136]] : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
// CHECK: } {limit_sub_block_id0}
#map = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<(d0) -> (d0 * 4)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_bwd_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c4 = arith.constant 4 : index
    return %c4 : index
  }
  func.func @_attn_bwd_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf8E5M2> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf8E5M2> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: memref<?xf32> {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 5.734400e+04 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.99999997E-7 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c28_i32 = arith.constant 28 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg17, %arg16 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = tensor.empty() : tensor<128x64xf32>
    %8 = tensor.empty() : tensor<32x128xf32>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 7
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 8
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_MTE3>] flag = 6
    scf.for %arg18 = %6 to %c8192_i32 step %c28_i32  : i32 {
      %9 = arith.divsi %arg18, %c8_i32 : i32
      %10 = arith.muli %9, %c8_i32 : i32
      %11 = arith.subi %arg18, %10 : i32
      %12 = arith.muli %9, %c1024_i32 : i32
      %13 = arith.remsi %9, %c8_i32 : i32
      %14 = arith.muli %13, %c65536_i32 : i32
      %15 = arith.divsi %9, %c8_i32 : i32
      %16 = arith.muli %15, %c524288_i32 : i32
      %17 = arith.addi %14, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.divsi %12, %c128_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.muli %11, %c128_i32 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = affine.apply #map()[%18, %22]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf8E5M2> to memref<128x64xf8E5M2, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x64xf8E5M2>
      hivm.hir.load ins(%reinterpret_cast : memref<128x64xf8E5M2, strided<[64, 1], offset: ?>>) outs(%alloc : memref<128x64xf8E5M2>) eviction_policy = <EvictFirst>
      %24 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf8E5M2>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %alloc_3 = memref.alloc() : memref<128x64xf16>
      hivm.hir.load ins(%reinterpret_cast_2 : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_3 : memref<128x64xf16>) eviction_policy = <EvictFirst>
      %25 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x64xf16>
      %26 = arith.divsi %21, %c128_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = affine.apply #map1()[%20, %27]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg13 to offset: [%28], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %29 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %30 = scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg20 = %7) -> (tensor<128x64xf32>)  : i32 {
        %31 = arith.muli %arg19, %c32_i32 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = affine.apply #map()[%18, %32]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 64], strides: [64, 1] : memref<?xf8E5M2> to memref<32x64xf8E5M2, strided<[64, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<32x64xf8E5M2>
        hivm.hir.load ins(%reinterpret_cast_6 : memref<32x64xf8E5M2, strided<[64, 1], offset: ?>>) outs(%alloc_7 : memref<32x64xf8E5M2>) eviction_policy = <EvictFirst>
        %34 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x64xf8E5M2>
        %35 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%34, %24, %true, %c32, %c64, %c128 : tensor<32x64xf8E5M2>, tensor<128x64xf8E5M2>, i1, index, index, index) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %alloc_8 = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<32x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%35 : tensor<32x128xf32>) outs(%alloc_8 : memref<32x128xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_10 = memref.alloc() : memref<32x64xf16>
        hivm.hir.load ins(%reinterpret_cast_9 : memref<32x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_10 : memref<32x64xf16>) eviction_policy = <EvictFirst>
        %36 = bufferization.to_tensor %alloc_10 restrict writable : memref<32x64xf16>
        %37 = tensor.empty() : tensor<32x128xf16>
        %38 = arith.cmpi eq, %arg19, %c0_i32 : i32
        %expanded = tensor.expand_shape %37 [[0], [1, 2]] output_shape [32, 8, 16] : tensor<32x128xf16> into tensor<32x8x16xf16>
        annotation.mark %expanded {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x8x16xf16>
        %39 = tensor.empty() : tensor<8x32x16xf16>
        %expanded_11 = tensor.expand_shape %39 [[0], [1, 2], [3]] output_shape [8, 2, 16, 16] : tensor<8x32x16xf16> into tensor<8x2x16x16xf16>
        annotation.mark %expanded_11 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x2x16x16xf16>
        %alloc_12 = memref.alloc() : memref<8x2x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8x2x16x16xf16, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %alloc_12 : memref<8x2x16x16xf16, #hivm.address_space<cbuf>> to memref<8x2x16x16xf16>
        %40 = bufferization.to_tensor %memspacecast restrict writable : memref<8x2x16x16xf16>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %41 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_already_inserted = true} ins(%40, %36, %38, %c128, %c32, %c64 : tensor<8x2x16x16xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%arg20 : tensor<128x64xf32>) -> tensor<128x64xf32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
        %42 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%36, %25, %true, %c32, %c64, %c128 : tensor<32x64xf16>, tensor<128x64xf16>, i1, index, index, index) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %alloc_13 = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<32x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%42 : tensor<32x128xf32>) outs(%alloc_13 : memref<32x128xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %43 = bufferization.alloc_tensor() : tensor<f32>
        %44 = hivm.hir.get_block_idx -> i64
        %45 = arith.index_cast %44 : i64 to index
        %46 = affine.apply #map2(%45)
        %view = memref.view %arg1[%46][] : memref<?xi8> to memref<f32>
        %47 = bufferization.to_tensor %view restrict writable : memref<f32>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 2
        %extracted = tensor.extract %47[] {"DuplicateTensorExtractForCube::newExtractLabel" = 1 : i32, "DuplicateTensorExtractForCube::visitedLabel" = 1 : i32, "PostCubeReplacement::visitedLabel" = 1 : i32} : tensor<f32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_MTE3>] flag = 6
        annotation.mark %extracted {"DuplicateTensorExtractForCube::replacementLabel" = 1 : i32} keys = [] values = [%extracted : f32] : f32
        %48 = arith.cmpf ogt, %extracted, %cst_1 : f32
        %49 = scf.if %48 -> (f32) {
          %58 = arith.divf %cst, %extracted : f32
          scf.yield %58 : f32
        } else {
          scf.yield %cst_0 : f32
        }
        %50 = tensor.empty() : tensor<32x128xf8E5M2>
        %51 = tensor.empty() : tensor<32x64xf32>
        %expanded_14 = tensor.expand_shape %50 [[0], [1, 2]] output_shape [32, 4, 32] : tensor<32x128xf8E5M2> into tensor<32x4x32xf8E5M2>
        annotation.mark %expanded_14 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x4x32xf8E5M2>
        %52 = tensor.empty() : tensor<4x32x32xf8E5M2>
        %expanded_15 = tensor.expand_shape %52 [[0], [1, 2], [3]] output_shape [4, 2, 16, 32] : tensor<4x32x32xf8E5M2> into tensor<4x2x16x32xf8E5M2>
        annotation.mark %expanded_15 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<4x2x16x32xf8E5M2>
        %alloc_16 = memref.alloc() : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        %memspacecast_17 = memref.memory_space_cast %alloc_16 : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>> to memref<4x2x16x32xf8E5M2>
        %53 = bufferization.to_tensor %memspacecast_17 restrict writable : memref<4x2x16x32xf8E5M2>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %54 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%53, %24, %true, %c32, %c128, %c64 : tensor<4x2x16x32xf8E5M2>, tensor<128x64xf8E5M2>, i1, index, index, index) outs(%51 : tensor<32x64xf32>) -> tensor<32x64xf32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 7
        %55 = arith.divf %29, %49 : f32
        annotation.mark %expanded_14 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x4x32xf8E5M2>
        annotation.mark %expanded_15 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<4x2x16x32xf8E5M2>
        %alloc_18 = memref.alloc() : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        %memspacecast_19 = memref.memory_space_cast %alloc_18 : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>> to memref<4x2x16x32xf8E5M2>
        %56 = bufferization.to_tensor %memspacecast_19 restrict writable : memref<4x2x16x32xf8E5M2>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %57 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_already_inserted = true} ins(%56, %34, %true, %c128, %c32, %c64 : tensor<4x2x16x32xf8E5M2>, tensor<32x64xf8E5M2>, i1, index, index, index) outs(%7 : tensor<128x64xf32>) -> tensor<128x64xf32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 8
        %alloc_20 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_20 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<128x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%57 : tensor<128x64xf32>) outs(%alloc_20 : memref<128x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %reinterpret_cast_21 = memref.reinterpret_cast %arg6 to offset: [%33], sizes: [32, 64], strides: [64, 1] : memref<?xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
        hivm.hir.set_atomic kind = <add>[type = f32]
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>} ins(%54 : tensor<32x64xf32>) outs(%reinterpret_cast_21 : memref<32x64xf32, strided<[64, 1], offset: ?>>) quant_scale = %55 : f32
        hivm.hir.set_atomic kind = <none>[type = f32]
        scf.yield %41 : tensor<128x64xf32>
      }
      %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%30 : tensor<128x64xf32>) outs(%reinterpret_cast_5 : memref<128x64xf32, strided<[64, 1], offset: ?>>)
    }
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
    return
  }
  func.func @_attn_bwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf8E5M2> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf8E5M2> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: memref<?xf32> {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %c-2139095040_i32 = arith.constant -2139095040 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 5.734400e+04 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.99999997E-7 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c28_i32 = arith.constant 28 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 0xFF800000 : f32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg17, %arg16 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = tensor.empty() : tensor<128x64xf32>
    %8 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %9 = tensor.empty() : tensor<32x128xf32>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
    scf.for %arg18 = %6 to %c8192_i32 step %c28_i32  : i32 {
      %10 = arith.divsi %arg18, %c8_i32 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %arg18, %11 : i32
      %13 = arith.muli %10, %c1024_i32 : i32
      %14 = arith.remsi %10, %c8_i32 : i32
      %15 = arith.muli %14, %c65536_i32 : i32
      %16 = arith.divsi %10, %c8_i32 : i32
      %17 = arith.muli %16, %c524288_i32 : i32
      %18 = arith.addi %15, %17 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.index_cast %13 : i32 to index
      %21 = arith.divsi %13, %c128_i32 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %12, %c128_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = affine.apply #map()[%19, %24]
      %26 = arith.divsi %23, %c128_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = affine.apply #map1()[%22, %27]
      %reinterpret_cast = memref.reinterpret_cast %arg13 to offset: [%28], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %29 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %30 = scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg20 = %8) -> (tensor<128x64xf32>)  : i32 {
        %31 = arith.muli %arg19, %c32_i32 : i32
        %32 = arith.divsi %31, %c128_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        %34 = affine.apply #map1()[%22, %33]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg12 to offset: [%34], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %35 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>>
        %36 = arith.index_cast %31 : i32 to index
        %alloc = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<32x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>
        %37 = bufferization.to_tensor %memspacecast restrict writable : memref<32x128xf32>
        %38 = tensor.empty() : tensor<32x128xi1>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %39 = hivm.hir.vcmp ins(%37, %37 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%38 : tensor<32x128xi1>) compare_mode = <ne> -> tensor<32x128xi1>
        %40 = hivm.hir.vsel ins(%39, %cst_2, %37 : tensor<32x128xi1>, f32, tensor<32x128xf32>) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %41 = hivm.hir.vmul ins(%40, %35 : tensor<32x128xf32>, f32) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %42 = hivm.hir.vmul ins(%41, %29 : tensor<32x128xf32>, f32) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %43 = affine.apply #map1()[%20, %36]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg9 to offset: [%43], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<32xf32>
        hivm.hir.load ins(%reinterpret_cast_6 : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc_7 : memref<32xf32>) eviction_policy = <EvictFirst>
        %44 = bufferization.to_tensor %alloc_7 restrict writable : memref<32xf32>
        %45 = hivm.hir.vmul ins(%42, %arg11 : tensor<32x128xf32>, f32) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %expanded = tensor.expand_shape %44 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %46 = hivm.hir.vsub ins(%45, %expanded : tensor<32x128xf32>, tensor<32x1xf32>) outs(%9 : tensor<32x128xf32>) broadcast = [1] -> tensor<32x128xf32>
        %47 = hivm.hir.vexp ins(%46 : tensor<32x128xf32>) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %48 = tensor.empty() : tensor<32x128xf16>
        %49 = hivm.hir.vcast ins(%47 : tensor<32x128xf32>) outs(%48 : tensor<32x128xf16>) -> tensor<32x128xf16>
        %expanded_8 = tensor.expand_shape %49 [[0], [1, 2]] output_shape [32, 8, 16] : tensor<32x128xf16> into tensor<32x8x16xf16>
        annotation.mark %expanded_8 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x8x16xf16>
        %50 = tensor.empty() : tensor<8x32x16xf16>
        %51 = hivm.hir.vtranspose ins(%expanded_8 : tensor<32x8x16xf16>) outs(%50 : tensor<8x32x16xf16>) permutation = [1, 0, 2] -> tensor<8x32x16xf16>
        %expanded_9 = tensor.expand_shape %51 [[0], [1, 2], [3]] output_shape [8, 2, 16, 16] : tensor<8x32x16xf16> into tensor<8x2x16x16xf16>
        annotation.mark %expanded_9 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x2x16x16xf16>
        %alloc_10 = memref.alloc() : memref<8x2x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8x2x16x16xf16, #hivm.address_space<cbuf>>
        %memspacecast_11 = memref.memory_space_cast %alloc_10 : memref<8x2x16x16xf16, #hivm.address_space<cbuf>> to memref<8x2x16x16xf16>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
        hivm.hir.copy ins(%expanded_9 : tensor<8x2x16x16xf16>) outs(%memspacecast_11 : memref<8x2x16x16xf16>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %alloc_12 = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<32x128xf32, #hivm.address_space<ub>>
        %memspacecast_13 = memref.memory_space_cast %alloc_12 : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>
        %52 = bufferization.to_tensor %memspacecast_13 restrict writable : memref<32x128xf32>
        %reinterpret_cast_14 = memref.reinterpret_cast %arg10 to offset: [%43], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %alloc_15 = memref.alloc() : memref<32xf32>
        hivm.hir.load ins(%reinterpret_cast_14 : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc_15 : memref<32xf32>) eviction_policy = <EvictFirst>
        %53 = bufferization.to_tensor %alloc_15 restrict writable : memref<32xf32>
        %expanded_16 = tensor.expand_shape %53 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %54 = hivm.hir.vsub ins(%52, %expanded_16 : tensor<32x128xf32>, tensor<32x1xf32>) outs(%9 : tensor<32x128xf32>) broadcast = [1] -> tensor<32x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        %55 = hivm.hir.vmul ins(%47, %54 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %56 = hivm.hir.vmul ins(%55, %arg11 : tensor<32x128xf32>, f32) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %57 = hivm.hir.vabs ins(%56 : tensor<32x128xf32>) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %collapsed = tensor.collapse_shape %57 [[0, 1]] : tensor<32x128xf32> into tensor<4096xf32>
        %58 = bufferization.alloc_tensor() : tensor<f32>
        %59 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%58 : tensor<f32>) -> tensor<f32>
        %60 = tensor.empty() : tensor<4096xi32>
        %61 = hivm.hir.bitcast %collapsed : tensor<4096xf32> -> tensor<4096xi32>
        %62 = hivm.hir.vbrc ins(%c2147483647_i32 : i32) outs(%60 : tensor<4096xi32>) -> tensor<4096xi32>
        %63 = hivm.hir.vand ins(%61, %62 : tensor<4096xi32>, tensor<4096xi32>) outs(%60 : tensor<4096xi32>) -> tensor<4096xi32>
        %64 = hivm.hir.vadd ins(%63, %c-2139095040_i32 : tensor<4096xi32>, i32) outs(%60 : tensor<4096xi32>) -> tensor<4096xi32>
        %65 = hivm.hir.vmin ins(%64, %c1_i32 : tensor<4096xi32>, i32) outs(%64 : tensor<4096xi32>) -> tensor<4096xi32>
        %66 = hivm.hir.vmax ins(%65, %c0_i32 : tensor<4096xi32>, i32) outs(%65 : tensor<4096xi32>) -> tensor<4096xi32>
        %67 = tensor.empty() : tensor<4096xf32>
        %68 = hivm.hir.vcast ins(%66 : tensor<4096xi32>) outs(%67 : tensor<4096xf32>) -> tensor<4096xf32>
        %69 = tensor.empty() : tensor<4096xi1>
        %70 = hivm.hir.vcmp ins(%68, %cst_2 : tensor<4096xf32>, f32) outs(%69 : tensor<4096xi1>) compare_mode = <ne> -> tensor<4096xi1>
        %71 = hivm.hir.vsel ins(%70, %cst_3, %collapsed : tensor<4096xi1>, f32, tensor<4096xf32>) outs(%67 : tensor<4096xf32>) -> tensor<4096xf32>
        %expanded_17 = tensor.expand_shape %59 [] output_shape [1] : tensor<f32> into tensor<1xf32>
        %72 = hivm.hir.vreduce <max> ins(%71 : tensor<4096xf32>) outs(%expanded_17 : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
        %collapsed_18 = tensor.collapse_shape %72 [] : tensor<1xf32> into tensor<f32>
        %extracted = tensor.extract %collapsed_18[] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<f32>
        %73 = hivm.hir.get_block_idx -> i64
        %74 = arith.index_cast %73 : i64 to index
        %75 = affine.apply #map2(%74)
        %view = memref.view %arg1[%75][] : memref<?xi8> to memref<f32>
        %76 = bufferization.to_tensor %view restrict writable : memref<f32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE3>] flag = 6
        %77 = hivm.hir.store ins(%collapsed_18 : tensor<f32>) outs(%76 : tensor<f32>) -> tensor<f32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 2
        annotation.mark %77 {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<f32>
        %78 = arith.cmpf ogt, %extracted, %cst_1 : f32
        %79 = scf.if %78 -> (f32) {
          %89 = arith.divf %cst, %extracted : f32
          scf.yield %89 : f32
        } else {
          scf.yield %cst_0 : f32
        }
        %80 = hivm.hir.vmul ins(%56, %79 : tensor<32x128xf32>, f32) outs(%9 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %81 = tensor.empty() : tensor<32x128xf8E5M2>
        %82 = hivm.hir.vcast ins(%80 : tensor<32x128xf32>) outs(%81 : tensor<32x128xf8E5M2>) -> tensor<32x128xf8E5M2>
        %expanded_19 = tensor.expand_shape %82 [[0], [1, 2]] output_shape [32, 4, 32] : tensor<32x128xf8E5M2> into tensor<32x4x32xf8E5M2>
        annotation.mark %expanded_19 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x4x32xf8E5M2>
        %83 = tensor.empty() : tensor<4x32x32xf8E5M2>
        %84 = hivm.hir.vtranspose ins(%expanded_19 : tensor<32x4x32xf8E5M2>) outs(%83 : tensor<4x32x32xf8E5M2>) permutation = [1, 0, 2] -> tensor<4x32x32xf8E5M2>
        %expanded_20 = tensor.expand_shape %84 [[0], [1, 2], [3]] output_shape [4, 2, 16, 32] : tensor<4x32x32xf8E5M2> into tensor<4x2x16x32xf8E5M2>
        annotation.mark %expanded_20 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<4x2x16x32xf8E5M2>
        %alloc_21 = memref.alloc() : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        annotation.mark %alloc_21 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        %memspacecast_22 = memref.memory_space_cast %alloc_21 : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>> to memref<4x2x16x32xf8E5M2>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 7
        hivm.hir.copy ins(%expanded_20 : tensor<4x2x16x32xf8E5M2>) outs(%memspacecast_22 : memref<4x2x16x32xf8E5M2>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        annotation.mark %expanded_19 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x4x32xf8E5M2>
        annotation.mark %expanded_20 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<4x2x16x32xf8E5M2>
        %alloc_23 = memref.alloc() : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        annotation.mark %alloc_23 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>>
        %memspacecast_24 = memref.memory_space_cast %alloc_23 : memref<4x2x16x32xf8E5M2, #hivm.address_space<cbuf>> to memref<4x2x16x32xf8E5M2>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 8
        hivm.hir.copy ins(%expanded_20 : tensor<4x2x16x32xf8E5M2>) outs(%memspacecast_24 : memref<4x2x16x32xf8E5M2>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %alloc_25 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_25 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<128x64xf32, #hivm.address_space<ub>>
        %memspacecast_26 = memref.memory_space_cast %alloc_25 : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %85 = bufferization.to_tensor %memspacecast_26 restrict writable : memref<128x64xf32>
        %86 = arith.divf %35, %79 : f32
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %87 = hivm.hir.vmul ins(%85, %86 : tensor<128x64xf32>, f32) outs(%7 : tensor<128x64xf32>) -> tensor<128x64xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
        %88 = hivm.hir.vadd ins(%arg20, %87 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%7 : tensor<128x64xf32>) -> tensor<128x64xf32>
        hivm.hir.set_atomic kind = <add>[type = f32]
        hivm.hir.set_atomic kind = <none>[type = f32]
        scf.yield %88 : tensor<128x64xf32>
      }
      %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%25], sizes: [128, 64], strides: [64, 1] : memref<?xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.store ins(%30 : tensor<128x64xf32>) outs(%reinterpret_cast_4 : memref<128x64xf32, strided<[64, 1], offset: ?>>)
    }
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE3>] flag = 6
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 7
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 8
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

// CHECK-LABEL: func.func @check_split_stride_store
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : i64
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i64
// CHECK: scf.for
// CHECK: %[[TILE_OFFSET_IDX:.*]] = affine.apply
// CHECK: %[[SRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<8x64xf16>) -> tensor<8x64xf16>
// CHECK: %[[TILE_OFFSET:.*]] = arith.index_cast %[[TILE_OFFSET_IDX]] : index to i64
// CHECK: %[[NEW_OFFSET:.*]] = arith.muli %[[TILE_OFFSET]], %[[C64]] : i64
// CHECK: %[[REMAINING:.*]] = arith.subi %[[C1024]], %[[TILE_OFFSET]] : i64
// CHECK: %[[BOUNDED:.*]] = arith.minsi %[[REMAINING]], %[[C8]] : i64
// CHECK: %[[NEW_NUMEL:.*]] = arith.maxsi %[[BOUNDED]], %{{.*}} : i64
// CHECK: hivm.hir.stride_store ins(%[[SRC]] : tensor<8x64xf16>) outs(%arg0 : memref<?xf16>) offset(%[[NEW_OFFSET]] : i64) strides([%[[C64]], %[[C1_I64]] : i64, i64]) numels([%[[NEW_NUMEL]], %[[C64]] : i64, i64]) {tiled_op}
// CHECK-NOT: limit_sub_block_id0
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_split_stride_store(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<16x64xf16>
    %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<16x64xf16>) -> tensor<16x64xf16>
    hivm.hir.stride_store ins(%1 : tensor<16x64xf16>) outs(%arg0 : memref<?xf16>) offset(%c0_i64 : i64) strides([%c64_i64, %c1_i64 : i64, i64]) numels([%c1024_i64, %c64_i64 : i64, i64])
    return
  }
}

// -----

// CHECK-LABEL: func.func @_hstu_attn_fwd_mix_aiv
// CHECK-DAG: annotation.mark {{.*}} {buffer_size_in_byte = 18560 : i64} : memref<?x80xf32>
// CHECK-DAG: annotation.mark {{.*}} {buffer_size_in_byte = 9280 : i64} : tensor<?x80xf16>
// CHECK: hivm.hir.store
// CHECK-NOT: limit_sub_block_id0
// CHECK: map_for_to_forall
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_hstu_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %c115 = arith.constant 115 : index
    %c320 = arith.constant 320 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c115_i32 = arith.constant 115 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c4_i64 = arith.constant 4 : i64
    %c3_i32 = arith.constant 3 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c80_i64 = arith.constant 80 : i64
    %c320_i64 = arith.constant 320 : i64
    %c256_i64 = arith.constant 256 : i64
    %c255_i32 = arith.constant 255 : i32
    %c255_i64 = arith.constant 255 : i64
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg17, %arg16 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = tensor.empty() : tensor<128x256xf32>
    %8 = arith.cmpi sle, %arg12, %c128_i32 : i32
    %9 = scf.if %8 -> (i64) {
      scf.yield %c4_i64 : i64
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [4], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: 4>>
      %29 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: 4>>
      scf.yield %29 : i64
    }
    %10 = arith.muli %9, %c4_i64 : i64
    %11 = arith.extsi %arg15 : i32 to i64
    %12 = arith.minsi %11, %10 : i64
    %13 = arith.divsi %10, %12 : i64
    %14 = arith.addi %13, %c1_i64 : i64
    %15 = arith.remsi %10, %12 : i64
    %16 = arith.extsi %6 : i32 to i64
    %17 = arith.cmpi slt, %16, %12 : i64
    %18 = arith.cmpi slt, %16, %15 : i64
    %19 = arith.muli %16, %14 : i64
    %20 = arith.muli %15, %14 : i64
    %21 = arith.subi %16, %15 : i64
    %22 = arith.muli %21, %13 : i64
    %23 = arith.addi %20, %22 : i64
    %24 = arith.select %18, %19, %23 : i64
    %25 = arith.select %17, %24, %c0_i64 : i64
    %26 = arith.select %18, %14, %13 : i64
    %27 = arith.select %17, %26, %c0_i64 : i64
    %28 = arith.cmpi sge, %16, %12 : i64
    scf.if %28 {
    } else {
      %29 = arith.cmpi sle, %27, %c0_i64 : i64
      scf.if %29 {
      } else {
        %30 = arith.addi %arg13, %c255_i32 : i32
        %31 = arith.divsi %30, %c256_i32 : i32
        %32 = arith.extsi %31 : i32 to i64
        %33 = arith.muli %27, %32 : i64
        scf.for %arg18 = %c0_i64 to %33 step %c1_i64  : i64 {
          %34 = arith.divsi %arg18, %32 : i64
          %35 = arith.addi %25, %34 : i64
          %36 = arith.remsi %arg18, %32 : i64
          %37 = arith.divsi %35, %9 : i64
          %38 = arith.remsi %35, %9 : i64
          %39:2 = scf.if %8 -> (i64, i64) {
            scf.yield %38, %c0_i64 : i64, i64
          } else {
            %72:2 = scf.for %arg19 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %c5_i32) -> (i32, i32)  : i32 {
              %78 = arith.addi %arg20, %arg21 : i32
              %79 = arith.divsi %78, %c2_i32 : i32
              %80 = arith.index_cast %79 : i32 to index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%80], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %81 = memref.load %reinterpret_cast_8[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %82 = arith.cmpi sle, %81, %38 : i64
              %83 = arith.select %82, %arg21, %79 : i32
              %84 = scf.if %82 -> (i32) {
                %85 = arith.addi %79, %c1_i32 : i32
                scf.yield %85 : i32
              } else {
                scf.yield %arg20 : i32
              }
              scf.yield %84, %83 : i32, i32
            }
            %73 = arith.subi %72#0, %c1_i32 : i32
            %74 = arith.extsi %73 : i32 to i64
            %75 = arith.index_cast %73 : i32 to index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%75], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %76 = memref.load %reinterpret_cast_7[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %77 = arith.subi %38, %76 : i64
            scf.yield %74, %77 : i64, i64
          }
          %40 = arith.index_cast %39#0 : i64 to index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %41 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %42 = arith.addi %40, %c1 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%42], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %43 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%40], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %44 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%42], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %45 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %46 = arith.subi %43, %41 : i64
          %47 = arith.subi %45, %44 : i64
          %48 = arith.muli %37, %c80_i64 : i64
          %49 = arith.muli %41, %c320_i64 : i64
          %50 = arith.addi %48, %49 : i64
          %51 = arith.index_cast %50 : i64 to index
          %52 = arith.addi %47, %c255_i64 : i64
          %53 = arith.divsi %52, %c256_i64 : i64
          %54 = arith.muli %39#1, %c128_i64 : i64
          %55 = arith.index_cast %54 : i64 to index
          %56 = arith.index_cast %46 : i64 to index
          %alloc = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x256xf32, #hivm.address_space<ub>>
          %memspacecast = memref.memory_space_cast %alloc : memref<128x256xf32, #hivm.address_space<ub>> to memref<128x256xf32>
          %57 = bufferization.to_tensor %memspacecast restrict writable : memref<128x256xf32>
          %58 = hivm.hir.vmul ins(%57, %arg10 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %59 = hivm.hir.vmul ins(%58, %cst : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %60 = hivm.hir.vexp ins(%59 : tensor<128x256xf32>) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %61 = hivm.hir.vadd ins(%60, %cst_0 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %62 = hivm.hir.vdiv ins(%58, %61 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %63 = hivm.hir.vmul ins(%62, %arg11 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %64 = tensor.empty() : tensor<128x256xf16>
          %65 = hivm.hir.vcast ins(%63 : tensor<128x256xf32>) outs(%64 : tensor<128x256xf16>) -> tensor<128x256xf16>
          %expanded = tensor.expand_shape %65 [[0, 1], [2, 3]] output_shape [8, 16, 16, 16] : tensor<128x256xf16> into tensor<8x16x16x16xf16>
          %66 = tensor.empty() : tensor<16x8x16x16xf16>
          %67 = hivm.hir.vtranspose ins(%expanded : tensor<8x16x16x16xf16>) outs(%66 : tensor<16x8x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<16x8x16x16xf16>
          %alloc_5 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %memspacecast_6 = memref.memory_space_cast %alloc_5 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<16x8x16x16xf16>
          %68 = bufferization.to_tensor %memspacecast_6 restrict writable : memref<16x8x16x16xf16>
          %69 = hivm.hir.copy ins(%67 : tensor<16x8x16x16xf16>) outs(%68 : tensor<16x8x16x16xf16>) -> tensor<16x8x16x16xf16>
          annotation.mark %69 : tensor<16x8x16x16xf16>
          hivm.hir.set_atomic kind = <add>[type = f32]
          hivm.hir.set_atomic kind = <none>[type = f32]
          %70 = arith.subi %53, %c1_i64 : i64
          %71 = arith.cmpi eq, %36, %70 : i64
          scf.if %71 {
            scf.for %arg19 = %c0_i32 to %c128_i32 step %c115_i32  : i32 {
              %72 = arith.index_cast %arg19 : i32 to index
              %73 = arith.addi %55, %72 : index
              %74 = arith.muli %73, %c320 : index
              %75 = arith.addi %51, %74 : index
              %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%75], sizes: [115, 80], strides: [320, 1] : memref<?xf32> to memref<115x80xf32, strided<[320, 1], offset: ?>>
              %alloc_8 = memref.alloc() : memref<115x80xf32>
              %76 = arith.addi %72, %c115 : index
              %77 = arith.addi %76, %55 : index
              %78 = arith.maxsi %73, %56 : index
              %79 = arith.minsi %77, %78 : index
              %80 = arith.subi %79, %73 : index
              %81 = arith.cmpi slt, %80, %c115 : index
              %subview = memref.subview %reinterpret_cast_7[0, 0] [%80, 80] [1, 1] : memref<115x80xf32, strided<[320, 1], offset: ?>> to memref<?x80xf32, strided<[320, 1], offset: ?>>
              %subview_9 = memref.subview %alloc_8[0, 0] [%80, 80] [1, 1] : memref<115x80xf32> to memref<?x80xf32, strided<[80, 1]>>
              hivm.hir.load ins(%subview : memref<?x80xf32, strided<[320, 1], offset: ?>>) outs(%subview_9 : memref<?x80xf32, strided<[80, 1]>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %81 : i1 eviction_policy = <EvictFirst>
              %82 = bufferization.to_tensor %alloc_8 restrict writable : memref<115x80xf32>
              %83 = tensor.empty() : tensor<115x80xf16>
              %84 = hivm.hir.vcast ins(%82 : tensor<115x80xf32>) outs(%83 : tensor<115x80xf16>) -> tensor<115x80xf16>
              %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [%75], sizes: [115, 80], strides: [320, 1] : memref<?xf16> to memref<115x80xf16, strided<[320, 1], offset: ?>>
              %extracted_slice = tensor.extract_slice %84[0, 0] [%80, 80] [1, 1] : tensor<115x80xf16> to tensor<?x80xf16>
              %subview_11 = memref.subview %reinterpret_cast_10[0, 0] [%80, 80] [1, 1] : memref<115x80xf16, strided<[320, 1], offset: ?>> to memref<?x80xf16, strided<[320, 1], offset: ?>>
              hivm.hir.store ins(%extracted_slice : tensor<?x80xf16>) outs(%subview_11 : memref<?x80xf16, strided<[320, 1], offset: ?>>)
            }
          }
        }
      }
    }
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

// CHECK-LABEL: func.func @check_stride_store_in_dynamic_shape_mix_aiv
// CHECK: %[[SUB_BLOCK:.*]] = hivm.hir.get_sub_block_idx
// CHECK: %[[SUB_BLOCK_IDX:.*]] = arith.index_cast %[[SUB_BLOCK]] : i64 to index
// CHECK: %[[IS_SUB_BLOCK_0:.*]] = arith.cmpi eq, %[[SUB_BLOCK_IDX]], %{{.*}} : index
// CHECK: scf.if %[[IS_SUB_BLOCK_0]] {
// CHECK:   hivm.hir.stride_store
// CHECK: } {limit_sub_block_id0}
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_stride_store_in_dynamic_shape_mix_aiv(%arg0: memref<?xf16>, %arg1: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty(%arg1) : tensor<?x64xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x64xf16>) -> tensor<?x64xf16>
    hivm.hir.stride_store ins(%1 : tensor<?x64xf16>) outs(%arg0 : memref<?xf16>) offset(%c0_i64 : i64) strides([%c64_i64, %c1_i64 : i64, i64]) numels([%c64_i64, %c64_i64 : i64, i64])
    return
  }
}

// -----

// CHECK-LABEL: func.func @triton_matmul_exp_mix_aiv
// CHECK: scf.for
// CHECK: annotation.mark {{.*}} {buffer_size_in_byte = 100 : i64} : memref<?xf16>
// CHECK: hivm.hir.load
// CHECK: annotation.mark {{.*}} {buffer_size_in_byte = 100 : i64} : memref<?xf16>
// CHECK-NOT: scf.if
// CHECK: hivm.hir.store
// CHECK-NOT: limit_sub_block_id0
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @triton_matmul_exp_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg11 : i32
    %5 = arith.remsi %4, %arg10 : i32
    %6 = arith.index_cast %5 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%6], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %alloc = memref.alloc() : memref<32x32xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<32x32xf16, strided<[32, 1], offset: ?>>) outs(%alloc : memref<32x32xf16>) eviction_policy = <EvictFirst>
    %7 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%6], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<32x32xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32x32xf16, strided<[32, 1], offset: ?>>) outs(%alloc_1 : memref<32x32xf16>) eviction_policy = <EvictFirst>
    %8 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xf16>
    %9 = tensor.empty() : tensor<32x32xf32>
    %10 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%8, %7, %true, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%6], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%10 : tensor<32x32xf32>) outs(%reinterpret_cast_2 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    return
  }
  func.func @triton_matmul_exp_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [99], strides: [1] : memref<?xf16> to memref<99xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<99xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<99xf16, strided<[1]>>) outs(%alloc : memref<99xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<99xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [99], strides: [1] : memref<?xf16> to memref<99xf16, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<99xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<99xf16, strided<[1]>>) outs(%alloc_1 : memref<99xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<99xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [99], strides: [1] : memref<?xf32> to memref<99xf32, strided<[1]>>
    %4 = tensor.empty() : tensor<99xf16>
    %5 = hivm.hir.vadd ins(%2, %3 : tensor<99xf16>, tensor<99xf16>) outs(%4 : tensor<99xf16>) -> tensor<99xf16>
    %6 = tensor.empty() : tensor<99xf32>
    %7 = hivm.hir.vcast ins(%5 : tensor<99xf16>) outs(%6 : tensor<99xf32>) -> tensor<99xf32>
    hivm.hir.store ins(%7 : tensor<99xf32>) outs(%reinterpret_cast_2 : memref<99xf32, strided<[1]>>)
    return
  }
}

// -----
// chunk_dqkwg AIC: UB fixpipe for tightly_coupled_buffer<1>/<2> follows each mark.
// chunk_dqkwg AIV: same buffer ids; UB uses memory_space_cast (no fixpipe on AIV).
// CHECK-LABEL: func.func @chunk_bwd_kernel_dqkwg_mix_aic(
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>
// CHECK-NEXT: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.*}} : tensor<32x16xf32>) outs(%{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>
// CHECK-NEXT: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.*}} : tensor<32x16xf32>) outs(%{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
// CHECK-LABEL: func.func @chunk_bwd_kernel_dqkwg_mix_aiv(
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>
// CHECK-NEXT: memref.memory_space_cast %{{.*}} : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>
// CHECK-NEXT: memref.memory_space_cast %{{.*}} : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
#map = affine_map<()[s0, s1] -> (s0 + s1 * 16)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
#map2 = affine_map<()[s0] -> (-s0 + 16)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 16)>
#map4 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 16)>
#map5 = affine_map<()[s0] -> (-s0 + (s0 floordiv 16) * 16 + 16)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1)>
#map7 = affine_map<()[s0] -> (s0 + 32)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_957c">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @chunk_bwd_kernel_dqkwg_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: f32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %c256_i64 = arith.constant 256 : i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg14, %arg15 : i32
    %1 = arith.muli %0, %arg16 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.remsi %3, %arg16 : i32
    %5 = arith.divsi %3, %arg16 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = arith.muli %arg16, %arg15 : i32
    %8 = arith.divsi %3, %7 : i32
    %9 = arith.remsi %8, %arg14 : i32
    %10 = tensor.empty() : tensor<32x32xf32>
    %11 = tensor.empty() : tensor<32x16xf32>
    %12 = arith.addi %arg13, %c31_i32 : i32
    %13 = arith.divsi %12, %c32_i32 : i32
    %14 = arith.muli %4, %13 : i32
    %15 = arith.addi %14, %6 : i32
    %16 = arith.muli %4, %arg13 : i32
    %17 = arith.muli %16, %c16_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.extsi %15 : i32 to i64
    %20 = arith.muli %19, %c256_i64 : i64
    %21 = arith.index_cast %20 : i64 to index
    %22 = arith.muli %6, %c32_i32 : i32
    %23 = arith.maxsi %22, %c0_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = affine.apply #map()[%18, %24]
    %26 = arith.index_cast %arg13 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [%25], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %27 = arith.muli %9, %c16_i32 : i32
    %alloc = memref.alloc() : memref<32x16xbf16>
    %28 = affine.apply #map1()[%26, %24]
    %29 = arith.maxsi %28, %c0 : index
    %30 = arith.minsi %29, %c32 : index
    %31 = arith.subi %c0_i32, %22 : i32
    %32 = arith.maxsi %31, %c0_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = arith.minsi %33, %30 : index
    %35 = affine.apply #map1()[%30, %34]
    %36 = arith.cmpi slt, %35, %c32 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%35, 16] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[%34, 0] [%35, 16] [1, 1] : memref<32x16xbf16> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    hivm.hir.load ins(%subview : memref<?x16xbf16, strided<[16, 1], offset: ?>>) outs(%subview_1 : memref<?x16xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %36 : i1 eviction_policy = <EvictFirst>
    %37 = bufferization.to_tensor %alloc restrict writable : memref<32x16xbf16>
    %alloc_2 = memref.alloc() : memref<32x16xbf16>
    %subview_3 = memref.subview %reinterpret_cast_0[0, 0] [%35, 16] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[%34, 0] [%35, 16] [1, 1] : memref<32x16xbf16> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    hivm.hir.load ins(%subview_3 : memref<?x16xbf16, strided<[16, 1], offset: ?>>) outs(%subview_4 : memref<?x16xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %36 : i1 eviction_policy = <EvictFirst>
    %38 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x16xbf16>
    %39 = arith.maxsi %27, %c0_i32 : i32
    %40 = arith.index_cast %39 : i32 to index
    %41 = affine.apply #map()[%21, %40]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [16, 16], strides: [16, 1] : memref<?xbf16> to memref<16x16xbf16, strided<[16, 1], offset: ?>>
    %alloc_6 = memref.alloc() : memref<16x16xbf16>
    %42 = affine.apply #map2()[%40]
    %43 = arith.maxsi %42, %c0 : index
    %44 = arith.minsi %43, %c16 : index
    %45 = arith.cmpi slt, %44, %c16 : index
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%44, 16] [1, 1] : memref<16x16xbf16, strided<[16, 1], offset: ?>> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    %subview_8 = memref.subview %alloc_6[0, 0] [%44, 16] [1, 1] : memref<16x16xbf16> to memref<?x16xbf16, strided<[16, 1]>>
    hivm.hir.load ins(%subview_7 : memref<?x16xbf16, strided<[16, 1], offset: ?>>) outs(%subview_8 : memref<?x16xbf16, strided<[16, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %45 : i1 eviction_policy = <EvictFirst>
    %46 = bufferization.to_tensor %alloc_6 restrict writable : memref<16x16xbf16>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%41], sizes: [16, 16], strides: [16, 1] : memref<?xbf16> to memref<16x16xbf16, strided<[16, 1], offset: ?>>
    %alloc_10 = memref.alloc() : memref<16x16xbf16>
    %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%44, 16] [1, 1] : memref<16x16xbf16, strided<[16, 1], offset: ?>> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    %subview_12 = memref.subview %alloc_10[0, 0] [%44, 16] [1, 1] : memref<16x16xbf16> to memref<?x16xbf16, strided<[16, 1]>>
    hivm.hir.load ins(%subview_11 : memref<?x16xbf16, strided<[16, 1], offset: ?>>) outs(%subview_12 : memref<?x16xbf16, strided<[16, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %45 : i1 eviction_policy = <EvictFirst>
    %47 = bufferization.to_tensor %alloc_10 restrict writable : memref<16x16xbf16>
    %48 = bufferization.alloc_tensor() : tensor<f32>
    %49 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%38, %37, %true, %c32, %c16, %c32 : tensor<32x16xbf16>, tensor<32x16xbf16>, i1, index, index, index) outs(%10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %alloc_13 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%49 : tensor<32x32xf32>) outs(%alloc_13 : memref<32x32xf32, #hivm.address_space<ub>>)
    %50 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%38, %46, %true, %c32, %c16, %c16 : tensor<32x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%11 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %alloc_14 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<32x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%50 : tensor<32x16xf32>) outs(%alloc_14 : memref<32x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %51 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%37, %47, %true, %c32, %c16, %c16 : tensor<32x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%11 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %alloc_15 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_15 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<32x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%51 : tensor<32x16xf32>) outs(%alloc_15 : memref<32x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %52 = affine.apply #map3()[%40, %18, %24]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg2 to offset: [%52], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %alloc_18 = memref.alloc() : memref<32x16xbf16>
    %53 = affine.apply #map4()[%26, %40, %24]
    %54 = arith.maxsi %53, %c0 : index
    %55 = arith.minsi %54, %c32 : index
    %56 = affine.apply #map5()[%40]
    %57 = arith.maxsi %56, %c0 : index
    %58 = arith.minsi %57, %c16 : index
    %59 = arith.minsi %33, %55 : index
    %60 = affine.apply #map1()[%55, %59]
    %61 = arith.subi %c0_i32, %27 : i32
    %62 = arith.maxsi %61, %c0_i32 : i32
    %63 = arith.index_cast %62 : i32 to index
    %64 = arith.minsi %63, %58 : index
    %65 = affine.apply #map1()[%58, %64]
    %66 = arith.cmpi slt, %60, %c32 : index
    %67 = arith.cmpi slt, %65, %c16 : index
    %68 = arith.ori %66, %67 : i1
    %subview_19 = memref.subview %reinterpret_cast_16[0, 0] [%60, %65] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    %subview_20 = memref.subview %alloc_18[%59, %64] [%60, %65] [1, 1] : memref<32x16xbf16> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    %69 = arith.remui %64, %c16 : index
    hivm.hir.load ins(%subview_19 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) outs(%subview_20 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %69 : index init_out_buffer = true init_condition = %68 : i1 eviction_policy = <EvictFirst>
    %70 = bufferization.to_tensor %alloc_18 restrict writable : memref<32x16xbf16>
    %alloc_21 = memref.alloc() : memref<32x16xbf16>
    %subview_22 = memref.subview %reinterpret_cast_17[0, 0] [%60, %65] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    %subview_23 = memref.subview %alloc_21[%59, %64] [%60, %65] [1, 1] : memref<32x16xbf16> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    hivm.hir.load ins(%subview_22 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) outs(%subview_23 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %69 : index init_out_buffer = true init_condition = %68 : i1 eviction_policy = <EvictFirst>
    %71 = bufferization.to_tensor %alloc_21 restrict writable : memref<32x16xbf16>
    %72 = bufferization.alloc_tensor() : tensor<f32>
    %73 = tensor.empty() : tensor<32x32xbf16>
    %expanded = tensor.expand_shape %73 [[0], [1, 2]] output_shape [32, 2, 16] : tensor<32x32xbf16> into tensor<32x2x16xbf16>
    annotation.mark %expanded {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x2x16xbf16>
    %74 = tensor.empty() : tensor<2x32x16xbf16>
    %expanded_24 = tensor.expand_shape %74 [[0], [1, 2], [3]] output_shape [2, 2, 16, 16] : tensor<2x32x16xbf16> into tensor<2x2x16x16xbf16>
    annotation.mark %expanded_24 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<2x2x16x16xbf16>
    %alloc_25 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_25 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    %memspacecast = memref.memory_space_cast %alloc_25 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
    %75 = bufferization.to_tensor %memspacecast restrict writable : memref<2x2x16x16xbf16>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 2
    %76 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%75, %71, %true, %c32, %c32, %c16 : tensor<2x2x16x16xbf16>, tensor<32x16xbf16>, i1, index, index, index) outs(%11 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %alloc_26 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_26 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<32x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%76 : tensor<32x16xf32>) outs(%alloc_26 : memref<32x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 3
    annotation.mark %expanded {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x2x16xbf16>
    annotation.mark %expanded_24 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<2x2x16x16xbf16>
    %alloc_27 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_27 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    %memspacecast_28 = memref.memory_space_cast %alloc_27 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
    %77 = bufferization.to_tensor %memspacecast_28 restrict writable : memref<2x2x16x16xbf16>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
    %78 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_already_inserted = true} ins(%77, %70, %true, %c32, %c32, %c16 : tensor<2x2x16x16xbf16>, tensor<32x16xbf16>, i1, index, index, index) outs(%11 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %alloc_29 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_29 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>} : memref<32x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%78 : tensor<32x16xf32>) outs(%alloc_29 : memref<32x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 5
    return
  }
  // expected-remark@+1 {{Selected tiling dim might have broadcast two different axis. Automatically disables strict mode.}}
  func.func @chunk_bwd_kernel_dqkwg_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: f32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_2 = arith.constant 0.000000e+00 : bf16
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %c256_i64 = arith.constant 256 : i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg14, %arg15 : i32
    %1 = arith.muli %0, %arg16 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.remsi %3, %arg16 : i32
    %5 = arith.divsi %3, %arg16 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = arith.muli %arg16, %arg15 : i32
    %8 = arith.divsi %3, %7 : i32
    %9 = arith.remsi %8, %arg14 : i32
    %10 = tensor.empty() : tensor<1xf32>
    %11 = tensor.empty() : tensor<32x32xf32>
    %12 = tensor.empty() : tensor<32x16xf32>
    %13 = tensor.empty() : tensor<32xf32>
    %14 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %15 = arith.addi %arg13, %c31_i32 : i32
    %16 = arith.divsi %15, %c32_i32 : i32
    %17 = arith.muli %4, %16 : i32
    %18 = arith.addi %17, %6 : i32
    %19 = arith.muli %4, %arg13 : i32
    %20 = arith.muli %19, %c16_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.extsi %18 : i32 to i64
    %23 = arith.muli %22, %c256_i64 : i64
    %24 = arith.index_cast %23 : i64 to index
    %25 = arith.muli %9, %arg13 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.muli %6, %c32_i32 : i32
    %28 = arith.maxsi %27, %c0_i32 : i32
    %29 = arith.index_cast %28 : i32 to index
    %30 = arith.index_cast %arg13 : i32 to index
    %31 = arith.muli %9, %c16_i32 : i32
    %32 = affine.apply #map1()[%30, %29]
    %33 = arith.maxsi %32, %c0 : index
    %34 = arith.minsi %33, %c32 : index
    %35 = arith.subi %c0_i32, %27 : i32
    %36 = arith.maxsi %35, %c0_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = arith.minsi %37, %34 : index
    %39 = affine.apply #map1()[%34, %38]
    %40 = arith.cmpi slt, %39, %c32 : index
    %41 = arith.maxsi %31, %c0_i32 : i32
    %42 = arith.index_cast %41 : i32 to index
    %43 = affine.apply #map()[%24, %42]
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%43], sizes: [16, 16], strides: [16, 1] : memref<?xbf16> to memref<16x16xbf16, strided<[16, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xbf16>
    %44 = affine.apply #map2()[%42]
    %45 = arith.maxsi %44, %c0 : index
    %46 = arith.minsi %45, %c16 : index
    %47 = arith.cmpi slt, %46, %c16 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%46, 16] [1, 1] : memref<16x16xbf16, strided<[16, 1], offset: ?>> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    %subview_3 = memref.subview %alloc[0, 0] [%46, 16] [1, 1] : memref<16x16xbf16> to memref<?x16xbf16, strided<[16, 1]>>
    hivm.hir.load ins(%subview : memref<?x16xbf16, strided<[16, 1], offset: ?>>) outs(%subview_3 : memref<?x16xbf16, strided<[16, 1]>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %47 : i1 eviction_policy = <EvictFirst>
    %48 = bufferization.to_tensor %alloc restrict writable : memref<16x16xbf16>
    %49 = tensor.empty() : tensor<16x16xbf16>
    %50 = hivm.hir.vtranspose ins(%48 : tensor<16x16xbf16>) outs(%49 : tensor<16x16xbf16>) permutation = [1, 0] -> tensor<16x16xbf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%43], sizes: [16, 16], strides: [16, 1] : memref<?xbf16> to memref<16x16xbf16, strided<[16, 1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<16x16xbf16>
    %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%46, 16] [1, 1] : memref<16x16xbf16, strided<[16, 1], offset: ?>> to memref<?x16xbf16, strided<[16, 1], offset: ?>>
    %subview_7 = memref.subview %alloc_5[0, 0] [%46, 16] [1, 1] : memref<16x16xbf16> to memref<?x16xbf16, strided<[16, 1]>>
    hivm.hir.load ins(%subview_6 : memref<?x16xbf16, strided<[16, 1], offset: ?>>) outs(%subview_7 : memref<?x16xbf16, strided<[16, 1]>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %47 : i1 eviction_policy = <EvictFirst>
    %51 = bufferization.to_tensor %alloc_5 restrict writable : memref<16x16xbf16>
    %52 = hivm.hir.vtranspose ins(%51 : tensor<16x16xbf16>) outs(%49 : tensor<16x16xbf16>) permutation = [1, 0] -> tensor<16x16xbf16>
    %53 = tensor.empty() : tensor<16x16xf32>
    %54 = hivm.hir.vcast ins(%50 : tensor<16x16xbf16>) outs(%53 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %55 = hivm.hir.vcast ins(%52 : tensor<16x16xbf16>) outs(%53 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %56 = hivm.hir.vmul ins(%54, %55 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%53 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %collapsed = tensor.collapse_shape %56 [[0, 1]] : tensor<16x16xf32> into tensor<256xf32>
    %57 = bufferization.alloc_tensor() : tensor<f32>
    %58 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%57 : tensor<f32>) -> tensor<f32>
    %expanded = tensor.expand_shape %58 [] output_shape [1] : tensor<f32> into tensor<1xf32>
    %59 = hivm.hir.vreduce <sum> ins(%collapsed : tensor<256xf32>) outs(%expanded : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
    %collapsed_8 = tensor.collapse_shape %59 [] : tensor<1xf32> into tensor<f32>
    %extracted = tensor.extract %collapsed_8[] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<f32>
    %inserted = tensor.insert %extracted into %10[%c0] : tensor<1xf32>
    %extracted_9 = tensor.extract %inserted[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
    %alloc_10 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<32x32xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_10 : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
    %60 = bufferization.to_tensor %memspacecast restrict writable : memref<32x32xf32>
    %alloc_11 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<32x16xf32, #hivm.address_space<ub>>
    %memspacecast_12 = memref.memory_space_cast %alloc_11 : memref<32x16xf32, #hivm.address_space<ub>> to memref<32x16xf32>
    %61 = bufferization.to_tensor %memspacecast_12 restrict writable : memref<32x16xf32>
    %alloc_13 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<32x16xf32, #hivm.address_space<ub>>
    %memspacecast_14 = memref.memory_space_cast %alloc_13 : memref<32x16xf32, #hivm.address_space<ub>> to memref<32x16xf32>
    %62 = bufferization.to_tensor %memspacecast_14 restrict writable : memref<32x16xf32>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %63 = affine.apply #map3()[%42, %21, %29]
    %reinterpret_cast_15 = memref.reinterpret_cast %arg2 to offset: [%63], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %alloc_17 = memref.alloc() : memref<32x16xbf16>
    %64 = affine.apply #map4()[%30, %42, %29]
    %65 = arith.maxsi %64, %c0 : index
    %66 = arith.minsi %65, %c32 : index
    %67 = affine.apply #map5()[%42]
    %68 = arith.maxsi %67, %c0 : index
    %69 = arith.minsi %68, %c16 : index
    %70 = arith.minsi %37, %66 : index
    %71 = affine.apply #map1()[%66, %70]
    %72 = arith.subi %c0_i32, %31 : i32
    %73 = arith.maxsi %72, %c0_i32 : i32
    %74 = arith.index_cast %73 : i32 to index
    %75 = arith.minsi %74, %69 : index
    %76 = affine.apply #map1()[%69, %75]
    %77 = arith.cmpi slt, %71, %c32 : index
    %78 = arith.cmpi slt, %76, %c16 : index
    %79 = arith.ori %77, %78 : i1
    %subview_18 = memref.subview %reinterpret_cast_15[0, 0] [%71, %76] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    %subview_19 = memref.subview %alloc_17[%70, %75] [%71, %76] [1, 1] : memref<32x16xbf16> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    %80 = arith.remui %75, %c16 : index
    hivm.hir.load ins(%subview_18 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) outs(%subview_19 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %80 : index init_out_buffer = true init_condition = %79 : i1 eviction_policy = <EvictFirst>
    %81 = bufferization.to_tensor %alloc_17 restrict writable : memref<32x16xbf16>
    %alloc_20 = memref.alloc() : memref<32x16xbf16>
    %subview_21 = memref.subview %reinterpret_cast_16[0, 0] [%71, %76] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    %subview_22 = memref.subview %alloc_20[%70, %75] [%71, %76] [1, 1] : memref<32x16xbf16> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    hivm.hir.load ins(%subview_21 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) outs(%subview_22 : memref<?x?xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %80 : index init_out_buffer = true init_condition = %79 : i1 eviction_policy = <EvictFirst>
    %82 = bufferization.to_tensor %alloc_20 restrict writable : memref<32x16xbf16>
    %reinterpret_cast_23 = memref.reinterpret_cast %arg9 to offset: [%63], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %reinterpret_cast_24 = memref.reinterpret_cast %arg10 to offset: [%63], sizes: [32, 16], strides: [16, 1] : memref<?xbf16> to memref<32x16xbf16, strided<[16, 1], offset: ?>>
    %83 = tensor.empty() : tensor<32xi32>
    %84 = hivm.hir.varange offset[%c0] strides[%c1] outs(%83 : tensor<32xi32>) -> tensor<32xi32>
    %85 = hivm.hir.vadd ins(%84, %27 : tensor<32xi32>, i32) outs(%83 : tensor<32xi32>) -> tensor<32xi32>
    %86 = tensor.empty() : tensor<32xi1>
    %87 = hivm.hir.vcmp ins(%85, %arg13 : tensor<32xi32>, i32) outs(%86 : tensor<32xi1>) compare_mode = <lt> -> tensor<32xi1>
    %88 = tensor.empty() : tensor<32x32xi32>
    // expected-warning@+1 {{Extract slice is not fully bubbled up}}
    %expanded_25 = tensor.expand_shape %85 [[0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
    %89 = hivm.hir.vbrc ins(%expanded_25 : tensor<32x1xi32>) outs(%88 : tensor<32x32xi32>) broadcast_dims = [1] -> tensor<32x32xi32>
    %expanded_26 = tensor.expand_shape %85 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
    %90 = hivm.hir.vbrc ins(%expanded_26 : tensor<1x32xi32>) outs(%88 : tensor<32x32xi32>) broadcast_dims = [0] -> tensor<32x32xi32>
    %91 = tensor.empty() : tensor<32x32xi1>
    %92 = hivm.hir.vcmp ins(%89, %90 : tensor<32x32xi32>, tensor<32x32xi32>) outs(%91 : tensor<32x32xi1>) compare_mode = <ge> -> tensor<32x32xi1>
    %93 = tensor.empty() : tensor<32xf16>
    %94 = hivm.hir.vcast ins(%87 : tensor<32xi1>) outs(%93 : tensor<32xf16>) round_mode = <trunc> -> tensor<32xf16>
    %95 = tensor.empty() : tensor<32x32xf16>
    // expected-warning@+1 {{Extract slice is not fully bubbled up}}
    %expanded_27 = tensor.expand_shape %94 [[0, 1]] output_shape [32, 1] : tensor<32xf16> into tensor<32x1xf16>
    %96 = hivm.hir.vbrc ins(%expanded_27 : tensor<32x1xf16>) outs(%95 : tensor<32x32xf16>) broadcast_dims = [1] -> tensor<32x32xf16>
    %97 = hivm.hir.vcmp ins(%96, %cst_0 : tensor<32x32xf16>, f16) outs(%91 : tensor<32x32xi1>) compare_mode = <ne> -> tensor<32x32xi1>
    %expanded_28 = tensor.expand_shape %94 [[0, 1]] output_shape [1, 32] : tensor<32xf16> into tensor<1x32xf16>
    %98 = hivm.hir.vbrc ins(%expanded_28 : tensor<1x32xf16>) outs(%95 : tensor<32x32xf16>) broadcast_dims = [0] -> tensor<32x32xf16>
    %99 = hivm.hir.vcmp ins(%98, %cst_0 : tensor<32x32xf16>, f16) outs(%91 : tensor<32x32xi1>) compare_mode = <ne> -> tensor<32x32xi1>
    %100 = hivm.hir.vand ins(%97, %99 : tensor<32x32xi1>, tensor<32x32xi1>) outs(%91 : tensor<32x32xi1>) -> tensor<32x32xi1>
    %101 = hivm.hir.vand ins(%92, %100 : tensor<32x32xi1>, tensor<32x32xi1>) outs(%91 : tensor<32x32xi1>) -> tensor<32x32xi1>
    %102 = arith.index_cast %19 : i32 to index
    %103 = affine.apply #map6()[%29, %102]
    %reinterpret_cast_29 = memref.reinterpret_cast %arg5 to offset: [%103], sizes: [32], strides: [1] : memref<?xbf16> to memref<32xbf16, strided<[1], offset: ?>>
    %alloc_30 = memref.alloc() : memref<32xbf16>
    %subview_31 = memref.subview %reinterpret_cast_29[0] [%39] [1] : memref<32xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
    %subview_32 = memref.subview %alloc_30[%38] [%39] [1] : memref<32xbf16> to memref<?xbf16, strided<[1], offset: ?>>
    %104 = arith.remui %38, %c16 : index
    hivm.hir.load ins(%subview_31 : memref<?xbf16, strided<[1], offset: ?>>) outs(%subview_32 : memref<?xbf16, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %104 : index init_out_buffer = true init_condition = %40 : i1 eviction_policy = <EvictFirst>
    %105 = bufferization.to_tensor %alloc_30 restrict writable : memref<32xbf16>
    %106 = arith.addi %27, %c32_i32 : i32
    %107 = arith.minsi %106, %arg13 : i32
    %108 = arith.subi %107, %c1_i32 : i32
    %109 = arith.index_cast %108 : i32 to index
    %110 = affine.apply #map6()[%102, %109]
    %reinterpret_cast_33 = memref.reinterpret_cast %arg5 to offset: [%110], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
    %111 = memref.load %reinterpret_cast_33[%c0] : memref<1xbf16, strided<[1], offset: ?>>
    %from_elements = tensor.from_elements %111 : tensor<1xbf16>
    %112 = hivm.hir.vcast ins(%from_elements : tensor<1xbf16>) outs(%10 : tensor<1xf32>) -> tensor<1xf32>
    %extracted_34 = tensor.extract %112[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
    %inserted_35 = tensor.insert %extracted_34 into %10[%c0] : tensor<1xf32>
    %113 = hivm.hir.vexp ins(%inserted_35 : tensor<1xf32>) outs(%10 : tensor<1xf32>) -> tensor<1xf32>
    %extracted_36 = tensor.extract %113[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
    %inserted_37 = tensor.insert %extracted_9 into %10[%c0] : tensor<1xf32>
    %inserted_38 = tensor.insert %extracted_36 into %10[%c0] : tensor<1xf32>
    %114 = hivm.hir.vmul ins(%inserted_37, %inserted_38 : tensor<1xf32>, tensor<1xf32>) outs(%10 : tensor<1xf32>) -> tensor<1xf32>
    %extracted_39 = tensor.extract %114[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
    %115 = hivm.hir.vcast ins(%105 : tensor<32xbf16>) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    // expected-warning@+1 {{Extract slice is not fully bubbled up}}
    %116 = hivm.hir.vexp ins(%115 : tensor<32xf32>) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %expanded_40 = tensor.expand_shape %116 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %117 = hivm.hir.vmul ins(%61, %expanded_40 : tensor<32x16xf32>, tensor<32x1xf32>) outs(%12 : tensor<32x16xf32>) broadcast = [1] -> tensor<32x16xf32>
    %118 = hivm.hir.vmul ins(%117, %arg12 : tensor<32x16xf32>, f32) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %119 = hivm.hir.vmul ins(%115, %cst : tensor<32xf32>, f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %120 = hivm.hir.vadd ins(%119, %extracted_34 : tensor<32xf32>, f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %121 = hivm.hir.vexp ins(%120 : tensor<32xf32>) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %122 = arith.index_cast %27 : i32 to index
    %123 = affine.apply #map7()[%122]
    %124 = arith.maxsi %122, %30 : index
    %125 = arith.minsi %123, %124 : index
    %126 = affine.apply #map1()[%125, %122]
    %extracted_slice = tensor.extract_slice %121[0] [%126] [1] : tensor<32xf32> to tensor<?xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %14[0] [%126] [1] : tensor<?xf32> into tensor<32xf32>
    %expanded_41 = tensor.expand_shape %inserted_slice [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
    %127 = hivm.hir.vmul ins(%62, %expanded_41 : tensor<32x16xf32>, tensor<32x1xf32>) outs(%12 : tensor<32x16xf32>) broadcast = [1] -> tensor<32x16xf32>
    %128 = hivm.hir.vcast ins(%82 : tensor<32x16xbf16>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %129 = hivm.hir.vmul ins(%127, %128 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %collapsed_42 = tensor.collapse_shape %129 [[0, 1]] : tensor<32x16xf32> into tensor<512xf32>
    %130 = bufferization.alloc_tensor() : tensor<f32>
    %131 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%130 : tensor<f32>) -> tensor<f32>
    %expanded_43 = tensor.expand_shape %131 [] output_shape [1] : tensor<f32> into tensor<1xf32>
    %132 = hivm.hir.vreduce <sum> ins(%collapsed_42 : tensor<512xf32>) outs(%expanded_43 : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
    %collapsed_44 = tensor.collapse_shape %132 [] : tensor<1xf32> into tensor<f32>
    %extracted_45 = tensor.extract %collapsed_44[] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<f32>
    %inserted_46 = tensor.insert %extracted_39 into %10[%c0] : tensor<1xf32>
    %inserted_47 = tensor.insert %extracted_45 into %10[%c0] : tensor<1xf32>
    %133 = hivm.hir.vadd ins(%inserted_46, %inserted_47 : tensor<1xf32>, tensor<1xf32>) outs(%10 : tensor<1xf32>) -> tensor<1xf32>
    %extracted_48 = tensor.extract %133[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
    %134 = tensor.empty() : tensor<32x32xbf16>
    %expanded_49 = tensor.expand_shape %115 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %expanded_50 = tensor.expand_shape %115 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
    %135 = hivm.hir.vsub ins(%expanded_49, %expanded_50 : tensor<32x1xf32>, tensor<1x32xf32>) outs(%11 : tensor<32x32xf32>) broadcast = [0, 1] -> tensor<32x32xf32>
    %136 = hivm.hir.vexp ins(%135 : tensor<32x32xf32>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %137 = hivm.hir.vmul ins(%60, %136 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %138 = hivm.hir.vsel ins(%101, %137, %cst_1 : tensor<32x32xi1>, tensor<32x32xf32>, f32) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %139 = hivm.hir.vmul ins(%138, %arg12 : tensor<32x32xf32>, f32) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %140 = hivm.hir.vcast ins(%139 : tensor<32x32xf32>) outs(%134 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %expanded_51 = tensor.expand_shape %140 [[0], [1, 2]] output_shape [32, 2, 16] : tensor<32x32xbf16> into tensor<32x2x16xbf16>
    annotation.mark %expanded_51 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x2x16xbf16>
    %141 = tensor.empty() : tensor<2x32x16xbf16>
    %142 = hivm.hir.vtranspose ins(%expanded_51 : tensor<32x2x16xbf16>) outs(%141 : tensor<2x32x16xbf16>) permutation = [1, 0, 2] -> tensor<2x32x16xbf16>
    %expanded_52 = tensor.expand_shape %142 [[0], [1, 2], [3]] output_shape [2, 2, 16, 16] : tensor<2x32x16xbf16> into tensor<2x2x16x16xbf16>
    annotation.mark %expanded_52 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<2x2x16x16xbf16>
    %alloc_53 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
    hivm.hir.copy ins(%expanded_52 : tensor<2x2x16x16xbf16>) outs(%memspacecast_54 : memref<2x2x16x16xbf16>)
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 2
    %alloc_55 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_55 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<32x16xf32, #hivm.address_space<ub>>
    %memspacecast_56 = memref.memory_space_cast %alloc_55 : memref<32x16xf32, #hivm.address_space<ub>> to memref<32x16xf32>
    %143 = bufferization.to_tensor %memspacecast_56 restrict writable : memref<32x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 3
    %144 = hivm.hir.vadd ins(%143, %118 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    annotation.mark %expanded_51 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<32x2x16xbf16>
    annotation.mark %expanded_52 {tiling_dim_mapping = {"1" = 1 : index}} : tensor<2x2x16x16xbf16>
    %alloc_57 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_57 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
    %memspacecast_58 = memref.memory_space_cast %alloc_57 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
    hivm.hir.copy ins(%expanded_52 : tensor<2x2x16x16xbf16>) outs(%memspacecast_58 : memref<2x2x16x16xbf16>)
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 4
    %alloc_59 = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_59 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>} : memref<32x16xf32, #hivm.address_space<ub>>
    %memspacecast_60 = memref.memory_space_cast %alloc_59 : memref<32x16xf32, #hivm.address_space<ub>> to memref<32x16xf32>
    %145 = bufferization.to_tensor %memspacecast_60 restrict writable : memref<32x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 5
    %146 = hivm.hir.vadd ins(%145, %127 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %147 = hivm.hir.vcast ins(%81 : tensor<32x16xbf16>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %148 = hivm.hir.vmul ins(%144, %147 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %expanded_61 = tensor.expand_shape %14 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %149 = hivm.hir.vreduce <sum> ins(%148 : tensor<32x16xf32>) outs(%expanded_61 : tensor<32x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<32x1xf32>
    %collapsed_62 = tensor.collapse_shape %149 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
    %150 = hivm.hir.vmul ins(%146, %128 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%12 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %151 = hivm.hir.vreduce <sum> ins(%150 : tensor<32x16xf32>) outs(%expanded_61 : tensor<32x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<32x1xf32>
    %collapsed_63 = tensor.collapse_shape %151 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
    %152 = hivm.hir.vsub ins(%collapsed_62, %collapsed_63 : tensor<32xf32>, tensor<32xf32>) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %153 = affine.apply #map8()[%29, %26, %102]
    %reinterpret_cast_64 = memref.reinterpret_cast %arg11 to offset: [%153], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %154 = hivm.hir.vadd ins(%152, %extracted_48 : tensor<32xf32>, f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
    %155 = arith.maxsi %122, %109 : index
    %156 = arith.minsi %123, %155 : index
    %157 = affine.apply #map1()[%156, %122]
    %extracted_slice_65 = tensor.extract_slice %152[0] [%157] [1] : tensor<32xf32> to tensor<?xf32>
    %inserted_slice_66 = tensor.insert_slice %extracted_slice_65 into %154[0] [%157] [1] : tensor<?xf32> into tensor<32xf32>
    %158 = tensor.empty() : tensor<32x16xbf16>
    %159 = hivm.hir.vcast ins(%144 : tensor<32x16xf32>) outs(%158 : tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %extracted_slice_67 = tensor.extract_slice %159[%70, %75] [%71, %76] [1, 1] : tensor<32x16xbf16> to tensor<?x?xbf16>
    %subview_68 = memref.subview %reinterpret_cast_23[0, 0] [%71, %76] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice_67 : tensor<?x?xbf16>) outs(%subview_68 : memref<?x?xbf16, strided<[16, 1], offset: ?>>)
    %160 = hivm.hir.vcast ins(%146 : tensor<32x16xf32>) outs(%158 : tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %extracted_slice_69 = tensor.extract_slice %160[%70, %75] [%71, %76] [1, 1] : tensor<32x16xbf16> to tensor<?x?xbf16>
    %subview_70 = memref.subview %reinterpret_cast_24[0, 0] [%71, %76] [1, 1] : memref<32x16xbf16, strided<[16, 1], offset: ?>> to memref<?x?xbf16, strided<[16, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice_69 : tensor<?x?xbf16>) outs(%subview_70 : memref<?x?xbf16, strided<[16, 1], offset: ?>>)
    %extracted_slice_71 = tensor.extract_slice %inserted_slice_66[%38] [%39] [1] : tensor<32xf32> to tensor<?xf32>
    %subview_72 = memref.subview %reinterpret_cast_64[0] [%39] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%extracted_slice_71 : tensor<?xf32>) outs(%subview_72 : memref<?xf32, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @custom_variadic_outs_mix_aiv(
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK:           %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK:           %[[SUB_BLOCK:.*]] = hivm.hir.get_sub_block_idx
// CHECK:           %[[SUB_BLOCK_IDX:.*]] = arith.index_cast %[[SUB_BLOCK]] : i64 to index
// CHECK:           %[[IS_SUB_BLOCK_0:.*]] = arith.cmpi eq, %[[SUB_BLOCK_IDX]], %[[C0]] : index
// CHECK:           %[[CUSTOM_RES:.*]]:2 = scf.if %[[IS_SUB_BLOCK_0]] -> (tensor<16xf32>, tensor<16xf32>) {
// CHECK:             %[[CUSTOM_THEN:.*]]:2 = hivm.hir.custom
// CHECK:             scf.yield %[[CUSTOM_THEN]]#0, %[[CUSTOM_THEN]]#1 : tensor<16xf32>, tensor<16xf32>
// CHECK:           } else {
// CHECK:             scf.yield %[[EMPTY0]], %[[EMPTY1]] : tensor<16xf32>, tensor<16xf32>
// CHECK:           } {limit_sub_block_id0}
// CHECK:           scf.if
// CHECK:             hivm.hir.store ins(%[[CUSTOM_RES]]#0
// CHECK:           } {limit_sub_block_id0}
// CHECK:           scf.if
// CHECK:             hivm.hir.store ins(%[[CUSTOM_RES]]#1
// CHECK:           } {limit_sub_block_id0}
// CHECK-NOT:       map_for_to_forall
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @custom_variadic_outs_mix_aiv(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %empty0 = tensor.empty() : tensor<16xf32>
    %empty1 = tensor.empty() : tensor<16xf32>
    %0:2 = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "custom_todo"}
        "my_custom_op" outs(%empty0, %empty1 : tensor<16xf32>, tensor<16xf32>)
        -> (tensor<16xf32>, tensor<16xf32>)
    hivm.hir.store ins(%0#0 : tensor<16xf32>) outs(%arg0 : memref<16xf32>)
    hivm.hir.store ins(%0#1 : tensor<16xf32>) outs(%arg1 : memref<16xf32>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @brc_two_dim_with_reduction_dim
// CHECK-NOT: scf.if
// CHECK: hivm.hir.vreduce <sum> ins(%[[VAL_24:.*]] : tensor<16x8xf32>) outs(%[[VAL_26:.*]] : tensor<1x8xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x8xf32>
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
    %10 = hivm.hir.vreduce <sum> ins(%9 : tensor<16x16xf32>) outs(%expanded_1 : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
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
// CHECK-LABEL: func.func @fwd_kernel_mix_aiv

// CHECK: %{{.*}} = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32x128xf32, #hivm.address_space<ub>>

// CHECK: %{{.*}} = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %{{.*}} : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>

// CHECK: %{{.*}} = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST_9:.*]] = memref.memory_space_cast %{{.*}} : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>

// CHECK: %{{.*}} = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST_11:.*]] = memref.memory_space_cast %{{.*}} : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

// CHECK: %{{.*}} = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST_13:.*]] = memref.memory_space_cast %{{.*}} : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

// CHECK: scf.if %{{.*}} {
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<32xf32>) outs(%[[MEMSPACECAST_11]] : memref<32xf32>) {tiled_op}
// CHECK: } else {
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<32xf32>) outs(%[[MEMSPACECAST_13]] : memref<32xf32>) {tiled_op}
// CHECK: } {ssbuffer.intra_buffer}

// CHECK: scf.if %{{.*}} {
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<32x128xf32>) outs(%[[MEMSPACECAST]] : memref<32x128xf32>) {tiled_op}
// CHECK: } else {
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<32x128xf32>) outs(%[[MEMSPACECAST_9]] : memref<32x128xf32>) {tiled_op}
// CHECK: } {ssbuffer.intra_buffer}

// CHECK: %{{.*}} = scf.if %{{.*}} -> (tensor<32x128xf32>) {
// CHECK: %{{.*}} = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable : memref<32x128xf32>
// CHECK: scf.yield %{{.*}} : tensor<32x128xf32>
// CHECK: } else {
// CHECK: %{{.*}} = bufferization.to_tensor %[[MEMSPACECAST_9]] restrict writable : memref<32x128xf32>
// CHECK: scf.yield %{{.*}} : tensor<32x128xf32>
// CHECK: } {ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}

// CHECK: %{{.*}} = scf.if %{{.*}} -> (tensor<32xf32>) {
// CHECK: %{{.*}} = bufferization.to_tensor %[[MEMSPACECAST_11]] restrict writable : memref<32xf32>
// CHECK: scf.yield %{{.*}} : tensor<32xf32>
// CHECK: } else {
// CHECK: %{{.*}} = bufferization.to_tensor %[[MEMSPACECAST_13]] restrict writable : memref<32xf32>
// CHECK: scf.yield %{{.*}} : tensor<32xf32>
// CHECK: } {ssbuffer.intraDeps = [1 : i32, 0 : i32], ssbuffer.intra_buffer}

// CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @fwd_kernel_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %true = arith.constant true
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c576_i32 = arith.constant 576 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg17, %arg16 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = arith.muli %arg12, %c8_i32 : i32
    %8 = arith.remsi %7, %c28_i32 : i32
    %9 = arith.cmpi eq, %8, %c0_i32 : i32
    %10 = arith.select %9, %c576_i32, %c512_i32 : i32
    %11 = arith.addi %10, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %12, %c3_i32 : i32
    %14 = arith.muli %13, %arg12 : i32
    scf.for %arg18 = %6 to %14 step %c28_i32  : i32 {
      %15 = arith.divsi %arg18, %12 : i32
      %16 = arith.remsi %arg18, %12 : i32
      %17 = arith.divsi %15, %arg12 : i32
      %18 = arith.index_cast %17 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %19 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %20 = arith.addi %18, %c1 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg10 to offset: [%20], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %21 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %22 = arith.subi %21, %19 : i32
      %23 = arith.muli %16, %c64_i32 : i32
      %24 = arith.cmpi slt, %23, %22 : i32
      %25 = arith.remsi %15, %arg12 : i32
      %26 = arith.muli %25, %arg13 : i32
      %27 = arith.divsi %26, %arg12 : i32
      scf.if %24 {
        %reinterpret_cast_1 = memref.reinterpret_cast %arg11 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %28 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg11 to offset: [%20], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %29 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %30 = arith.subi %29, %28 : i32
        %31 = arith.extsi %23 : i32 to i64
        %32 = arith.extsi %30 : i32 to i64
        %33 = arith.cmpi slt, %31, %32 : i64
        scf.if %33 {
          %34 = arith.extsi %19 : i32 to i64
          %35 = arith.extsi %28 : i32 to i64
          %36 = arith.extsi %arg12 : i32 to i64
          %37 = arith.muli %34, %36 : i64
          %38 = arith.muli %37, %c64_i64 : i64
          %39 = arith.index_cast %38 : i64 to index
          %40 = arith.muli %25, %c64_i32 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = arith.addi %39, %41 : index
          %43 = arith.muli %arg12, %c64_i32 : i32
          %44 = arith.maxsi %23, %c0_i32 : i32
          %45 = arith.index_cast %44 : i32 to index
          %46 = arith.index_cast %43 : i32 to index
          %47 = arith.muli %45, %46 : index
          %48 = arith.addi %47, %42 : index
          %49 = arith.index_cast %22 : i32 to index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%48], sizes: [64, 64], strides: [%46, 1] : memref<?xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
          %50 = arith.extsi %arg13 : i32 to i64
          %51 = arith.muli %35, %50 : i64
          %52 = arith.muli %51, %c64_i64 : i64
          %53 = arith.index_cast %52 : i64 to index
          %54 = arith.muli %27, %c64_i32 : i32
          %55 = arith.index_cast %54 : i32 to index
          %56 = arith.addi %53, %55 : index
          %57 = arith.muli %arg13, %c64_i32 : i32
          %alloc = memref.alloc() : memref<64x64xf16>
          %58 = arith.divsi %47, %46 : index
          %59 = arith.subi %49, %58 : index
          %60 = arith.maxsi %59, %c0 : index
          %61 = arith.minsi %60, %c64 : index
          %62 = arith.subi %c0_i32, %23 : i32
          %63 = arith.maxsi %62, %c0_i32 : i32
          %64 = arith.index_cast %63 : i32 to index
          %65 = arith.minsi %64, %61 : index
          %66 = arith.subi %61, %65 : index
          %subview = memref.subview %reinterpret_cast_3[0, 0] [%66, 64] [1, 1] : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
          %subview_4 = memref.subview %alloc[%65, 0] [%66, 64] [1, 1] : memref<64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
          %67 = arith.cmpi slt, %66, %c64 : index
          hivm.hir.load ins(%subview : memref<?x64xf16, strided<[?, 1], offset: ?>>) outs(%subview_4 : memref<?x64xf16, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %67 : i1 eviction_policy = <EvictFirst>
          %68 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
          %alloc_5 = memref.alloc() {ssbuffer.crossDeps = [0 : i32, 1 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          %alloc_6 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x64xf32, #hivm.address_space<ub>>
          %alloc_7 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
          %69:3 = scf.for %arg19 = %23 to %30 step %c128_i32 iter_args(%arg20 = %23, %arg21 = %23, %arg22 = %23) -> (i32, i32, i32)  : i32 {
            %70 = arith.index_cast %30 : i32 to index
            %71 = arith.maxsi %arg21, %c0_i32 : i32
            %72 = arith.index_cast %71 : i32 to index
            %73 = arith.index_cast %57 : i32 to index
            %74 = arith.muli %72, %73 : index
            %75 = arith.maxsi %arg20, %c0_i32 : i32
            %76 = arith.index_cast %75 : i32 to index
            %77 = arith.muli %76, %73 : index
            %78 = arith.divsi %77, %73 : index
            %79 = arith.subi %70, %78 : index
            %80 = arith.maxsi %79, %c0 : index
            %81 = arith.minsi %80, %c128 : index
            %82 = arith.subi %c0_i32, %arg20 : i32
            %83 = arith.maxsi %82, %c0_i32 : i32
            %84 = arith.index_cast %83 : i32 to index
            %85 = arith.minsi %84, %81 : index
            %86 = arith.subi %81, %85 : index
            %87 = arith.cmpi slt, %86, %c128 : index
            %88 = arith.divsi %74, %73 : index
            %89 = arith.subi %70, %88 : index
            %90 = arith.maxsi %89, %c0 : index
            %91 = arith.minsi %90, %c128 : index
            %92 = arith.subi %c0_i32, %arg21 : i32
            %93 = arith.maxsi %92, %c0_i32 : i32
            %94 = arith.index_cast %93 : i32 to index
            %95 = arith.minsi %94, %91 : index
            %96 = arith.subi %91, %95 : index
            %97 = arith.cmpi slt, %96, %c128 : index
            %98 = arith.addi %arg20, %c128_i32 : i32
            %99 = arith.addi %arg21, %c128_i32 : i32
            %100 = arith.addi %arg22, %c128_i32 : i32
            %alloc_8 = memref.alloc() : memref<128x64xf16>
            %101 = arith.addi %77, %56 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%101], sizes: [128, 64], strides: [%73, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
            %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%86, 64] [1, 1] : memref<128x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
            %subview_11 = memref.subview %alloc_8[%85, 0] [%86, 64] [1, 1] : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
            hivm.hir.load ins(%subview_10 : memref<?x64xf16, strided<[?, 1], offset: ?>>) outs(%subview_11 : memref<?x64xf16, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %87 : i1 eviction_policy = <EvictFirst>
            %102 = bufferization.to_tensor %alloc_8 restrict writable : memref<128x64xf16>
            %103 = tensor.empty() : tensor<64x128xf32>
            %104 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose} ins(%68, %102, %true, %c64, %c64, %c128 : tensor<64x64xf16>, tensor<128x64xf16>, i1, index, index, index) outs(%103 : tensor<64x128xf32>) -> tensor<64x128xf32>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%104 : tensor<64x128xf32>) outs(%alloc_7 : memref<64x128xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
            %105 = hivm.hir.convert_layout %alloc_5 output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.crossDeps = [0 : i32, 0 : i32]} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %105 : memref<64x128xf16, #hivm.address_space<cbuf>> to memref<64x128xf16>
            %106 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf16>
            %alloc_12 = memref.alloc() : memref<128x64xf16>
            %107 = arith.addi %74, %56 : index
            %reinterpret_cast_13 = memref.reinterpret_cast %arg4 to offset: [%107], sizes: [128, 64], strides: [%73, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
            %subview_14 = memref.subview %reinterpret_cast_13[0, 0] [%96, 64] [1, 1] : memref<128x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
            %subview_15 = memref.subview %alloc_12[%95, 0] [%96, 64] [1, 1] : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
            hivm.hir.load ins(%subview_14 : memref<?x64xf16, strided<[?, 1], offset: ?>>) outs(%subview_15 : memref<?x64xf16, strided<[64, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %97 : i1 eviction_policy = <EvictFirst>
            %108 = bufferization.to_tensor %alloc_12 restrict writable : memref<128x64xf16>
            %109 = tensor.empty() : tensor<64x64xf32>
            %110 = hivm.hir.mmadL1 {already_set_real_mkn} ins(%106, %108, %true, %c64, %c128, %c64 : tensor<64x128xf16>, tensor<128x64xf16>, i1, index, index, index) outs(%109 : tensor<64x64xf32>) -> tensor<64x64xf32>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%110 : tensor<64x64xf32>) outs(%alloc_6 : memref<64x64xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
            scf.yield %98, %99, %100 : i32, i32, i32
          } {ssbuffer.main_loop = 0 : i32, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        }
      }
    }
    return
  }
  func.func @fwd_kernel_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant -1.07374182E+9 : f32
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c576_i32 = arith.constant 576 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %cst_1 = arith.constant 0xFF800000 : f32
    %c32 = arith.constant 32 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg17, %arg16 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg15 : i32
    %7 = tensor.empty() : tensor<64xf32>
    %8 = hivm.hir.vbrc ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
    %9 = tensor.empty() : tensor<64x128xf32>
    %10 = tensor.empty() : tensor<64x64xf32>
    %11 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %12 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
    %13 = arith.muli %arg12, %c8_i32 : i32
    %14 = arith.remsi %13, %c28_i32 : i32
    %15 = arith.cmpi eq, %14, %c0_i32 : i32
    %16 = arith.select %15, %c576_i32, %c512_i32 : i32
    %17 = arith.addi %16, %c63_i32 : i32
    %18 = arith.divsi %17, %c64_i32 : i32
    %19 = arith.muli %18, %c3_i32 : i32
    %20 = arith.muli %19, %arg12 : i32
    scf.for %arg18 = %6 to %20 step %c28_i32  : i32 {
      %21 = arith.divsi %arg18, %18 : i32
      %22 = arith.remsi %arg18, %18 : i32
      %23 = arith.divsi %21, %arg12 : i32
      %24 = arith.remsi %21, %arg12 : i32
      %25 = arith.index_cast %23 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%25], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %26 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %27 = arith.addi %25, %c1 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg10 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %28 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %29 = arith.subi %28, %26 : i32
      %30 = arith.muli %22, %c64_i32 : i32
      %31 = arith.cmpi slt, %30, %29 : i32
      scf.if %31 {
        %reinterpret_cast_3 = memref.reinterpret_cast %arg11 to offset: [%25], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %32 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg11 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %33 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %34 = arith.subi %33, %32 : i32
        %35 = arith.extsi %30 : i32 to i64
        %36 = arith.extsi %34 : i32 to i64
        %37 = arith.cmpi slt, %35, %36 : i64
        scf.if %37 {
          %38 = arith.muli %arg12, %c64_i32 : i32
          %39 = arith.maxsi %30, %c0_i32 : i32
          %40 = arith.index_cast %39 : i32 to index
          %41 = arith.index_cast %38 : i32 to index
          %42 = arith.muli %40, %41 : index
          %43 = arith.index_cast %29 : i32 to index
          %44 = arith.muli %23, %16 : i32
          %45 = arith.muli %44, %c512_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = arith.divsi %42, %41 : index
          %48 = arith.subi %43, %47 : index
          %49 = arith.maxsi %48, %c0 : index
          %50 = arith.minsi %49, %c64 : index
          %51 = arith.subi %c0_i32, %30 : i32
          %52 = arith.maxsi %51, %c0_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = arith.minsi %53, %50 : index
          %55 = arith.subi %50, %54 : index
          %alloc = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_5 = memref.alloc() {ssbuffer.crossDeps = [1 : i32, 1 : i32]} : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
          %alloc_6 = memref.alloc() {ssbuffer.crossDeps = [2 : i32, 1 : i32]} : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          %alloc_7 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<64x128xf32, #hivm.address_space<ub>>
          %memspacecast = memref.memory_space_cast %alloc_7 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
          %alloc_8 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<64x128xf32, #hivm.address_space<ub>>
          %memspacecast_9 = memref.memory_space_cast %alloc_8 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
          %alloc_10 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<64xf32, #hivm.address_space<ub>>
          %memspacecast_11 = memref.memory_space_cast %alloc_10 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
          %alloc_12 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>} : memref<64xf32, #hivm.address_space<ub>>
          %memspacecast_13 = memref.memory_space_cast %alloc_12 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
          %56:6 = scf.for %arg19 = %30 to %34 step %c128_i32 iter_args(%arg20 = %11, %arg21 = %12, %arg22 = %8, %arg23 = %30, %arg24 = %30, %arg25 = %30) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, i32)  : i32 {
            %83 = arith.maxsi %arg25, %c0_i32 : i32
            %84 = arith.index_cast %83 : i32 to index
            %85 = arith.muli %40, %c512 : index
            %86 = arith.addi %85, %46 : index
            %87 = arith.index_cast %34 : i32 to index
            %88 = arith.addi %86, %84 {ssbuffer.dep_mark = [1 : i32]} : index
            %89 = arith.subi %88, %46 : index
            %90 = arith.divsi %89, %c512 : index
            %91 = arith.subi %43, %90 : index
            %92 = arith.maxsi %91, %c0 : index
            %93 = arith.minsi %92, %c64 : index
            %94 = arith.remsi %89, %c512 : index
            %95 = arith.subi %87, %94 : index
            %96 = arith.maxsi %95, %c0 : index
            %97 = arith.minsi %96, %c128 : index
            %98 = arith.minsi %53, %93 {ssbuffer.dep_mark = [4 : i32]} : index
            %99 = arith.subi %93, %98 {ssbuffer.dep_mark = [2 : i32]} : index
            %100 = arith.subi %c0_i32, %arg25 : i32
            %101 = arith.maxsi %100, %c0_i32 : i32
            %102 = arith.index_cast %101 : i32 to index
            %103 = arith.minsi %102, %97 {ssbuffer.dep_mark = [5 : i32]} : index
            %104 = arith.subi %97, %103 {ssbuffer.dep_mark = [3 : i32]} : index
            %105 = arith.cmpi slt, %99, %c64 : index
            %106 = arith.cmpi slt, %104, %c128 : index
            %107 = arith.ori %105, %106 : i1
            %108 = arith.addi %arg23, %c128_i32 : i32
            %109 = arith.addi %arg24, %c128_i32 : i32
            %110 = arith.addi %arg25, %c128_i32 : i32
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
            %memspacecast_18 = memref.memory_space_cast %alloc_6 {ssbuffer.crossDeps = [2 : i32, 0 : i32]} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
            %111 = bufferization.to_tensor %memspacecast_18 restrict writable : memref<64x128xf32>
            %alloc_19 = memref.alloc() : memref<64x128xi8>
            %reinterpret_cast_20 = memref.reinterpret_cast %arg9 to offset: [%88], sizes: [64, 128], strides: [512, 1] {ssbuffer.dep_mark = [1 : i32]} : memref<?xi8> to memref<64x128xi8, strided<[512, 1], offset: ?>>
            %subview_21 = memref.subview %reinterpret_cast_20[0, 0] [%99, %104] [1, 1] {ssbuffer.dep_mark = [2 : i32, 3 : i32]} : memref<64x128xi8, strided<[512, 1], offset: ?>> to memref<?x?xi8, strided<[512, 1], offset: ?>>
            %subview_22 = memref.subview %alloc_19[%98, %103] [%99, %104] [1, 1] {ssbuffer.dep_mark = [2 : i32, 3 : i32, 4 : i32, 5 : i32]} : memref<64x128xi8> to memref<?x?xi8, strided<[128, 1], offset: ?>>
            %112 = arith.remui %103, %c32 : index
            hivm.hir.load ins(%subview_21 : memref<?x?xi8, strided<[512, 1], offset: ?>>) outs(%subview_22 : memref<?x?xi8, strided<[128, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %112 : index init_out_buffer = true init_condition = %107 : i1 eviction_policy = <EvictFirst>
            %113 = bufferization.to_tensor %alloc_19 restrict writable : memref<64x128xi8>
            %114 = hivm.hir.vmul ins(%111, %arg14 : tensor<64x128xf32>, f32) outs(%9 : tensor<64x128xf32>) -> tensor<64x128xf32>
            %115 = tensor.empty() : tensor<64x128xi1>
            %116 = hivm.hir.vcmp ins(%113, %c0_i8 : tensor<64x128xi8>, i8) outs(%115 : tensor<64x128xi1>) compare_mode = <ne> -> tensor<64x128xi1>
            %117 = hivm.hir.vsel ins(%116, %cst_0, %cst : tensor<64x128xi1>, f32, f32) outs(%9 : tensor<64x128xf32>) -> tensor<64x128xf32>
            %118 = hivm.hir.vadd ins(%114, %117 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%9 : tensor<64x128xf32>) -> tensor<64x128xf32>
            %119 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
            %expanded_23 = tensor.expand_shape %119 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
            %120 = hivm.hir.vreduce <max> ins(%118 : tensor<64x128xf32>) outs(%expanded_23 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
            %collapsed = tensor.collapse_shape %120 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
            %121 = hivm.hir.vmax ins(%arg22, %collapsed : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
            %122 = arith.subi %arg19, %30 : i32
            %123 = arith.divui %122, %c128_i32 : i32
            %124 = arith.remsi %123, %c2_i32 : i32
            %125 = arith.cmpi eq, %124, %c0_i32 : i32
            scf.if %125 {
              hivm.hir.copy ins(%121 : tensor<64xf32>) outs(%memspacecast_11 : memref<64xf32>)
            } else {
              hivm.hir.copy ins(%121 : tensor<64xf32>) outs(%memspacecast_13 : memref<64xf32>)
            } {ssbuffer.intra_buffer}
            %expanded_24 = tensor.expand_shape %121 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
            %126 = hivm.hir.vsub ins(%118, %expanded_24 : tensor<64x128xf32>, tensor<64x1xf32>) outs(%9 : tensor<64x128xf32>) broadcast = [1] -> tensor<64x128xf32>
            %127 = hivm.hir.vexp ins(%126 : tensor<64x128xf32>) outs(%9 : tensor<64x128xf32>) -> tensor<64x128xf32>
            scf.if %125 {
              hivm.hir.copy ins(%127 : tensor<64x128xf32>) outs(%memspacecast : memref<64x128xf32>)
            } else {
              hivm.hir.copy ins(%127 : tensor<64x128xf32>) outs(%memspacecast_9 : memref<64x128xf32>)
            } {ssbuffer.intra_buffer}
            %128 = tensor.empty() : tensor<64x128xf16>
            %129 = hivm.hir.vcast ins(%127 : tensor<64x128xf32>) outs(%128 : tensor<64x128xf16>) -> tensor<64x128xf16>
            %expanded_25 = tensor.expand_shape %129 [[0], [1, 2]] output_shape [64, 8, 16] : tensor<64x128xf16> into tensor<64x8x16xf16>
            %130 = tensor.empty() : tensor<8x64x16xf16>
            %131 = hivm.hir.vtranspose ins(%expanded_25 : tensor<64x8x16xf16>) outs(%130 : tensor<8x64x16xf16>) permutation = [1, 0, 2] -> tensor<8x64x16xf16>
            %expanded_26 = tensor.expand_shape %131 [[0], [1, 2], [3]] output_shape [8, 4, 16, 16] : tensor<8x64x16xf16> into tensor<8x4x16x16xf16>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
            hivm.hir.copy ins(%expanded_26 : tensor<8x4x16x16xf16>) outs(%alloc : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
            %memspacecast_27 = memref.memory_space_cast %alloc_5 {ssbuffer.crossDeps = [1 : i32, 0 : i32]} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
            %132 = bufferization.to_tensor %memspacecast_27 restrict writable : memref<64x64xf32>
            %133 = scf.if %125 -> (tensor<64x128xf32>) {
              %142 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
              scf.yield %142 : tensor<64x128xf32>
            } else {
              %142 = bufferization.to_tensor %memspacecast_9 restrict writable : memref<64x128xf32>
              scf.yield %142 : tensor<64x128xf32>
            } {ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}
            %expanded_28 = tensor.expand_shape %12 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
            %134 = hivm.hir.vreduce <sum> ins(%133 : tensor<64x128xf32>) outs(%expanded_28 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
            %collapsed_29 = tensor.collapse_shape %134 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
            %135 = scf.if %125 -> (tensor<64xf32>) {
              %142 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<64xf32>
              scf.yield %142 : tensor<64xf32>
            } else {
              %142 = bufferization.to_tensor %memspacecast_13 restrict writable : memref<64xf32>
              scf.yield %142 : tensor<64xf32>
            } {ssbuffer.intraDeps = [1 : i32, 0 : i32], ssbuffer.intra_buffer}
            %136 = hivm.hir.vsub ins(%arg22, %135 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
            %137 = hivm.hir.vexp ins(%136 : tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
            %expanded_30 = tensor.expand_shape %137 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
            %138 = hivm.hir.vmul ins(%arg20, %expanded_30 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%10 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
            %139 = hivm.hir.vadd ins(%132, %138 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %140 = hivm.hir.vmul ins(%arg21, %137 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
            %141 = hivm.hir.vadd ins(%140, %collapsed_29 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
            scf.yield %139, %141, %121, %108, %109, %110 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, i32
          } {ssbuffer.main_loop = 0 : i32, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          %57 = arith.extsi %26 : i32 to i64
          %58 = arith.extsi %arg12 : i32 to i64
          %59 = arith.muli %57, %58 : i64
          %60 = arith.muli %59, %c64_i64 : i64
          %61 = arith.index_cast %60 : i64 to index
          %62 = arith.muli %24, %c64_i32 : i32
          %63 = arith.index_cast %62 : i32 to index
          %64 = arith.addi %61, %63 : index
          %65 = arith.addi %42, %64 : index
          %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [64, 64], strides: [%41, 1] : memref<?xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
          %66 = arith.index_cast %59 : i64 to index
          %67 = arith.index_cast %24 : i32 to index
          %68 = arith.addi %66, %67 : index
          %69 = arith.index_cast %arg12 : i32 to index
          %70 = arith.muli %40, %69 : index
          %71 = arith.addi %70, %68 : index
          %reinterpret_cast_15 = memref.reinterpret_cast %arg6 to offset: [%71], sizes: [64], strides: [%69] : memref<?xf32> to memref<64xf32, strided<[?], offset: ?>>
          %expanded = tensor.expand_shape %56#1 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
          %72 = hivm.hir.vdiv ins(%56#0, %expanded : tensor<64x64xf32>, tensor<64x1xf32>) outs(%10 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
          %73 = hivm.hir.vln ins(%56#1 : tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
          %74 = hivm.hir.vadd ins(%56#2, %73 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
          %75 = tensor.empty() : tensor<64x64xf16>
          %76 = hivm.hir.vcast ins(%72 : tensor<64x64xf32>) outs(%75 : tensor<64x64xf16>) -> tensor<64x64xf16>
          %extracted_slice = tensor.extract_slice %76[%54, 0] [%55, 64] [1, 1] : tensor<64x64xf16> to tensor<?x64xf16>
          %subview = memref.subview %reinterpret_cast_14[0, 0] [%55, 64] [1, 1] : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
          hivm.hir.store ins(%extracted_slice : tensor<?x64xf16>) outs(%subview : memref<?x64xf16, strided<[?, 1], offset: ?>>)
          %77 = arith.divsi %70, %69 : index
          %78 = arith.subi %43, %77 : index
          %79 = arith.maxsi %78, %c0 : index
          %80 = arith.minsi %79, %c64 : index
          %81 = arith.minsi %53, %80 : index
          %82 = arith.subi %80, %81 : index
          %extracted_slice_16 = tensor.extract_slice %74[%81] [%82] [1] : tensor<64xf32> to tensor<?xf32>
          %subview_17 = memref.subview %reinterpret_cast_15[0] [%82] [1] : memref<64xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
          hivm.hir.store ins(%extracted_slice_16 : tensor<?xf32>) outs(%subview_17 : memref<?xf32, strided<[?], offset: ?>>)
        }
      }
    }
    return
  }
}

// -----
 	 
 	 // CHECK-LABEL:   func.func @contiguous_load_dot_to_indirect_store_kernel_mix_aiv(
 	 // CHECK:           hivm.hir.sync_block_wait
 	 // CHECK:           hivm.hir.vcast ins(%{{.*}} : memref<16x64xf32, #hivm.address_space<ub>>) outs(%{{.*}} : memref<16x64xf16, #hivm.address_space<ub>>)
 	 // CHECK:           bufferization.to_tensor %{{.*}} restrict writable : memref<16x64xf16>
 	 // CHECK:           hivm.hir.get_sub_block_idx
 	 // CHECK:           scf.if %{{.*}} {
 	 // CHECK:             hivm.hir.indirect_store ins(%{{.*}} : tensor<16x64xf16>, %{{.*}} : tensor<16x64xi64>, %{{.*}} : tensor<16x64xi8>) outs(%arg4 : memref<?xf16>) {hivm.vf_mode = #hivm.vf_mode<SIMT>}
 	 // CHECK:           } {limit_sub_block_id0}
 	 func.func @contiguous_load_dot_to_indirect_store_kernel_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {DirectlyUsedGMArgIdxList = [4], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<MIX>, mix_mode = "mix", parallel_mode = "mix_simd_simt"} {
 	     %cst = arith.constant 0.000000e+00 : f16
 	     %c0 = arith.constant 0 : index
 	     %c1 = arith.constant 1 : index
 	     %c16_i64 = arith.constant 16 : i64
 	     %c64_i64 = arith.constant 64 : i64
 	     %c16_i32 = arith.constant 16 : i32
 	     hivm.hir.set_ctrl false at ctrl[60]
 	     hivm.hir.set_ctrl true at ctrl[48]
 	     %0 = arith.muli %arg6, %arg7 : i32
 	     %1 = arith.muli %0, %arg8 : i32
 	     annotation.mark %1 {logical_block_num} : i32
 	     %2 = hivm.hir.get_block_idx -> i64
 	     %3 = arith.trunci %2 : i64 to i32
 	     %4 = arith.muli %arg8, %arg7 : i32
 	     %5 = arith.divsi %3, %4 : i32
 	     %6 = arith.remsi %5, %arg6 : i32
 	     %7 = tensor.empty() : tensor<16x1xi64>
 	     %8 = arith.muli %6, %c16_i32 : i32
 	     %9 = tensor.empty() : tensor<64xi32>
 	     %10 = hivm.hir.varange offset[%c0] strides[%c1] outs(%9 : tensor<64xi32>) -> tensor<64xi32>
 	     %11 = tensor.empty() : tensor<64xi64>
 	     %12 = hivm.hir.vcast ins(%10 : tensor<64xi32>) outs(%11 : tensor<64xi64>) -> tensor<64xi64>
 	     %13 = tensor.empty() : tensor<16x64xf16>
 	     %alloc = memref.alloc() : memref<16x64xf16, #hivm.address_space<ub>>
 	     annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x64xf16, #hivm.address_space<ub>>
 	     %memspacecast = memref.memory_space_cast %alloc : memref<16x64xf16, #hivm.address_space<ub>> to memref<16x64xf16>
 	     %alloc_0 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
 	     annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x64xf32, #hivm.address_space<ub>>
 	     hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
 	     hivm.hir.vcast ins(%alloc_0 : memref<16x64xf32, #hivm.address_space<ub>>) outs(%alloc : memref<16x64xf16, #hivm.address_space<ub>>)
 	     %14 = bufferization.to_tensor %memspacecast restrict writable : memref<16x64xf16>
 	     %15 = arith.index_cast %8 : i32 to index
 	     %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [16], strides: [1] : memref<?xi64> to memref<16xi64, strided<[1], offset: ?>>
 	     %alloc_1 = memref.alloc() : memref<16xi64>
 	     hivm.hir.load ins(%reinterpret_cast : memref<16xi64, strided<[1], offset: ?>>) outs(%alloc_1 : memref<16xi64>) eviction_policy = <EvictFirst>
 	     %16 = bufferization.to_tensor %alloc_1 restrict writable : memref<16xi64>
 	     %17 = tensor.empty() : tensor<16xi1>
 	     %18 = hivm.hir.vcmp ins(%16, %c16_i64 : tensor<16xi64>, i64) outs(%17 : tensor<16xi1>) compare_mode = <lt> -> tensor<16xi1>
 	     %expanded = tensor.expand_shape %16 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
 	     %19 = hivm.hir.vmul ins(%expanded, %c64_i64 : tensor<16x1xi64>, i64) outs(%7 : tensor<16x1xi64>) -> tensor<16x1xi64>
 	     %expanded_2 = tensor.expand_shape %12 [[0, 1]] output_shape [1, 64] : tensor<64xi64> into tensor<1x64xi64>
 	     %20 = tensor.empty() : tensor<16x64xi64>
 	     %21 = hivm.hir.vbrc ins(%19 : tensor<16x1xi64>) outs(%20 : tensor<16x64xi64>) broadcast_dims = [1] -> tensor<16x64xi64>
 	     %22 = hivm.hir.vbrc ins(%expanded_2 : tensor<1x64xi64>) outs(%20 : tensor<16x64xi64>) broadcast_dims = [0] -> tensor<16x64xi64>
 	     %23 = hivm.hir.vadd ins(%21, %22 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%20 : tensor<16x64xi64>) -> tensor<16x64xi64>
 	     %24 = tensor.empty() : tensor<1x64xi1>
 	     %25 = hivm.hir.vcmp ins(%expanded_2, %c64_i64 : tensor<1x64xi64>, i64) outs(%24 : tensor<1x64xi1>) compare_mode = <lt> -> tensor<1x64xi1>
 	     %26 = tensor.empty() : tensor<16x64xi1>
 	     %27 = tensor.empty() : tensor<16xf16>
 	     %28 = hivm.hir.vcast ins(%18 : tensor<16xi1>) outs(%27 : tensor<16xf16>) round_mode = <trunc> -> tensor<16xf16>
 	     %expanded_3 = tensor.expand_shape %28 [[0, 1]] output_shape [16, 1] : tensor<16xf16> into tensor<16x1xf16>
 	     %29 = hivm.hir.vbrc ins(%expanded_3 : tensor<16x1xf16>) outs(%13 : tensor<16x64xf16>) broadcast_dims = [1] -> tensor<16x64xf16>
 	     %30 = hivm.hir.vcmp ins(%29, %cst : tensor<16x64xf16>, f16) outs(%26 : tensor<16x64xi1>) compare_mode = <ne> -> tensor<16x64xi1>
 	     %collapsed = tensor.collapse_shape %25 [[0, 1]] : tensor<1x64xi1> into tensor<64xi1>
 	     %31 = tensor.empty() : tensor<64xf16>
 	     %32 = hivm.hir.vcast ins(%collapsed : tensor<64xi1>) outs(%31 : tensor<64xf16>) round_mode = <trunc> -> tensor<64xf16>
 	     %expanded_4 = tensor.expand_shape %32 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
 	     %33 = hivm.hir.vbrc ins(%expanded_4 : tensor<1x64xf16>) outs(%13 : tensor<16x64xf16>) broadcast_dims = [0] -> tensor<16x64xf16>
 	     %34 = hivm.hir.vcmp ins(%33, %cst : tensor<16x64xf16>, f16) outs(%26 : tensor<16x64xi1>) compare_mode = <ne> -> tensor<16x64xi1>
 	     %35 = hivm.hir.vand ins(%30, %34 : tensor<16x64xi1>, tensor<16x64xi1>) outs(%26 : tensor<16x64xi1>) -> tensor<16x64xi1>
 	     %36 = tensor.empty() : tensor<16x64xi8>
 	     %37 = hivm.hir.vcast ins(%35 : tensor<16x64xi1>) outs(%36 : tensor<16x64xi8>) -> tensor<16x64xi8>
 	     hivm.hir.indirect_store ins(%14 : tensor<16x64xf16>, %23 : tensor<16x64xi64>, %37 : tensor<16x64xi8>) outs(%arg4 : memref<?xf16>) {hivm.vf_mode = #hivm.vf_mode<SIMT>}
 	     return
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
// CHECK: hivm.hir.vreduce <sum> ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0]
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
