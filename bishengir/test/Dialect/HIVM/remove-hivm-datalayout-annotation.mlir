// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-remove-layout-annotation -split-input-file | FileCheck %s

// -----
// CHECK-LABEL:   func.func @triton_conv1d_3d_fp16_aligned_groups_mix_aic
// CHECK:           %{{.*}} = memref_ext.alloc_workspace() from %{{.*}} offset = [%{{.*}}] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x1x128x16xf16, #hivm.address_space<gm>>
// CHECK:           annotation.mark %{{.*}} : memref<2x2x1x128x16xf16, #hivm.address_space<gm>>
// CHECK:           %{{.*}} = memref_ext.alloc_workspace() from %{{.*}} offset = [%{{.*}}] : from memref<?xi8, #hivm.address_space<gm>> to memref<1x1x3x32x16xf16, #hivm.address_space<gm>>
// CHECK:           annotation.mark %{{.*}} : memref<1x1x3x32x16xf16, #hivm.address_space<gm>>
#map = affine_map<()[s0, s1] -> ((s0 + 15) floordiv 16)>
#map1 = affine_map<()[s0, s1] -> ((s1 + 15) floordiv 16)>
#map2 = affine_map<()[s0, s1] -> (s0 floordiv 16)>
#map3 = affine_map<()[s0, s1] -> (s0 mod 16)>
#map4 = affine_map<()[s0, s1] -> (s1 floordiv 16)>
#map5 = affine_map<()[s0, s1] -> (s1 mod 16)>
func.func @triton_conv1d_3d_fp16_aligned_groups_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c16384 = arith.constant 16384 : index
  %c19456 = arith.constant 19456 : index
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x1x128x16xf16, #hivm.address_space<gm>>
  annotation.mark %2 {layout = #hivm.data_layout<NC1HWC0>} : memref<2x2x1x128x16xf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x1x128x16xf16, #hivm.address_space<cbuf>>
  hivm.hir.load ins(%2 : memref<2x2x1x128x16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<2x2x1x128x16xf16, #hivm.address_space<cbuf>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %3 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<1x1x3x32x16xf16, #hivm.address_space<gm>>
  annotation.mark %3 {layout = #hivm.data_layout<C1HWNC0>} : memref<1x1x3x32x16xf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x3x32x16xf16, #hivm.address_space<cbuf>>
  hivm.hir.load ins(%3 : memref<1x1x3x32x16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<1x1x3x32x16xf16, #hivm.address_space<cbuf>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %4 = affine.apply #map()[%c128, %c64]
  %5 = affine.apply #map1()[%c128, %c64]
  %c16 = arith.constant 16 : index
  %c16_1 = arith.constant 16 : index
  %alloc_2 = memref.alloc(%5, %4, %c16, %c16_1) {alignment = 64 : i64} : memref<?x?x?x?xf32, #hivm.address_space<cc>>
  hivm.hir.Conv1dL1 {fixpipe_already_inserted = true, groups = 2 : i32, outputAlreadyNormalized, padding = 0 : i32} ins(%alloc, %alloc_0, %true : memref<2x2x1x128x16xf16, #hivm.address_space<cbuf>>, memref<1x1x3x32x16xf16, #hivm.address_space<cbuf>>, i1) outs(%alloc_2 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  %c126 = arith.constant 126 : index
  %c64_3 = arith.constant 64 : index
  %6 = affine.apply #map()[%c126, %c64_3]
  %7 = affine.apply #map1()[%c126, %c64_3]
  %c16_4 = arith.constant 16 : index
  %c16_5 = arith.constant 16 : index
  %c0_6 = arith.constant 0 : index
  %c0_7 = arith.constant 0 : index
  %8 = affine.apply #map2()[%c0_6, %c0_7]
  %9 = affine.apply #map3()[%c0_6, %c0_7]
  %10 = affine.apply #map4()[%c0_6, %c0_7]
  %11 = affine.apply #map5()[%c0_6, %c0_7]
  %subview = memref.subview %alloc_2[%10, %8, %9, %11] [%7, %6, %c16_4, %c16_5] [1, 1, 1, 1] : memref<?x?x?x?xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
  %12 = memref_ext.alloc_workspace() from %arg2 offset = [%c19456] : from memref<?xi8, #hivm.address_space<gm>> to memref<126x64xf16, #hivm.address_space<gm>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%12 : memref<126x64xf16, #hivm.address_space<gm>>)
  annotation.mark %12 : memref<126x64xf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  return
}
