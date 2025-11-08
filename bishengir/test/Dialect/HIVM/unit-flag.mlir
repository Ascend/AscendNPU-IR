// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-inject-sync{enable-unit-flag=true}))" | FileCheck %s --check-prefixes="CHECK,CHECK-UF-ON"
// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-inject-sync{enable-unit-flag=false}))" | FileCheck %s --check-prefixes="CHECK,CHECK-UF-OFF"

// CHECK: @_attn_fwd_mix_aic
func.func @_attn_fwd_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.shape_0 = 0 : i32, tt.shape_1 = 0 : i32, tt.shape_2 = 0 : i32, tt.shape_3 = 0 : i32, tt.shape_4 = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.shape_0 = 0 : i32, tt.shape_1 = 0 : i32, tt.shape_2 = 0 : i32, tt.shape_3 = 0 : i32, tt.shape_4 = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
  %c442368_i64 = arith.constant 442368 : i64
  %c278528_i64 = arith.constant 278528 : i64
  %c147456_i64 = arith.constant 147456 : i64
  %c311296_i64 = arith.constant 311296 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c294912_i64 = arith.constant 294912 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c131072_i64 = arith.constant 131072 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c32_i32 = arith.constant 32 : i32
  %true = arith.constant true
  %c32 = arith.constant 32 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg10, %arg9 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg8 : i32
  hivm.hir.set_mask_norm
  %5 = arith.muli %4, %c32_i32 : i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
  %6 = arith.index_cast %5 : i32 to index
  %7 = hivm.hir.pointer_cast(%c0_i64) : memref<16x2x16x16xf32, #hivm.address_space<cc>>
  %cast = memref.cast %7 : memref<16x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  %8 = arith.index_cast %0 : i64 to index
  %9 = affine.apply affine_map<()[s0] -> (s0 * 81920)>()[%8]
  %view = memref.view %arg1[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x256xf32, #hivm.address_space<gm>>
  %10 = affine.apply affine_map<()[s0] -> (s0 * 81920 + 32768)>()[%8]
  %view_0 = memref.view %arg1[%10][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x256xf16, #hivm.address_space<gm>>
  %11 = affine.apply affine_map<()[s0] -> (s0 * 81920 + 49152)>()[%8]
  %view_1 = memref.view %arg1[%11][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x256xf32, #hivm.address_space<gm>>
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
//   CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
    %12 = arith.divsi %arg11, %c2_i32 : i32
    %13 = arith.remsi %arg11, %c2_i32 : i32
    %14 = arith.extsi %12 : i32 to i64
    %15 = arith.muli %14, %c131072_i64 : i64
    %16 = arith.extsi %13 : i32 to i64
    %17 = arith.muli %16, %c65536_i64 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%19, %6]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 256], strides: [256, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%19], sizes: [256, 256], strides: [256, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [256, 256], strides: [256, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %21 = hivm.hir.pointer_cast(%c0_i64, %c294912_i64) : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    %cast_4 = memref.cast %21 : memref<16x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    %22 = hivm.hir.pointer_cast(%c16384_i64, %c311296_i64) : memref<16x16x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<16x16x16x16xf16, #hivm.address_space<cbuf>>
    %cast_5 = memref.cast %22 : memref<16x16x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag[<ENABLED_WITH_UPDATE>]
    hivm.hir.mmadL1 {b_transpose} ins(%cast_4, %cast_5, %true, %c32, %c256, %c256 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    // CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag[<ENABLED_WITH_UPDATE>]
    hivm.hir.fixpipe {enable_nz2nd} ins(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<32x256xf32, #hivm.address_space<gm>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    annotation.mark %view : memref<32x256xf32, #hivm.address_space<gm>>
    annotation.mark %view : memref<32x256xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
    %23 = hivm.hir.pointer_cast(%c147456_i64, %c0_i64) : memref<16x16x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<16x16x16x16xf16, #hivm.address_space<cbuf>>
    %cast_6 = memref.cast %23 : memref<16x16x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_2 : memref<256x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
    %24 = hivm.hir.pointer_cast(%c278528_i64, %c442368_i64) : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %24 : memref<16x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%view_0 : memref<32x256xf16, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag[<ENABLED_WITH_UPDATE>]
    hivm.hir.mmadL1 ins(%cast_7, %cast_6, %true, %c32, %c256, %c256 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    // CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag[<ENABLED_WITH_UPDATE>]
    hivm.hir.fixpipe {enable_nz2nd} ins(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_1 : memref<32x256xf32, #hivm.address_space<gm>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    annotation.mark %view_1 : memref<32x256xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
  }
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
//   CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
  return
}

// -----

// CHECK: @matmul_x_w_bias_down_up_fused_layer_1_kernel_mix_aic
func.func @matmul_x_w_bias_down_up_fused_layer_1_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<19xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
  %c12288_i64 = arith.constant 12288 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c1_i32 = arith.constant 1 : i32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c31_i32 = arith.constant 31 : i32
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.divsi %1, %arg18 : i32
  %3 = arith.remsi %2, %arg17 : i32
  %4 = arith.muli %arg18, %arg17 : i32
  %5 = arith.divsi %1, %4 : i32
  %6 = arith.remsi %5, %arg16 : i32
  hivm.hir.set_mask_norm
  %7 = arith.muli %6, %c32_i32 : i32
  %8 = arith.muli %3, %c32_i32 : i32
  %9 = arith.index_cast %7 : i32 to index
  %10 = arith.index_cast %arg11 : i32 to index
  %11 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%9, %10]
  %12 = arith.index_cast %arg12 : i32 to index
  %13 = arith.index_cast %8 : i32 to index
  %14 = arith.index_cast %arg13 : i32 to index
  %15 = arith.addi %arg10, %c31_i32 : i32
  %16 = arith.divsi %15, %c32_i32 : i32
  %17 = arith.muli %arg12, %c32_i32 : i32
  %18 = arith.muli %arg13, %c32_i32 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [%10, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %cast = memref.cast %reinterpret_cast : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [32, 32], strides: [%12, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %cast_1 = memref.cast %reinterpret_cast_0 : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32, 64], strides: [%14, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, 1]>, #hivm.address_space<gm>>
  %cast_3 = memref.cast %reinterpret_cast_2 : memref<32x64xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %19 = hivm.hir.pointer_cast(%c8192_i64) : memref<2x2x16x16xf32, #hivm.address_space<cc>>
  %cast_4 = memref.cast %19 : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  %20 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
  %cast_5 = memref.cast %20 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  %21 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%9]
  %22 = arith.index_cast %arg8 : i32 to index
  %23 = arith.maxsi %9, %22 : index
  %24 = arith.minsi %21, %23 : index
  %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %9]
  %26 = arith.index_cast %arg10 : i32 to index
  %27 = arith.minsi %25, %c32 : index
  %28 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%13]
  %29 = arith.index_cast %arg9 : i32 to index
  %30 = arith.maxsi %13, %29 : index
  %31 = arith.minsi %28, %30 : index
  %32 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%31, %13]
  %33 = arith.minsi %32, %c32 : index
  %34 = arith.index_cast %17 : i32 to index
  %35 = arith.index_cast %18 : i32 to index
  %36:9 = scf.for %arg19 = %c0_i32 to %16 step %c1_i32 iter_args(%arg20 = %cast, %arg21 = %cast_1, %arg22 = %cast_3, %arg23 = %11, %arg24 = %c0, %arg25 = %13, %arg26 = %c0, %arg27 = %c0, %arg28 = %c0) -> (memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, index, index, index, index, index)  : i32 {
    %45 = arith.muli %arg19, %c32_i32 : i32
    %46 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    %cast_13 = memref.cast %46 : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    %47 = arith.index_cast %45 : i32 to index
    %48 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%47]
    %49 = arith.maxsi %47, %26 : index
    %50 = arith.minsi %48, %49 : index
    %51 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %47]
    %52 = arith.minsi %51, %c32 : index
    %subview_14 = memref.subview %arg20[0, 0] [%27, %52] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %53 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%27]
    %54 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%52]
    %subview_15 = memref.subview %46[0, 0, 0, 0] [%54, %53, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
    %cast_16 = memref.cast %subview_15 : memref<?x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%subview_14 : memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %55 = hivm.hir.pointer_cast(%c2048_i64, %c18432_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %55 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    %cast_17 = memref.cast %55 : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    %subview_18 = memref.subview %arg21[0, 0] [%52, %33] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %56 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%33]
    %subview_19 = memref.subview %55[0, 0, 0, 0] [%56, %54, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
    %cast_20 = memref.cast %subview_19 : memref<?x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%subview_18 : memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%cast_20 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %57 = hivm.hir.pointer_cast(%c4096_i64, %c20480_i64) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %57 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
    %cast_21 = memref.cast %57 : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
    %subview_22 = memref.subview %arg22[0, 0] [%51, 64] [1, 1] : memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %58 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%50, %47]
    %subview_23 = memref.subview %57[0, 0, 0, 0] [4, %58, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
    %cast_24 = memref.cast %subview_23 : memref<4x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%subview_22 : memref<?x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%cast_24 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %59 = arith.cmpi eq, %arg19, %c0_i32 : i32
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag[<ENABLED_ONLY_LAST_ITER>]
    hivm.hir.mmadL1 ins(%cast_13, %cast_17, %59, %27, %52, %33 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag[<ENABLED_ONLY_LAST_ITER>]
    hivm.hir.mmadL1 ins(%cast_13, %cast_21, %59, %27, %52, %c64 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 32)>()[%arg24, %arg23]
    %reinterpret_cast_25 = memref.reinterpret_cast %arg2 to offset: [%60], sizes: [32, 32], strides: [%10, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_26 = memref.cast %reinterpret_cast_25 : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %61 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg26, %arg25, %34]
    %reinterpret_cast_27 = memref.reinterpret_cast %arg3 to offset: [%61], sizes: [32, 32], strides: [%12, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_28 = memref.cast %reinterpret_cast_27 : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %62 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg28, %arg27, %35]
    %reinterpret_cast_29 = memref.reinterpret_cast %arg5 to offset: [%62], sizes: [32, 64], strides: [%14, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_30 = memref.cast %reinterpret_cast_29 : memref<32x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    scf.yield %cast_26, %cast_28, %cast_30, %60, %c0, %61, %c0, %62, %c0 : memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, index, index, index, index, index
  }
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  %37 = arith.index_cast %0 : i64 to index
  %38 = affine.apply affine_map<()[s0] -> (s0 * 12288)>()[%37]
  %view = memref.view %arg1[%38][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x32xf32, #hivm.address_space<gm>>
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
//   CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag[<ENABLED_WITH_UPDATE>, %{{.*}}]
  hivm.hir.fixpipe {enable_nz2nd} ins(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<32x32xf32, #hivm.address_space<gm>>)
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  annotation.mark %view : memref<32x32xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
  %39 = arith.index_cast %arg14 : i32 to index
  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%13], sizes: [64, 32], strides: [%39, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %40 = hivm.hir.pointer_cast(%c8192_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
  %cast_7 = memref.cast %40 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
  %subview = memref.subview %reinterpret_cast_6[0, 0] [64, %32] [1, 1] : memref<64x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %41 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%31, %13]
  %subview_8 = memref.subview %40[0, 0, 0, 0] [%41, 4, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<?x4x16x16xf16, strided<[1024, 256, 16, 1]>, #hivm.address_space<cbuf>>
  %cast_9 = memref.cast %subview_8 : memref<?x4x16x16xf16, strided<[1024, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<64x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  %42 = affine.apply affine_map<()[s0] -> (s0 * 12288 + 4096)>()[%37]
  %view_10 = memref.view %arg1[%42][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf16, #hivm.address_space<gm>>
//   CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag[<ENABLED_WITH_UPDATE>, %{{.*}}]
  hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_10 : memref<32x64xf16, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  %43 = hivm.hir.pointer_cast(%c12288_i64) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
  %cast_11 = memref.cast %43 : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%view_10 : memref<32x64xf16, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
//   CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag[<ENABLED_WITH_UPDATE>]
  hivm.hir.mmadL1 ins(%cast_11, %cast_7, %true, %c32, %c64, %32 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
//   CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
  %44 = affine.apply affine_map<()[s0] -> (s0 * 12288 + 8192)>()[%37]
  %view_12 = memref.view %arg1[%44][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x32xf32, #hivm.address_space<gm>>
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
//   CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
//   CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag[<ENABLED_WITH_UPDATE>]
  hivm.hir.fixpipe {enable_nz2nd} ins(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_12 : memref<32x32xf32, #hivm.address_space<gm>>)
  annotation.mark %view_12 : memref<32x32xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
  return
}
