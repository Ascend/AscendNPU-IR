// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-inject-sync{enable-unit-flag=true}))" | FileCheck %s --check-prefixes="CHECK,CHECK-UF-ON"
// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-inject-sync{enable-unit-flag=false}))" | FileCheck %s --check-prefixes="CHECK,CHECK-UF-OFF"
// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-unit-flag=false ignore-workspace-func-args=true}))" | FileCheck %s --check-prefixes="CHECK,CHECK-UF-OFF"
// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-unit-flag=true ignore-workspace-func-args=true}))" | FileCheck %s --check-prefixes="CHECK,CHECK-UF-ON"

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
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT0:EVENT_ID[0-7]]]>]
//   CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
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
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT0]]>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 {b_transpose} ins(%cast_4, %cast_5, %true, %c32, %c256, %c256 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
    // CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<32x256xf32, #hivm.address_space<gm>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT1:EVENT_ID[0-7]]]>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
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
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT1]]>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 ins(%cast_7, %cast_6, %true, %c32, %c256, %c256 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT2:EVENT_ID[0-7]]]>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
    // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT2]]>]
    // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
    // CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_1 : memref<32x256xf32, #hivm.address_space<gm>>)
    // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT0]]>]
    // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
    annotation.mark %view_1 : memref<32x256xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
  }
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT0]]>]
//   CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
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
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<DISABLED>, #hivm.unit_flag<ENABLED_WITH_UPDATE>]) unit_flag_cond(%{{.*}}, %{{.*}})
    hivm.hir.mmadL1 ins(%cast_13, %cast_17, %59, %27, %52, %33 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<DISABLED>, #hivm.unit_flag<ENABLED_WITH_UPDATE>]) unit_flag_cond(%{{.*}}, %{{.*}})
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
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT0:EVENT_ID[0-7]]]>]
  %37 = arith.index_cast %0 : i64 to index
  %38 = affine.apply affine_map<()[s0] -> (s0 * 12288)>()[%37]
  %view = memref.view %arg1[%38][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x32xf32, #hivm.address_space<gm>>
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT0]]>]
//   CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>]) unit_flag_cond(%{{.*}})
  hivm.hir.fixpipe {enable_nz2nd} ins(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<32x32xf32, #hivm.address_space<gm>>)
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT1:EVENT_ID[0-7]]]>]
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
//   CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>]) unit_flag_cond(%{{.*}})
  hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_10 : memref<32x64xf16, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  %43 = hivm.hir.pointer_cast(%c12288_i64) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
  %cast_11 = memref.cast %43 : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%view_10 : memref<32x64xf16, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT1]]>]
//   CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.mmadL1 ins(%cast_11, %cast_7, %true, %c32, %c64, %32 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
//   CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT2:EVENT_ID[0-7]]]>]
//   CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
  %44 = affine.apply affine_map<()[s0] -> (s0 * 12288 + 8192)>()[%37]
  %view_12 = memref.view %arg1[%44][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x32xf32, #hivm.address_space<gm>>
//   CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT2]]>]
//   CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{[0-7]}>]
//   CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.fixpipe {enable_nz2nd} ins(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_12 : memref<32x32xf32, #hivm.address_space<gm>>)
  annotation.mark %view_12 : memref<32x32xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
  return
}

// -----

// CHECK: @_fwd_kernel_mix_aic
func.func @_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg11: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg12: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<21xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, parallel_mode = "simd"} {
  %c4096_i64 = arith.constant 4096 : i64
  %c55296_i64 = arith.constant 55296 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c47104_i64 = arith.constant 47104 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c43008_i64 = arith.constant 43008 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c192_i32 = arith.constant 192 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %false = arith.constant false
  %c1_i32 = arith.constant 1 : i32
  %c64 = arith.constant 64 : index
  %c64_i32 = arith.constant 64 : i32
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg18, %arg19 : i32
  %1 = arith.muli %0, %arg20 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg20 : i32
  %5 = arith.divsi %3, %arg20 : i32
  %6 = arith.remsi %5, %arg19 : i32
  %7 = arith.muli %arg20, %arg19 : i32
  %8 = arith.divsi %3, %7 : i32
  %9 = arith.remsi %8, %arg18 : i32
  %10 = arith.index_cast %9 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  %11 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %12 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%10]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %13 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %14 = arith.subi %13, %11 : i32
  %15 = arith.muli %4, %c32_i32 : i32
  %16 = arith.addi %11, %15 : i32
  %17 = arith.muli %6, %c192_i32 : i32
  %18 = arith.index_cast %16 : i32 to index
  %19 = arith.index_cast %17 : i32 to index
  %20 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 24576)>()[%19, %18]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [32, 128], strides: [24576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
  %21 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>>
  %cast = memref.cast %21 : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  %22 = arith.index_cast %15 : i32 to index
  %23 = arith.index_cast %14 : i32 to index
  %24 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%22, %23]
  %25 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%24, %22]
  %26 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%25, %22]
  %27 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%25, %22]
  %28 = arith.cmpi slt, %27, %c32 : index
  %subview = memref.subview %reinterpret_cast_1[0, 0] [%27, 128] [1, 1] : memref<32x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
  %29 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%27]
  %subview_2 = memref.subview %21[0, 0, 0, 0] [8, %29, 16, 16] [1, 1, 1, 1] : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
  %cast_3 = memref.cast %subview_2 : memref<8x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  scf.if %28 {
    %collapse_shape = memref.collapse_shape %21 [[0, 1, 2, 3]] : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<4096xbf16, #hivm.address_space<cbuf>>
    hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<4096xbf16, #hivm.address_space<cbuf>>)
  }
  hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_3 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 24576 + 128)>()[%19, %18]
  %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [32, 64], strides: [24576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
  %31 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
  %cast_5 = memref.cast %31 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  %32 = arith.cmpi slt, %26, %c32 : index
  %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%26, 64] [1, 1] : memref<32x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
  %33 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%25, %22]
  %subview_7 = memref.subview %31[0, 0, 0, 0] [4, %33, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
  %cast_8 = memref.cast %subview_7 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  scf.if %32 {
    %collapse_shape = memref.collapse_shape %31 [[0, 1, 2, 3]] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<2048xbf16, #hivm.address_space<cbuf>>
    hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<2048xbf16, #hivm.address_space<cbuf>>)
  }
  hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<?x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  %34 = arith.addi %4, %c1_i32 : i32
  %35 = arith.muli %34, %c32_i32 : i32
  %36 = arith.minsi %14, %35 : i32
  %37 = arith.muli %6, %c128_i32 : i32
  %38 = arith.index_cast %2 : i64 to index
  %39 = affine.apply affine_map<()[s0] -> (s0 * 45056 + 36864)>()[%38]
  %view = memref.view %arg2[%39][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x32x32xf32, #hivm.address_space<gm>>
  %40 = affine.apply affine_map<()[s0] -> (s0 * 45056)>()[%38]
  %view_9 = memref.view %arg2[%40][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x32x32xbf16, #hivm.address_space<gm>>
  %41 = affine.apply affine_map<()[s0] -> (s0 * 45056 + 4096)>()[%38]
  %view_10 = memref.view %arg2[%41][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x32x128xf32, #hivm.address_space<gm>>
  %42 = arith.index_cast %36 : i32 to index
  %43 = arith.index_cast %37 : i32 to index
  scf.for %arg21 = %c0_i32 to %36 step %c64_i32  : i32 {
    %44 = arith.index_cast %arg21 : i32 to index
    %45 = affine.min affine_map<(d0)[s0] -> (s0, d0 + 64)>(%44)[%42]
    %46 = affine.apply affine_map<(d0, d1) -> ((d0 - d1) ceildiv 32)>(%45, %44)
    annotation.mark %view : memref<2x32x32xf32, #hivm.address_space<gm>>
    annotation.mark %view_9 : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %view_10 : memref<2x32x128xf32, #hivm.address_space<gm>>
    scf.for %arg22 = %c0 to %46 step %c1 {
      %47 = hivm.hir.pointer_cast(%c12288_i64, %c34816_i64) : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %47 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>>
      %48 = hivm.hir.pointer_cast(%c20480_i64, %c43008_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %48 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %49 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg22)[%44]
      %50 = arith.index_cast %49 : index to i32
      %51 = affine.max affine_map<(d0)[s0, s1] -> (s0, d0 * 32 + s1)>(%arg22)[%42, %44]
      %52 = affine.min affine_map<(d0)[s0, s1] -> (s0, d0 * 32 + s1 + 32)>(%arg22)[%51, %44]
      %53 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * -32 + s0 - s1)>(%arg22)[%52, %44]
      %54 = affine.min affine_map<(d0)[s0, s1] -> (d0 * -32 + s0 - s1, 32)>(%arg22)[%52, %44]
      %55 = arith.cmpi slt, %54, %c32 : index
      %56 = arith.addi %11, %50 : i32
      %57 = arith.index_cast %56 : i32 to index
      %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 24576)>()[%19, %57]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [32, 128], strides: [24576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_12 = memref.cast %47 : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%54, 128] [1, 1] : memref<32x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
      %59 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%54]
      %subview_14 = memref.subview %47[0, 0, 0, 0] [8, %59, 16, 16] [1, 1, 1, 1] : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_15 = memref.cast %subview_14 : memref<8x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %55 {
        %collapse_shape_23 = memref.collapse_shape %47 [[0, 1, 2, 3]] : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<4096xbf16, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape_23 : memref<4096xbf16, #hivm.address_space<cbuf>>)
      }
      hivm.hir.nd2nz {dst_continuous} ins(%subview_13 : memref<?x128xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_15 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %60 = hivm.hir.pointer_cast(%c0_i64) : memref<2x2x16x16xf32, #hivm.address_space<cc>>
      %cast_16 = memref.cast %60 : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT0:EVENT_ID[0-7]]]>]
      // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
      // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<DISABLED>, #hivm.unit_flag<ENABLED_WITHOUT_UPDATE>, #hivm.unit_flag<ENABLED_WITHOUT_UPDATE>]) unit_flag_cond(%{{.*}}, %{{.*}}, %true{{.*}})
      hivm.hir.mmadL1 {b_transpose} ins(%cast, %cast_12, %true, %27, %c128, %54 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 24576 + 128)>()[%19, %57]
      %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [32, 64], strides: [24576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_18 = memref.cast %48 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      %62 = arith.cmpi slt, %53, %c32 : index
      %subview_19 = memref.subview %reinterpret_cast_17[0, 0] [%53, 64] [1, 1] : memref<32x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>
      %63 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 * -32 + s0 - s1 + 15) floordiv 16)>(%arg22)[%52, %44]
      %subview_20 = memref.subview %48[0, 0, 0, 0] [4, %63, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_21 = memref.cast %subview_20 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %62 {
        %collapse_shape_23 = memref.collapse_shape %48 [[0, 1, 2, 3]] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<2048xbf16, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape_23 : memref<2048xbf16, #hivm.address_space<cbuf>>)
      }
      hivm.hir.nd2nz {dst_continuous} ins(%subview_19 : memref<?x64xbf16, strided<[24576, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_21 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
      hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%cast_5, %cast_18, %false, %26, %c64, %53 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT1:EVENT_ID[0-7]]]>]
      // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
      %subview_22 = memref.subview %view[%arg22, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview_22 [[0, 1], [2]] : memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      %64 = arith.index_cast %arg22 : index to i64
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %64
      %65 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%arg22]
      %66 = arith.index_cast %65 : index to i64
      // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT1]]>]
      // CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
      hivm.hir.fixpipe {enable_nz2nd} ins(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%collapse_shape : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
      // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT0]]>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %66
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
    scf.for %arg22 = %c0 to %46 step %c1 {
      %47 = hivm.hir.pointer_cast(%c32768_i64, %c55296_i64) : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %47 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      %48 = hivm.hir.pointer_cast(%c24576_i64, %c47104_i64) : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %48 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>>
      %49 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg22)[%44]
      %50 = arith.index_cast %49 : index to i32
      %51 = affine.max affine_map<(d0)[s0, s1] -> (s0, d0 * 32 + s1)>(%arg22)[%42, %44]
      %52 = affine.min affine_map<(d0)[s0, s1] -> (s0, d0 * 32 + s1 + 32)>(%arg22)[%51, %44]
      %53 = affine.min affine_map<(d0)[s0, s1] -> (d0 * -32 + s0 - s1, 32)>(%arg22)[%52, %44]
      %54 = arith.cmpi slt, %53, %c32 : index
      %55 = arith.addi %11, %50 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 16384)>()[%43, %56]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg5 to offset: [%57], sizes: [32, 128], strides: [16384, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x128xbf16, strided<[16384, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_12 = memref.cast %48 : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%53, 128] [1, 1] : memref<32x128xbf16, strided<[16384, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[16384, 1], offset: ?>, #hivm.address_space<gm>>
      %58 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%53]
      %subview_14 = memref.subview %48[0, 0, 0, 0] [8, %58, 16, 16] [1, 1, 1, 1] : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_15 = memref.cast %subview_14 : memref<8x?x16x16xbf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %54 {
        %collapse_shape_20 = memref.collapse_shape %48 [[0, 1, 2, 3]] : memref<8x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<4096xbf16, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape_20 : memref<4096xbf16, #hivm.address_space<cbuf>>)
      }
      hivm.hir.nd2nz {dst_continuous} ins(%subview_13 : memref<?x128xbf16, strided<[16384, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_15 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %subview_16 = memref.subview %view_9[%arg22, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      %59 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%arg22]
      %60 = arith.index_cast %59 : index to i64
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = %60
      %61 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg22]
      %62 = arith.index_cast %61 : index to i64
      %cast_17 = memref.cast %47 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_16 : memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_17 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %62
      %63 = hivm.hir.pointer_cast(%c4096_i64) : memref<8x2x16x16xf32, #hivm.address_space<cc>>
      %cast_18 = memref.cast %63 : memref<8x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT2:EVENT_ID[0-7]]]>]
      // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
      // CHECK-UF-ON: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
      hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_17, %cast_12, %true, %c32, %c32, %c128 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT3:EVENT_ID[0-7]]]>]
      // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
      %subview_19 = memref.subview %view_10[%arg22, 0, 0] [1, 32, 128] [1, 1, 1] : memref<2x32x128xf32, #hivm.address_space<gm>> to memref<1x32x128xf32, strided<[4096, 128, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview_19 [[0, 1], [2]] : memref<1x32x128xf32, strided<[4096, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %64 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%arg22]
      %65 = arith.index_cast %64 : index to i64
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %65
      // CHECK-UF-OFF: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <[[EVENT3]]>]
      // CHECK-UF-ON-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID{{[0-7]}}>]
      // CHECK-UF-ON: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
      hivm.hir.fixpipe {enable_nz2nd} ins(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%collapse_shape : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      // CHECK-UF-OFF: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[EVENT2]]>]
      // CHECK-UF-ON-NOT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID{{[0-7]}}>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %60
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  return
}

