// RUN: bishengir-opt -hivm-graph-sync-solver -split-input-file %s | FileCheck %s

// Regression test for a multi-buffer backward-sync deadlock.
//
// The MTE1->MTE2 "buffer free" sync of a double-buffered L1 input is a backward
// WAR between the consuming mmad's L0 load (set, inside the MmadL1 loop) and the
// producing nd2nz (wait, at the outer subloop level). When the set-side event
// id is reused via reuseConflictPair, the reused pair takes over the nd2nz-side
// set op but used to keep a stale setOnLastIterOnly. Codegen then wrapped the
// re-arm set_flag in an `scf.if <last-iteration>` guard while the wait_flag ran
// every iteration, so from the 2nd iteration the wait never observed a set and
// the MTE2 pipe deadlocked (seen with enable-auto-multi-buffer=True).
//
// The re-arm set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %dyn] must stay unconditional in
// the loop body. Anchor on the COLUMN_SPLIT fixpipe right before it and assert
// no scf.if guard sits between that point and the set.

// CHECK-LABEL: func.func @chunk_transform_qk_bwd_kernel_prepare_mix_aic
// CHECK: dual_dst_mode = <COLUMN_SPLIT>
// CHECK-NOT: scf.if
// CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %{{[0-9]+}}]

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
func.func @chunk_transform_qk_bwd_kernel_prepare_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg14: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg15: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg16: f32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<21xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c123744_i64 = arith.constant 123744 : i64
  %c151552_i64 = arith.constant 151552 : i64
  %c126976_i64 = arith.constant 126976 : i64
  %c118784_i64 = arith.constant 118784 : i64
  %c147456_i64 = arith.constant 147456 : i64
  %c114688_i64 = arith.constant 114688 : i64
  %c109024_i64 = arith.constant 109024 : i64
  %c110592_i64 = arith.constant 110592 : i64
  %c102400_i64 = arith.constant 102400 : i64
  %c100704_i64 = arith.constant 100704 : i64
  %c94208_i64 = arith.constant 94208 : i64
  %c86016_i64 = arith.constant 86016 : i64
  %c69632_i64 = arith.constant 69632 : i64
  %c84064_i64 = arith.constant 84064 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c57344_i64 = arith.constant 57344 : i64
  %c71776_i64 = arith.constant 71776 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c40960_i64 = arith.constant 40960 : i64
  %c55136_i64 = arith.constant 55136 : i64
  %c46944_i64 = arith.constant 46944 : i64
  %c36864_i64 = arith.constant 36864 : i64
  %c42848_i64 = arith.constant 42848 : i64
  %c28672_i64 = arith.constant 28672 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c22112_i64 = arith.constant 22112 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c139264_i64 = arith.constant 139264 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c135168_i64 = arith.constant 135168 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c131072_i64 = arith.constant 131072 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %true = arith.constant true
  %cst = arith.constant 0.000000e+00 : f16
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c1024_i32 = arith.constant 1024 : i32
  %c63_i32 = arith.constant 63 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0 = arith.constant 0 : index
  %0 = arith.muli %arg18, %arg19 : i32
  %1 = arith.muli %0, %arg20 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.addi %arg17, %c63_i32 : i32
  %5 = arith.divsi %4, %c64_i32 : i32
  %6 = arith.index_cast %arg17 : i32 to index
  scf.for %arg21 = %3 to %1 step %c32_i32  : i32 {
    %7 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %8 = hivm.hir.pointer_cast(%c126976_i64, %c151552_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    %9 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    %10 = hivm.hir.pointer_cast(%c114688_i64, %c147456_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    %11 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    %12 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %13 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    %14 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    %15 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %16 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %17 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %18 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %19 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %20 = hivm.hir.pointer_cast(%c8192_i64, %c139264_i64) : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    %21 = hivm.hir.pointer_cast(%c4096_i64, %c135168_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    %22 = hivm.hir.pointer_cast(%c0_i64, %c131072_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %23 = arith.remsi %arg21, %arg18 : i32
    %24 = arith.divsi %arg21, %arg18 : i32
    %25 = arith.remsi %24, %arg19 : i32
    %26 = arith.divsi %25, %c4_i32 : i32
    %27 = arith.remsi %25, %c4_i32 : i32
    %28 = arith.divsi %27, %c4_i32 : i32
    %29 = arith.muli %26, %arg17 : i32
    %30 = arith.muli %26, %5 : i32
    %31 = arith.muli %29, %c4_i32 : i32
    %32 = arith.addi %31, %27 : i32
    %33 = arith.addi %29, %28 : i32
    %34 = arith.muli %32, %c32_i32 : i32
    %35 = arith.index_cast %34 : i32 to index
    %36 = arith.muli %33, %c32_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = arith.addi %30, %23 : i32
    %39 = arith.addi %38, %28 : i32
    %40 = arith.muli %39, %c1024_i32 : i32
    %41 = arith.index_cast %40 : i32 to index
    %42 = arith.muli %23, %c64_i32 : i32
    %43 = arith.maxsi %42, %c0_i32 : i32
    %44 = arith.index_cast %43 : i32 to index
    %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%35, %44]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [64, 32], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32)>()[%37, %44]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [64, 32], strides: [32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %47 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%6, %44]
    %48 = arith.maxsi %47, %c0 : index
    %49 = arith.minsi %48, %c64 : index
    %50 = arith.subi %c0_i32, %42 : i32
    %51 = arith.maxsi %50, %c0_i32 : i32
    %52 = arith.index_cast %51 : i32 to index
    %53 = arith.minsi %52, %49 : index
    %54 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%49, %53]
    %55 = arith.cmpi slt, %54, %c64 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%54, 32] [1, 1] : memref<64x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %cast = memref.cast %subview : memref<?x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %56 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1) ceildiv 16)>()[%49, %53]
    %57 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%53]
    %58 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%53]
    %subview_2 = memref.subview %22[0, %57, %58, 0] [2, %56, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_3 = memref.cast %subview_2 : memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %55 {
      %collapse_shape = memref.collapse_shape %22 [[0, 1, 2, 3]] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> into memref<2048xf16, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<2048xf16, #hivm.address_space<cbuf>>)
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%cast : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_3 : memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%46], sizes: [64, 32], strides: [32, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %reinterpret_cast_4[0, 0] [%54, 32] [1, 1] : memref<64x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_6 = memref.cast %subview_5 : memref<?x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %21[0, %57, %58, 0] [2, %56, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_8 = memref.cast %subview_7 : memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %55 {
      %collapse_shape = memref.collapse_shape %21 [[0, 1, 2, 3]] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> into memref<2048xf16, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<2048xf16, #hivm.address_space<cbuf>>)
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%cast_6 : memref<?x?xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
    %subview_9 = memref.subview %reinterpret_cast_1[0, 0] [%54, 32] [1, 1] : memref<64x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_10 = memref.cast %subview_9 : memref<?x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %20[0, %57, %58, 0] [4, %56, 16, 8] [1, 1, 1, 1] : memref<4x4x16x8xf32, #hivm.address_space<cbuf>> to memref<4x?x16x8xf32, strided<[512, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_12 = memref.cast %subview_11 : memref<4x?x16x8xf32, strided<[512, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %55 {
      %collapse_shape = memref.collapse_shape %20 [[0, 1, 2, 3]] : memref<4x4x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>>)
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%cast_10 : memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_12 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>)
    %59 = hivm.hir.pointer_cast(%c16384_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %59 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true, normalized_in_L0C} ins(%22, %59, %true, %c64, %c32, %c64 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%19 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
    %60 = hivm.hir.pointer_cast(%c22112_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %60 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index} : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%19 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%60 : memref<32x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %61 = hivm.hir.pointer_cast(%c20480_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %61 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %62 = hivm.hir.pointer_cast(%c28672_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %62 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%61, %62, %true, %c64, %c64, %c64 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%18 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
    %63 = hivm.hir.pointer_cast(%c42848_i64) : memref<32x64xf16, #hivm.address_space<ub>>
    annotation.mark %63 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, hivm.tiling_dim = 0 : index} : memref<32x64xf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%18 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%63 : memref<32x64xf16, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    %64 = hivm.hir.pointer_cast(%c36864_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %64 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, hivm.tiling_dim = 1 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true, normalized_in_L0C} ins(%64, %21, %true, %c64, %c32, %c64 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%17 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
    %65 = hivm.hir.pointer_cast(%c46944_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %65 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, hivm.tiling_dim = 0 : index} : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%17 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%65 : memref<32x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true, normalized_in_L0C} ins(%22, %21, %true, %c64, %c32, %c64 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%16 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
    %66 = hivm.hir.pointer_cast(%c55136_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %66 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, hivm.tiling_dim = 0 : index} : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%16 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%66 : memref<32x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    %67 = hivm.hir.pointer_cast(%c40960_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %67 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %68 = hivm.hir.pointer_cast(%c49152_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %68 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%67, %68, %true, %c64, %c64, %c64 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%15 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
    %69 = hivm.hir.pointer_cast(%c71776_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %69 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, hivm.tiling_dim = 0 : index} : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%15 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%69 : memref<32x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    %70 = hivm.hir.pointer_cast(%c57344_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %70 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %71 = hivm.hir.pointer_cast(%c65536_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %71 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 1 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%70, %71, %true, %c64, %c64, %c32 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%14 : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
    %72 = hivm.hir.pointer_cast(%c84064_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    annotation.mark %72 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>, hivm.tiling_dim = 0 : index} : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%14 : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%72 : memref<32x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %73 = arith.remsi %27, %c4_i32 : i32
    %74 = arith.cmpi eq, %73, %c0_i32 : i32
    scf.if %74 {
      %77 = hivm.hir.pointer_cast(%c69632_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %77 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 1 : index} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%77, %20, %true, %c64, %c64, %c32 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>, memref<4x4x16x8xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%13 : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
      %78 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32)>()[%41, %44]
      %reinterpret_cast_27 = memref.reinterpret_cast %arg9 to offset: [%78], sizes: [64, 32], strides: [32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %13[0, %57, %58, 0] [2, %56, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<2x?x16x16xf32, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_29 = memref.subview %reinterpret_cast_27[0, 0] [%54, 32] [1, 1] : memref<64x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_30 = memref.cast %subview_29 : memref<?x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%subview_28 : memref<2x?x16x16xf32, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>) outs(%cast_30 : memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
      %79 = hivm.hir.pointer_cast(%c86016_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %79 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<15>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %80 = hivm.hir.pointer_cast(%c94208_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %80 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<16>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%79, %80, %true, %c64, %c64, %c64 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%12 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
      %81 = hivm.hir.pointer_cast(%c100704_i64) : memref<32x64xf16, #hivm.address_space<ub>>
      annotation.mark %81 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<17>, hivm.tiling_dim = 0 : index} : memref<32x64xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%12 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%81 : memref<32x64xf16, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
      %82 = hivm.hir.pointer_cast(%c102400_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %82 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<18>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      %83 = hivm.hir.pointer_cast(%c110592_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %83 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<19>, hivm.tiling_dim = 1 : index} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%83, %82, %true, %c32, %c64, %c64 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%11 : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
      %84 = hivm.hir.pointer_cast(%c109024_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %84 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<20>, hivm.tiling_dim = 1 : index} : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%11 : memref<4x2x16x16xf32, #hivm.address_space<cc>>) outs(%84 : memref<32x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    }
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [%45], sizes: [64, 32], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_14 = memref.subview %reinterpret_cast_13[0, 0] [%54, 32] [1, 1] : memref<64x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_15 = memref.cast %subview_14 : memref<?x32xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_16 = memref.subview %10[0, %57, %58, 0] [2, %56, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_17 = memref.cast %subview_16 : memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %55 {
      %collapse_shape = memref.collapse_shape %10 [[0, 1, 2, 3]] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> into memref<2048xf16, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<2048xf16, #hivm.address_space<cbuf>>)
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%cast_15 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_17 : memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
    %75 = hivm.hir.pointer_cast(%c118784_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %75 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<21>, hivm.tiling_dim = 1 : index} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%75, %10, %true, %c64, %c64, %c32 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%9 : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
    %reinterpret_cast_18 = memref.reinterpret_cast %arg14 to offset: [%45], sizes: [64, 32], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_19 = memref.subview %9[0, %57, %58, 0] [2, %56, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<2x?x16x16xf32, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_20 = memref.subview %reinterpret_cast_18[0, 0] [%54, 32] [1, 1] : memref<64x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_21 = memref.cast %subview_20 : memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%subview_19 : memref<2x?x16x16xf32, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>) outs(%cast_21 : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 32], strides: [32, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_23 = memref.subview %reinterpret_cast_22[0, 0] [%54, 32] [1, 1] : memref<64x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_24 = memref.cast %subview_23 : memref<?x32xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_25 = memref.subview %8[0, %57, %58, 0] [2, %56, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_26 = memref.cast %subview_25 : memref<2x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %55 {
      %collapse_shape = memref.collapse_shape %8 [[0, 1, 2, 3]] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> into memref<2048xf16, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<2048xf16, #hivm.address_space<cbuf>>)
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%cast_24 : memref<?x?xf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_26 : memref<?x?x16x16xf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
    hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true, normalized_in_L0C} ins(%10, %8, %true, %c64, %c32, %c64 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%7 : memref<4x4x16x16xf32, #hivm.address_space<cc>>)
    %76 = hivm.hir.pointer_cast(%c123744_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %76 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<22>, hivm.tiling_dim = 0 : index} : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%76 : memref<32x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
  return
}
}
