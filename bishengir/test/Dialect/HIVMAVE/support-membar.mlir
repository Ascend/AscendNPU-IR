// RUN: bishengir-opt  -convert-hivmave-to-ave-intrin %s | FileCheck %s
#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<()[s0] -> (s0 * 4)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
#map4 = affine_map<()[s0] -> (s0 + 4)>
#map5 = affine_map<()[s0, s1] -> (s0 - s1)>
#map6 = affine_map<()[s0, s1] -> ((s1 * s1) * s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @triton_unk_fused_sum_5(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c32_i64 = arith.constant 32 : i64
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c6_i32 = arith.constant 6 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.divsi %1, %arg10 : i32
    %3 = arith.remsi %2, %arg9 : i32
    %4 = arith.muli %arg10, %arg9 : i32
    %5 = arith.divsi %1, %4 : i32
    %6 = arith.remsi %5, %arg8 : i32
    %7 = arith.muli %3, %c6_i32 : i32
    // CHECK: "hivm_regbaseintrins.intr.hivm.mem.bar.vv.all"
    ave.hir.membar %c0_i32
    %8 = arith.addi %6, %c1_i32 : i32
    %9 = arith.minsi %8, %arg5 : i32
    %10 = arith.cmpi slt, %6, %9 : i32
    %11 = arith.addi %7, %c6_i32 : i32
    %12 = arith.minsi %11, %arg6 : i32
    %13 = arith.muli %6, %c128_i32 : i32
    %14 = arith.muli %6, %c16_i32 : i32
    %15 = arith.index_cast %13 : i32 to index
    %16 = arith.index_cast %12 : i32 to index
    %17 = arith.index_castui %10 : i1 to index
    %18 = affine.apply #map2()[%17]
    %19 = arith.minsi %17, %c1 : index
    %20 = arith.cmpi slt, %19, %c1 : index
    %21 = affine.apply #map()[%19]
    %22 = hivm.hir.pointer_cast(%c0_i64) : memref<4xf32, #hivm.address_space<ub>>
    %23 = arith.index_cast %14 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %24 = arith.muli %arg11, %c4_i32 : i32
      %25 = arith.addi %7, %24 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = affine.apply #map3()[%26, %15]
      %28 = hivm.hir.pointer_cast(%c32_i64) : memref<8x8x1xf32, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %28 : memref<8x8x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8, 4], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %29 = affine.apply #map4()[%26]
      %30 = arith.maxsi %26, %16 : index
      %31 = arith.minsi %29, %30 : index
      %32 = affine.apply #map5()[%31, %26]
      %33 = arith.minsi %32, %18 : index
      %34 = arith.cmpi slt, %33, %c4 : index
      %35 = arith.ori %20, %34 : i1
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg3 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%27], sizes: [%21, %33], strides: [16, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_5, %offset_6, %sizes_7:3, %strides_8:3 = memref.extract_strided_metadata %28 : memref<8x8x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [0], sizes: [%21, %33], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %reinterpret_cast_4 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_10 = memref.cast %reinterpret_cast_9 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast, %cast_10, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %36 = affine.apply #map3()[%26, %23]
      %37 = affine.apply #map6()[%33, %19]
      %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %22 : memref<4xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [%37], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %base_buffer_16, %offset_17, %sizes_18, %strides_19 = memref.extract_strided_metadata %arg4 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_20 = memref.reinterpret_cast %base_buffer_16 to offset: [%36], sizes: [%37], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_21 = memref.cast %reinterpret_cast_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %reinterpret_cast_20 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_21, %cast_22, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    return
  }
}