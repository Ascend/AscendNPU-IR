// RUN: bishengir-opt %s -create-preload -split-input-file  | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 1835008 + 1048576)>
#map1 = affine_map<()[s0] -> (s0 * 1835008 + 1572864)>
#map2 = affine_map<()[s0] -> (s0 * 1835008)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map4 = affine_map<()[s0] -> (s0 * 64)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1)>
#map6 = affine_map<()[s0] -> (s0 * 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.0.0">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_sdpa_infer_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c1835008 = arith.constant 1835008 : index
    return %c1835008 : index
  }
  func.func @_sdpa_infer_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
// CHECK-LABEL:   func.func @_sdpa_infer_kernel_mix_aic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
// CHECK-SAME:                                          %[[VAL_1:.*]]: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
// CHECK-SAME:                                          %[[VAL_2:.*]]: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>},
// CHECK-DAG:       %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[C512_I32:.*]] = arith.constant 512 : i32
// CHECK-DAG:       %[[C1024_I32:.*]] = arith.constant 1024 : i32
// CHECK-DAG:       %[[C8192_I32:.*]] = arith.constant 8192 : i32
// CHECK-DAG:       %[[C10240_I32:.*]] = arith.constant 10240 : i32
// CHECK:           %[[PRELOAD_WORKSPACE1:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_46:.*]]][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xbf16, #hivm.address_space<gm>>
// CHECK:           %[[PRELOAD_WORKSPACE2:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_48:.*]]][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x128xf32, #hivm.address_space<gm>>
// CHECK:           %[[PRELOAD_WORKSPACE3:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_50:.*]]][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_76:.*]]:2 = scf.for %[[PRELOAD3_IND_VAR:.*]] = %[[C0_I32]] to %[[C10240_I32]] step %[[C512_I32]] iter_args(%[[ITER1:.*]] = %[[C0_I32]], %[[ITER2:.*]] = %[[C0_I32]]) -> (i32, i32)  : i32 {
// CHECK:             %[[PRELOAD1_IND_VAR:.*]] = arith.subi %[[PRELOAD3_IND_VAR]], %[[C1024_I32]] : i32
// CHECK-DAG:         annotation.mark %[[PRELOAD_WORKSPACE1]] : memref<4x128x512xbf16, #hivm.address_space<gm>>
// CHECK-DAG:         annotation.mark %[[PRELOAD_WORKSPACE2]] : memref<4x128x128xf32, #hivm.address_space<gm>>
// CHECK-DAG:         annotation.mark %[[PRELOAD_WORKSPACE3]] : memref<4x128x512xf32, #hivm.address_space<gm>>
// CHECK:             %[[PRELOAD3_LB:.*]] = arith.cmpi sge, %[[PRELOAD3_IND_VAR]], %[[C0_I32]] : i32
// CHECK:             %[[PRELOAD3_UB:.*]] = arith.cmpi slt, %[[PRELOAD3_IND_VAR]], %[[C8192_I32]] : i32
// CHECK:             %[[PRELOAD3_COND:.*]] = arith.andi %[[PRELOAD3_LB]], %[[PRELOAD3_UB]] : i1
// CHECK:             %[[ITER2_RET:.*]] = scf.if %[[PRELOAD3_COND]] -> (i32) {
// CHECK:               %[[VAL_92:.*]] = arith.divsi %[[PRELOAD3_IND_VAR]], %[[C512_I32]] : i32
// CHECK:               %[[VAL_93:.*]] = arith.remsi %[[VAL_92]], %[[C4_I32]] : i32
// CHECK:               %[[PRELOAD3_OFFSET:.*]] = arith.index_cast %[[VAL_93]] : i32 to index
// CHECK:               %[[VAL_95:.*]] = memref.subview %[[PRELOAD_WORKSPACE3]]{{\[}}%[[PRELOAD3_OFFSET]], 0, 0] [1, 128, 512] [1, 1, 1] : memref<4x128x512xf32, #hivm.address_space<gm>> to memref<1x128x512xf32, strided<[65536, 512, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:               %[[PRELOAD3_ITER2:.*]] = arith.addi %[[ITER2]], %[[C512_I32]] : i32
// CHECK:               scf.yield %[[PRELOAD3_ITER2]] : i32
// CHECK:             } else {
// CHECK:               scf.yield %[[ITER2]] : i32
// CHECK:             }
// CHECK:             %[[PRELOAD1_LB:.*]] = arith.cmpi sge, %[[PRELOAD1_IND_VAR]], %[[C0_I32]] : i32
// CHECK:             %[[PRELOAD1_UB:.*]] = arith.cmpi slt, %[[PRELOAD1_IND_VAR]], %[[C8192_I32]] : i32
// CHECK:             %[[PRELOAD1_COND:.*]] = arith.andi %[[PRELOAD1_LB]], %[[PRELOAD1_UB]] : i1
// CHECK:             %[[ITER1_RET:.*]] = scf.if %[[PRELOAD1_COND]] -> (i32) {
// CHECK:               %[[VAL_113:.*]] = arith.divsi %[[PRELOAD1_IND_VAR]], %[[C512_I32]] : i32
// CHECK:               %[[VAL_114:.*]] = arith.remsi %[[VAL_113]], %[[C4_I32]] : i32
// CHECK:               %[[PRELOAD1_OFFSET:.*]] = arith.index_cast %[[VAL_114]] : i32 to index
// CHECK-DAG:           %[[VAL_116:.*]] = memref.subview %[[PRELOAD_WORKSPACE1]]{{\[}}%[[PRELOAD1_OFFSET]], 0, 0] [1, 128, 512] [1, 1, 1] : memref<4x128x512xbf16, #hivm.address_space<gm>> to memref<128x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK-DAG:           %[[VAL_118:.*]] = memref.subview %[[PRELOAD_WORKSPACE2]]{{\[}}%[[PRELOAD1_OFFSET]], 0, 0] [1, 128, 128] [1, 1, 1] : memref<4x128x128xf32, #hivm.address_space<gm>> to memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:               %[[PRELOAD1_ITER1:.*]] = arith.addi %[[ITER1]], %[[C512_I32]] : i32
// CHECK:               scf.yield %[[PRELOAD1_ITER1]] : i32
// CHECK:             } else {
// CHECK:               scf.yield %[[ITER1]] : i32
// CHECK:             }
// CHECK:             scf.yield %[[ITER1_RET]], %[[ITER2_RET]] : i32, i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
  func.func @_sdpa_infer_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: f32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.enable_saving_ub, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c458752_i64 = arith.constant 458752 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c327680_i64 = arith.constant 327680 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c294912_i64 = arith.constant 294912 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2097152_i64 = arith.constant 2097152 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg12, %arg11 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg10 : i32
    %7 = arith.index_cast %2 : i64 to index
    %8 = affine.apply #map()[%7]
    %view = memref.view %arg2[%8][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xbf16, #hivm.address_space<gm>>
    %9 = affine.apply #map1()[%7]
    %view_0 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x128xf32, #hivm.address_space<gm>>
    %10 = affine.apply #map2()[%7]
    %view_1 = memref.view %arg2[%10][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xf32, #hivm.address_space<gm>>
    scf.for %arg13 = %6 to %c512_i32 step %arg10  : i32 {
      %11 = hivm.hir.pointer_cast(%c0_i64, %c262144_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
      %12 = arith.divsi %arg13, %c64_i32 : i32
      %13 = arith.remsi %arg13, %c64_i32 : i32
      %14 = arith.divsi %12, %c8_i32 : i32
      %15 = arith.remsi %12, %c8_i32 : i32
      %16 = arith.divsi %15, %c4_i32 : i32
      %17 = arith.extsi %14 : i32 to i64
      %18 = arith.muli %17, %c8388608_i64 : i64
      %19 = arith.extsi %15 : i32 to i64
      %20 = arith.muli %19, %c1048576_i64 : i64
      %21 = arith.addi %18, %20 : i64
      %22 = arith.muli %17, %c2097152_i64 : i64
      %23 = arith.extsi %16 : i32 to i64
      %24 = arith.muli %23, %c1048576_i64 : i64
      %25 = arith.addi %22, %24 : i64
      %26 = arith.index_cast %21 : i64 to index
      %27 = arith.muli %13, %c128_i32 : i32
      %28 = arith.maxsi %27, %c0_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = affine.apply #map3()[%26, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.index_cast %25 : i64 to index
      %cast = memref.cast %11 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      %32:2 = scf.for %arg14 = %c0_i32 to %c8192_i32 step %c512_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
        %33 = hivm.hir.pointer_cast(%c65536_i64, %c327680_i64) : memref<8x32x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<8x32x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %view_1 : memref<4x128x512xf32, #hivm.address_space<gm>>
        annotation.mark %view : memref<4x128x512xbf16, #hivm.address_space<gm>>
        annotation.mark %view_0 : memref<4x128x128xf32, #hivm.address_space<gm>>
        %34 = scope.scope : () -> i32 {
          %36 = arith.index_cast %arg16 : i32 to index
          %37 = affine.apply #map3()[%31, %36]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%37], sizes: [512, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<512x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview = memref.subview %view_1[0, 0, 0] [1, 128, 512] [1, 1, 1] {hivm.preload_workspace} : memref<4x128x512xf32, #hivm.address_space<gm>> to memref<1x128x512xf32, strided<[65536, 512, 1]>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x512xf32, strided<[65536, 512, 1]>, #hivm.address_space<gm>> into memref<128x512xf32, strided<[512, 1]>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          scf.for %arg17 = %c0 to %c512 step %c128 {
            %39 = hivm.hir.pointer_cast(%c32768_i64, %c294912_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
            %subview_3 = memref.subview %reinterpret_cast_2[%arg17, 0] [128, 128] [1, 1] : memref<512x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            %cast_4 = memref.cast %39 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_3 : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
            %40 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
            %cast_5 = memref.cast %40 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {b_transpose, cube_producer_to_fuse_0, fixpipe_already_inserted = true} ins(%cast, %cast_4, %true, %c128, %c128, %c128 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
            %subview_6 = memref.subview %collapse_shape[0, %arg17] [128, 128] [1, 1] : memref<128x512xf32, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x128xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_0} ins(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%subview_6 : memref<128x128xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          }
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 3
          %38 = arith.addi %arg16, %c512_i32 : i32
          scope.return %38 : i32
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
        %35 = scope.scope : () -> i32 {
          %36 = arith.index_cast %arg15 : i32 to index
          %37 = affine.apply #map3()[%31, %36]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [512, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<512x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %cast_3 = memref.cast %33 : memref<8x32x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_2 : memref<512x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_3 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          %subview = memref.subview %view_0[0, 0, 0] [1, 128, 128] [1, 1, 1] {hivm.preload_workspace} : memref<4x128x128xf32, #hivm.address_space<gm>> to memref<1x128x128xf32, strided<[16384, 128, 1]>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x128xf32, strided<[16384, 128, 1]>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
          %subview_4 = memref.subview %view[0, 0, 0] [1, 128, 512] [1, 1, 1] {cube_producer_to_fuse_2, hivm.preload_workspace} : memref<4x128x512xbf16, #hivm.address_space<gm>> to memref<128x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          scf.for %arg17 = %c0 to %c128 step %c64 {
            %39 = hivm.hir.pointer_cast(%c196608_i64, %c458752_i64) : memref<32x4x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<32x4x16x16xbf16, #hivm.address_space<cbuf>>
            %subview_5 = memref.subview %subview_4[%arg17, 0] [64, 512] [1, 1] : memref<128x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<64x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
            %cast_6 = memref.cast %39 : memref<32x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<64x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
            %40 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x4x16x16xf32, #hivm.address_space<cc>>
            %cast_7 = memref.cast %40 : memref<8x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {cube_producer_to_fuse_2, fixpipe_already_inserted = true, hivm.tile_mix_cube_num = 2 : i32} ins(%cast_6, %cast_3, %true, %c64, %c512, %c128 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
            %subview_8 = memref.subview %collapse_shape[%arg17, 0] [64, 128] [1, 1] : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_2_0} ins(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%subview_8 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          }
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 3
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
          %38 = arith.addi %arg15, %c512_i32 : i32
          scope.return %38 : i32
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
        scf.yield %35, %34 : i32, i32
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    }
    return
  }
// CHECK-LABEL:   func.func @_sdpa_infer_kernel_mix_aiv(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
// CHECK-SAME:                                          %[[VAL_1:.*]]: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
// CHECK-SAME:                                          %[[VAL_2:.*]]: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>},
// CHECK-DAG:       %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[C512_I32:.*]] = arith.constant 512 : i32
// CHECK-DAG:       %[[C1536_I32:.*]] = arith.constant 1536 : i32
// CHECK-DAG:       %[[C8192_I32:.*]] = arith.constant 8192 : i32
// CHECK-DAG:       %[[C10240_I32:.*]] = arith.constant 10240 : i32
// CHECK-DAG:       %[[ITER1_ADDR:.*]] = arith.constant 98560 : i64
// CHECK-DAG:       %[[ITER2_ADDR:.*]] = arith.constant 149312 : i64
// CHECK-DAG:       %[[INIT1:.*]] = hivm.hir.pointer_cast(%[[ITER1_ADDR]]) : memref<64xf32, #hivm.address_space<ub>>
// CHECK-DAG:       %[[INIT2:.*]] = hivm.hir.pointer_cast(%[[ITER2_ADDR]]) : memref<64x128xf32, #hivm.address_space<ub>>
// CHECK-DAG:       %[[LOCALBUFFER1_ADDR0:.*]] = arith.constant 99584 : i64
// CHECK-DAG:       %[[LOCALBUFFER1_ADDR1:.*]] = arith.constant 99328 : i64
// CHECK-DAG:       %[[LOCALBUFFER1_ADDR2:.*]] = arith.constant 99072 : i64
// CHECK-DAG:       %[[LOCALBUFFER1_ADDR3:.*]] = arith.constant 98816 : i64
// CHECK:           %[[PRELOAD_WORKSPACE1:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_61:.*]]][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xbf16, #hivm.address_space<gm>>
// CHECK:           %[[PRELOAD_WORKSPACE2:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_63:.*]]][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x128xf32, #hivm.address_space<gm>>
// CHECK:           %[[PRELOAD_WORKSPACE3:.*]] = memref.view %[[VAL_2]]{{\[}}%[[VAL_62:.*]]][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xf32, #hivm.address_space<gm>>
// CHECK:             %[[VAL_92:.*]]:2 = scf.for %[[IND_VAR:.*]] = %[[C0_I32]] to %[[C10240_I32]] step %[[C512_I32]] iter_args(%[[ITER1:.*]] = %[[INIT1]], %[[ITER2:.*]] = %[[INIT2]]) -> (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>)  : i32 {
// CHECK:               %[[PRELOAD2_IND_VAR:.*]] = arith.subi %[[IND_VAR]], %[[C512_I32]] : i32
// CHECK:               %[[PRELOAD0_IND_VAR:.*]] = arith.subi %[[IND_VAR]], %[[C1536_I32]] : i32
// CHECK:               %[[PRELOAD2_LOCAL_BUFFER1:.*]] = hivm.hir.pointer_cast(%[[LOCALBUFFER1_ADDR0]], %[[LOCALBUFFER1_ADDR3]], %[[LOCALBUFFER1_ADDR2]], %[[LOCALBUFFER1_ADDR1]]) : memref<64xf32, #hivm.address_space<ub>>
// CHECK:               %[[PRELOAD0_LOCAL_BUFFER1:.*]] = hivm.hir.pointer_cast(%[[LOCALBUFFER1_ADDR2]], %[[LOCALBUFFER1_ADDR1]], %[[LOCALBUFFER1_ADDR0]], %[[LOCALBUFFER1_ADDR3]]) : memref<64xf32, #hivm.address_space<ub>>
// CHECK:               annotation.mark %[[PRELOAD0_LOCAL_BUFFER1]] {hivm.multi_buffer = 4 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
// CHECK:               annotation.mark %[[PRELOAD2_LOCAL_BUFFER1]] {hivm.multi_buffer = 4 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
// CHECK-DAG:           annotation.mark %[[PRELOAD_WORKSPACE1]] : memref<4x128x512xbf16, #hivm.address_space<gm>>
// CHECK-DAG:           annotation.mark %[[PRELOAD_WORKSPACE2]] : memref<4x128x128xf32, #hivm.address_space<gm>>
// CHECK-DAG:           annotation.mark %[[PRELOAD_WORKSPACE3]] : memref<4x128x512xf32, #hivm.address_space<gm>>
// CHECK:               %[[PRELOAD2_LB:.*]] = arith.cmpi sge, %[[PRELOAD2_IND_VAR]], %[[C0_I32]] : i32
// CHECK:               %[[PRELOAD2_UB:.*]] = arith.cmpi slt, %[[PRELOAD2_IND_VAR]], %[[C8192_I32]] : i32
// CHECK:               %[[PRELOAD2_COND:.*]] = arith.andi %[[PRELOAD2_LB]], %[[PRELOAD2_UB]] : i1
// CHECK:               %[[ITER1_RET:.*]] = scf.if %[[PRELOAD2_COND]] -> (memref<64xf32, #hivm.address_space<ub>>) {
// CHECK:                 %[[VAL_113:.*]] = arith.divsi %[[PRELOAD2_IND_VAR]], %[[C512_I32]] : i32
// CHECK:                 %[[VAL_114:.*]] = arith.remsi %[[VAL_113]], %[[C4_I32]] : i32
// CHECK:                 %[[PRELOAD2_OFFSET:.*]] = arith.index_cast %[[VAL_114]] : i32 to index
// CHECK:                 %[[VAL_116:.*]] = memref.subview %[[PRELOAD_WORKSPACE1]]{{\[}}%[[PRELOAD2_OFFSET]], 0, 0] [1, 128, 512] [1, 1, 1] : memref<4x128x512xbf16, #hivm.address_space<gm>> to memref<1x128x512xbf16, strided<[65536, 512, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:                 %[[VAL_120:.*]] = memref.subview %[[PRELOAD_WORKSPACE3]][0, %[[VAL_69:.*]], 0] [4, 64, 512] [1, 1, 1] {to_be_bubbled_slice} : memref<4x128x512xf32, #hivm.address_space<gm>> to memref<4x64x512xf32, strided<[65536, 512, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:                 %[[VAL_121:.*]] = memref.subview %[[VAL_120]]{{\[}}%[[PRELOAD2_OFFSET]], 0, 0] [1, 64, 512] [1, 1, 1] : memref<4x64x512xf32, strided<[65536, 512, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:                 %[[PRELOAD2_ITER1:.*]] = hivm.hir.pointer_cast(%[[ITER1_ADDR]]) : memref<64xf32, #hivm.address_space<ub>>
// CHECK:                 scf.for %[[VAL_125:.*]] = %[[VAL_50:.*]] to %[[VAL_36:.*]] step %[[VAL_51:.*]] {
// CHECK:                   %[[VAL_160:.*]] = memref.subview %[[PRELOAD2_LOCAL_BUFFER1]]{{\[}}%[[VAL_129:.*]]] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK:                   %[[VAL_161:.*]] = memref.subview %[[ITER1]]{{\[}}%[[VAL_129:.*]]] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK:                 }
// CHECK:                 scf.yield %[[PRELOAD2_ITER1]] : memref<64xf32, #hivm.address_space<ub>>
// CHECK:               } else {
// CHECK:                 scf.yield %[[ITER1]] : memref<64xf32, #hivm.address_space<ub>>
// CHECK:               }
// CHECK:               %[[PRELOAD0_LB:.*]] = arith.cmpi sge, %[[PRELOAD0_IND_VAR]], %[[C0_I32]] : i32
// CHECK:               %[[PRELOAD0_UB:.*]] = arith.cmpi slt, %[[PRELOAD0_IND_VAR]], %[[C8192_I32]] : i32
// CHECK:               %[[PRELOAD0_COND:.*]] = arith.andi %[[PRELOAD0_LB]], %[[PRELOAD0_UB]] : i1
// CHECK:               %[[ITER2_RET:.*]] = scf.if %[[PRELOAD0_COND]] -> (memref<64x128xf32, #hivm.address_space<ub>>) {
// CHECK:                 %[[VAL_171:.*]] = memref.expand_shape %[[PRELOAD0_LOCAL_BUFFER1]] {{\[\[}}0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
// CHECK:                 hivm.hir.vmul ins(%[[ITER2]], %[[VAL_171]] : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%[[VAL_172:.*]] : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%[[VAL_173:.*]] : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
// CHECK:                 %[[VAL_174:.*]] = memref.subview %[[PRELOAD_WORKSPACE2]][0, %[[VAL_69]], 0] [4, 64, 128] [1, 1, 1] {to_be_bubbled_slice} : memref<4x128x128xf32, #hivm.address_space<gm>> to memref<4x64x128xf32, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:                 %[[VAL_175:.*]] = arith.divsi %[[PRELOAD0_IND_VAR]], %[[C512_I32]] : i32
// CHECK:                 %[[VAL_176:.*]] = arith.remsi %[[VAL_175]], %[[C4_I32]] : i32
// CHECK:                 %[[PRELOAD0_OFFSET:.*]] = arith.index_cast %[[VAL_176]] : i32 to index
// CHECK:                 %[[VAL_178:.*]] = memref.subview %[[VAL_174]]{{\[}}%[[PRELOAD0_OFFSET]], 0, 0] [1, 64, 128] [1, 1, 1] : memref<4x64x128xf32, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK:                 %[[PRELOAD0_ITER2:.*]] = hivm.hir.pointer_cast(%[[ITER2_ADDR]]) : memref<64x128xf32, #hivm.address_space<ub>>
// CHECK:                 scf.yield %[[PRELOAD0_ITER2]] : memref<64x128xf32, #hivm.address_space<ub>>
// CHECK:               } else {
// CHECK:                 scf.yield %[[ITER2]] : memref<64x128xf32, #hivm.address_space<ub>>
// CHECK:               }
// CHECK:               scf.yield %[[ITER1_RET]], %[[ITER2_RET]] : memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>
// CHECK:             }
// CHECK:           return
// CHECK:         }
  func.func @_sdpa_infer_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: f32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.enable_saving_ub, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c32768_i64 = arith.constant 32768 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c149248_i64 = arith.constant 149248 : i64
    %c116480_i64 = arith.constant 116480 : i64
    %c49152_i64 = arith.constant 49152 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c81920_i64 = arith.constant 81920 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %c99584_i64 = arith.constant 99584 : i64
    %c99328_i64 = arith.constant 99328 : i64
    %c99072_i64 = arith.constant 99072 : i64
    %c98816_i64 = arith.constant 98816 : i64
    %c116224_i64 = arith.constant 116224 : i64
    %c99840_i64 = arith.constant 99840 : i64
    %c98560_i64 = arith.constant 98560 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c149312_i64 = arith.constant 149312 : i64
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8388608_i64 = arith.constant 8388608 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg12, %arg11 : i32
    %5 = hivm.hir.pointer_cast(%c149312_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c98304_i64) : memref<64xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c98560_i64) : memref<64xf32, #hivm.address_space<ub>>
    %8 = arith.index_cast %2 : i64 to index
    %9 = affine.apply #map()[%8]
    %10 = affine.apply #map2()[%8]
    %11 = affine.apply #map1()[%8]
    %view = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xbf16, #hivm.address_space<gm>>
    %view_3 = memref.view %arg2[%11][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x128xf32, #hivm.address_space<gm>>
    %view_4 = memref.view %arg2[%10][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x128x512xf32, #hivm.address_space<gm>>
    %12 = hivm.hir.get_sub_block_idx -> i64
    %13 = arith.index_cast %12 : i64 to index
    %14 = affine.apply #map4()[%13]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    annotation.mark %1 {logical_block_num} : i32
    %15 = arith.divsi %3, %4 : i32
    %16 = arith.remsi %15, %arg10 : i32
    scf.for %arg13 = %16 to %c512_i32 step %arg10  : i32 {
      %17 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
      annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<64x128xbf16, #hivm.address_space<ub>>
      %18 = arith.divsi %arg13, %c64_i32 : i32
      %19 = arith.remsi %arg13, %c64_i32 : i32
      %20 = arith.divsi %18, %c8_i32 : i32
      %21 = arith.remsi %18, %c8_i32 : i32
      %22 = arith.extsi %20 : i32 to i64
      %23 = arith.muli %22, %c8388608_i64 : i64
      %24 = arith.extsi %21 : i32 to i64
      %25 = arith.muli %24, %c1048576_i64 : i64
      %26 = arith.addi %23, %25 : i64
      %27 = arith.index_cast %26 : i64 to index
      %28 = arith.muli %19, %c128_i32 : i32
      %29 = arith.maxsi %28, %c0_i32 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = affine.apply #map3()[%27, %30]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%31], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %32 = arith.muli %19, %c1048576_i32 : i32
      %33 = arith.index_cast %32 : i32 to index
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%7 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%6 : memref<64xf32, #hivm.address_space<ub>>)
      %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
      %34:3 = scf.for %arg14 = %c0_i32 to %c8192_i32 step %c512_i32 iter_args(%arg15 = %7, %arg16 = %5, %arg17 = %6) -> (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i32 {
        %37 = hivm.hir.pointer_cast(%c98816_i64, %c99072_i64, %c99328_i64, %c99584_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %37 {hivm.multi_buffer = 4 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        %38 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %view_4 : memref<4x128x512xf32, #hivm.address_space<gm>>
        annotation.mark %view : memref<4x128x512xbf16, #hivm.address_space<gm>>
        annotation.mark %view_3 : memref<4x128x128xf32, #hivm.address_space<gm>>
        %39:3 = scope.scope : () -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
          %41 = arith.index_cast %arg14 : i32 to index
          %42 = affine.apply #map5()[%33, %41]
          %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%42], sizes: [128, 512], strides: [8192, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<128x512xi8, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_9 = memref.subview %view[0, 0, 0] [1, 128, 512] [1, 1, 1] {hivm.preload_workspace} : memref<4x128x512xbf16, #hivm.address_space<gm>> to memref<1x128x512xbf16, strided<[65536, 512, 1]>, #hivm.address_space<gm>>
          %collapse_shape_10 = memref.collapse_shape %subview_9 [[0, 1], [2]] : memref<1x128x512xbf16, strided<[65536, 512, 1]>, #hivm.address_space<gm>> into memref<128x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 3
          %subview_11 = memref.subview %reinterpret_cast_8[%14, 0] [64, 512] [1, 1] : memref<128x512xi8, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x512xi8, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %43 = hivm.hir.pointer_cast(%c99840_i64) : memref<16x512xf16, #hivm.address_space<ub>>
          %subview_12 = memref.subview %view_4[0, %14, 0] [4, 64, 512] [1, 1, 1] {to_be_bubbled_slice} : memref<4x128x512xf32, #hivm.address_space<gm>> to memref<4x64x512xf32, strided<[65536, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_13 = memref.subview %subview_12[0, 0, 0] [1, 64, 512] [1, 1, 1] {hivm.preload_workspace} : memref<4x64x512xf32, strided<[65536, 512, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_14 = memref.subview %collapse_shape_10[%14, 0] [64, 512] [1, 1] {to_be_bubbled_slice} : memref<128x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<64x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %44 = hivm.hir.pointer_cast(%c116224_i64) : memref<64xf32, #hivm.address_space<ub>>
          %45 = hivm.hir.pointer_cast(%c98560_i64) : memref<64xf32, #hivm.address_space<ub>>
          scf.for %arg18 = %c0 to %c4 step %c1 {
            %46 = hivm.hir.pointer_cast(%c16384_i64, %c49152_i64) : memref<16x512xf32, #hivm.address_space<ub>>
            annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<16x512xf32, #hivm.address_space<ub>>
            %47 = hivm.hir.pointer_cast(%c0_i64, %c8192_i64) : memref<16x512xi8, #hivm.address_space<ub>>
            annotation.mark %47 {hivm.multi_buffer = 2 : i32} : memref<16x512xi8, #hivm.address_space<ub>>
            %48 = hivm.hir.pointer_cast(%c0_i64, %c81920_i64) : memref<16x512xbf16, #hivm.address_space<ub>>
            annotation.mark %48 {hivm.multi_buffer = 2 : i32} : memref<16x512xbf16, #hivm.address_space<ub>>
            %49 = affine.apply #map6()[%arg18]
            %subview_15 = memref.subview %subview_11[%49, 0] [16, 512] [1, 1] : memref<64x512xi8, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x512xi8, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.load ins(%subview_15 : memref<16x512xi8, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) outs(%47 : memref<16x512xi8, #hivm.address_space<ub>>) {vector_producer_to_fuse_1} init_out_buffer = false may_implicit_transpose_with_last_axis = false
            %50 = hivm.hir.pointer_cast(%c81920_i64) : memref<16x512xf16, #hivm.address_space<ub>>
            %collapse_shape_16 = memref.collapse_shape %47 [[0, 1]] : memref<16x512xi8, #hivm.address_space<ub>> into memref<8192xi8, #hivm.address_space<ub>>
            %collapse_shape_17 = memref.collapse_shape %50 [[0, 1]] : memref<16x512xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
            hivm.hir.vcast {vector_producer_to_fuse_1} ins(%collapse_shape_16 : memref<8192xi8, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<8192xf16, #hivm.address_space<ub>>)
            %51 = hivm.hir.pointer_cast(%c81920_i64) : memref<16x512xi1, #hivm.address_space<ub>>
            %collapse_shape_18 = memref.collapse_shape %43 [[0, 1]] : memref<16x512xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
            hivm.hir.vbrc {vector_producer_to_fuse_1} ins(%cst : f16) outs(%collapse_shape_18 : memref<8192xf16, #hivm.address_space<ub>>)
            %collapse_shape_19 = memref.collapse_shape %51 [[0, 1]] : memref<16x512xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
            hivm.hir.vcmp {vector_producer_to_fuse_1} ins(%collapse_shape_17, %collapse_shape_18 : memref<8192xf16, #hivm.address_space<ub>>, memref<8192xf16, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<8192xi1, #hivm.address_space<ub>>)
            %52 = hivm.hir.pointer_cast(%c81920_i64) : memref<16x512xi1, #hivm.address_space<ub>>
            %collapse_shape_20 = memref.collapse_shape %52 [[0, 1]] : memref<16x512xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
            hivm.hir.vnot {vector_producer_to_fuse_1} ins(%collapse_shape_19 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<8192xi1, #hivm.address_space<ub>>)
            %subview_21 = memref.subview %subview_13[%49, 0] [16, 512] [1, 1] : memref<64x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_22 = memref.collapse_shape %subview_21 [[0, 1]] : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_23 = memref.collapse_shape %46 [[0, 1]] : memref<16x512xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.load ins(%collapse_shape_22 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_23 : memref<8192xf32, #hivm.address_space<ub>>) {vector_producer_to_fuse_1} init_out_buffer = false may_implicit_transpose_with_last_axis = false
            %collapse_shape_24 = memref.collapse_shape %46 [[0, 1]] : memref<16x512xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.vmul {vector_producer_to_fuse_1} ins(%collapse_shape_23, %arg9 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>)
            %53 = hivm.hir.pointer_cast(%c116480_i64) : memref<16x512xf32, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %53 [[0, 1]] : memref<16x512xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            %54 = hivm.hir.pointer_cast(%c149248_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.vsel {vector_producer_to_fuse_1} ins(%collapse_shape_20, %cst_1, %collapse_shape_24 : memref<8192xi1, #hivm.address_space<ub>>, f32, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<16xf32, #hivm.address_space<ub>>)
            %55 = hivm.hir.pointer_cast(%c0_i64) : memref<16x1xf32, #hivm.address_space<ub>>
            %56 = hivm.hir.pointer_cast(%c64_i64) : memref<1024xf32, #hivm.address_space<ub>>
            hivm.hir.vreduce {vector_producer_to_fuse_1} <max> ins(%53 : memref<16x512xf32, #hivm.address_space<ub>>) outs(%55 : memref<16x1xf32, #hivm.address_space<ub>>) temp_buffer(%56 : memref<1024xf32, #hivm.address_space<ub>>) reduce_dims = [1]
            %collapse_shape_26 = memref.collapse_shape %55 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
            %subview_27 = memref.subview %arg17[%49] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            %subview_28 = memref.subview %44[%49] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmax {vector_producer_to_fuse_1} ins(%subview_27, %collapse_shape_26 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%subview_28 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>)
            %expand_shape_29 = memref.expand_shape %subview_28 [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            %57 = hivm.hir.pointer_cast(%c116480_i64) : memref<16x512xf32, #hivm.address_space<ub>>
            %58 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
            hivm.hir.vsub {vector_producer_to_fuse_1} ins(%53, %expand_shape_29 : memref<16x512xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%57 : memref<16x512xf32, #hivm.address_space<ub>>) temp_buffer(%58 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
            %59 = hivm.hir.pointer_cast(%c116480_i64) : memref<16x512xf32, #hivm.address_space<ub>>
            %collapse_shape_30 = memref.collapse_shape %57 [[0, 1]] : memref<16x512xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %59 [[0, 1]] : memref<16x512xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.vexp {vector_producer_to_fuse_1} ins(%collapse_shape_30 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<8192xf32, #hivm.address_space<ub>>)
            %collapse_shape_32 = memref.collapse_shape %48 [[0, 1]] : memref<16x512xbf16, #hivm.address_space<ub>> into memref<8192xbf16, #hivm.address_space<ub>>
            hivm.hir.vcast {vector_producer_to_fuse_1} ins(%collapse_shape_31 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<8192xbf16, #hivm.address_space<ub>>)
            %subview_33 = memref.subview %subview_14[%49, 0] [16, 512] [1, 1] : memref<64x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_34 = memref.collapse_shape %subview_33 [[0, 1]] : memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.store ins(%collapse_shape_32 : memref<8192xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_1_0, tiled_op}
            %60 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.vsub {vector_producer_to_fuse_1} ins(%subview_27, %subview_28 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%60 : memref<16xf32, #hivm.address_space<ub>>)
            %subview_35 = memref.subview %37[%49] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp {vector_producer_to_fuse_1} ins(%60 : memref<16xf32, #hivm.address_space<ub>>) outs(%subview_35 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>)
            %subview_36 = memref.subview %arg15[%49] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            %61 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.vmul {vector_producer_to_fuse_1} ins(%subview_36, %subview_35 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%61 : memref<16xf32, #hivm.address_space<ub>>)
            %62 = hivm.hir.pointer_cast(%c64_i64) : memref<16x1xf32, #hivm.address_space<ub>>
            %63 = hivm.hir.pointer_cast(%c128_i64) : memref<1024xf32, #hivm.address_space<ub>>
            hivm.hir.vreduce {vector_producer_to_fuse_1} <sum> ins(%59 : memref<16x512xf32, #hivm.address_space<ub>>) outs(%62 : memref<16x1xf32, #hivm.address_space<ub>>) temp_buffer(%63 : memref<1024xf32, #hivm.address_space<ub>>) reduce_dims = [1]
            %collapse_shape_37 = memref.collapse_shape %62 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
            %subview_38 = memref.subview %45[%49] [16] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vadd {vector_producer_to_fuse_1} ins(%61, %collapse_shape_37 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%subview_38 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>)
          }
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 4
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          hivm.hir.copy ins(%44 : memref<64xf32, #hivm.address_space<ub>>) outs(%arg17 : memref<64xf32, #hivm.address_space<ub>>)
          scope.return %37, %arg17, %45 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
        %40 = scope.scope : () -> memref<64x128xf32, #hivm.address_space<ub>> {
          %expand_shape_8 = memref.expand_shape %39#0 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
          %41 = hivm.hir.pointer_cast(%c65536_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %42 = hivm.hir.pointer_cast(%c0_i64) : memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%arg16, %expand_shape_8 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%41 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 3
          %subview_9 = memref.subview %view_3[0, %14, 0] [4, 64, 128] [1, 1, 1] {to_be_bubbled_slice} : memref<4x128x128xf32, #hivm.address_space<gm>> to memref<4x64x128xf32, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %subview_9[0, 0, 0] [1, 64, 128] [1, 1, 1] {hivm.preload_workspace} : memref<4x64x128xf32, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1]] : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_12 = memref.collapse_shape %38 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.load ins(%collapse_shape_11 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_12 : memref<8192xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
          %43 = hivm.hir.pointer_cast(%c149312_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_13 = memref.collapse_shape %41 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %collapse_shape_14 = memref.collapse_shape %43 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vadd ins(%collapse_shape_12, %collapse_shape_13 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<8192xf32, #hivm.address_space<ub>>)
          scope.return %43 : memref<64x128xf32, #hivm.address_space<ub>>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
        scf.yield %39#2, %40, %39#1 : memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
      }
      annotation.mark %34#2 {hivm.tile_and_bind_leaf} : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      %expand_shape = memref.expand_shape %34#0 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
      %35 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %36 = hivm.hir.pointer_cast(%c98816_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.vdiv ins(%34#1, %expand_shape : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%35 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%36 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
      %collapse_shape_5 = memref.collapse_shape %35 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %collapse_shape_6 = memref.collapse_shape %17 [[0, 1]] : memref<64x128xbf16, #hivm.address_space<ub>> into memref<8192xbf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%collapse_shape_5 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<8192xbf16, #hivm.address_space<ub>>)
      %subview = memref.subview %reinterpret_cast[%14, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_7 = memref.collapse_shape %subview [[0, 1]] : memref<64x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_6 : memref<8192xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    }
    return
  }
}