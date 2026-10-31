// RUN: bishengir-opt -outline-cv-scopes %s | FileCheck %s

// CHECK-LABEL: func.func @_attn_fwd
// CHECK: scf.for {{.*}} iter_args(%[[ACC0:.*]] = {{.*}}, %[[ACC1:.*]] = {{.*}}, %[[ACC2:.*]] = {{.*}}, %[[CNT0:.*]] = {{.*}}, %[[CNT1:.*]] = {{.*}}) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, i32, i32) : i32 {
// CHECK:   %[[S0:.*]] = scope.scope : () -> tensor<64x64xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<64x64xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S1:.*]] = scope.scope : () -> i32 {
// CHECK:     %[[ADD0:.*]] = arith.addi %[[CNT0]], {{.*}} : i32
// CHECK:     scope.return %[[ADD0]] : i32
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK:   %[[S2:.*]] = scope.scope : () -> i32 {
// CHECK:     %[[ADD1:.*]] = arith.addi %[[CNT1]], {{.*}} : i32
// CHECK:     scope.return %[[ADD1]] : i32
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK:   %[[S3:.*]] = scope.scope : () -> tensor<64x64xf32> {
// CHECK:     hivm.hir.vmul ins(%[[S0]], {{.*}})
// CHECK:     scope.return {{.*}} : tensor<64x64xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S4:.*]] = scope.scope : () -> tensor<64xf32> {
// CHECK:     hivm.hir.vreduce <max> ins(%[[S3]] : tensor<64x64xf32>)
// CHECK:     hivm.hir.vmax ins(%[[ACC2]], {{.*}})
// CHECK:     scope.return {{.*}} : tensor<64xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S5:.*]] = scope.scope : () -> tensor<64x64xf32> {
// CHECK:     hivm.hir.vsub ins(%[[S3]], {{.*}})
// CHECK:     hivm.hir.vexp
// CHECK:     scope.return {{.*}} : tensor<64x64xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S6:.*]] = scope.scope : () -> tensor<64xf32> {
// CHECK:     hivm.hir.vsub ins(%[[ACC2]], %[[S4]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:     hivm.hir.vexp
// CHECK:     scope.return {{.*}} : tensor<64xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S7:.*]] = scope.scope : () -> tensor<4x4x16x16xf16> {
// CHECK:     hivm.hir.vcast {{.*}} ins(%[[S5]] : tensor<64x64xf32>)
// CHECK:     hivm.hir.vtranspose
// CHECK:     hivm.hir.copy
// CHECK:     scope.return {{.*}} : tensor<4x4x16x16xf16>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S8:.*]] = scope.scope : () -> tensor<64xf32> {
// CHECK:     hivm.hir.vreduce <sum> ins(%[[S5]] : tensor<64x64xf32>)
// CHECK:     hivm.hir.vmul ins(%[[ACC0]], %[[S6]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:     hivm.hir.vadd
// CHECK:     scope.return {{.*}} : tensor<64xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S9:.*]] = scope.scope : () -> tensor<64x128xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     hivm.hir.mmadL1 {{.*}} ins(%[[S7]], {{.*}})
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<64x128xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S10:.*]] = scope.scope : () -> tensor<64x128xf32> {
// CHECK:     hivm.hir.vmul ins(%[[ACC1]], {{.*}})
// CHECK:     hivm.hir.vadd ins(%[[S9]], {{.*}})
// CHECK:     scope.return {{.*}} : tensor<64x128xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S8]], %[[S10]], %[[S4]], %[[S1]], %[[S2]] : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, i32, i32
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true, hivm.cv_pipelined_loop, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
func.func @_attn_fwd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c256_i32 = arith.constant 256 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32768_i64 = arith.constant 32768 : i64
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_0 = arith.constant 5.000000e-01 : f32
  %cst_1 = arith.constant 0xFF800000 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  %c28_i32 = arith.constant 28 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %4 = arith.remsi %arg10, %arg7 : i32
    %5 = tensor.empty() : tensor<64x128xf32>
    %6 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%5 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %7 = tensor.empty() : tensor<64xf32>
    %8 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_1 : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
    %9 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_2 : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
    scf.for %arg11 = %4 to %c4_i32 step %c32_i32  : i32 {
      %10 = arith.divsi %arg11, %c4_i32 : i32
      %11 = arith.remsi %arg11, %c4_i32 : i32
      %12 = arith.extsi %10 : i32 to i64
      %13 = arith.muli %12, %c32768_i64 : i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.muli %11, %c64_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%14, %16]
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [64, 128], strides: [128, 1] : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [64, 128], strides: [128, 1] : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %alloc = memref.alloc() : memref<8x4x16x16xf16>
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<64x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc : memref<8x4x16x16xf16>)
      %18 = bufferization.to_tensor %alloc restrict writable : memref<8x4x16x16xf16>
      %19:5 = scf.for %arg12 = %c0_i32 to %c256_i32 step %c64_i32 iter_args(%arg13 = %9, %arg14 = %6, %arg15 = %8, %arg16 = %c0_i32, %arg17 = %c0_i32) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, i32, i32)  : i32 {
        %28 = arith.index_cast %arg17 : i32 to index
        %29 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%14, %28]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [64, 128], strides: [128, 1] : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %30 = arith.index_cast %arg16 : i32 to index
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%14, %30]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [64, 128], strides: [128, 1] : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<8x4x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_5 : memref<64x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc_7 : memref<8x4x16x16xf16>)
        %32 = bufferization.to_tensor %alloc_7 restrict writable : memref<8x4x16x16xf16>
        %33 = tensor.empty() : tensor<4x4x16x16xf32>
        %34 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%18, %32, %true, %c64, %c128, %c64 : tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>, i1, index, index, index) outs(%33 : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
        %alloc_8 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x64xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_8 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %35 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%34 : tensor<4x4x16x16xf32>) outs(%alloc_8 : memref<64x64xf32, #hivm.address_space<ub>>)
        %36 = tensor.empty() : tensor<64x64xf32>
        %37 = hivm.hir.vmul ins(%35, %cst_0 : tensor<64x64xf32>, f32) outs(%36 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %expanded_9 = tensor.expand_shape %8 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %38 = hivm.hir.vreduce <max> ins(%37 : tensor<64x64xf32>) outs(%expanded_9 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
        %collapsed = tensor.collapse_shape %38 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %39 = hivm.hir.vmax ins(%arg15, %collapsed : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %expanded_10 = tensor.expand_shape %39 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %40 = hivm.hir.vsub ins(%37, %expanded_10 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%36 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
        %41 = hivm.hir.vexp ins(%40 : tensor<64x64xf32>) outs(%36 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %42 = tensor.empty() : tensor<64x64xf16>
        %43 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%41 : tensor<64x64xf32>) outs(%42 : tensor<64x64xf16>) -> tensor<64x64xf16>
        %alloc_11 = memref.alloc() : memref<8x4x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_6 : memref<64x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc_11 : memref<8x4x16x16xf16>)
        %44 = bufferization.to_tensor %alloc_11 restrict writable : memref<8x4x16x16xf16>
        %45 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %expanded_12 = tensor.expand_shape %45 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %46 = hivm.hir.vreduce <sum> ins(%41 : tensor<64x64xf32>) outs(%expanded_12 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
        %collapsed_13 = tensor.collapse_shape %46 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %47 = hivm.hir.vsub ins(%arg15, %39 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %48 = hivm.hir.vexp ins(%47 : tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %49 = hivm.hir.vmul ins(%arg13, %48 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %50 = hivm.hir.vadd ins(%49, %collapsed_13 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %expanded_14 = tensor.expand_shape %48 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %51 = hivm.hir.vmul ins(%arg14, %expanded_14 : tensor<64x128xf32>, tensor<64x1xf32>) outs(%5 : tensor<64x128xf32>) broadcast = [1] -> tensor<64x128xf32>
        %expanded_15 = tensor.expand_shape %43 [[0], [1, 2]] output_shape [64, 4, 16] : tensor<64x64xf16> into tensor<64x4x16xf16>
        %52 = tensor.empty() : tensor<4x64x16xf16>
        %53 = hivm.hir.vtranspose ins(%expanded_15 : tensor<64x4x16xf16>) outs(%52 : tensor<4x64x16xf16>) permutation = [1, 0, 2] -> tensor<4x64x16xf16>
        %expanded_16 = tensor.expand_shape %53 [[0], [1, 2], [3]] output_shape [4, 4, 16, 16] : tensor<4x64x16xf16> into tensor<4x4x16x16xf16>
        %alloc_17 = memref.alloc() : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
        %54 = bufferization.to_tensor %alloc_17 restrict writable : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_16 : tensor<4x4x16x16xf16>) outs(%alloc_17 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %55 = tensor.empty() : tensor<8x4x16x16xf32>
        %56 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%54, %44, %true, %c64, %c64, %c128 : tensor<4x4x16x16xf16>, tensor<8x4x16x16xf16>, i1, index, index, index) outs(%55 : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        %alloc_18 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x128xf32, #hivm.address_space<ub>>
        %memspacecast_19 = memref.memory_space_cast %alloc_18 : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %57 = bufferization.to_tensor %memspacecast_19 restrict writable : memref<64x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%56 : tensor<8x4x16x16xf32>) outs(%alloc_18 : memref<64x128xf32, #hivm.address_space<ub>>)
        %58 = hivm.hir.vadd ins(%57, %51 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%5 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %59 = arith.addi %arg16, %c64_i32 : i32
        %60 = arith.addi %arg17, %c64_i32 : i32
        scf.yield %50, %58, %39, %59, %60 : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, i32, i32
      } {fixpipe_for_mmad_result_already_inserted = true, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
      %20 = hivm.hir.vln ins(%19#0 : tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
      %21 = hivm.hir.vadd ins(%19#2, %20 : tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
      %expanded = tensor.expand_shape %19#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %22 = hivm.hir.vdiv ins(%19#1, %expanded : tensor<64x128xf32>, tensor<64x1xf32>) outs(%5 : tensor<64x128xf32>) broadcast = [1] -> tensor<64x128xf32>
      %23 = arith.muli %10, %c256_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %16]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%21 : tensor<64xf32>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>>)
      %26 = tensor.empty() : tensor<64x128xf16>
      %27 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%22 : tensor<64x128xf32>) outs(%26 : tensor<64x128xf16>) -> tensor<64x128xf16>
      hivm.hir.store ins(%27 : tensor<64x128xf16>) outs(%reinterpret_cast_3 : memref<64x128xf16, strided<[128, 1], offset: ?>>)
    }
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  return
}
}
