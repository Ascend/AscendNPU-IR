// RUN: bishengir-opt %s -hivm-bind-sub-block='batch-matmul=true' -split-input-file -verify-diagnostics | FileCheck %s
// Case 3 (rank-3 ND [batch, M, N]): a batched matmul whose result feeds a
// vector op drains L0C into a rank-3 UB buffer.  The AIV consumer tiles the
// batch axis, which is neither rank-2 nor rank-1 and so cannot be expressed as
// a ROW/COLUMN split -> tileAndSliceFailure -> CV1:1 fallback.  Letting this
// through instead would leave the drain NO_DUAL while the AIV side stays
// sub-tiled, so one sub-block reads UB that was never written.

// CHECK-LABEL: func.func @_batched_mix_aic(
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.+}} : tensor<2x64x32xf32>) outs(%{{.+}} : memref<2x64x32xf32, #hivm.address_space<ub>>){{$}}
// CHECK-LABEL: func.func @_batched_mix_aiv(
// CHECK-NOT: mapping = [#hivm.sub_block
#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0, s1] -> (s0 * s1)>
#map2 = affine_map<()[s0] -> (s0 * 2048)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @_batched_backup(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module">, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.anchor {id = 0 : i64}
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c28_i32 = arith.constant 28 : i32
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg13 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg13, %arg10 : i32
      %5 = arith.muli %4, %c16_i32 : i32
      hivm.hir.anchor {id = 3 : i64}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        hivm.hir.anchor {id = 4 : i64}
        %alloc = memref.alloc() : memref<2x64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x64x32xf32, #hivm.address_space<ub>>
        hivm.hir.anchor {id = 5 : i64}
        %6 = scope.scope : () -> tensor<2x1x32xf32> {
          hivm.hir.anchor {id = 6 : i64}
          %8 = arith.muli %arg14, %c2_i32 : i32
          %9 = arith.addi %5, %8 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = affine.apply #map()[%10]
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%11], sizes: [2, 1, 32], strides: [32, 32, 1] : memref<?xf16> to memref<2x1x32xf16, strided<[32, 32, 1], offset: ?>>
          %alloc_0 = memref.alloc() : memref<2x1x32xf16>
          hivm.hir.anchor {id = 7 : i64}
          hivm.hir.load ins(%reinterpret_cast : memref<2x1x32xf16, strided<[32, 32, 1], offset: ?>>) outs(%alloc_0 : memref<2x1x32xf16>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} eviction_policy = <EvictFirst> core_type = <VECTOR>
          hivm.hir.anchor {id = 8 : i64}
          %12 = bufferization.to_tensor %alloc_0 restrict writable : memref<2x1x32xf16>
          %13 = tensor.empty() : tensor<2x1x32xf32>
          hivm.hir.anchor {id = 9 : i64}
          %14 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%12 : tensor<2x1x32xf16>) outs(%13 : tensor<2x1x32xf32>) -> tensor<2x1x32xf32>
          hivm.hir.anchor {id = 10 : i64}
          scope.return %14 : tensor<2x1x32xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
        hivm.hir.anchor {id = 11 : i64}
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        scope.scope : () -> () {
          hivm.hir.anchor {id = 12 : i64}
          %8 = arith.muli %arg14, %c2_i32 : i32
          %9 = arith.addi %5, %8 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = arith.index_cast %arg6 : i32 to index
          %12 = affine.apply #map1()[%10, %11]
          %13 = arith.index_cast %arg7 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [2, 64, 32], strides: [%11, %13, 1] : memref<?xf16> to memref<2x64x32xf16, strided<[?, ?, 1], offset: ?>>
          %alloc_0 = memref.alloc() : memref<2x64x32xf16>
          hivm.hir.anchor {id = 13 : i64}
          hivm.hir.load ins(%reinterpret_cast : memref<2x64x32xf16, strided<[?, ?, 1], offset: ?>>) outs(%alloc_0 : memref<2x64x32xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} eviction_policy = <EvictFirst>
          hivm.hir.anchor {id = 14 : i64}
          %14 = bufferization.to_tensor %alloc_0 restrict writable : memref<2x64x32xf16>
          %15 = arith.index_cast %arg8 : i32 to index
          %16 = affine.apply #map1()[%10, %15]
          %17 = arith.index_cast %arg9 : i32 to index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [2, 32, 32], strides: [%15, %17, 1] : memref<?xf16> to memref<2x32x32xf16, strided<[?, ?, 1], offset: ?>>
          %alloc_2 = memref.alloc() : memref<2x32x32xf16>
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.load ins(%reinterpret_cast_1 : memref<2x32x32xf16, strided<[?, ?, 1], offset: ?>>) outs(%alloc_2 : memref<2x32x32xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} eviction_policy = <EvictFirst>
          hivm.hir.anchor {id = 16 : i64}
          %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x32x32xf16>
          %19 = tensor.empty() : tensor<2x64x32xf32>
          hivm.hir.anchor {id = 17 : i64}
          %20 = hivm.hir.batchMmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%14, %18, %true, %c64, %c32, %c32 : tensor<2x64x32xf16>, tensor<2x32x32xf16>, i1, index, index, index) outs(%19 : tensor<2x64x32xf32>) -> tensor<2x64x32xf32>
          hivm.hir.anchor {id = 18 : i64}
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x32xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%20 : tensor<2x64x32xf32>) outs(%alloc : memref<2x64x32xf32, #hivm.address_space<ub>>)
          hivm.hir.anchor {id = 20 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
        hivm.hir.anchor {id = 21 : i64}
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %memspacecast = memref.memory_space_cast %alloc : memref<2x64x32xf32, #hivm.address_space<ub>> to memref<2x64x32xf32>
        %7 = bufferization.to_tensor %memspacecast restrict writable : memref<2x64x32xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        hivm.hir.anchor {id = 22 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 23 : i64}
          %8 = arith.muli %arg14, %c2_i32 : i32
          %9 = arith.addi %5, %8 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = tensor.empty() : tensor<2x64x32xf32>
          hivm.hir.anchor {id = 24 : i64}
          %12 = hivm.hir.vadd ins(%7, %6 : tensor<2x64x32xf32>, tensor<2x1x32xf32>) outs(%11 : tensor<2x64x32xf32>) broadcast = [1] -> tensor<2x64x32xf32>
          hivm.hir.anchor {id = 25 : i64}
          %13 = affine.apply #map2()[%10]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1], offset: ?>>
          hivm.hir.store ins(%12 : tensor<2x64x32xf32>) outs(%reinterpret_cast : memref<2x64x32xf32, strided<[2048, 32, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 26 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 27 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 28 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.anchor {id = 29 : i64}
    return
  }
  func.func @_batched_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    hivm.hir.anchor {id = 0 : i64}
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg13 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg13, %arg10 : i32
      %5 = arith.muli %4, %c16_i32 : i32
      hivm.hir.anchor {id = 3 : i64}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        hivm.hir.anchor {id = 4 : i64}
        %alloc = memref.alloc() : memref<2x64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x64x32xf32, #hivm.address_space<ub>>
        hivm.hir.anchor {id = 5 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 6 : i64}
          hivm.hir.anchor {id = 7 : i64}
          hivm.hir.anchor {id = 8 : i64}
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
        hivm.hir.anchor {id = 11 : i64}
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        scope.scope : () -> () {
          hivm.hir.anchor {id = 12 : i64}
          %6 = arith.muli %arg14, %c2_i32 : i32
          %7 = arith.addi %5, %6 : i32
          %8 = arith.index_cast %7 : i32 to index
          %9 = arith.index_cast %arg6 : i32 to index
          %10 = affine.apply #map1()[%8, %9]
          %11 = arith.index_cast %arg7 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%10], sizes: [2, 64, 32], strides: [%9, %11, 1] : memref<?xf16> to memref<2x64x32xf16, strided<[?, ?, 1], offset: ?>>
          %alloc_0 = memref.alloc() : memref<2x64x32xf16>
          hivm.hir.anchor {id = 13 : i64}
          hivm.hir.load ins(%reinterpret_cast : memref<2x64x32xf16, strided<[?, ?, 1], offset: ?>>) outs(%alloc_0 : memref<2x64x32xf16>) eviction_policy = <EvictFirst>
          hivm.hir.anchor {id = 14 : i64}
          %12 = bufferization.to_tensor %alloc_0 restrict writable : memref<2x64x32xf16>
          %13 = arith.index_cast %arg8 : i32 to index
          %14 = affine.apply #map1()[%8, %13]
          %15 = arith.index_cast %arg9 : i32 to index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [2, 32, 32], strides: [%13, %15, 1] : memref<?xf16> to memref<2x32x32xf16, strided<[?, ?, 1], offset: ?>>
          %alloc_2 = memref.alloc() : memref<2x32x32xf16>
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.load ins(%reinterpret_cast_1 : memref<2x32x32xf16, strided<[?, ?, 1], offset: ?>>) outs(%alloc_2 : memref<2x32x32xf16>) eviction_policy = <EvictFirst>
          hivm.hir.anchor {id = 16 : i64}
          %16 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x32x32xf16>
          %17 = tensor.empty() : tensor<2x64x32xf32>
          hivm.hir.anchor {id = 17 : i64}
          %18 = hivm.hir.batchMmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%12, %16, %true, %c64, %c32, %c32 : tensor<2x64x32xf16>, tensor<2x32x32xf16>, i1, index, index, index) outs(%17 : tensor<2x64x32xf32>) -> tensor<2x64x32xf32>
          hivm.hir.anchor {id = 18 : i64}
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x32xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 19 : i64}
          // expected-warning @below {{The tilingDim in AIC does not match row_split or column split!}}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%18 : tensor<2x64x32xf32>) outs(%alloc : memref<2x64x32xf32, #hivm.address_space<ub>>)
          hivm.hir.anchor {id = 20 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
        hivm.hir.anchor {id = 21 : i64}
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        hivm.hir.anchor {id = 22 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 23 : i64}
          hivm.hir.anchor {id = 24 : i64}
          hivm.hir.anchor {id = 25 : i64}
          hivm.hir.anchor {id = 26 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 27 : i64}
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 28 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.anchor {id = 29 : i64}
    return
  }
  func.func @_batched_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    hivm.hir.anchor {id = 0 : i64}
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg13 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg13, %arg10 : i32
      %5 = arith.muli %4, %c16_i32 : i32
      hivm.hir.anchor {id = 3 : i64}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        hivm.hir.anchor {id = 4 : i64}
        %alloc = memref.alloc() : memref<2x64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x64x32xf32, #hivm.address_space<ub>>
        hivm.hir.anchor {id = 5 : i64}
        %6 = scope.scope : () -> tensor<2x1x32xf32> {
          hivm.hir.anchor {id = 6 : i64}
          %8 = arith.muli %arg14, %c2_i32 : i32
          %9 = arith.addi %5, %8 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = affine.apply #map()[%10]
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%11], sizes: [2, 1, 32], strides: [32, 32, 1] : memref<?xf16> to memref<2x1x32xf16, strided<[32, 32, 1], offset: ?>>
          %alloc_0 = memref.alloc() : memref<2x1x32xf16>
          hivm.hir.anchor {id = 7 : i64}
          hivm.hir.load ins(%reinterpret_cast : memref<2x1x32xf16, strided<[32, 32, 1], offset: ?>>) outs(%alloc_0 : memref<2x1x32xf16>) eviction_policy = <EvictFirst> core_type = <VECTOR>
          hivm.hir.anchor {id = 8 : i64}
          %12 = bufferization.to_tensor %alloc_0 restrict writable : memref<2x1x32xf16>
          %13 = tensor.empty() : tensor<2x1x32xf32>
          hivm.hir.anchor {id = 9 : i64}
          %14 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%12 : tensor<2x1x32xf16>) outs(%13 : tensor<2x1x32xf32>) -> tensor<2x1x32xf32>
          hivm.hir.anchor {id = 10 : i64}
          scope.return %14 : tensor<2x1x32xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
        hivm.hir.anchor {id = 11 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 12 : i64}
          hivm.hir.anchor {id = 13 : i64}
          hivm.hir.anchor {id = 14 : i64}
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.anchor {id = 16 : i64}
          hivm.hir.anchor {id = 17 : i64}
          hivm.hir.anchor {id = 18 : i64}
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x64x32xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.anchor {id = 20 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
        hivm.hir.anchor {id = 21 : i64}
        %memspacecast = memref.memory_space_cast %alloc : memref<2x64x32xf32, #hivm.address_space<ub>> to memref<2x64x32xf32>
        %7 = bufferization.to_tensor %memspacecast restrict writable : memref<2x64x32xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        hivm.hir.anchor {id = 22 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 23 : i64}
          %8 = arith.muli %arg14, %c2_i32 : i32
          %9 = arith.addi %5, %8 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = tensor.empty() : tensor<2x64x32xf32>
          hivm.hir.anchor {id = 24 : i64}
          %12 = hivm.hir.vadd ins(%7, %6 : tensor<2x64x32xf32>, tensor<2x1x32xf32>) outs(%11 : tensor<2x64x32xf32>) broadcast = [1] -> tensor<2x64x32xf32>
          hivm.hir.anchor {id = 25 : i64}
          %13 = affine.apply #map2()[%10]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1], offset: ?>>
          hivm.hir.store ins(%12 : tensor<2x64x32xf32>) outs(%reinterpret_cast : memref<2x64x32xf32, strided<[2048, 32, 1], offset: ?>>)
          hivm.hir.anchor {id = 26 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 27 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 28 : i64}
    } {autoblockify.subloop}
    hivm.hir.anchor {id = 29 : i64}
    return
  }
}
