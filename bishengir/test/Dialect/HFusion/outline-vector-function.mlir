// RUN: bishengir-opt %s -outline-vector-function -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

#map = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0) -> (d0, 0)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1)>
#map7 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8388608_i64 = arith.constant 8388608 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c28_i32 = arith.constant 28 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg10, %arg9 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg8 : i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_S>] flag = 2
    scf.for %arg11 = %6 to %c65536_i32 step %c28_i32  : i32 {
      %7 = arith.divsi %arg11, %c64_i32 : i32
      %8 = arith.remsi %arg11, %c64_i32 : i32
      %9 = arith.divsi %7, %c8_i32 : i32
      %10 = arith.remsi %7, %c8_i32 : i32
      %11 = arith.extsi %9 : i32 to i64
      %12 = arith.muli %11, %c8388608_i64 : i64
      %13 = arith.extsi %10 : i32 to i64
      %14 = arith.muli %13, %c1048576_i64 : i64
      %15 = arith.addi %12, %14 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.muli %8, %c128_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = affine.apply #map()[%16, %18]
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xf16>
      hivm.hir.load ins(%reinterpret_cast : memref<128x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc : memref<128x128xf16>) eviction_policy = <EvictFirst>
      %20 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
      %21:2 = scf.for %arg12 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
        %22 = arith.index_cast %arg14 : i32 to index
        %23 = affine.apply #map()[%16, %22]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %24 = arith.index_cast %arg13 : i32 to index
        %25 = affine.apply #map()[%16, %24]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<128x128xf16>
        hivm.hir.load ins(%reinterpret_cast_0 : memref<128x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc_2 : memref<128x128xf16>) eviction_policy = <EvictFirst>
        %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<128x128xf16>
        %27 = tensor.empty() : tensor<128x128xf32>
        %28 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%20, %26, %true, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index) outs(%27 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %alloc_3 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_S>] flag = 0
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%28 : tensor<128x128xf32>) outs(%alloc_3 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %alloc_4 = memref.alloc() : memref<128x128xf16>
        hivm.hir.load ins(%reinterpret_cast_1 : memref<128x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc_4 : memref<128x128xf16>) eviction_policy = <EvictFirst>
        %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<128x128xf16>
        %alloc_5 = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %alloc_5 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<8x8x16x16xf16>
        %30 = bufferization.to_tensor %memspacecast restrict writable : memref<8x8x16x16xf16>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
        %31 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%30, %29, %true, %c128, %c128, %c128 : tensor<8x8x16x16xf16>, tensor<128x128xf16>, i1, index, index, index) outs(%27 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_S>] flag = 2
        %alloc_6 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_S>] flag = 3
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%31 : tensor<128x128xf32>) outs(%alloc_6 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %32 = arith.addi %arg13, %c128_i32 : i32
        %33 = arith.addi %arg14, %c128_i32 : i32
        scf.yield %32, %33 : i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    }
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_S>] flag = 3
    return
  }
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c28_i32 = arith.constant 28 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg10, %arg9 : i32
    %5 = tensor.empty() : tensor<64x128xf32>
    %6 = scf.for %arg11 = %c0 to %c64 step %c1 iter_args(%arg12 = %5) -> (tensor<64x128xf32>) {
      %13 = scf.for %arg13 = %c0 to %c128 step %c64 iter_args(%arg14 = %arg12) -> (tensor<64x128xf32>) {
        %extracted_slice = tensor.extract_slice %arg14[%arg11, %arg13] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
        %c1_3 = arith.constant 1 : index
        %c64_4 = arith.constant 64 : index
        %c0_5 = arith.constant 0 : index
        %cst_6 = arith.constant 0.000000e+00 : f32
        %14 = vector.transfer_read %extracted_slice[%c0_5, %c0_5], %cst_6 : tensor<1x64xf32>, vector<1x64xf32>
        %c0_7 = arith.constant 0 : index
        %cst_8 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
        %15 = vector.transfer_write %cst_8, %extracted_slice[%c0_7, %c0_7] : vector<1x64xf32>, tensor<1x64xf32>
        %inserted_slice = tensor.insert_slice %15 into %arg14[%arg11, %arg13] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
        scf.yield %inserted_slice : tensor<64x128xf32>
      }
      scf.yield %13 : tensor<64x128xf32>
    } {"outlined-loop-target-9"}
    %7 = tensor.empty() : tensor<64xf32>
    %8:2 = scf.for %arg11 = %c0 to %c64 step %c64 iter_args(%arg12 = %7, %arg13 = %7) -> (tensor<64xf32>, tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg12[%arg11] [64] [1] : tensor<64xf32> to tensor<64xf32>
      %c64_3 = arith.constant 64 : index
      %c0_4 = arith.constant 0 : index
      %cst_5 = arith.constant 0.000000e+00 : f32
      %13 = vector.transfer_read %extracted_slice[%c0_4], %cst_5 : tensor<64xf32>, vector<64xf32>
      %c0_6 = arith.constant 0 : index
      %cst_7 = arith.constant dense<0xFF800000> : vector<64xf32>
      %14 = vector.transfer_write %cst_7, %extracted_slice[%c0_6] : vector<64xf32>, tensor<64xf32>
      %inserted_slice = tensor.insert_slice %14 into %arg12[%arg11] [64] [1] : tensor<64xf32> into tensor<64xf32>
      %extracted_slice_8 = tensor.extract_slice %arg13[%arg11] [64] [1] : tensor<64xf32> to tensor<64xf32>
      %c64_9 = arith.constant 64 : index
      %c0_10 = arith.constant 0 : index
      %cst_11 = arith.constant 0.000000e+00 : f32
      %15 = vector.transfer_read %extracted_slice_8[%c0_10], %cst_11 : tensor<64xf32>, vector<64xf32>
      %c0_12 = arith.constant 0 : index
      %cst_13 = arith.constant dense<1.000000e+00> : vector<64xf32>
      %16 = vector.transfer_write %cst_13, %extracted_slice_8[%c0_12] : vector<64xf32>, tensor<64xf32>
      %inserted_slice_14 = tensor.insert_slice %16 into %arg13[%arg11] [64] [1] : tensor<64xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %inserted_slice_14 : tensor<64xf32>, tensor<64xf32>
    } {"outlined-loop-target-10"}
    %expanded = tensor.expand_shape %8#1 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %9 = tensor.empty() : tensor<64x8x16xf16>
    %10 = tensor.empty() : tensor<8x64x16xf16>
    %11 = tensor.empty() : tensor<64x128xf16>
    %12 = tensor.empty() : tensor<1x64xf32>
    scf.for %arg11 = %c0 to %c2 step %c1 {
      %13 = affine.apply #map1()[%arg11]
      %14 = affine.apply #map2()[%arg11]
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      annotation.mark %1 {logical_block_num} : i32
      %15 = arith.divsi %3, %4 : i32
      %16 = arith.remsi %15, %arg8 : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 3
      %17 = arith.muli %13, %c16 : index
      scf.for %arg12 = %16 to %c65536_i32 step %c28_i32  : i32 {
        %18 = arith.divsi %arg12, %c64_i32 : i32
        %19 = arith.remsi %arg12, %c64_i32 : i32
        %20 = arith.divsi %18, %c8_i32 : i32
        %21 = arith.remsi %18, %c8_i32 : i32
        %22 = arith.extsi %20 : i32 to i64
        %23 = arith.muli %22, %c8388608_i64 : i64
        %24 = arith.extsi %21 : i32 to i64
        %25 = arith.muli %24, %c1048576_i64 : i64
        %26 = arith.addi %23, %25 : i64
        %27 = arith.index_cast %26 : i64 to index
        %28 = arith.muli %19, %c128_i32 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = affine.apply #map()[%27, %29]
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%30], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %31:5 = scf.for %arg13 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg14 = %expanded, %arg15 = %6, %arg16 = %8#0, %arg17 = %c0_i32, %arg18 = %c0_i32) -> (tensor<64x1xf32>, tensor<64x128xf32>, tensor<64xf32>, i32, i32)  : i32 {
          %collapsed_5 = tensor.collapse_shape %arg14 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
          %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x128xf32, #hivm.address_space<ub>>
          %memspacecast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
          %37 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %38 = scf.for %arg19 = %c0 to %c64 step %c1 iter_args(%arg20 = %5) -> (tensor<64x128xf32>) {
            %48 = scf.for %arg21 = %c0 to %c128 step %c64 iter_args(%arg22 = %arg20) -> (tensor<64x128xf32>) {
              %extracted_slice = tensor.extract_slice %37[%arg19, %arg21] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %extracted_slice_12 = tensor.extract_slice %arg22[%arg19, %arg21] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %c1_13 = arith.constant 1 : index
              %c64_14 = arith.constant 64 : index
              %c0_15 = arith.constant 0 : index
              %cst_16 = arith.constant 0.000000e+00 : f32
              %49 = vector.transfer_read %extracted_slice[%c0_15, %c0_15], %cst_16 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_17 = arith.constant 0.000000e+00 : f32
              %50 = vector.transfer_read %extracted_slice_12[%c0_15, %c0_15], %cst_17 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_18 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
              %51 = arith.mulf %49, %cst_18 : vector<1x64xf32>
              %c0_19 = arith.constant 0 : index
              %52 = vector.transfer_write %51, %extracted_slice_12[%c0_19, %c0_19] : vector<1x64xf32>, tensor<1x64xf32>
              %inserted_slice = tensor.insert_slice %52 into %arg22[%arg19, %arg21] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
              scf.yield %inserted_slice : tensor<64x128xf32>
            }
            scf.yield %48 : tensor<64x128xf32>
          } {"outlined-loop-target-6"}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
          %39 = scf.for %arg19 = %c0 to %c64 step %c1 iter_args(%arg20 = %7) -> (tensor<64xf32>) {
            %extracted_slice = tensor.extract_slice %arg20[%arg19] [1] [1] : tensor<64xf32> to tensor<1xf32>
            %c1_12 = arith.constant 1 : index
            %c64_13 = arith.constant 64 : index
            %c0_14 = arith.constant 0 : index
            %cst_15 = arith.constant 0.000000e+00 : f32
            %48 = vector.transfer_read %12[%c0_14, %c0_14], %cst_15 : tensor<1x64xf32>, vector<1x64xf32>
            %c0_16 = arith.constant 0 : index
            %cst_17 = arith.constant dense<0xFF800000> : vector<1x64xf32>
            %49 = vector.transfer_write %cst_17, %12[%c0_16, %c0_16] : vector<1x64xf32>, tensor<1x64xf32>
            %50 = scf.for %arg21 = %c0 to %c128 step %c64 iter_args(%arg22 = %49) -> (tensor<1x64xf32>) {
              %extracted_slice_24 = tensor.extract_slice %38[%arg19, %arg21] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %c1_25 = arith.constant 1 : index
              %c64_26 = arith.constant 64 : index
              %c0_27 = arith.constant 0 : index
              %cst_28 = arith.constant 0.000000e+00 : f32
              %55 = vector.transfer_read %extracted_slice_24[%c0_27, %c0_27], %cst_28 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_29 = arith.constant 0.000000e+00 : f32
              %56 = vector.transfer_read %arg22[%c0_27, %c0_27], %cst_29 : tensor<1x64xf32>, vector<1x64xf32>
              %57 = arith.maximumf %55, %56 {reductionOp} : vector<1x64xf32>
              %c0_30 = arith.constant 0 : index
              %58 = vector.transfer_write %57, %arg22[%c0_30, %c0_30] : vector<1x64xf32>, tensor<1x64xf32>
              scf.yield %58 : tensor<1x64xf32>
            } {reductionLoop}
            %c1_18 = arith.constant 1 : index
            %c64_19 = arith.constant 64 : index
            %c0_20 = arith.constant 0 : index
            %cst_21 = arith.constant 0.000000e+00 : f32
            %51 = vector.transfer_read %50[%c0_20, %c0_20], %cst_21 : tensor<1x64xf32>, vector<1x64xf32>
            %cst_22 = arith.constant 0.000000e+00 : f32
            %52 = vector.transfer_read %extracted_slice[%c0_20], %cst_22 : tensor<1xf32>, vector<1xf32>
            %53 = vector.multi_reduction <maximumf>, %51, %52 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
            %c0_23 = arith.constant 0 : index
            %54 = vector.transfer_write %53, %extracted_slice[%c0_23] : vector<1xf32>, tensor<1xf32>
            %inserted_slice = tensor.insert_slice %54 into %arg20[%arg19] [1] [1] : tensor<1xf32> into tensor<64xf32>
            scf.yield %inserted_slice : tensor<64xf32>
          } {"outlined-loop-target-5"}
          %40 = scf.for %arg19 = %c0 to %c64 step %c64 iter_args(%arg20 = %7) -> (tensor<64xf32>) {
            %extracted_slice = tensor.extract_slice %arg16[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %extracted_slice_12 = tensor.extract_slice %39[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %extracted_slice_13 = tensor.extract_slice %arg20[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %c64_14 = arith.constant 64 : index
            %c0_15 = arith.constant 0 : index
            %cst_16 = arith.constant 0.000000e+00 : f32
            %48 = vector.transfer_read %extracted_slice[%c0_15], %cst_16 : tensor<64xf32>, vector<64xf32>
            %cst_17 = arith.constant 0.000000e+00 : f32
            %49 = vector.transfer_read %extracted_slice_12[%c0_15], %cst_17 : tensor<64xf32>, vector<64xf32>
            %cst_18 = arith.constant 0.000000e+00 : f32
            %50 = vector.transfer_read %extracted_slice_13[%c0_15], %cst_18 : tensor<64xf32>, vector<64xf32>
            %51 = arith.maximumf %48, %49 : vector<64xf32>
            %c0_19 = arith.constant 0 : index
            %52 = vector.transfer_write %51, %extracted_slice_13[%c0_19] : vector<64xf32>, tensor<64xf32>
            %inserted_slice = tensor.insert_slice %52 into %arg20[%arg19] [64] [1] : tensor<64xf32> into tensor<64xf32>
            scf.yield %inserted_slice : tensor<64xf32>
          } {"outlined-loop-target-4"}
          %41:2 = scf.for %arg19 = %c0 to %c64 step %c1 iter_args(%arg20 = %7, %arg21 = %10) -> (tensor<64xf32>, tensor<8x64x16xf16>) {
            %extracted_slice = tensor.extract_slice %arg20[%arg19] [1] [1] : tensor<64xf32> to tensor<1xf32>
            %c1_12 = arith.constant 1 : index
            %c64_13 = arith.constant 64 : index
            %c0_14 = arith.constant 0 : index
            %cst_15 = arith.constant 0.000000e+00 : f32
            %48 = vector.transfer_read %12[%c0_14, %c0_14], %cst_15 : tensor<1x64xf32>, vector<1x64xf32>
            %c0_16 = arith.constant 0 : index
            %cst_17 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
            %49 = vector.transfer_write %cst_17, %12[%c0_16, %c0_16] : vector<1x64xf32>, tensor<1x64xf32>
            %extracted_slice_18 = tensor.extract_slice %40[%arg19] [1] [1] : tensor<64xf32> to tensor<1xf32>
            %50:2 = scf.for %arg22 = %c0 to %c8 step %c4 iter_args(%arg23 = %49, %arg24 = %arg21) -> (tensor<1x64xf32>, tensor<8x64x16xf16>) {
              %55 = affine.apply #map3(%arg22)
              %extracted_slice_25 = tensor.extract_slice %38[%arg19, %55] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %extracted_slice_26 = tensor.extract_slice %5[%arg19, %55] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %c1_27 = arith.constant 1 : index
              %c64_28 = arith.constant 64 : index
              %c0_29 = arith.constant 0 : index
              %cst_30 = arith.constant 0.000000e+00 : f32
              %56 = vector.transfer_read %extracted_slice_25[%c0_29, %c0_29], %cst_30 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_31 = arith.constant 0.000000e+00 : f32
              %57 = vector.transfer_read %extracted_slice_18[%c0_29], %cst_31 {in_bounds = [false, true], permutation_map = #map4} : tensor<1xf32>, vector<1x64xf32>
              %cst_32 = arith.constant 0.000000e+00 : f32
              %58 = vector.transfer_read %extracted_slice_26[%c0_29, %c0_29], %cst_32 : tensor<1x64xf32>, vector<1x64xf32>
              %59 = arith.subf %56, %57 : vector<1x64xf32>
              %60 = math.exp %59 : vector<1x64xf32>
              %c0_33 = arith.constant 0 : index
              %61 = vector.transfer_write %60, %extracted_slice_26[%c0_33, %c0_33] : vector<1x64xf32>, tensor<1x64xf32>
              %c1_34 = arith.constant 1 : index
              %c64_35 = arith.constant 64 : index
              %c0_36 = arith.constant 0 : index
              %cst_37 = arith.constant 0.000000e+00 : f32
              %62 = vector.transfer_read %61[%c0_36, %c0_36], %cst_37 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_38 = arith.constant 0.000000e+00 : f32
              %63 = vector.transfer_read %arg23[%c0_36, %c0_36], %cst_38 : tensor<1x64xf32>, vector<1x64xf32>
              %64 = arith.addf %62, %63 {reductionOp} : vector<1x64xf32>
              %c0_39 = arith.constant 0 : index
              %65 = vector.transfer_write %64, %arg23[%c0_39, %c0_39] : vector<1x64xf32>, tensor<1x64xf32>
              %expanded_40 = tensor.expand_shape %61 [[0], [1, 2]] output_shape [64, 8, 16] : tensor<1x64xf32> into tensor<1x4x16xf32>
              %extracted_slice_41 = tensor.extract_slice %9[%arg19, %arg22, 0] [1, 4, 16] [1, 1, 1] : tensor<64x8x16xf16> to tensor<1x4x16xf16>
              %c1_42 = arith.constant 1 : index
              %c4_43 = arith.constant 4 : index
              %c16_44 = arith.constant 16 : index
              %c0_45 = arith.constant 0 : index
              %cst_46 = arith.constant 0.000000e+00 : f32
              %66 = vector.transfer_read %expanded_40[%c0_45, %c0_45, %c0_45], %cst_46 : tensor<1x4x16xf32>, vector<1x4x16xf32>
              %cst_47 = arith.constant 0.000000e+00 : f16
              %67 = vector.transfer_read %extracted_slice_41[%c0_45, %c0_45, %c0_45], %cst_47 : tensor<1x4x16xf16>, vector<1x4x16xf16>
              %68 = arith.truncf %66 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x4x16xf32> to vector<1x4x16xf16>
              %c0_48 = arith.constant 0 : index
              %69 = vector.transfer_write %68, %extracted_slice_41[%c0_48, %c0_48, %c0_48] : vector<1x4x16xf16>, tensor<1x4x16xf16>
              %extracted_slice_49 = tensor.extract_slice %arg24[%arg22, %arg19, 0] [4, 1, 16] [1, 1, 1] : tensor<8x64x16xf16> to tensor<4x1x16xf16>
              %c4_50 = arith.constant 4 : index
              %c1_51 = arith.constant 1 : index
              %c16_52 = arith.constant 16 : index
              %c0_53 = arith.constant 0 : index
              %cst_54 = arith.constant 0.000000e+00 : f16
              %70 = vector.transfer_read %69[%c0_53, %c0_53, %c0_53], %cst_54 {permutation_map = #map5} : tensor<1x4x16xf16>, vector<4x1x16xf16>
              %cst_55 = arith.constant 0.000000e+00 : f16
              %71 = vector.transfer_read %extracted_slice_49[%c0_53, %c0_53, %c0_53], %cst_55 : tensor<4x1x16xf16>, vector<4x1x16xf16>
              %c0_56 = arith.constant 0 : index
              %72 = vector.transfer_write %70, %extracted_slice_49[%c0_56, %c0_56, %c0_56] : vector<4x1x16xf16>, tensor<4x1x16xf16>
              %inserted_slice_57 = tensor.insert_slice %72 into %arg24[%arg22, %arg19, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<8x64x16xf16>
              scf.yield %65, %inserted_slice_57 : tensor<1x64xf32>, tensor<8x64x16xf16>
            } {reductionLoop, unroll_for_vsstb}
            %c1_19 = arith.constant 1 : index
            %c64_20 = arith.constant 64 : index
            %c0_21 = arith.constant 0 : index
            %cst_22 = arith.constant 0.000000e+00 : f32
            %51 = vector.transfer_read %50#0[%c0_21, %c0_21], %cst_22 : tensor<1x64xf32>, vector<1x64xf32>
            %cst_23 = arith.constant 0.000000e+00 : f32
            %52 = vector.transfer_read %extracted_slice[%c0_21], %cst_23 : tensor<1xf32>, vector<1xf32>
            %53 = vector.multi_reduction <add>, %51, %52 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
            %c0_24 = arith.constant 0 : index
            %54 = vector.transfer_write %53, %extracted_slice[%c0_24] : vector<1xf32>, tensor<1xf32>
            %inserted_slice = tensor.insert_slice %54 into %arg20[%arg19] [1] [1] : tensor<1xf32> into tensor<64xf32>
            scf.yield %inserted_slice, %50#1 : tensor<64xf32>, tensor<8x64x16xf16>
          } {"outlined-loop-target-2"}
          %42:2 = scf.for %arg19 = %c0 to %c64 step %c64 iter_args(%arg20 = %7, %arg21 = %7) -> (tensor<64xf32>, tensor<64xf32>) {
            %extracted_slice = tensor.extract_slice %collapsed_5[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %extracted_slice_12 = tensor.extract_slice %arg16[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %extracted_slice_13 = tensor.extract_slice %40[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %extracted_slice_14 = tensor.extract_slice %arg21[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %c64_15 = arith.constant 64 : index
            %c0_16 = arith.constant 0 : index
            %cst_17 = arith.constant 0.000000e+00 : f32
            %48 = vector.transfer_read %extracted_slice_12[%c0_16], %cst_17 : tensor<64xf32>, vector<64xf32>
            %cst_18 = arith.constant 0.000000e+00 : f32
            %49 = vector.transfer_read %extracted_slice_13[%c0_16], %cst_18 : tensor<64xf32>, vector<64xf32>
            %cst_19 = arith.constant 0.000000e+00 : f32
            %50 = vector.transfer_read %extracted_slice_14[%c0_16], %cst_19 : tensor<64xf32>, vector<64xf32>
            %51 = arith.subf %48, %49 : vector<64xf32>
            %52 = math.exp %51 : vector<64xf32>
            %c0_20 = arith.constant 0 : index
            %53 = vector.transfer_write %52, %extracted_slice_14[%c0_20] : vector<64xf32>, tensor<64xf32>
            %extracted_slice_21 = tensor.extract_slice %41#0[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %extracted_slice_22 = tensor.extract_slice %arg20[%arg19] [64] [1] : tensor<64xf32> to tensor<64xf32>
            %c64_23 = arith.constant 64 : index
            %c0_24 = arith.constant 0 : index
            %cst_25 = arith.constant 0.000000e+00 : f32
            %54 = vector.transfer_read %extracted_slice[%c0_24], %cst_25 : tensor<64xf32>, vector<64xf32>
            %cst_26 = arith.constant 0.000000e+00 : f32
            %55 = vector.transfer_read %53[%c0_24], %cst_26 : tensor<64xf32>, vector<64xf32>
            %cst_27 = arith.constant 0.000000e+00 : f32
            %56 = vector.transfer_read %extracted_slice_21[%c0_24], %cst_27 : tensor<64xf32>, vector<64xf32>
            %cst_28 = arith.constant 0.000000e+00 : f32
            %57 = vector.transfer_read %extracted_slice_22[%c0_24], %cst_28 : tensor<64xf32>, vector<64xf32>
            %58 = arith.mulf %54, %55 : vector<64xf32>
            %59 = arith.addf %58, %56 : vector<64xf32>
            %c0_29 = arith.constant 0 : index
            %60 = vector.transfer_write %59, %extracted_slice_22[%c0_29] : vector<64xf32>, tensor<64xf32>
            %inserted_slice = tensor.insert_slice %60 into %arg20[%arg19] [64] [1] : tensor<64xf32> into tensor<64xf32>
            %inserted_slice_30 = tensor.insert_slice %53 into %arg21[%arg19] [64] [1] : tensor<64xf32> into tensor<64xf32>
            scf.yield %inserted_slice, %inserted_slice_30 : tensor<64xf32>, tensor<64xf32>
          } {"outlined-loop-target-1"}
          %alloc_6 = memref.alloc() : memref<8x128x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8x128x16xf16, #hivm.address_space<cbuf>>
          %memspacecast_7 = memref.memory_space_cast %alloc_6 : memref<8x128x16xf16, #hivm.address_space<cbuf>> to memref<8x128x16xf16>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
          %subview_8 = memref.subview %memspacecast_7[0, %17, 0] [8, 64, 16] [1, 1, 1] : memref<8x128x16xf16> to memref<8x64x16xf16, strided<[2048, 16, 1], offset: ?>>
          hivm.hir.copy ins(%41#1 : tensor<8x64x16xf16>) outs(%subview_8 : memref<8x64x16xf16, strided<[2048, 16, 1], offset: ?>>) {tiled_op}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
          %alloc_9 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
          %memspacecast_10 = memref.memory_space_cast %alloc_9 : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
          %43 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<64x128xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %44 = scf.for %arg19 = %c0 to %c64 step %c1 iter_args(%arg20 = %5) -> (tensor<64x128xf32>) {
            %extracted_slice = tensor.extract_slice %42#1[%arg19] [1] [1] : tensor<64xf32> to tensor<1xf32>
            %48 = scf.for %arg21 = %c0 to %c128 step %c64 iter_args(%arg22 = %arg20) -> (tensor<64x128xf32>) {
              %extracted_slice_12 = tensor.extract_slice %43[%arg19, %arg21] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %extracted_slice_13 = tensor.extract_slice %arg15[%arg19, %arg21] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %extracted_slice_14 = tensor.extract_slice %arg22[%arg19, %arg21] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
              %c1_15 = arith.constant 1 : index
              %c64_16 = arith.constant 64 : index
              %c0_17 = arith.constant 0 : index
              %cst_18 = arith.constant 0.000000e+00 : f32
              %49 = vector.transfer_read %extracted_slice_12[%c0_17, %c0_17], %cst_18 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_19 = arith.constant 0.000000e+00 : f32
              %50 = vector.transfer_read %extracted_slice_13[%c0_17, %c0_17], %cst_19 : tensor<1x64xf32>, vector<1x64xf32>
              %cst_20 = arith.constant 0.000000e+00 : f32
              %51 = vector.transfer_read %extracted_slice[%c0_17], %cst_20 {in_bounds = [false, true], permutation_map = #map4} : tensor<1xf32>, vector<1x64xf32>
              %cst_21 = arith.constant 0.000000e+00 : f32
              %52 = vector.transfer_read %extracted_slice_14[%c0_17, %c0_17], %cst_21 : tensor<1x64xf32>, vector<1x64xf32>
              %53 = arith.mulf %50, %51 : vector<1x64xf32>
              %54 = arith.addf %49, %53 : vector<1x64xf32>
              %c0_22 = arith.constant 0 : index
              %55 = vector.transfer_write %54, %extracted_slice_14[%c0_22, %c0_22] : vector<1x64xf32>, tensor<1x64xf32>
              %inserted_slice = tensor.insert_slice %55 into %arg22[%arg19, %arg21] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
              scf.yield %inserted_slice : tensor<64x128xf32>
            }
            scf.yield %48 : tensor<64x128xf32>
          } {"outlined-loop-target-3"}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 3
          %45 = arith.addi %arg17, %c128_i32 : i32
          %46 = arith.addi %arg18, %c128_i32 : i32
          %expanded_11 = tensor.expand_shape %42#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
          %47 = hivm.hir.copy ins(%40 : tensor<64xf32>) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
          scf.yield %expanded_11, %44, %47, %45, %46 : tensor<64x1xf32>, tensor<64x128xf32>, tensor<64xf32>, i32, i32
        } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
        %collapsed = tensor.collapse_shape %31#0 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %32 = scf.for %arg13 = %c0 to %c64 step %c64 iter_args(%arg14 = %7) -> (tensor<64xf32>) {
          %extracted_slice = tensor.extract_slice %31#2[%arg13] [64] [1] : tensor<64xf32> to tensor<64xf32>
          %extracted_slice_5 = tensor.extract_slice %collapsed[%arg13] [64] [1] : tensor<64xf32> to tensor<64xf32>
          %extracted_slice_6 = tensor.extract_slice %arg14[%arg13] [64] [1] : tensor<64xf32> to tensor<64xf32>
          %c64_7 = arith.constant 64 : index
          %c0_8 = arith.constant 0 : index
          %cst_9 = arith.constant 0.000000e+00 : f32
          %37 = vector.transfer_read %extracted_slice[%c0_8], %cst_9 : tensor<64xf32>, vector<64xf32>
          %cst_10 = arith.constant 0.000000e+00 : f32
          %38 = vector.transfer_read %extracted_slice_5[%c0_8], %cst_10 : tensor<64xf32>, vector<64xf32>
          %cst_11 = arith.constant 0.000000e+00 : f32
          %39 = vector.transfer_read %extracted_slice_6[%c0_8], %cst_11 : tensor<64xf32>, vector<64xf32>
          %40 = math.log %38 : vector<64xf32>
          %41 = arith.addf %37, %40 : vector<64xf32>
          %c0_12 = arith.constant 0 : index
          %42 = vector.transfer_write %41, %extracted_slice_6[%c0_12] : vector<64xf32>, tensor<64xf32>
          %inserted_slice = tensor.insert_slice %42 into %arg14[%arg13] [64] [1] : tensor<64xf32> into tensor<64xf32>
          scf.yield %inserted_slice : tensor<64xf32>
        } {"outlined-loop-target-7"}
        %33 = arith.muli %18, %c8192_i32 : i32
        %34 = arith.index_cast %33 : i32 to index
        %35 = affine.apply #map6()[%34, %29]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%35], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        %subview = memref.subview %reinterpret_cast_3[%14] [64] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
        hivm.hir.store ins(%32 : tensor<64xf32>) outs(%subview : memref<64xf32, strided<[1], offset: ?>>) {tiled_op}
        %36 = scf.for %arg13 = %c0 to %c64 step %c1 iter_args(%arg14 = %11) -> (tensor<64x128xf16>) {
          %extracted_slice = tensor.extract_slice %31#0[%arg13, 0] [1, 1] [1, 1] : tensor<64x1xf32> to tensor<1x1xf32>
          %37 = scf.for %arg15 = %c0 to %c128 step %c64 iter_args(%arg16 = %arg14) -> (tensor<64x128xf16>) {
            %extracted_slice_5 = tensor.extract_slice %31#1[%arg13, %arg15] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
            %extracted_slice_6 = tensor.extract_slice %arg16[%arg13, %arg15] [1, 64] [1, 1] : tensor<64x128xf16> to tensor<1x64xf16>
            %c1_7 = arith.constant 1 : index
            %c64_8 = arith.constant 64 : index
            %c0_9 = arith.constant 0 : index
            %cst_10 = arith.constant 0.000000e+00 : f32
            %38 = vector.transfer_read %extracted_slice_5[%c0_9, %c0_9], %cst_10 : tensor<1x64xf32>, vector<1x64xf32>
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.transfer_read %extracted_slice[%c0_9, %c0_9], %cst_11 {in_bounds = [false, true], permutation_map = #map7} : tensor<1x1xf32>, vector<1x64xf32>
            %cst_12 = arith.constant 0.000000e+00 : f16
            %40 = vector.transfer_read %extracted_slice_6[%c0_9, %c0_9], %cst_12 : tensor<1x64xf16>, vector<1x64xf16>
            %41 = arith.divf %38, %39 : vector<1x64xf32>
            %42 = arith.truncf %41 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
            %c0_13 = arith.constant 0 : index
            %43 = vector.transfer_write %42, %extracted_slice_6[%c0_13, %c0_13] : vector<1x64xf16>, tensor<1x64xf16>
            %inserted_slice = tensor.insert_slice %43 into %arg16[%arg13, %arg15] [1, 64] [1, 1] : tensor<1x64xf16> into tensor<64x128xf16>
            scf.yield %inserted_slice : tensor<64x128xf16>
          }
          scf.yield %37 : tensor<64x128xf16>
        } {"outlined-loop-target-8"}
        %subview_4 = memref.subview %reinterpret_cast[%14, 0] [64, 128] [1, 1] : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        hivm.hir.store ins(%36 : tensor<64x128xf16>) outs(%subview_4 : memref<64x128xf16, strided<[128, 1], offset: ?>>) {tiled_op}
      }
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_S>] flag = 2
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}
// CHECK: func.func @_attn_fwd_mix_aiv_outlined_vf_0
// CHECK: func.func @_attn_fwd_mix_aiv_outlined_vf_1
// CHECK: func.func @_attn_fwd_mix_aiv_outlined_vf_2
// CHECK: func.func @_attn_fwd_mix_aiv_outlined_vf_3
// CHECK: func.func @_attn_fwd_mix_aiv_outlined_vf_4
// CHECK: func.func @_attn_fwd_mix_aiv_outlined_vf_5