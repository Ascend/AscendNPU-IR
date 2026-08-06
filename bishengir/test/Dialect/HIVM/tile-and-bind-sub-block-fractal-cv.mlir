// RUN: bishengir-opt %s -hivm-bind-sub-block -split-input-file -verify-diagnostics | FileCheck %s

// Test: 4D fractal [N1, M1, 16, 16] CV tightly-coupled UB buffer. When the
// AIV consumer is sub-tiled along a block dim, the AIC fixpipe must
// dual-deliver (ROW/COLUMN_SPLIT); NO_DUAL would starve the other sub-block.
// Case 1 (even N1 = 6): consumer tiles dim0 -> fixpipe gets
// dual_dst_mode = <COLUMN_SPLIT> and the dst view is halved to [3,10,16,16].

// CHECK-LABEL: func.func @s_CVC_vreshape1d_kernel_mix_aic(
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3x10x16x16xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]] {{.*}}tiledAlloc{{.*}} : memref<3x10x16x16xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe ins(%{{.+}} : tensor<6x10x16x16xf32>) outs(%[[ALLOC]] : memref<3x10x16x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
// CHECK-LABEL: func.func @s_CVC_vreshape1d_kernel_mix_aiv(
// CHECK: scf.for %{{.+}} = %c0 to %c2 step %c1
// CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 36 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 36 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 72 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9599">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @s_CVC_vreshape1d_kernel_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c96 = arith.constant 96 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [20, 10, 16, 16], strides: [2560, 256, 16, 1] : memref<?xf16> to memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<20x10x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>) outs(%alloc : memref<20x10x16x16xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<20x10x16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [6, 20, 16, 16], strides: [5120, 256, 16, 1] : memref<?xf16> to memref<6x20x16x16xf16, strided<[5120, 256, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<6x20x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<6x20x16x16xf16, strided<[5120, 256, 16, 1]>>) outs(%alloc_1 : memref<6x20x16x16xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<6x20x16x16xf16>
    %4 = tensor.empty() : tensor<6x10x16x16xf32>
    %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%2, %3, %true, %c160, %c320, %c96 : tensor<20x10x16x16xf16>, tensor<6x20x16x16xf16>, i1, index, index, index) outs(%4 : tensor<6x10x16x16xf32>) -> tensor<6x10x16x16xf32>
    %alloc_2 = memref.alloc() : memref<6x10x16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<6x10x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe ins(%5 : tensor<6x10x16x16xf32>) outs(%alloc_2 : memref<6x10x16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<?xf16> to memref<4x6x16x16xf16, strided<[1536, 256, 16, 1]>>
    %alloc_4 = memref.alloc() : memref<4x6x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_3 : memref<4x6x16x16xf16, strided<[1536, 256, 16, 1]>>) outs(%alloc_4 : memref<4x6x16x16xf16>) eviction_policy = <EvictFirst>
    %6 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x6x16x16xf16>
    %alloc_5 = memref.alloc() : memref<6x10x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<6x10x16x16xf16, #hivm.address_space<cbuf>>
    %7 = bufferization.to_tensor %alloc_5 restrict writable : memref<6x10x16x16xf16, #hivm.address_space<cbuf>>
    %8 = tensor.empty() : tensor<4x10x16x16xf32>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    %9 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%7, %6, %true, %c160, %c96, %c64 : tensor<6x10x16x16xf16>, tensor<4x6x16x16xf16>, i1, index, index, index) outs(%8 : tensor<4x10x16x16xf32>) -> tensor<4x10x16x16xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 10, 16, 16], strides: [2560, 256, 16, 1] : memref<?xf32> to memref<4x10x16x16xf32, strided<[2560, 256, 16, 1]>>
    hivm.hir.fixpipe ins(%9 : tensor<4x10x16x16xf32>) outs(%reinterpret_cast_6 : memref<4x10x16x16xf32, strided<[2560, 256, 16, 1]>>)
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
  func.func @s_CVC_vreshape1d_kernel_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [20, 10, 16, 16], strides: [2560, 256, 16, 1] : memref<?xf16> to memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<20x10x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>) outs(%alloc : memref<20x10x16x16xf16>) eviction_policy = <EvictFirst>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [6, 20, 16, 16], strides: [5120, 256, 16, 1] : memref<?xf16> to memref<6x20x16x16xf16, strided<[5120, 256, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<6x20x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<6x20x16x16xf16, strided<[5120, 256, 16, 1]>>) outs(%alloc_1 : memref<6x20x16x16xf16>) eviction_policy = <EvictFirst>
    %alloc_2 = memref.alloc() : memref<6x10x16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<6x10x16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_2 : memref<6x10x16x16xf32, #hivm.address_space<ub>> to memref<6x10x16x16xf32>
    %2 = bufferization.to_tensor %memspacecast restrict writable : memref<6x10x16x16xf32>
    %collapsed = tensor.collapse_shape %2 [[0, 1, 2, 3]] : tensor<6x10x16x16xf32> into tensor<15360xf32>
    %3 = tensor.empty() : tensor<15360xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %4 = hivm.hir.vexp ins(%collapsed : tensor<15360xf32>) outs(%3 : tensor<15360xf32>) -> tensor<15360xf32>
    %5 = tensor.empty() : tensor<15360xf16>
    %6 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%4 : tensor<15360xf32>) outs(%5 : tensor<15360xf16>) -> tensor<15360xf16>
    %expanded = tensor.expand_shape %6 [[0, 1, 2, 3]] output_shape [6, 10, 16, 16] : tensor<15360xf16> into tensor<6x10x16x16xf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<?xf16> to memref<4x6x16x16xf16, strided<[1536, 256, 16, 1]>>
    %alloc_4 = memref.alloc() : memref<4x6x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_3 : memref<4x6x16x16xf16, strided<[1536, 256, 16, 1]>>) outs(%alloc_4 : memref<4x6x16x16xf16>) eviction_policy = <EvictFirst>
    %alloc_5 = memref.alloc() : memref<6x10x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<6x10x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%expanded : tensor<6x10x16x16xf16>) outs(%alloc_5 : memref<6x10x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}



// -----

// Case 2 (odd N1 = 5): cannot halve -> tileAndSliceFailure -> CV1:1 fallback;
// the fixpipe keeps the full [5,10,16,16] view with NO_DUAL, no sub-block loop.

// CHECK-LABEL: func.func @s_CV_epilogue_kernel_mix_aic(
// CHECK: hivm.hir.fixpipe ins(%{{.+}} : tensor<5x10x16x16xf32>) outs(%{{.+}} : memref<5x10x16x16xf32, #hivm.address_space<ub>>){{$}}
// CHECK-LABEL: func.func @s_CV_epilogue_kernel_mix_aiv(
// CHECK-NOT: mapping = [#hivm.sub_block
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 36 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 36 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 72 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9599">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @s_CV_epilogue_kernel_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [20, 10, 16, 16], strides: [2560, 256, 16, 1] : memref<?xf16> to memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<20x10x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>) outs(%alloc : memref<20x10x16x16xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<20x10x16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [5, 20, 16, 16], strides: [5120, 256, 16, 1] : memref<?xf16> to memref<5x20x16x16xf16, strided<[5120, 256, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<5x20x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<5x20x16x16xf16, strided<[5120, 256, 16, 1]>>) outs(%alloc_1 : memref<5x20x16x16xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<5x20x16x16xf16>
    %4 = tensor.empty() : tensor<5x10x16x16xf32>
    %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%2, %3, %true, %c160, %c320, %c80 : tensor<20x10x16x16xf16>, tensor<5x20x16x16xf16>, i1, index, index, index) outs(%4 : tensor<5x10x16x16xf32>) -> tensor<5x10x16x16xf32>
    %alloc_2 = memref.alloc() : memref<5x10x16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<5x10x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe ins(%5 : tensor<5x10x16x16xf32>) outs(%alloc_2 : memref<5x10x16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
  func.func @s_CV_epilogue_kernel_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [20, 10, 16, 16], strides: [2560, 256, 16, 1] : memref<?xf16> to memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<20x10x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<20x10x16x16xf16, strided<[2560, 256, 16, 1]>>) outs(%alloc : memref<20x10x16x16xf16>) eviction_policy = <EvictFirst>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [5, 20, 16, 16], strides: [5120, 256, 16, 1] : memref<?xf16> to memref<5x20x16x16xf16, strided<[5120, 256, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<5x20x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<5x20x16x16xf16, strided<[5120, 256, 16, 1]>>) outs(%alloc_1 : memref<5x20x16x16xf16>) eviction_policy = <EvictFirst>
    %2 = tensor.empty() : tensor<5x10x16x16xf32>
    %alloc_2 = memref.alloc() : memref<5x10x16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<5x10x16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_2 : memref<5x10x16x16xf32, #hivm.address_space<ub>> to memref<5x10x16x16xf32>
    %3 = bufferization.to_tensor %memspacecast restrict writable : memref<5x10x16x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %4 = hivm.hir.vexp ins(%3 : tensor<5x10x16x16xf32>) outs(%2 : tensor<5x10x16x16xf32>) -> tensor<5x10x16x16xf32>
    %5 = tensor.empty() : tensor<5x10x16x16xf16>
    %6 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%4 : tensor<5x10x16x16xf32>) outs(%5 : tensor<5x10x16x16xf16>) -> tensor<5x10x16x16xf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [5, 10, 16, 16], strides: [2560, 256, 16, 1] : memref<?xf16> to memref<5x10x16x16xf16, strided<[2560, 256, 16, 1]>>
    hivm.hir.store ins(%6 : tensor<5x10x16x16xf16>) outs(%reinterpret_cast_3 : memref<5x10x16x16xf16, strided<[2560, 256, 16, 1]>>)
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}


