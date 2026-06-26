// RUN: bishengir-opt %s --hivm-bind-sub-block='enable-tile=false' -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @check_tiling_dim_mapping_aic(
// CHECK-NOT: tiling_dim_mapping
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @check_tiling_dim_mapping_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc : memref<16x16xf16>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf16>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_3 = memref.alloc() : memref<16x16xf32>
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = hivm.hir.mmadL1 ins(%2, %3, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_4 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_4 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %36 = tensor.empty() : tensor<16x16xf16>
    %expanded = tensor.expand_shape %36 [[0, 1], [2, 3]] output_shape [1, 16, 1, 16] : tensor<16x16xf16> into tensor<1x16x1x16xf16>
    annotation.mark %expanded {tiling_dim_mapping = {"1" = 1 : index}} : tensor<1x16x1x16xf16>
    // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins({{.*}} : tensor<16x16xf32>) outs({{.*}} : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6 : tensor<16x16xf32>) outs(%alloc_4 : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %7 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %8 = tensor.empty() : tensor<16x16xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    return
  }
  // CHECK-LABEL:   func.func @check_tiling_dim_mapping_aiv(
  // CHECK: scf.if
  // CHECK: hivm.hir.store
  // CHECK: limit_sub_block_id0
  func.func @check_tiling_dim_mapping_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xf16>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_3 = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_3 : memref<16x16xf32>)
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    %alloc_4 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_4 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %6 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %7 = tensor.empty() : tensor<16x16xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %8 = hivm.hir.vadd ins(%6, %4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    hivm.hir.store ins(%8 : tensor<16x16xf32>) outs(%reinterpret_cast_5 : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
}