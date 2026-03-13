// RUN: bishengir-opt --hivm-split-mix-kernel --debug-only=hivm-split-mix-kernel %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @triton_matmul_exp_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @triton_matmul_exp(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg11 : i32
    %5 = arith.remsi %4, %arg10 : i32
    %6 = arith.muli %arg11, %arg10 : i32
    %7 = arith.divsi %3, %6 : i32
    %8 = arith.remsi %7, %arg9 : i32
    %9 = tensor.empty() : tensor<1x1xf32>
    %10 = arith.muli %8, %c4_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [1, 4], strides: [4, 1] : memref<?xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x4xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<1x4xf32, strided<[4, 1], offset: ?>>) outs(%alloc : memref<1x4xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %12 = bufferization.to_tensor %alloc restrict writable : memref<1x4xf32>
    %13 = arith.index_cast %arg8 : i32 to index
    %14 = arith.index_cast %5 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [4, 1], strides: [%13, 1] : memref<?xf32> to memref<4x1xf32, strided<[?, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<4x1xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<4x1xf32, strided<[?, 1], offset: ?>>) outs(%alloc_1 : memref<4x1xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %15 = bufferization.to_tensor %alloc_1 restrict writable : memref<4x1xf32>
    %16 = arith.muli %8, %arg8 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.addi %17, %14 : index
    %19 = tensor.empty() : tensor<1x1xf32>
    %20 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%12, %15, %true, %c1, %c4, %c1 : tensor<1x4xf32>, tensor<4x1xf32>, i1, index, index, index) outs(%19 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%18], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
    hivm.hir.fixpipe {enable_nz2nd} ins(%20 : tensor<1x1xf32>) outs(%reinterpret_cast_2 : memref<1x1xf32, strided<[1, 1], offset: ?>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %alloc_3 = memref.alloc() : memref<1x1xf32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    hivm.hir.load ins(%reinterpret_cast_2 : memref<1x1xf32, strided<[1, 1], offset: ?>>) outs(%alloc_3 : memref<1x1xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %21 = bufferization.to_tensor %alloc_3 restrict writable : memref<1x1xf32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " acc_11_reload: ", tcoretype = #hivm.tcore_type<VECTOR>} %21 : tensor<1x1xf32>
    %22 = hivm.hir.vexp ins(%21 : tensor<1x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
    hivm.hir.store ins(%22 : tensor<1x1xf32>) outs(%reinterpret_cast_4 : memref<1x1xf32, strided<[1, 1], offset: ?>>)
    return
  }
}

// CHECK-LABEL: func.func @triton_matmul_exp_mix_aiv
// CHECK: hivm.hir.debug {{.*}} tcoretype = #hivm.tcore_type<VECTOR>