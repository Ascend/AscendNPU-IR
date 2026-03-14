// RUN: bishengir-opt -hivm-normalize-convops %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// -----
// CHECK-LABEL: func.func @triton_conv1d_2d_fp16_nobias
// test NormalizeConvResultTypePattern
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<30x128xf16>, tensor<30x15x5xf16>, i1) outs(%{{.*}} : tensor<30x126xf32>) -> tensor<30x126xf32>
// CHECK: %{{.*}} = tensor.empty() : tensor<30x126xf16>
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30x126xf32>) outs(%{{.*}} : tensor<30x126xf16>) -> tensor<30x126xf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_2d_fp16_nobias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [30, 128], strides: [128, 1] : memref<?xf16> to memref<30x128xf16, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<30x128xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<30x128xf16, strided<[128, 1]>>) outs(%alloc : memref<30x128xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<30x128xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [30, 15, 5], strides: [75, 5, 1] : memref<?xf16> to memref<30x15x5xf16, strided<[75, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<30x15x5xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<30x15x5xf16, strided<[75, 5, 1]>>) outs(%alloc_1 : memref<30x15x5xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<30x15x5xf16>
    %4 = tensor.empty() : tensor<30x126xf16>
    %5 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%2, %3, %true : tensor<30x128xf16>, tensor<30x15x5xf16>, i1) outs(%4 : tensor<30x126xf16>) -> tensor<30x126xf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [30, 126], strides: [126, 1] : memref<?xf16> to memref<30x126xf16, strided<[126, 1]>>
    hivm.hir.store ins(%5 : tensor<30x126xf16>) outs(%reinterpret_cast_4 : memref<30x126xf16, strided<[126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_2d_bf16_nobias
// test NormalizeConvResultTypePattern
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<30x128xbf16>, tensor<30x15x5xbf16>, i1) outs(%{{.*}} : tensor<30x126xf32>) -> tensor<30x126xf32>
// CHECK: %{{.*}} = tensor.empty() : tensor<30x126xbf16>
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30x126xf32>) outs(%{{.*}} : tensor<30x126xbf16>) -> tensor<30x126xbf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_2d_bf16_nobias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [30, 128], strides: [128, 1] : memref<?xbf16> to memref<30x128xbf16, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<30x128xbf16>
    hivm.hir.load ins(%reinterpret_cast : memref<30x128xbf16, strided<[128, 1]>>) outs(%alloc : memref<30x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<30x128xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [30, 15, 5], strides: [75, 5, 1] : memref<?xbf16> to memref<30x15x5xbf16, strided<[75, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<30x15x5xbf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<30x15x5xbf16, strided<[75, 5, 1]>>) outs(%alloc_1 : memref<30x15x5xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<30x15x5xbf16>
    %4 = tensor.empty() : tensor<30x126xbf16>
    %5 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%2, %3, %true : tensor<30x128xbf16>, tensor<30x15x5xbf16>, i1) outs(%4 : tensor<30x126xbf16>) -> tensor<30x126xbf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [30, 126], strides: [126, 1] : memref<?xbf16> to memref<30x126xbf16, strided<[126, 1]>>
    hivm.hir.store ins(%5 : tensor<30x126xbf16>) outs(%reinterpret_cast_4 : memref<30x126xbf16, strided<[126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_2d_fp32_nobias
// test NormalizeConvResultTypePattern
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<30x128xf32>, tensor<30x15x5xf32>, i1) outs(%{{.*}} : tensor<30x126xf32>) -> tensor<30x126xf32>
// CHECK-NOT: hivm.hir.vcast
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_2d_fp32_nobias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [30, 128], strides: [128, 1] : memref<?xf32> to memref<30x128xf32, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<30x128xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<30x128xf32, strided<[128, 1]>>) outs(%alloc : memref<30x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<30x128xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [30, 15, 5], strides: [75, 5, 1] : memref<?xf32> to memref<30x15x5xf32, strided<[75, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<30x15x5xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<30x15x5xf32, strided<[75, 5, 1]>>) outs(%alloc_1 : memref<30x15x5xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<30x15x5xf32>
    %4 = tensor.empty() : tensor<30x126xf32>
    %5 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%2, %3, %true : tensor<30x128xf32>, tensor<30x15x5xf32>, i1) outs(%4 : tensor<30x126xf32>) -> tensor<30x126xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [30, 126], strides: [126, 1] : memref<?xf32> to memref<30x126xf32, strided<[126, 1]>>
    hivm.hir.store ins(%5 : tensor<30x126xf32>) outs(%reinterpret_cast_4 : memref<30x126xf32, strided<[126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_2d_fp16_bias
// test NormalizeConvResultTypePattern + DecomposeConv1dWithBiasPattern
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<30x128xf16>, tensor<30x15x5xf16>, i1) outs(%{{.*}} : tensor<30x126xf32>) -> tensor<30x126xf32>
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30x126xf32>) outs(%{{.*}} : tensor<30x126xf16>) -> tensor<30x126xf16>
// CHECK: %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf16> into tensor<30x1xf16>
// CHECK: %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<30x126xf16>, tensor<30x1xf16>) outs(%{{.*}} : tensor<30x126xf16>) broadcast = [1] -> tensor<30x126xf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_2d_fp16_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [30, 128], strides: [128, 1] : memref<?xf16> to memref<30x128xf16, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<30x128xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<30x128xf16, strided<[128, 1]>>) outs(%alloc : memref<30x128xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<30x128xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [30, 15, 5], strides: [75, 5, 1] : memref<?xf16> to memref<30x15x5xf16, strided<[75, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<30x15x5xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<30x15x5xf16, strided<[75, 5, 1]>>) outs(%alloc_1 : memref<30x15x5xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<30x15x5xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [30], strides: [1] : memref<?xf16> to memref<30xf16, strided<[1]>>
    %alloc_3 = memref.alloc() : memref<30xf16>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<30xf16, strided<[1]>>) outs(%alloc_3 : memref<30xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<30xf16>
    %5 = tensor.empty() : tensor<30x126xf16>
    %6 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%2, %3, %true, %4 : tensor<30x128xf16>, tensor<30x15x5xf16>, i1, tensor<30xf16>) outs(%5 : tensor<30x126xf16>) -> tensor<30x126xf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [30, 126], strides: [126, 1] : memref<?xf16> to memref<30x126xf16, strided<[126, 1]>>
    hivm.hir.store ins(%6 : tensor<30x126xf16>) outs(%reinterpret_cast_4 : memref<30x126xf16, strided<[126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_2d_bf16_bias
// test NormalizeConvResultTypePattern + DecomposeConv1dWithBiasPattern
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30xbf16>) outs(%{{.*}} : tensor<30xf32>) -> tensor<30xf32>
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<30x128xbf16>, tensor<30x15x5xbf16>, i1) outs(%{{.*}} : tensor<30x126xf32>) -> tensor<30x126xf32>
// CHECK: %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
// CHECK: %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<30x126xf32>, tensor<30x1xf32>) outs(%{{.*}} : tensor<30x126xf32>) broadcast = [1] -> tensor<30x126xf32>
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<30x126xf32>) outs(%{{.*}} : tensor<30x126xbf16>) -> tensor<30x126xbf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_2d_bf16_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [30, 128], strides: [128, 1] : memref<?xbf16> to memref<30x128xbf16, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<30x128xbf16>
    hivm.hir.load ins(%reinterpret_cast : memref<30x128xbf16, strided<[128, 1]>>) outs(%alloc : memref<30x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<30x128xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [30, 15, 5], strides: [75, 5, 1] : memref<?xbf16> to memref<30x15x5xbf16, strided<[75, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<30x15x5xbf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<30x15x5xbf16, strided<[75, 5, 1]>>) outs(%alloc_1 : memref<30x15x5xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<30x15x5xbf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [30], strides: [1] : memref<?xbf16> to memref<30xbf16, strided<[1]>>
    %alloc_3 = memref.alloc() : memref<30xbf16>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<30xbf16, strided<[1]>>) outs(%alloc_3 : memref<30xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<30xbf16>
    %5 = tensor.empty() : tensor<30x126xbf16>
    %6 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%2, %3, %true, %4 : tensor<30x128xbf16>, tensor<30x15x5xbf16>, i1, tensor<30xbf16>) outs(%5 : tensor<30x126xbf16>) -> tensor<30x126xbf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [30, 126], strides: [126, 1] : memref<?xbf16> to memref<30x126xbf16, strided<[126, 1]>>
    hivm.hir.store ins(%6 : tensor<30x126xbf16>) outs(%reinterpret_cast_4 : memref<30x126xbf16, strided<[126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_2d_fp32_bias
// test NormalizeConvResultTypePattern + DecomposeConv1dWithBiasPattern
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<30x128xf32>, tensor<30x15x5xf32>, i1) outs(%{{.*}} : tensor<30x126xf32>) -> tensor<30x126xf32>
// CHECK: %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
// CHECK: %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<30x126xf32>, tensor<30x1xf32>) outs(%{{.*}} : tensor<30x126xf32>) broadcast = [1] -> tensor<30x126xf32>
// CHECK-NOT: hivm.hir.vcast
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_2d_fp32_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [30, 128], strides: [128, 1] : memref<?xf32> to memref<30x128xf32, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<30x128xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<30x128xf32, strided<[128, 1]>>) outs(%alloc : memref<30x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<30x128xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [30, 15, 5], strides: [75, 5, 1] : memref<?xf32> to memref<30x15x5xf32, strided<[75, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<30x15x5xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<30x15x5xf32, strided<[75, 5, 1]>>) outs(%alloc_1 : memref<30x15x5xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<30x15x5xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [30], strides: [1] : memref<?xf32> to memref<30xf32, strided<[1]>>
    %alloc_3 = memref.alloc() : memref<30xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<30xf32, strided<[1]>>) outs(%alloc_3 : memref<30xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<30xf32>
    %5 = tensor.empty() : tensor<30x126xf32>
    %6 = hivm.hir.Conv1dL1 {groups = 2 : i32, padding = 1 : i32} ins(%2, %3, %true, %4 : tensor<30x128xf32>, tensor<30x15x5xf32>, i1, tensor<30xf32>) outs(%5 : tensor<30x126xf32>) -> tensor<30x126xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [30, 126], strides: [126, 1] : memref<?xf32> to memref<30x126xf32, strided<[126, 1]>>
    hivm.hir.store ins(%6 : tensor<30x126xf32>) outs(%reinterpret_cast_4 : memref<30x126xf32, strided<[126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_3d_fp16_bias
// test NormalizeConvResultTypePattern + DecomposeConv1dWithBiasPattern
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x32x128xf16>, tensor<32x32x5xf16>, i1) outs(%{{.*}} : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<2x32x126xf32>) outs(%{{.*}} : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
// CHECK: %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf16> into tensor<1x32x1xf16>
// CHECK: %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x32x126xf16>, tensor<1x32x1xf16>) outs(%{{.*}} : tensor<2x32x126xf16>) broadcast = [0, 2] -> tensor<2x32x126xf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_3d_fp16_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<?xf16> to memref<2x32x128xf16, strided<[4096, 128, 1]>>
    %alloc = memref.alloc() : memref<2x32x128xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<2x32x128xf16, strided<[4096, 128, 1]>>) outs(%alloc : memref<2x32x128xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2x32x128xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32, 5], strides: [160, 5, 1] : memref<?xf16> to memref<32x32x5xf16, strided<[160, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<32x32x5xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32x32x5xf16, strided<[160, 5, 1]>>) outs(%alloc_1 : memref<32x32x5xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32x5xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1]>>
    %alloc_3 = memref.alloc() : memref<32xf16>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<32xf16, strided<[1]>>) outs(%alloc_3 : memref<32xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<32xf16>
    %5 = tensor.empty() : tensor<2x32x126xf16>
    %6 = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%2, %3, %true, %4 : tensor<2x32x128xf16>, tensor<32x32x5xf16>, i1, tensor<32xf16>) outs(%5 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [2, 32, 126], strides: [4032, 126, 1] : memref<?xf16> to memref<2x32x126xf16, strided<[4032, 126, 1]>>
    hivm.hir.store ins(%6 : tensor<2x32x126xf16>) outs(%reinterpret_cast_4 : memref<2x32x126xf16, strided<[4032, 126, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @triton_conv1d_3d_bf16_bias
// test NormalizeConvResultTypePattern + DecomposeConv1dWithBiasPattern
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<32xbf16>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x32x128xbf16>, tensor<32x32x5xbf16>, i1) outs(%{{.*}} : tensor<2x32x126xf32>) -> tensor<2x32x126xf32>
// CHECK: %{{.*}} = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 32, 1] : tensor<32xf32> into tensor<1x32x1xf32>
// CHECK: %{{.*}} = hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<2x32x126xf32>, tensor<1x32x1xf32>) outs(%{{.*}} : tensor<2x32x126xf32>) broadcast = [0, 2] -> tensor<2x32x126xf32>
// CHECK: %{{.*}} = hivm.hir.vcast ins(%{{.*}} : tensor<2x32x126xf32>) outs(%{{.*}} : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, memref.memref_as_ptr} {
  func.func @triton_conv1d_3d_bf16_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<?xbf16> to memref<2x32x128xbf16, strided<[4096, 128, 1]>>
    %alloc = memref.alloc() : memref<2x32x128xbf16>
    hivm.hir.load ins(%reinterpret_cast : memref<2x32x128xbf16, strided<[4096, 128, 1]>>) outs(%alloc : memref<2x32x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2x32x128xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32, 5], strides: [160, 5, 1] : memref<?xbf16> to memref<32x32x5xbf16, strided<[160, 5, 1]>>
    %alloc_1 = memref.alloc() : memref<32x32x5xbf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32x32x5xbf16, strided<[160, 5, 1]>>) outs(%alloc_1 : memref<32x32x5xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32x5xbf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32], strides: [1] : memref<?xbf16> to memref<32xbf16, strided<[1]>>
    %alloc_3 = memref.alloc() : memref<32xbf16>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<32xbf16, strided<[1]>>) outs(%alloc_3 : memref<32xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<32xbf16>
    %5 = tensor.empty() : tensor<2x32x126xbf16>
    %6 = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%2, %3, %true, %4 : tensor<2x32x128xbf16>, tensor<32x32x5xbf16>, i1, tensor<32xbf16>) outs(%5 : tensor<2x32x126xbf16>) -> tensor<2x32x126xbf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [2, 32, 126], strides: [4032, 126, 1] : memref<?xbf16> to memref<2x32x126xbf16, strided<[4032, 126, 1]>>
    hivm.hir.store ins(%6 : tensor<2x32x126xbf16>) outs(%reinterpret_cast_4 : memref<2x32x126xbf16, strided<[4032, 126, 1]>>)
    return
  }
}