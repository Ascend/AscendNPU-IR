// RUN: bishengir-opt --hivm-split-mix-kernel --debug-only=hivm-split-mix-kernel %s
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
func.func @triton_device_print_0(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c-1_i64 = arith.constant -1 : i64
    %c1011 = arith.constant 1011 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = tensor.empty() : tensor<1011xi32>
    %3 = hivm.hir.varange offset[%c0] strides[%c1] outs(%2 : tensor<1011xi32>) -> tensor<1011xi32>
    %expanded = tensor.expand_shape %3 [[0, 1, 2]] output_shape [1, 1011, 1] : tensor<1011xi32> into tensor<1x1011x1xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1011, 1], strides: [1011, 1, 1] : memref<?xi8> to memref<1x1011x1xi8, strided<[1011, 1, 1]>>
    %alloc = memref.alloc() : memref<1x1011x1xi8>
    hivm.hir.load ins(%reinterpret_cast : memref<1x1011x1xi8, strided<[1011, 1, 1]>>) outs(%alloc : memref<1x1011x1xi8>) eviction_policy = <EvictFirst>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<1x1011x1xi8>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " x0_idx : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %expanded : tensor<1x1011x1xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1] : memref<?xi8> to memref<1x1x1xi8, strided<[1, 1, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1x1xi8>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<1x1x1xi8, strided<[1, 1, 1]>>) outs(%alloc_1 : memref<1x1x1xi8>) eviction_policy = <EvictFirst>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x1xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 1011, 1], strides: [1011, 1, 1] : memref<?xi32> to memref<1x1011x1xi32, strided<[1011, 1, 1]>>
    %extracted_slice = tensor.extract_slice %4[0, 0, 0] [1, 1011, 1] [1, 1, 1] : tensor<1x1011x1xi8> to tensor<1011x1xi8>
    %extracted_slice_3 = tensor.extract_slice %5[0, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xi8> to tensor<1x1xi8>
    %6 = tensor.empty() : tensor<1011x1xi32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, batch_matmul} ins(%extracted_slice, %extracted_slice_3, %true, %c1011, %c1, %c1 : tensor<1011x1xi8>, tensor<1x1xi8>, i1, index, index, index) outs(%6 : tensor<1011x1xi32>) -> tensor<1011x1xi32>
    %collapse_shape = memref.collapse_shape %reinterpret_cast_2 [[0, 1], [2]] : memref<1x1011x1xi32, strided<[1011, 1, 1]>> into memref<1011x1xi32, strided<[1, 1]>>
    %cast = memref.cast %collapse_shape : memref<1011x1xi32, strided<[1, 1]>> to memref<1011x1xi32, strided<[1, 1], offset: ?>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<1011x1xi32>) outs(%cast : memref<1011x1xi32, strided<[1, 1], offset: ?>>)
    return
}
}
// CHECK-LABEL: func.func @triton_device_print_0_mix_aic
// CHECK-NOT: hivm.hir.debug