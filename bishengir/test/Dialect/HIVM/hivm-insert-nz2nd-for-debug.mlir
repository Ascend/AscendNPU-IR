// RUN: bishengir-opt --hivm-insert-nz2nd-for-debug %s | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, memref.memref_as_ptr} {
  func.func @triton_device_print_0(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 32], strides: [32, 1] : memref<?xi8> to memref<16x32xi8, strided<[32, 1]>>
    %alloc = memref.alloc() : memref<16x32xi8>
    hivm.hir.load ins(%reinterpret_cast : memref<16x32xi8, strided<[32, 1]>>) outs(%alloc : memref<16x32xi8>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x32xi8>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = "  \0A: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %c0_i32 : i32
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " X0_M : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %c16_i32 : i32
    hivm.hir.debug {debugtype = "print", hex = true, prefix = " X0_M (hex):\0A: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %c16_i32 : i32
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xi8> to memref<32x16xi8, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<32x16xi8>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32x16xi8, strided<[16, 1]>>) outs(%alloc_1 : memref<32x16xi8>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x16xi8>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %3 : tensor<32x16xi8>
    %4 = tensor.empty() : tensor<16x16xi32>
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c16_2 = arith.constant 16 : index
    %5 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%2, %3, %true, %c16, %c32, %c16_2 : tensor<16x32xi8>, tensor<32x16xi8>, i1, index, index, index) outs(%4 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %6 = tensor.empty() : tensor<16x16xi32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi32> to memref<16x16xi32, strided<[16, 1]>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<16x16xi32>) outs(%reinterpret_cast_3 : memref<16x16xi32, strided<[16, 1]>>)
    %7 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<16x16xi32>) outs(%6 : tensor<16x16xi32>) -> tensor<16x16xi32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " 3: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %7 : tensor<16x16xi32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " 3: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %3 : tensor<32x16xi8>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " 3: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %c16_i32 : i32
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " 3: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %3 : tensor<32x16xi8>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " 3: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %2 : tensor<16x32xi8>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " 3: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %3 : tensor<32x16xi8>
    return
  }
}

// CHECK-LABEL: func.func @triton_device_print_0
// CHECK: %[[B:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<32x16xi8>
// CHECK: %[[BNZ:.*]] = hivm.hir.nz2nd ins(%[[B]] : tensor<32x16xi8>) outs(%{{.*}} : tensor<32x16xi8>) -> tensor<32x16xi8>
// CHECK-COUNT-4: hivm.hir.debug {{.*}} %[[BNZ]] : tensor<32x16xi8>