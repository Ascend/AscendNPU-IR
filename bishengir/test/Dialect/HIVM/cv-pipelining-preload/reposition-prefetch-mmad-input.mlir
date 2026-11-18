// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=true" | FileCheck %s

// CHECK-LABEL: func.func @prefetch_mmad_input(
// CHECK: hivm.hir.vtranspose
// CHECK-SAME: permutation = [1, 0, 2]
// CHECK: hivm.hir.copy
// CHECK-SAME: "hivm.inserted-copy"
// CHECK: hivm.hir.vtranspose
// CHECK-SAME: permutation = [1, 0, 2]
// CHECK: hivm.hir.copy
// CHECK-SAME: "hivm.inserted-copy"
// CHECK: hivm.hir.vadd
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.load_store_buf_count = 1 : i32} {
  func.func @prefetch_mmad_input(%arg0: memref<512x512xf32>, %arg1: memref<512x512xf32>, %arg2: memref<512x512xf32>, %arg3: memref<512x512xf32>, %arg4: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %0 = tensor.empty() : tensor<4x32x8xf32>
    %1 = tensor.empty() : tensor<16x32x8xf32>
    %2 = tensor.empty() : tensor<8x2x16x16xf32>
    %alloc = memref.alloc() : memref<32x32xf32>
    %subview = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<512x512xf32> to memref<32x32xf32, strided<[512, 1]>>
    %subview_0 = memref.subview %alloc[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
    memref.copy %subview, %subview_0 : memref<32x32xf32, strided<[512, 1]>> to memref<32x32xf32, strided<[32, 1]>>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32>
    %alloc_12 = memref.alloc() : memref<32x32xf32>
    %subview_13 = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<512x512xf32> to memref<32x32xf32, strided<[512, 1]>>
    %subview_14 = memref.subview %alloc_12[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
    memref.copy %subview_13, %subview_14 : memref<32x32xf32, strided<[512, 1]>> to memref<32x32xf32, strided<[32, 1]>>
    %13 = bufferization.to_tensor %alloc_12 restrict writable : memref<32x32xf32>
    %alloc_18 = memref.alloc() : memref<32x128xf32>
    %subview_19 = memref.subview %arg3[0, 0] [32, 128] [1, 1] : memref<512x512xf32> to memref<32x128xf32, strided<[512, 1]>>
    %subview_20 = memref.subview %alloc_18[0, 0] [32, 128] [1, 1] : memref<32x128xf32> to memref<32x128xf32, strided<[128, 1]>>
    memref.copy %subview_19, %subview_20 : memref<32x128xf32, strided<[512, 1]>> to memref<32x128xf32, strided<[128, 1]>>
    %16 = bufferization.to_tensor %alloc_18 restrict writable : memref<32x128xf32>
    %expanded = tensor.expand_shape %3 [[0], [1, 2]] output_shape [32, 4, 8] : tensor<32x32xf32> into tensor<32x4x8xf32>
    %4 = hivm.hir.vtranspose ins(%expanded : tensor<32x4x8xf32>) outs(%0 : tensor<4x32x8xf32>) permutation = [1, 0, 2] -> tensor<4x32x8xf32>
    %expanded_1 = tensor.expand_shape %4 [[0], [1, 2], [3]] output_shape [4, 2, 16, 8] : tensor<4x32x8xf32> into tensor<4x2x16x8xf32>
    %alloc_2 = memref.alloc() : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<17>} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%expanded_1 : tensor<4x2x16x8xf32>) outs(%alloc_2 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %5 = bufferization.to_tensor %alloc_2 restrict writable : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %alloc_3 = memref.alloc() : memref<32x128xf32>
    %subview_4 = memref.subview %arg1[0, 0] [32, 128] [1, 1] : memref<512x512xf32> to memref<32x128xf32, strided<[512, 1]>>
    %subview_5 = memref.subview %alloc_3[0, 0] [32, 128] [1, 1] : memref<32x128xf32> to memref<32x128xf32, strided<[128, 1]>>
    memref.copy %subview_4, %subview_5 : memref<32x128xf32, strided<[512, 1]>> to memref<32x128xf32, strided<[128, 1]>>
    %6 = bufferization.to_tensor %alloc_3 restrict writable : memref<32x128xf32>
    %expanded_6 = tensor.expand_shape %6 [[0], [1, 2]] output_shape [32, 16, 8] : tensor<32x128xf32> into tensor<32x16x8xf32>
    %7 = hivm.hir.vtranspose ins(%expanded_6 : tensor<32x16x8xf32>) outs(%1 : tensor<16x32x8xf32>) permutation = [1, 0, 2] -> tensor<16x32x8xf32>
    %expanded_7 = tensor.expand_shape %7 [[0], [1, 2], [3]] output_shape [16, 2, 16, 8] : tensor<16x32x8xf32> into tensor<16x2x16x8xf32>
    %alloc_8 = memref.alloc() : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<16>} : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%expanded_7 : tensor<16x2x16x8xf32>) outs(%alloc_8 : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %8 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>
    %9 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%5, %8, %true, %c32, %c32, %c128 : tensor<4x2x16x8xf32>, tensor<16x2x16x8xf32>, i1, index, index, index) outs(%2 : tensor<8x2x16x16xf32>) -> tensor<8x2x16x16xf32>
    %alloc_9 = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>} : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%9 : tensor<8x2x16x16xf32>) outs(%alloc_9 : memref<32x128xf32, #hivm.address_space<ub>>)
    %memspacecast = memref.memory_space_cast %alloc_9 : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>
    %10 = bufferization.to_tensor %memspacecast restrict writable : memref<32x128xf32>
    %alloc_10 = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>} : memref<32x128xf32, #hivm.address_space<ub>>
    %memspacecast_11 = memref.memory_space_cast %alloc_10 : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>
    %11 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<32x128xf32>
    %12 = hivm.hir.vadd ins(%10, %arg4 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%11 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %expanded_15 = tensor.expand_shape %13 [[0], [1, 2]] output_shape [32, 4, 8] : tensor<32x32xf32> into tensor<32x4x8xf32>
    %14 = hivm.hir.vtranspose ins(%expanded_15 : tensor<32x4x8xf32>) outs(%0 : tensor<4x32x8xf32>) permutation = [1, 0, 2] -> tensor<4x32x8xf32>
    %expanded_16 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [4, 2, 16, 8] : tensor<4x32x8xf32> into tensor<4x2x16x8xf32>
    %alloc_17 = memref.alloc() : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<19>} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%expanded_16 : tensor<4x2x16x8xf32>) outs(%alloc_17 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %15 = bufferization.to_tensor %alloc_17 restrict writable : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %expanded_21 = tensor.expand_shape %16 [[0], [1, 2]] output_shape [32, 16, 8] : tensor<32x128xf32> into tensor<32x16x8xf32>
    %17 = hivm.hir.vtranspose ins(%expanded_21 : tensor<32x16x8xf32>) outs(%1 : tensor<16x32x8xf32>) permutation = [1, 0, 2] -> tensor<16x32x8xf32>
    %expanded_22 = tensor.expand_shape %17 [[0], [1, 2], [3]] output_shape [16, 2, 16, 8] : tensor<16x32x8xf32> into tensor<16x2x16x8xf32>
    %alloc_23 = memref.alloc() : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_23 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<18>} : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%expanded_22 : tensor<16x2x16x8xf32>) outs(%alloc_23 : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %18 = bufferization.to_tensor %alloc_23 restrict writable : memref<16x2x16x8xf32, #hivm.address_space<cbuf>>
    %19 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%15, %18, %true, %c32, %c32, %c128 : tensor<4x2x16x8xf32>, tensor<16x2x16x8xf32>, i1, index, index, index) outs(%2 : tensor<8x2x16x16xf32>) -> tensor<8x2x16x16xf32>
    %alloc_24 = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_24 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>} : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%19 : tensor<8x2x16x16xf32>) outs(%alloc_24 : memref<32x128xf32, #hivm.address_space<ub>>)
    %memspacecast_25 = memref.memory_space_cast %alloc_24 : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x128xf32>
    %20 = bufferization.to_tensor %memspacecast_25 restrict writable : memref<32x128xf32>
    return %12 : tensor<32x128xf32>
  }
}
