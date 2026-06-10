// RUN: bishengir-opt --hivm-tile-batchmm-into-loop %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK: func.func @test_tile_batchMmadL1(%[[DST:.*]]: memref<2x256x256xf16>)
func.func @test_tile_batchMmadL1(%dst : memref<2x256x256xf16>) {
  // CHECK-DAG: %[[MA:.*]] = tensor.empty() : tensor<2x256x128xf16>
  // CHECK-DAG: %[[MB:.*]] = tensor.empty() : tensor<2x128x256xf16>
  %ma = tensor.empty() : tensor<2x256x128xf16>
  %mb = tensor.empty() : tensor<2x128x256xf16>
  %mc = tensor.empty() : tensor<2x256x256xf32>
  %true = arith.constant true
  %M = arith.constant 256 : index
  %K = arith.constant 128 : index
  %N = arith.constant 256 : index
  // CHECK: scf.for %[[ITERATOR:.*]] =
  // CHECK:   %[[EXT_MA:.*]] = tensor.extract_slice %[[MA]][%[[ITERATOR]], 0, 0]
  // CHECK:   %[[EXT_MB:.*]] = tensor.extract_slice %[[MB]][%[[ITERATOR]], 0, 0]
  // CHECK:   %[[MC:.*]] = tensor.empty() : tensor<256x256xf32>


  // CHECK:   %[[RES:.*]] = hivm.hir.mmadL1 {batch_matmul} ins(%[[EXT_MA]], %[[EXT_MB]]
  // CHECK-SAME:                            outs(%[[MC]]
  // CHECK:   %[[SUBVIEW_DST:.*]] = memref.subview %[[DST]][%[[ITERATOR]], 0, 0]
  // CHECK:   %[[COLLAPSE_DST:.*]] = memref.collapse_shape %[[SUBVIEW_DST]]
  // CHECK:   hivm.hir.fixpipe
  // CHECK-SAME: ins(%[[RES]]
  // CHECK-SAME: outs(%[[COLLAPSE_DST]]
  %result = hivm.hir.batchMmadL1 ins(%ma, %mb, %true, %M, %K, %N: tensor<2x256x128xf16>, tensor<2x128x256xf16>, i1, index, index, index)
                              outs(%mc: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  hivm.hir.fixpipe {enable_nz2nd} ins(%result : tensor<2x256x256xf32>) outs(%dst : memref<2x256x256xf16>)
  return
}

// -----
module {
  // CHECK-LABEL: func.func @test_tile_mix_cv
  func.func @test_tile_mix_cv(%arg2: memref<?xf32>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf32> , %arg6: i32, %arg7: i32, %arg8: i32) {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [3, 16, 32], strides: [512, 32, 1] : memref<?xf16> to memref<3x16x32xf16, strided<[512, 32, 1]>>
    %alloc = memref.alloc() : memref<3x16x32xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<3x16x32xf16, strided<[512, 32, 1]>>) outs(%alloc : memref<3x16x32xf16>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<3x16x32xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [3, 32, 16], strides: [512, 16, 1] : memref<?xf16> to memref<3x32x16xf16, strided<[512, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<3x32x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<3x32x16xf16, strided<[512, 16, 1]>>) outs(%alloc_1 : memref<3x32x16xf16>)
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<3x32x16xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1] : memref<?xf32> to memref<3x16x16xf32, strided<[256, 16, 1]>>
    %alloc_3 = memref.alloc() : memref<3x16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<3x16x16xf32, strided<[256, 16, 1]>>) outs(%alloc_3 : memref<3x16x16xf32>)
    %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<3x16x16xf32>
    // CHECK: %[[WORKSPACE_TENSOR:.*]] = tensor.empty() : tensor<3x16x16xf32>
    // CHECK: scf.for %[[INDUCTION_VAR:.*]] = %c0 to %c3 step %c1
    // CHECK-SAME: iter_args(%[[ITERATION:.*]] = %[[WORKSPACE_TENSOR]])
    // CHECK: %[[EXT_MA:.*]] = tensor.extract_slice{{.*}}[%[[INDUCTION_VAR]], 0, 0]
    // CHECK: %[[EXT_MB:.*]] = tensor.extract_slice{{.*}}[%[[INDUCTION_VAR]], 0, 0]
    // CHECK: %[[MATMUL_RES:.*]] = hivm.hir.mmadL1 {batch_matmul} ins(%[[EXT_MA]], %[[EXT_MB]]
    // CHECK: %[[EXT_WS:.*]] = tensor.extract_slice %[[ITERATION]][%[[INDUCTION_VAR]], 0, 0]
    // CHECK: %[[FIX_RES:.*]] = hivm.hir.fixpipe
    // CHECK-SAME: ins(%[[MATMUL_RES]]
    // CHECK-SAME: outs(%[[EXT_WS]]
    // CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[FIX_RES]] into %[[ITERATION]]
    // CHECK: scf.yield %[[INSERT]]
    %3 = tensor.empty() : tensor<3x16x16xf32>
    %4 = hivm.hir.batchMmadL1 ins(%0, %1, %true, %c16, %c32, %c16 : tensor<3x16x32xf16>, tensor<3x32x16xf16>, i1, index, index, index) outs(%3 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
    %5 = tensor.empty() : tensor<3x16x16xf32>
    %6 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<3x16x16xf32>) outs(%5 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
    %7 = tensor.empty() : tensor<3x16x16xf32>
    %8 = hivm.hir.vadd ins(%6, %2 : tensor<3x16x16xf32>, tensor<3x16x16xf32>) outs(%7 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1] : memref<?xf32> to memref<3x16x16xf32, strided<[256, 16, 1]>>
    hivm.hir.store ins(%8 : tensor<3x16x16xf32>) outs(%reinterpret_cast_4 : memref<3x16x16xf32, strided<[256, 16, 1]>>)
    return
  }
}

// -----
module {
// CHECK-LABEL:   func.func @test_tile_batchMmadL1_debug(
// CHECK-SAME:                                           %[[VAL_0:.*]]: memref<2x256x256xf16>) -> tensor<2x256x256xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 256 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant true
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<2x256x128xf16>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<2x128x256xf16>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<2x256x256xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<2x256x256xf32>
// CHECK:           %[[VAL_11:.*]] = tensor.empty() : tensor<2x256x256xf32>
// CHECK:           %[[VAL_12:.*]]:2 = scf.for %[[VAL_13:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_1]] iter_args(%[[VAL_14:.*]] = %[[VAL_10]], %[[VAL_15:.*]] = %[[VAL_9]]) -> (tensor<2x256x256xf32>, tensor<2x256x256xf32>) {
// CHECK:             %[[VAL_16:.*]] = tensor.extract_slice %[[VAL_7]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 128] [1, 1, 1] : tensor<2x256x128xf16> to tensor<256x128xf16>
// CHECK:             %[[VAL_17:.*]] = tensor.extract_slice %[[VAL_8]]{{\[}}%[[VAL_13]], 0, 0] [1, 128, 256] [1, 1, 1] : tensor<2x128x256xf16> to tensor<128x256xf16>
// CHECK:             %[[VAL_18:.*]] = tensor.empty() : tensor<256x256xf32>
// CHECK:             %[[VAL_19:.*]] = hivm.hir.mmadL1 {batch_matmul} ins(%[[VAL_16]], %[[VAL_17]], %[[VAL_6]], %[[VAL_5]], %[[VAL_4]], %[[VAL_5]] : tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index) outs(%[[VAL_18]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK:             %[[VAL_20:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : memref<2x256x256xf16> to memref<1x256x256xf16, strided<[65536, 256, 1], offset: ?>>
// CHECK:             %[[VAL_21:.*]] = memref.collapse_shape %[[VAL_20]] {{\[\[}}0, 1], [2]] : memref<1x256x256xf16, strided<[65536, 256, 1], offset: ?>> into memref<256x256xf16, strided<[256, 1], offset: ?>>
// CHECK:             hivm.hir.fixpipe {enable_nz2nd} ins(%[[VAL_19]] : tensor<256x256xf32>) outs(%[[VAL_21]] : memref<256x256xf16, strided<[256, 1], offset: ?>>)
// CHECK:             %[[VAL_22:.*]] = tensor.extract_slice %[[VAL_14]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : tensor<2x256x256xf32> to tensor<256x256xf32>
// CHECK:             %[[VAL_23:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_19]] : tensor<256x256xf32>) outs(%[[VAL_22]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK:             %[[VAL_24:.*]] = tensor.insert_slice %[[VAL_23]] into %[[VAL_14]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : tensor<256x256xf32> into tensor<2x256x256xf32>
// CHECK:             %[[VAL_25:.*]] = tensor.extract_slice %[[VAL_15]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : tensor<2x256x256xf32> to tensor<256x256xf32>
// CHECK:             %[[VAL_26:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_19]] : tensor<256x256xf32>) outs(%[[VAL_25]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK:             %[[VAL_27:.*]] = tensor.insert_slice %[[VAL_26]] into %[[VAL_15]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : tensor<256x256xf32> into tensor<2x256x256xf32>
// CHECK:             scf.yield %[[VAL_24]], %[[VAL_27]] : tensor<2x256x256xf32>, tensor<2x256x256xf32>
// CHECK:           }
// CHECK:           hivm.hir.debug {debugtype = "print", hex = true, prefix = " ret (hex)\0A: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %[[VAL_28:.*]]#1 : tensor<2x256x256xf32>
// CHECK:           %[[VAL_29:.*]] = hivm.hir.vcumsum ins(%[[VAL_28]]#0 : tensor<2x256x256xf32>) outs(%[[VAL_11]] : tensor<2x256x256xf32>) cum_dims = [0] reverse = false -> tensor<2x256x256xf32>
// CHECK:           return %[[VAL_29]] : tensor<2x256x256xf32>
// CHECK:         }

func.func @test_tile_batchMmadL1_debug(%dst : memref<2x256x256xf16>) -> tensor<2x256x256xf32> {
  %ma = tensor.empty() : tensor<2x256x128xf16>
  %mb = tensor.empty() : tensor<2x128x256xf16>
  %mc = tensor.empty() : tensor<2x256x256xf32>
  %true = arith.constant true
  %M = arith.constant 256 : index
  %K = arith.constant 128 : index
  %N = arith.constant 256 : index
  %result = hivm.hir.batchMmadL1 {fixpipe_already_inserted = true} ins(%ma, %mb, %true, %M, %K, %N: tensor<2x256x128xf16>, tensor<2x128x256xf16>, i1, index, index, index)
                              outs(%mc: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>

  %tmp = tensor.empty() : tensor<2x256x256xf32>
  %debug_print = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%result : tensor<2x256x256xf32>) outs(%tmp : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  hivm.hir.debug {debugtype = "print", hex = true, prefix = " ret (hex)\0A: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %debug_print : tensor<2x256x256xf32>

  %tmp_1 = tensor.empty() : tensor<2x256x256xf32>
  %tmp_2 = tensor.empty() : tensor<2x256x256xf32>
  %tmp_3 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%result : tensor<2x256x256xf32>) outs(%tmp_1 : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  %tmp_4 = hivm.hir.vcumsum ins(%tmp_3 : tensor<2x256x256xf32>) outs(%tmp_2: tensor<2x256x256xf32>) cum_dims = [0] reverse = false -> tensor<2x256x256xf32>

  hivm.hir.fixpipe {enable_nz2nd} ins(%result : tensor<2x256x256xf32>) outs(%dst : memref<2x256x256xf16>)
  return %tmp_4 : tensor<2x256x256xf32>
}
}

// -----
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<
    #dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>,
    #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>,
    #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>,
    #dlti.dl_entry<"UB_SIZE", 2031616 : i32>,
    #dlti.dl_entry<"L1_SIZE", 4194304 : i32>,
    #dlti.dl_entry<"L0A_SIZE", 524288 : i32>,
    #dlti.dl_entry<"L0B_SIZE", 524288 : i32>,
    #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>,
    #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>,
    #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>,
    #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>,
    #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>,
    #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>,
    #dlti.dl_entry<"ARCH", "dav-c310">
  >>,
  hacc.target = #hacc.target<"Ascend950PR_957c">
} {
  // CHECK-LABEL: func.func @triton_device_print_5
  // Batch matmul must be lowered into a loop first.
  // CHECK: scf.for
  // CHECK:   hivm.hir.mmadL1 {batch_matmul}
  // CHECK: }

  // Debug loop must remain after the lowered batch matmul loop.
  // CHECK: scf.for
  // CHECK:   hivm.hir.debug
  // CHECK: }
  func.func @triton_device_print_5(
      %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
      %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
      %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
      %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
      %arg5: i32,
      %arg6: i32,
      %arg7: i32)
    attributes {
      SyncBlockLockArgIdx = 0 : i64,
      WorkspaceArgIdx = 1 : i64,
      func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>,
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      mix_mode = "mix",
      parallel_mode = "simd"
    } {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %true = arith.constant true

    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]

    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32

    %reinterpret_cast = memref.reinterpret_cast %arg2
      to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
      : memref<?xf16> to memref<1x1x1xf16, strided<[1, 1, 1]>>
    %alloc = memref.alloc() : memref<1x1x1xf16>
    hivm.hir.load
      ins(%reinterpret_cast : memref<1x1x1xf16, strided<[1, 1, 1]>>)
      outs(%alloc : memref<1x1x1xf16>)
      eviction_policy = <EvictFirst> core_type = <CUBE>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1x1x1xf16>

    %reinterpret_cast_0 = memref.reinterpret_cast %arg3
      to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
      : memref<?xf16> to memref<1x1x1xf16, strided<[1, 1, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1x1xf16>
    hivm.hir.load
      ins(%reinterpret_cast_0 : memref<1x1x1xf16, strided<[1, 1, 1]>>)
      outs(%alloc_1 : memref<1x1x1xf16>)
      eviction_policy = <EvictFirst> core_type = <CUBE>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x1xf16>

    %4 = tensor.empty() : tensor<1x1x1xf32>
    %5 = hivm.hir.batchMmadL1 {
      already_set_real_mkn,
      fixpipe_for_result_already_inserted = true,
      normalized_in_L0C
    } ins(%2, %3, %true, %c1, %c1, %c1
      : tensor<1x1x1xf16>, tensor<1x1x1xf16>, i1, index, index, index)
      outs(%4 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>

    %alloc_2 = memref.alloc() : memref<1x1x1xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_2
      : memref<1x1x1xf32, #hivm.address_space<ub>> to memref<1x1x1xf32>
    %6 = bufferization.to_tensor %memspacecast restrict writable : memref<1x1x1xf32>

    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%5 : tensor<1x1x1xf32>)
      outs(%alloc_2 : memref<1x1x1xf32, #hivm.address_space<ub>>)

    %reinterpret_cast_3 = memref.reinterpret_cast %arg4
      to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
      : memref<?xf32> to memref<1x1x1xf32, strided<[1, 1, 1]>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%5 : tensor<1x1x1xf32>)
      outs(%reinterpret_cast_3 : memref<1x1x1xf32, strided<[1, 1, 1]>>)

    scf.for %arg8 = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
      hivm.hir.debug {
        debugtype = "print",
        hex = false,
        memscope = #hivm.address_space<ub>,
        prefix = " ret :\0A: ",
        tcoretype = #hivm.tcore_type<VECTOR>
      } %6 : tensor<1x1x1xf32>
    }

    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}
