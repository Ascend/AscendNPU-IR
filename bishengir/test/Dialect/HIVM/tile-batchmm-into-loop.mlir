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


  // CHECK:   %[[RES:.*]] = hivm.hir.mmadL1 ins(%[[EXT_MA]], %[[EXT_MB]]
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
    // CHECK: %[[MATMUL_RES:.*]] = hivm.hir.mmadL1 ins(%[[EXT_MA]], %[[EXT_MB]]
    // CHECK: %[[EXT_WS:.*]] = tensor.extract_slice %[[ITERATION]][%[[INDUCTION_VAR]], 0, 0]
    // CHECK: %[[FIX_RES:.*]] = hivm.hir.fixpipe
    // CHECK-SAME: ins(%[[MATMUL_RES]]
    // CHECK-SAME: outs(%[[EXT_WS]]
    // CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[FIX_RES]] into %[[ITERATION]][%[[INDUCTION_VAR]], 0, 0] [1, 16, 16] [1, 1, 1] {elide_after_bufferize}
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


  // CHECK:   %[[RES:.*]] = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%[[EXT_MA]], %[[EXT_MB]]
  // CHECK-SAME:                            outs(%[[MC]]
  // CHECK:   %[[SUBVIEW_DST:.*]] = memref.subview %[[DST]][%[[ITERATOR]], 0, 0]
  // CHECK:   %[[COLLAPSE_DST:.*]] = memref.collapse_shape %[[SUBVIEW_DST]]
  // CHECK:   hivm.hir.fixpipe
  // CHECK-SAME: ins(%[[RES]]
  // CHECK-SAME: outs(%[[COLLAPSE_DST]]
  %result = hivm.hir.batchMmadL1 {fixpipe_already_inserted = true} ins(%ma, %mb, %true, %M, %K, %N: tensor<2x256x128xf16>, tensor<2x128x256xf16>, i1, index, index, index)
                              outs(%mc: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  hivm.hir.fixpipe {enable_nz2nd} ins(%result : tensor<2x256x256xf32>) outs(%dst : memref<2x256x256xf16>)
  return
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
// CHECK:             %[[VAL_19:.*]] = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%[[VAL_16]], %[[VAL_17]], %[[VAL_6]], %[[VAL_5]], %[[VAL_4]], %[[VAL_5]] : tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index) outs(%[[VAL_18]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK:             %[[VAL_20:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : memref<2x256x256xf16> to memref<1x256x256xf16, strided<[65536, 256, 1], offset: ?>>
// CHECK:             %[[VAL_21:.*]] = memref.collapse_shape %[[VAL_20]] {{\[\[}}0, 1], [2]] : memref<1x256x256xf16, strided<[65536, 256, 1], offset: ?>> into memref<256x256xf16, strided<[256, 1], offset: ?>>
// CHECK:             hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_19]] : tensor<256x256xf32>) outs(%[[VAL_21]] : memref<256x256xf16, strided<[256, 1], offset: ?>>)
// CHECK:             %[[VAL_22:.*]] = tensor.extract_slice %[[VAL_14]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : tensor<2x256x256xf32> to tensor<256x256xf32>
// CHECK:             %[[VAL_23:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_19]] : tensor<256x256xf32>) outs(%[[VAL_22]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK:             %[[VAL_24:.*]] = tensor.insert_slice %[[VAL_23]] into %[[VAL_14]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] {elide_after_bufferize} : tensor<256x256xf32> into tensor<2x256x256xf32>
// CHECK:             %[[VAL_25:.*]] = tensor.extract_slice %[[VAL_15]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] : tensor<2x256x256xf32> to tensor<256x256xf32>
// CHECK:             %[[VAL_26:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_19]] : tensor<256x256xf32>) outs(%[[VAL_25]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK:             %[[VAL_27:.*]] = tensor.insert_slice %[[VAL_26]] into %[[VAL_15]]{{\[}}%[[VAL_13]], 0, 0] [1, 256, 256] [1, 1, 1] {elide_after_bufferize} : tensor<256x256xf32> into tensor<2x256x256xf32>
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