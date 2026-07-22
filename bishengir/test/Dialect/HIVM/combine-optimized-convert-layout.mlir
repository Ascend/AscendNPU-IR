// RUN: bishengir-opt %s --canonicalize --hivm-combine-optimized-convert-layout --split-input-file | FileCheck %s

// CHECK:   func.func @fold_one_use_subview(%[[VAL_0:.*]]: memref<16x16xf16, strided<[?, 1], offset: ?>>, %[[VAL_1:.*]]: memref<16x16xf16, strided<[?, 1], offset:
// CHECK: %[[VAL_4:.*]] = memref.subview %[[VAL_0]][0, 0] {{\[}}%{{.*}}, 16] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x16xf16, strided<[?, 1], offset: ?>>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<1x1x16x16xf16>
// CHECK: %[[VAL_7:.*]] = memref.subview %[[VAL_6]][0, 0, 0, 0] [1, %{{.*}}, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xf16> to memref<1x?x16x16xf16, strided<[256, 256, 16, 1]>>
// CHECK: hivm.hir.nd2nz {dst_continuous} ins(%[[VAL_4]] : memref<?x16xf16, strided<[?, 1], offset: ?>>) outs(%[[VAL_7]] : memref<1x?x16x16xf16, strided<[256, 256, 16, 1]>>)
// CHECK: %[[VAL_8:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<1x1x16x16xf16>
// CHECK: return %[[VAL_8]] : tensor<1x1x16x16xf16>
func.func @fold_one_use_subview(%arg0: memref<16x16xf16, strided<[?, 1], offset: ?>>, %arg1: memref<16x16xf16, strided<[?, 1], offset: ?>>, %arg2: index, %arg3: i1) -> tensor<1x1x16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<16x16xf16>
  %subview = memref.subview %arg0[0, 0] [%arg2, 16] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x16xf16, strided<[?, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%arg2, 16] [1, 1] : memref<16x16xf16> to memref<?x16xf16, strided<[16, 1]>>
  hivm.hir.load ins(%subview : memref<?x16xf16, strided<[?, 1], offset: ?>>) outs(%subview_0 : memref<?x16xf16, strided<[16, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %arg3 : i1
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
  %1 = hivm.hir.convert_layout %0 output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND, transpose = false>} : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  return %1 : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @fold_convert_fixpipe
// CHECK-SAME: %[[DSTR:.*]]: memref<128x256xf16
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.*}} : tensor<16x8x16x16xf32>) outs(%[[DSTR]] : memref<128x256xf16
func.func @fold_convert_fixpipe(%dst: memref<128x256xf16, strided<[1024, 1], offset: ?>>) {
  %mmad_out = arith.constant dense<0.0> : tensor<16x8x16x16xf32>
  %conv = hivm.hir.convert_layout %mmad_out output_shape [128, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<16x8x16x16xf32>) -> tensor<128x256xf32>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%conv : tensor<128x256xf32>) outs(%dst : memref<128x256xf16, strided<[1024, 1], offset: ?>>)
  return
}

// -----

// CHECK-LABEL: func.func @fold_convert_extract_slice_fixpipe
// CHECK-SAME: %[[DSTR:.*]]: memref<128x256xf16
// CHECK-SAME: %[[S0:.*]]: index, %[[S1:.*]]: index
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: %[[FR_SLICE:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0] {{\[}}%{{.*}}, %{{.*}}, 16, 16] [1, 1, 1, 1] : tensor<16x8x16x16xf32> to tensor<?x?x16x16xf32>
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[FR_SLICE]] : tensor<?x?x16x16xf32>)
func.func @fold_convert_extract_slice_fixpipe(%dst: memref<128x256xf16, strided<[1024, 1], offset: ?>>, %s0: index, %s1: index) {
  %c0 = arith.constant 0 : index
  %mmad_out = arith.constant dense<0.0> : tensor<16x8x16x16xf32>
  %conv = hivm.hir.convert_layout %mmad_out output_shape [128, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<16x8x16x16xf32>) -> tensor<128x256xf32>
  %slice = tensor.extract_slice %conv[0, 0] [%s0, %s1] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
  %subview = memref.subview %dst[0, 0] [%s0, %s1] [1, 1] : memref<128x256xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf16, strided<[1024, 1], offset: ?>>)
  return
}

// -----


// CHECK: func.func @fold_two_use_split(%[[VAL_0:.*]]: memref<16x256xbf16, strided<[?, 1], offset: ?>>,
// CHECK-DAG: %[[VAL_5:.*]] = memref.alloc() : memref<16x256xbf16>
// CHECK-DAG: %[[VAL_6:.*]] = memref.subview %[[VAL_0]][0, 0] {{\[}}%{{.*}}, 256] [1, 1] : memref<16x256xbf16, strided<[?, 1], offset: ?>> to memref<?x256xbf16, strided<[?, 1], offset: ?>>
// CHECK: %[[VAL_7:.*]] = memref.subview %[[VAL_5]][0, 0] {{\[}}%{{.*}}, 256] [1, 1] : memref<16x256xbf16> to memref<?x256xbf16, strided<[256, 1]>>

// Create a new container to store the nd2nz with fractal layout
// CHECK: %[[VAL_9:.*]] = memref.alloc() : memref<16x1x16x16xbf16>
// CHECK: %[[VAL_10:.*]] = memref.subview %[[VAL_9]][0, 0, 0, 0] [16, %{{.*}}, 16, 16] [1, 1, 1, 1] : memref<16x1x16x16xbf16> to memref<16x?x16x16xbf16, strided<[256, 256, 16, 1]>>
// CHECK: hivm.hir.nd2nz {dst_continuous} ins(%[[VAL_6]] : memref<?x256xbf16, strided<[?, 1], offset: ?>>) outs(%[[VAL_10]] : memref<16x?x16x16xbf16, strided<[256, 256, 16, 1]>>)

// Convert both AIV path (VTranspose) and AIC path (nd2nz, load) to tensor
// CHECK-DAG: %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<16x1x16x16xbf16>
// CHECK-DAG: %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_5]] restrict writable : memref<16x256xbf16>
// CHECK: %[[VAL_14:.*]] = hivm.hir.vtranspose ins(%[[VAL_12]] : tensor<16x256xbf16>) outs(%{{.*}} : tensor<256x16xbf16>) permutation = [1, 0] -> tensor<256x16xbf16>
// CHECK: %[[VAL_15:.*]] = hivm.hir.convert_layout %[[VAL_14]] output_shape [2, 16, 16, 8]
// CHECK-SAME: (tensor<256x16xbf16>) -> tensor<2x16x16x8xbf16>
// CHECK: return %[[VAL_11]], %[[VAL_15]] : tensor<16x1x16x16xbf16>, tensor<2x16x16x8xbf16>
func.func @fold_two_use_split(%arg0: memref<16x256xbf16, strided<[?, 1], offset: ?>>, %arg1: index, %arg2: i1) -> (tensor<16x1x16x16xbf16>, tensor<2x16x16x8xbf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<16x256xbf16>
  %subview = memref.subview %arg0[0, 0] [%arg1, 256] [1, 1] : memref<16x256xbf16, strided<[?, 1], offset: ?>> to memref<?x256xbf16, strided<[?, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%arg1, 256] [1, 1] : memref<16x256xbf16> to memref<?x256xbf16, strided<[256, 1]>>
  hivm.hir.load ins(%subview : memref<?x256xbf16, strided<[?, 1], offset: ?>>) outs(%subview_0 : memref<?x256xbf16, strided<[256, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %arg2 : i1
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x256xbf16>
  %1 = hivm.hir.convert_layout %0 output_shape [16, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x256xbf16>) -> tensor<16x1x16x16xbf16>
  %2 = tensor.empty() : tensor<256x16xbf16>
  %3 = hivm.hir.vtranspose ins(%0 : tensor<16x256xbf16>) outs(%2 : tensor<256x16xbf16>) permutation = [1, 0] -> tensor<256x16xbf16>
  %4 = hivm.hir.convert_layout %3 output_shape [2, 16, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>} : (tensor<256x16xbf16>) -> tensor<2x16x16x8xbf16>
  return %1, %4 : tensor<16x1x16x16xbf16>, tensor<2x16x16x8xbf16>
}

// -----

// CHECK-LABEL: func.func @fold_subview_annotation_mark
// CHECK-NOT: hivm.hir.load
// CHECK: %[[FRACTAL_ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16>
// CHECK: hivm.hir.nd2nz {dst_continuous} ins(
// CHECK: %[[NEW_TENSOR:.*]] = bufferization.to_tensor %[[FRACTAL_ALLOC]] restrict writable
// CHECK: annotation.mark %[[NEW_TENSOR]] {dot_pad_only_k} : tensor<1x1x16x16xf16>
func.func @fold_subview_annotation_mark(%arg0: memref<16x16xf16, strided<[?, 1], offset: ?>>, %arg1: index) -> tensor<1x1x16x16xf16> {
  %alloc = memref.alloc() : memref<16x16xf16>
  %subview_in = memref.subview %arg0[0, 0] [%arg1, 16] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x16xf16, strided<[?, 1], offset: ?>>
  %subview_out = memref.subview %alloc[0, 0] [%arg1, 16] [1, 1] : memref<16x16xf16> to memref<?x16xf16, strided<[16, 1]>>
  hivm.hir.load ins(%subview_in : memref<?x16xf16, strided<[?, 1], offset: ?>>) outs(%subview_out : memref<?x16xf16, strided<[16, 1]>>)
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
  %1 = hivm.hir.convert_layout %0 output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  annotation.mark %0 {dot_pad_only_k} : tensor<16x16xf16>
  return %1 : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @fold_direct_load_annotation_mark
// CHECK-NOT: hivm.hir.load
// CHECK: %[[FRACTAL_ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16>
// CHECK: hivm.hir.nd2nz {dst_continuous} ins(
// CHECK: %[[NEW_TENSOR:.*]] = bufferization.to_tensor %[[FRACTAL_ALLOC]] restrict writable
// CHECK: annotation.mark %[[NEW_TENSOR]] {dot_pad_only_k} : tensor<1x1x16x16xf16>
func.func @fold_direct_load_annotation_mark(%arg0: memref<16x16xf16, strided<[?, 1], offset: ?>>) -> tensor<1x1x16x16xf16> {
  %alloc = memref.alloc() : memref<16x16xf16>
  hivm.hir.load ins(%arg0 : memref<16x16xf16, strided<[?, 1], offset: ?>>) outs(%alloc : memref<16x16xf16>)
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
  %1 = hivm.hir.convert_layout %0 output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  annotation.mark %0 {dot_pad_only_k} : tensor<16x16xf16>
  return %1 : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @fold_tensor_load_convert_layout
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x4x16x8xf32>
// CHECK: %[[RES:.*]] = hivm.hir.nd2nz {dst_continuous} ins(%{{.*}} : tensor<64x64xf32>) outs(%[[EMPTY]] : tensor<8x4x16x8xf32>) -> tensor<8x4x16x8xf32>
// CHECK: return %[[RES]] : tensor<8x4x16x8xf32>
func.func @fold_tensor_load_convert_layout(%src: tensor<64x64xf32>, %l1_init: tensor<64x64xf32>) -> tensor<8x4x16x8xf32> {
  %load = hivm.hir.load ins(%src : tensor<64x64xf32>) outs(%l1_init : tensor<64x64xf32>) {"inserted-load"} core_type = <CUBE> -> tensor<64x64xf32>
  %conv = hivm.hir.convert_layout %load output_shape [8, 4, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>} : (tensor<64x64xf32>) -> tensor<8x4x16x8xf32>
  return %conv : tensor<8x4x16x8xf32>
}

// -----

// Regression for the row-wise A operand gather shape from err-kernel.ttadapter:
// the load writes %alloc through a subview inside the inner scf.for, while
// to_tensor/convert_layout are outside that loop. Folding this convert_layout
// after the nested load would create a tensor value inside the loop and use it
// outside the loop.
// CHECK-LABEL: func.func @do_not_fold_nested_row_load_from_case
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32>
// CHECK: scf.for
// CHECK: hivm.hir.load
// CHECK-NOT: hivm.hir.nd2nz
// CHECK: %[[A_TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<64x32xf32>
// CHECK: %[[A_FRACTAL:.*]] = hivm.hir.convert_layout %[[A_TENSOR]] output_shape [4, 4, 16, 8]
// CHECK-SAME: (tensor<64x32xf32>) -> tensor<4x4x16x8xf32>
// CHECK: hivm.hir.copy ins(%[[A_FRACTAL]] : tensor<4x4x16x8xf32>)
func.func @do_not_fold_nested_row_load_from_case(%arg2: memref<?xf32>, %arg5: i32, %arg13: i32, %base_m: i32, %base_k: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %alloc = memref.alloc() : memref<64x32xf32>
  scf.for %arg25 = %c0 to %c64 step %c1 {
    %row_i32 = arith.index_cast %arg25 : index to i32
    %row = arith.addi %base_m, %row_i32 : i32
    %wrapped_row = arith.remsi %row, %arg5 : i32
    %row_offset_i32 = arith.muli %wrapped_row, %arg13 : i32
    %row_offset = arith.index_cast %row_offset_i32 : i32 to index
    %base_k_index = arith.index_cast %base_k : i32 to index
    %src_offset = arith.addi %row_offset, %base_k_index : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%src_offset], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
    %subview = memref.subview %alloc[%arg25, 0] [1, 32] [1, 1] : memref<64x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
    hivm.hir.load ins(%reinterpret_cast : memref<1x32xf32, strided<[32, 1], offset: ?>>) outs(%subview : memref<1x32xf32, strided<[32, 1], offset: ?>>) left_padding_num = %c0 : index core_type = <VECTOR>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %a_tensor = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %a_fractal = hivm.hir.convert_layout %a_tensor output_shape [4, 4, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, not_to_propagate_up = true, srcLayout = #hivm.data_layout<ND>} : (tensor<64x32xf32>) -> tensor<4x4x16x8xf32>
  %cbuf = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%a_fractal : tensor<4x4x16x8xf32>) outs(%cbuf : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>) {"inserted-copy"}
  return
}
