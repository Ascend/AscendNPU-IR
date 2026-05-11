// RUN: bishengir-opt %s  --hivm-bubble-up-extract-slice --split-input-file -verify-diagnostics  | FileCheck %s

// CHECK-LABEL:   func.func @bubble_up_hivm(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vln ins(%[[VAL_1]] : tensor<2xf32>) outs(%[[VAL_2]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return %[[VAL_3]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm(%arg0: tensor<4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() :  tensor<4xf32>
    %29 = hivm.hir.vln ins(%arg0: tensor<4xf32>) outs(%0  : tensor<4xf32>) -> tensor<4xf32>
    %extracted_slice = tensor.extract_slice %29[1] [2] [1] {to_be_bubbled_slice} : tensor<4xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce2(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<5x4xf32>) -> tensor<1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1x2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_1]] : tensor<5x2xf32>) outs(%[[VAL_2]] : tensor<1x2xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x2xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x2xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce2(%arg0: tensor<5x4xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%0 : tensor<1x4xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x4xf32>
    %extracted_slice = tensor.extract_slice %51[0, 0] [1,2] [1,1] {to_be_bubbled_slice} : tensor<1x4xf32> to tensor<1x2xf32>
    return %extracted_slice : tensor<1x2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_vbrc(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<1x4xf32>) -> tensor<5x2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vbrc ins(%[[VAL_1]] : tensor<1x2xf32>) outs(%[[VAL_2]] : tensor<5x2xf32>) broadcast_dims = [0] -> tensor<5x2xf32>
// CHECK:           return %[[VAL_3]] : tensor<5x2xf32>
// CHECK:         }
func.func @bubble_up_hivm_vbrc(%arg0: tensor<1x4xf32>) -> tensor<5x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5x4xf32>
    %35 = hivm.hir.vbrc ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<5x4xf32>) broadcast_dims = [0] -> tensor<5x4xf32>
    %extracted_slice = tensor.extract_slice %35[0, 0] [5, 2] [1,1] {to_be_bubbled_slice} : tensor<5x4xf32> to tensor<5x2xf32>
    return %extracted_slice : tensor<5x2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_collapse_shape(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<64x1xf32>) -> tensor<32xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1]] : tensor<32x1xf32> into tensor<32xf32>
// CHECK:           return %[[VAL_2]] : tensor<32xf32>
// CHECK:         }

func.func @bubble_up_collapse_shape(%arg0: tensor<64x1xf32>) -> tensor<32xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
  %extracted_slice_10 = tensor.extract_slice %collapsed[0] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
  return %extracted_slice_10 : tensor<32xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_expand_shape(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<64xf32>) -> tensor<32x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x1xf32>
// CHECK:         }
func.func @bubble_up_expand_shape(%arg0: tensor<64xf32>) -> tensor<32x1xf32> {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %extracted_slice_10 = tensor.extract_slice %expanded[0, 0] [32, 1] [1, 1] {to_be_bubbled_slice} : tensor<64x1xf32> to tensor<32x1xf32>
    return %extracted_slice_10 : tensor<32x1xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_vinterleave(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<32x16x1xi32>) -> tensor<16x16x2xi32> {
// CHECK:          %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:          %[[VAL_2:.*]] = tensor.empty() : tensor<16x16x2xi32>
// CHECK:          %[[VAL_3:.*]] = hivm.hir.vinterleave ins(%[[VAL_1:.*]], %[[VAL_1:.*]] : tensor<16x16x1xi32>, tensor<16x16x1xi32>) outs(%[[VAL_2:.*]] : tensor<16x16x2xi32>) interleave_channel_nums = 2 -> tensor<16x16x2xi32>
// CHECK:          return %[[VAL_3:.*]] : tensor<16x16x2xi32>
// CHECK:         }
func.func @bubble_up_vinterleave(%arg0: tensor<32x16x1xi32>) -> tensor<16x16x2xi32> {
  %62 = tensor.empty() : tensor<32x16x2xi32>
  %63 = hivm.hir.vinterleave ins(%arg0, %arg0 : tensor<32x16x1xi32>, tensor<32x16x1xi32>) outs(%62 : tensor<32x16x2xi32>) interleave_channel_nums = 2 -> tensor<32x16x2xi32>
  %extracted_slice = tensor.extract_slice %63[0, 0, 0] [16, 16, 2] [1, 1, 1] {to_be_bubbled_slice} : tensor<32x16x2xi32> to tensor<16x16x2xi32>
  return %extracted_slice : tensor<16x16x2xi32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_for_loop3(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<64x32xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<32x16xf32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: tensor<64x16xf32>) -> (tensor<32x32xf32>, index) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_7:.*]]:2 = scf.for %[[VAL_8:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]], %[[VAL_10:.*]] = %[[VAL_3]]) -> (tensor<32x32xf32>, index) {
// CHECK-DAG:             %[[VAL_12:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK-DAG:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_4]] : index
// CHECK-DAG:             %[[VAL_13:.*]] = hivm.hir.vln ins(%[[VAL_9]] : tensor<32x32xf32>) outs(%[[VAL_12]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:             scf.yield %[[VAL_13]], %[[VAL_11]] : tensor<32x32xf32>, index
// CHECK:           } {to_be_tiled_op}
// CHECK:           return %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<32x32xf32>, index
// CHECK:         }
func.func @bubble_up_for_loop3(%arg0: tensor<64x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<64x16xf32>) -> (tensor<32x32xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_counter = arith.constant 0 : index
  %result:2 = scf.for %i = %c0 to %c10 step %c1
    iter_args(%temp = %arg0, %counter = %init_counter) -> (tensor<64x32xf32>, index) {
        %expanded = tensor.empty() : tensor<64x32xf32>
    %29 = hivm.hir.vln ins(%temp: tensor<64x32xf32>) outs(%expanded : tensor<64x32xf32>) -> tensor<64x32xf32>
    %new_counter = arith.addi %counter, %c1 : index
    scf.yield %29, %new_counter : tensor<64x32xf32>, index
  } {to_be_tiled_op}
  %extracted_slice_10 = tensor.extract_slice %result#0[0, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xf32> to tensor<32x32xf32>
  return  %extracted_slice_10  , %result#1 : tensor<32x32xf32>, index
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce1(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<5x4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x1xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_1]] : tensor<2x4xf32>) outs(%[[VAL_2]] : tensor<2x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<2x1xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_3]] {{\[\[}}0, 1]] : tensor<2x1xf32> into tensor<2xf32>
// CHECK:           return %[[VAL_4]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce1(%arg0: tensor<5x4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5xf32>
    %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [5, 1] : tensor<5xf32> into tensor<5x1xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%expanded : tensor<5x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<5x1xf32>
    %collapsed = tensor.collapse_shape %51 [[0, 1]] : tensor<5x1xf32> into tensor<5xf32>

    %extracted_slice = tensor.extract_slice %collapsed[0] [2] [1] {to_be_bubbled_slice} : tensor<5xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_reduce(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<5x4xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x1xf32>
// CHECK:           %[[VAL_3:.*]] = hivm.hir.vreduce <sum> ins(%[[VAL_1]] : tensor<2x4xf32>) outs(%[[VAL_2]] : tensor<2x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<2x1xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_3]] {{\[\[}}0, 1]] : tensor<2x1xf32> into tensor<2xf32>
// CHECK:           return %[[VAL_4]] : tensor<2xf32>
// CHECK:         }
func.func @bubble_up_hivm_reduce(%arg0: tensor<5x4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5xf32>
    %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [5, 1] : tensor<5xf32> into tensor<5x1xf32>
    %51 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x4xf32>) outs(%expanded : tensor<5x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<5x1xf32>
    %collapsed = tensor.collapse_shape %51 [[0, 1]] : tensor<5x1xf32> into tensor<5xf32>

    %extracted_slice = tensor.extract_slice %collapsed[0] [2] [1] {to_be_bubbled_slice} : tensor<5xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
}


// -----
// CHECK-LABEL:   func.func @bubble_up_varange(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<128xi32>) -> tensor<64xi32> {
// CHECK:            %c1 = arith.constant 1 : index
// CHECK:            %c64 = arith.constant 64 : index
// CHECK:            %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0:.*]][64] [64] [1] {to_be_bubbled_slice} : tensor<128xi32> to tensor<64xi32>
// CHECK:            %[[VAL_2:.*]] = hivm.hir.varange offset[%c64] strides[%c1] outs(%[[VAL_1:.*]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:            return %[[VAL_2:.*]] : tensor<64xi32>
// CHECK:        }
func.func @bubble_up_varange(%arg0 : tensor<128xi32>) -> tensor<64xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index 
    %14 = hivm.hir.varange offset[%c0] strides[%c1] outs(%arg0 : tensor<128xi32>) -> tensor<128xi32>
    %extracted_slice = tensor.extract_slice %14[64] [64] [1] {to_be_bubbled_slice} : tensor<128xi32> to tensor<64xi32>
    return %extracted_slice : tensor<64xi32>
}


// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_fixpipe(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<128x128xf32>) -> tensor<64x128xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_0:.*]] : tensor<128x128xf32>) outs(%[[VAL_1:.*]] : tensor<64x128xf32>) dual_dst_mode = <ROW_SPLIT> -> tensor<64x128xf32>
// CHECK:           return %[[VAL_2:.*]] : tensor<64x128xf32>
// CHECK:         }
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @bubble_up_hivm_fixpipe(%arg0 : tensor<128x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0] [64,128] [1,1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
    return %extracted_slice : tensor<64x128xf32>
  }
}

// -----
// CHECK-LABEL:   func.func @bubble_up_hivm_bitcast(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<32x64xf32>) -> tensor<16x64xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][1, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : tensor<32x64xf32> to tensor<16x64xf32>
// CHECK:           %[[VAL_2:.*]] = hivm.hir.bitcast %[[VAL_1]] : tensor<16x64xf32> -> tensor<16x64xi32>
// CHECK:           return %[[VAL_2]] : tensor<16x64xi32>
// CHECK:         }
func.func @bubble_up_hivm_bitcast(%arg0: tensor<32x64xf32>) -> tensor<16x64xi32> {
    %1 = hivm.hir.bitcast %arg0 : tensor<32x64xf32> -> tensor<32x64xi32>
    %extracted_slice = tensor.extract_slice %1[1,0] [16,64] [1,1] {to_be_bubbled_slice} : tensor<32x64xi32> to tensor<16x64xi32>
    return %extracted_slice : tensor<16x64xi32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_if(
// CHECK-SAME:                            %[[VAL_0:.*]]: i1,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<8x8xf32>,
// CHECK-SAME:                            %[[VAL_2:.*]]: tensor<8x8xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_3:.*]] = scf.if %[[VAL_0]] -> (tensor<4x8xf32>) {
// CHECK:             %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]][0, 0] [4, 8] [1, 1] {to_be_bubbled_slice} : tensor<8x8xf32> to tensor<4x8xf32>
// CHECK:             scf.yield %[[VAL_4]] : tensor<4x8xf32>
// CHECK:           } else {
// CHECK:             %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]][0, 0] [4, 8] [1, 1] {to_be_bubbled_slice} : tensor<8x8xf32> to tensor<4x8xf32>
// CHECK:             scf.yield %[[VAL_5]] : tensor<4x8xf32>
// CHECK:           }
// CHECK:           return %[[VAL_3]] : tensor<4x8xf32>
// CHECK:         }
func.func @bubble_up_if(%cond: i1,
                        %arg0: tensor<8x8xf32>,
                        %arg1: tensor<8x8xf32>)
    -> tensor<4x8xf32> {

  %r = scf.if %cond -> tensor<8x8xf32> {
    scf.yield %arg0 : tensor<8x8xf32>
  } else {
    scf.yield %arg1 : tensor<8x8xf32>
  }

  %slice = tensor.extract_slice %r[0, 0] [4, 8] [1, 1] {to_be_bubbled_slice}
      : tensor<8x8xf32> to tensor<4x8xf32>

  return %slice : tensor<4x8xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_select(
// CHECK-SAME:                                %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                %[[VAL_1:.*]]: tensor<64x128xf32>,
// CHECK-SAME:                                %[[VAL_2:.*]]: tensor<64x128xf32>) -> tensor<32x128xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_1]][0, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf32> to tensor<32x128xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_2]][0, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf32> to tensor<32x128xf32>
// CHECK:           %[[VAL_5:.*]] = arith.select %[[VAL_0]], %[[VAL_3]], %[[VAL_4]] : tensor<32x128xf32>
// CHECK:           return %[[VAL_5]] : tensor<32x128xf32>
// CHECK:         }
func.func @bubble_up_select(
    %cond: i1,
    %arg0: tensor<64x128xf32>,
    %arg1: tensor<64x128xf32>)
    -> tensor<32x128xf32> {

  %sel = arith.select %cond, %arg0, %arg1
      : tensor<64x128xf32>

  %slice = tensor.extract_slice %sel[0, 0] [32, 128] [1, 1] {to_be_bubbled_slice}
      : tensor<64x128xf32> to tensor<32x128xf32>

  return %slice : tensor<32x128xf32>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_empty_odd_buffer_size(
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%arg0) : tensor<?x80xf16>
// CHECK:           annotation.mark %[[EMPTY]] {buffer_size_in_byte = 9280 : i64} : tensor<?x80xf16>
func.func @bubble_up_empty_odd_buffer_size(%arg0: index) -> tensor<?x80xf16> {
  %empty = tensor.empty() : tensor<115x80xf16>
  %slice = tensor.extract_slice %empty[0, 0] [%arg0, 80] [1, 1] {to_be_bubbled_slice}
      : tensor<115x80xf16> to tensor<?x80xf16>
  return %slice : tensor<?x80xf16>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_alloc_odd_buffer_size(
// CHECK-SAME:                                                    %[[SRC:.*]]: memref<?xf16>,
// CHECK-SAME:                                                    %[[SIZE:.*]]: index) -> tensor<?x80xf16> {
// CHECK:           %[[SLICED_ALLOC:.*]] = memref.alloc(%[[SIZE]]) : memref<?x80xf16>
// CHECK:           annotation.mark %[[SLICED_ALLOC]] {buffer_size_in_byte = 9280 : i64} : memref<?x80xf16>
// CHECK:           hivm.hir.load ins({{.*}} : memref<?x80xf16, strided<[80, 1], offset: ?>>) outs(%[[SLICED_ALLOC]] : memref<?x80xf16>)
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[SLICED_ALLOC]] restrict writable : memref<?x80xf16>
// CHECK:           return %[[TENSOR]] : tensor<?x80xf16>
// CHECK:         }
func.func @bubble_up_alloc_odd_buffer_size(
    %arg0: memref<?xf16>, %arg1: index) -> tensor<?x80xf16> {
  %alloc = memref.alloc() : memref<115x80xf16>
  %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [115, 80], strides: [80, 1]
      : memref<?xf16> to memref<115x80xf16, strided<[80, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<115x80xf16, strided<[80, 1], offset: ?>>) outs(%alloc : memref<115x80xf16>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<115x80xf16>
  %slice = tensor.extract_slice %tensor[0, 0] [%arg1, 80] [1, 1] {to_be_bubbled_slice}
      : tensor<115x80xf16> to tensor<?x80xf16>
  return %slice : tensor<?x80xf16>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_subview_alloc_odd_buffer_size(
// CHECK-SAME:                                                            %[[SRC:.*]]: memref<?xf16>,
// CHECK-SAME:                                                            %[[SIZE:.*]]: index) -> tensor<?x80xf16> {
// CHECK:           %[[SLICED_ALLOC:.*]] = memref.alloc(%[[SIZE]]) : memref<?x80xf16>
// CHECK:           annotation.mark %[[SLICED_ALLOC]] {buffer_size_in_byte = 9280 : i64} : memref<?x80xf16>
// CHECK:           %[[DST_SUBVIEW:.*]] = memref.subview %[[SLICED_ALLOC]]
// CHECK:           hivm.hir.load ins({{.*}} : memref<?x80xf16, strided<[80, 1], offset: ?>>) outs(%[[DST_SUBVIEW]] : memref<?x80xf16, strided<[80, 1]>>)
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[SLICED_ALLOC]] restrict writable : memref<?x80xf16>
// CHECK:           return %[[TENSOR]] : tensor<?x80xf16>
// CHECK:         }
func.func @bubble_up_subview_alloc_odd_buffer_size(
    %arg0: memref<?xf16>, %arg1: index) -> tensor<?x80xf16> {
  %alloc = memref.alloc() : memref<115x80xf16>
  %dst = memref.subview %alloc[0, 0] [115, 80] [1, 1]
      : memref<115x80xf16> to memref<115x80xf16, strided<[80, 1]>>
  %src_base = memref.reinterpret_cast %arg0 to offset: [0], sizes: [115, 80], strides: [80, 1]
      : memref<?xf16> to memref<115x80xf16, strided<[80, 1], offset: ?>>
  %src = memref.subview %src_base[0, 0] [115, 80] [1, 1]
      : memref<115x80xf16, strided<[80, 1], offset: ?>> to memref<115x80xf16, strided<[80, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<115x80xf16, strided<[80, 1], offset: ?>>) outs(%dst : memref<115x80xf16, strided<[80, 1]>>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<115x80xf16>
  %slice = tensor.extract_slice %tensor[0, 0] [%arg1, 80] [1, 1] {to_be_bubbled_slice}
      : tensor<115x80xf16> to tensor<?x80xf16>
  return %slice : tensor<?x80xf16>
}

// -----
// CHECK-LABEL:   func.func @bubble_up_extract_of_insert_same_dim_dynamic(
// CHECK:           %[[VAL_0:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %arg0[%arg3] [32] [1] : tensor<64xf32> to tensor<32xf32>
// CHECK:           %[[VAL_2:.*]] = arith.minsi %arg3, %arg2 : index
// CHECK:           %[[VAL_3:.*]] = arith.subi %arg2, %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.minsi %[[VAL_3]], %[[VAL_0]] : index
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]][0] [%[[VAL_4]]] [1] : tensor<32xf32> to tensor<?xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice %arg1[%arg3] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.insert_slice %[[VAL_5]] into %[[VAL_6]][0] [%[[VAL_4]]] [1] : tensor<?xf32> into tensor<32xf32>
// CHECK:           return %[[VAL_7]] : tensor<32xf32>
// CHECK:         }
func.func @bubble_up_extract_of_insert_same_dim_dynamic(
    %arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: index, %arg3: index)
    -> tensor<32xf32> {
  %0 = tensor.extract_slice %arg0[0] [%arg2] [1]
      : tensor<64xf32> to tensor<?xf32>
  %1 = tensor.insert_slice %0 into %arg1[0] [%arg2] [1]
      : tensor<?xf32> into tensor<64xf32>
  %2 = tensor.extract_slice %1[%arg3] [32] [1] {to_be_bubbled_slice}
      : tensor<64xf32> to tensor<32xf32>
  return %2 : tensor<32xf32>
}

// -----
// ExtractSliceBubbleUpStrategy: Extract→Extract different dim
// Parent extracts on dim0, to_be_bubbled_slice extracts on dim1 (no overlap)
// After bubble-up: swap extraction order, extract dim1 first then dim0
// CHECK-LABEL:   func.func @bubble_up_extract_of_extract_different_dim(
// CHECK:           %[[VAL_0:.*]] = tensor.extract_slice %arg0[0, %arg2] [64, 8] [1, 1] {to_be_bubbled_slice} : tensor<64x16xf32> to tensor<64x8xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][%arg1, 0] [32, 8] [1, 1] : tensor<64x8xf32> to tensor<32x8xf32>
// CHECK:           return %[[VAL_1]] : tensor<32x8xf32>
// CHECK:         }
func.func @bubble_up_extract_of_extract_different_dim(
    %arg0: tensor<64x16xf32>, %arg1: index, %arg2: index) -> tensor<32x8xf32> {
  %0 = tensor.extract_slice %arg0[%arg1, 0] [32, 16] [1, 1]
      : tensor<64x16xf32> to tensor<32x16xf32>
  %1 = tensor.extract_slice %0[0, %arg2] [32, 8] [1, 1] {to_be_bubbled_slice}
      : tensor<32x16xf32> to tensor<32x8xf32>
  return %1 : tensor<32x8xf32>
}

// -----
// ExtractSliceBubbleUpStrategy: Extract→Extract rank-reduced
// Parent does rank-reduce (4x16x16 -> 16x16), to_be_bubbled_slice extracts from result
// After bubble-up: extract from original tensor first, then rank-reduce
// CHECK-LABEL:   func.func @bubble_up_extract_of_extract_rank_reduced(
// CHECK:           %[[VAL_0:.*]] = tensor.extract_slice %arg0[0, %arg2, 0] [4, 8, 8] [1, 1, 1] {to_be_bubbled_slice} : tensor<4x16x16xf32> to tensor<4x8x8xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, %arg1, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x8xf32> to tensor<8x8xf32>
// CHECK:           return %[[VAL_1]] : tensor<8x8xf32>
// CHECK:         }
func.func @bubble_up_extract_of_extract_rank_reduced(
    %arg0: tensor<4x16x16xf32>, %arg1: index, %arg2: index) -> tensor<8x8xf32> {
  %0 = tensor.extract_slice %arg0[0, %arg1, 0] [1, 16, 16] [1, 1, 1]
      : tensor<4x16x16xf32> to tensor<16x16xf32>
  %1 = tensor.extract_slice %0[%arg2, 0] [8, 8] [1, 1] {to_be_bubbled_slice}
      : tensor<16x16xf32> to tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----
// InsertSliceBubbleUpStrategy: Insert→Extract ranked-reduce (static)
// Insert does rank-expand (8x8 -> 4x8x8), extract on dim1 only (4x8x8 -> 4x4x8)
// After bubble-up: extract from source and dest, then insert
// CHECK-LABEL:   func.func @bubble_up_insert_rank_reduced(
// CHECK:           %[[VAL_0:.*]] = tensor.extract_slice %arg0[0, %arg2] [4, 8] [1, 1] {to_be_bubbled_slice} : tensor<8x8xf32> to tensor<4x8xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %arg1[0, 0, %arg2] [4, 4, 8] [1, 1, 1] {to_be_bubbled_slice} : tensor<4x8x8xf32> to tensor<4x4x8xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.insert_slice %[[VAL_0]] into %[[VAL_1]][0, %arg2, 0] [1, 4, 8] [1, 1, 1] : tensor<4x8xf32> into tensor<4x4x8xf32>
// CHECK:           return %[[VAL_2]] : tensor<4x4x8xf32>
// CHECK:         }
func.func @bubble_up_insert_rank_reduced(
    %arg0: tensor<8x8xf32>, %arg1: tensor<4x8x8xf32>, %arg2: index) -> tensor<4x4x8xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, %arg2, 0] [1, 8, 8] [1, 1, 1]
      : tensor<8x8xf32> into tensor<4x8x8xf32>
  %1 = tensor.extract_slice %0[0, 0, %arg2] [4, 4, 8] [1, 1, 1] {to_be_bubbled_slice}
      : tensor<4x8x8xf32> to tensor<4x4x8xf32>
  return %1 : tensor<4x4x8xf32>
}

// -----
// InsertSliceBubbleUpStrategy: Insert→Extract same dim, non-tiling, source dimSize=1
// Insert scalar (1xf32) into tensor at offset, then extract from result
// After bubble-up: extract from dest first, then insert scalar into extracted slice
// CHECK-LABEL:   func.func @bubble_up_insert_scalar_same_dim(
// CHECK:           %[[VAL_0:.*]] = tensor.extract_slice %arg1[%arg2] [8] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<8xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.insert_slice %arg0 into %[[VAL_0]][%arg2] [1] [1] : tensor<1xf32> into tensor<8xf32>
// CHECK:           return %[[VAL_1]] : tensor<8xf32>
// CHECK:         }
func.func @bubble_up_insert_scalar_same_dim(
    %arg0: tensor<1xf32>, %arg1: tensor<64xf32>, %arg2: index) -> tensor<8xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[%arg2] [1] [1]
      : tensor<1xf32> into tensor<64xf32>
  %1 = tensor.extract_slice %0[%arg2] [8] [1] {to_be_bubbled_slice}
      : tensor<64xf32> to tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----
// InsertSliceBubbleUpStrategy: Insert→Extract different dim (static source)
// Insert on dim0 only (8x16 into 16x16), extract on dim1 only (16x16 -> 16x8)
// After bubble-up: extract from dest and source, then insert
// CHECK-LABEL:   func.func @bubble_up_insert_extract_different_dim(
// CHECK:           %[[VAL_0:.*]] = tensor.extract_slice %arg1[0, %arg3] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %arg0[0, %arg3] [8, 8] [1, 1] {to_be_bubbled_slice} : tensor<8x16xf32> to tensor<8x8xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_0]][%arg2, 0] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<16x8xf32>
// CHECK:           return %[[VAL_2]] : tensor<16x8xf32>
// CHECK:         }
func.func @bubble_up_insert_extract_different_dim(
    %arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>, %arg2: index, %arg3: index) -> tensor<16x8xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[%arg2, 0] [8, 16] [1, 1]
      : tensor<8x16xf32> into tensor<16x16xf32>
  %1 = tensor.extract_slice %0[0, %arg3] [16, 8] [1, 1] {to_be_bubbled_slice}
      : tensor<16x16xf32> to tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}
