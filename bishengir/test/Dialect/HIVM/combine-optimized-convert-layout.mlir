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
  hivm.hir.load ins(%subview : memref<?x16xf16, strided<[?, 1], offset: ?>>) outs(%subview_0 : memref<?x16xf16, strided<[16, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %arg3 : i1 eviction_policy = <EvictFirst>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
  %1 = hivm.hir.convert_layout %0 output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND, transpose = false>} : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  return %1 : tensor<1x1x16x16xf16>
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
  hivm.hir.load ins(%subview : memref<?x256xbf16, strided<[?, 1], offset: ?>>) outs(%subview_0 : memref<?x256xbf16, strided<[256, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %arg2 : i1 eviction_policy = <EvictFirst>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x256xbf16>
  %1 = hivm.hir.convert_layout %0 output_shape [16, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x256xbf16>) -> tensor<16x1x16x16xbf16>
  %2 = tensor.empty() : tensor<256x16xbf16>
  %3 = hivm.hir.vtranspose ins(%0 : tensor<16x256xbf16>) outs(%2 : tensor<256x16xbf16>) permutation = [1, 0] -> tensor<256x16xbf16>
  %4 = hivm.hir.convert_layout %3 output_shape [2, 16, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>} : (tensor<256x16xbf16>) -> tensor<2x16x16x8xbf16>
  return %1, %4 : tensor<16x1x16x16xbf16>, tensor<2x16x16x8xbf16>
}
