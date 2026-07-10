// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// TODO: add more testcase for indirect_load for its input operands
// TODO: add testcase for addition convert layout to each inputs operands

// CHECK-LABEL: @no_insert_store_between_extract_and_non_vector_non_cube_user(
// CHECK: %[[EXTRACTED_INDEX:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<52xi64>
// CHECK: %{{[A-Za-z0-9_]+}} = arith.index_cast %[[EXTRACTED_INDEX:.*]] : i64 to index
// CHECK-NOT: %{{[A-Za-z0-9_]+}} = hivm.hir.store ins(%input_indices : tensor<52xi64>) outs(%{{[A-Za-z0-9_]+}} : tensor<52xi64>) -> tensor<52xi64>
func.func @no_insert_store_between_extract_and_non_vector_non_cube_user(%input_indices: tensor<52xi64>, %reinterpret_cast_1: memref<16384x768xf16, strided<[768, 1]>>, %alloc_2: memref<52x768xf16>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c52 = arith.constant 52 : index
  %empty = tensor.empty() : tensor<52xi64>
  %loaded = hivm.hir.load ins(%input_indices : tensor<52xi64>) outs(%empty : tensor<52xi64>) -> tensor<52xi64>
  scf.for %arg10 = %c0 to %c52 step %c1 {
    %extracted = tensor.extract %loaded[%arg10] : tensor<52xi64>
    %26 = arith.index_cast %extracted : i64 to index
    %subview_5 = memref.subview %reinterpret_cast_1[%26, 0] [1, 768] [1, 1] : memref<16384x768xf16, strided<[768, 1]>> to memref<1x768xf16, strided<[768, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_2[%arg10, 0] [1, 768] [1, 1] : memref<52x768xf16> to memref<1x768xf16, strided<[768, 1], offset: ?>>
    annotation.mark %subview_6 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x768xf16, strided<[768, 1], offset: ?>>
    hivm.hir.load ins(%subview_5 : memref<1x768xf16, strided<[768, 1], offset: ?>>) outs(%subview_6 : memref<1x768xf16, strided<[768, 1], offset: ?>>) left_padding_num = %c0 : index
  }
  return
}

// -----
// CHECK-LABEL: @insert_store_between_vector_and_cube_grandchild
// CHECK: %[[VEC_RESULT:.*]] = hivm.hir.vmul
// CHECK: %[[EXTRACTED:.*]] = tensor.extract
// CHECK: %[[VEC:.*]] = tensor.splat
// CHECK: %[[SUM:.*]] = hivm.hir.vadd
// CHECK: %{{[A-Za-z0-9_]+}} = hivm.hir.store
// CHECK: %{{[A-Za-z0-9_]+}} = hivm.hir.load
// CHECK: %{{[A-Za-z0-9_]+}} = hivm.hir.mmadL1
func.func @insert_store_between_vector_and_cube_grandchild(%arg0: tensor<16x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<16x1xf32>, %arg3: memref<16x16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = hivm.hir.vmul ins(%arg0, %arg0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<16x16xf32> into tensor<256xf32>
  scf.for %arg4 = %c0 to %c16 step %c1 {
    %extracted = tensor.extract %collapsed[%arg4] : tensor<256xf32>
    %splat = tensor.splat %extracted : tensor<1x16xf32>
    %2 = tensor.empty() : tensor<1x16xf32>
    %3 = hivm.hir.vadd ins(%splat, %arg1 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%2 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %4 = tensor.empty() : tensor<1x16xf32>
    %5 = hivm.hir.store ins(%3 : tensor<1x16xf32>) outs(%4 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %6 = tensor.empty() : tensor<1x16xf32>
    %7 = hivm.hir.load ins(%5 : tensor<1x16xf32>) outs(%6 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %8 = tensor.empty() : tensor<16x16xf32>
    %9 = hivm.hir.mmadL1 ins(%arg2, %7, %true, %c16, %c16, %c16 : tensor<16x1xf32>, tensor<1x16xf32>, i1, index, index, index) outs(%8 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%arg4], sizes: [16, 16], strides: [16, 1] : memref<16x16x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
    hivm.hir.store ins(%9 : tensor<16x16xf32>) outs(%reinterpret_cast : memref<16x16xf32, strided<[16, 1], offset: ?>>)
  }
  return
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_vector(
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf16>, %[[ARG1:.*]]: memref<?xi8>)
func.func @insert_load_between_fixpipe_and_vector(%arg0: memref<?xf16>, %arg1: memref<?xi8>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 2.000000e+00 : f16
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
  %0 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<16x16xf16, strided<[16, 1]>>
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xf16>
  %3 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%1 : tensor<16x16xf32>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %4 = tensor.empty() : tensor<16x16xf16>
  %5 = hivm.hir.load ins(%3 : tensor<16x16xf16>) outs(%4 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %6 = hivm.hir.vmul ins(%5, %cst : tensor<16x16xf16>, f16) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [1] : memref<?xi8> to memref<512xi8, strided<[1]>>
  %view = memref.view %reinterpret_cast_0[%c0][] : memref<512xi8, strided<[1]>> to memref<16x16xf16>
  hivm.hir.store ins(%6 : tensor<16x16xf16>) outs(%view : memref<16x16xf16>)
  return
}

// -----
// CHECK-LABEL: @insert_store_between_vector_and_load
// CHECK: %[[VAL1:.*]] = hivm.hir.vmul
// CHECK: %[[VAL2:.*]] = hivm.hir.store
// CHECK: %[[VAL3:.*]] = tensor.empty
// CHECK: %[[VAL4:.*]] = hivm.hir.load
func.func @insert_store_between_vector_and_load(%arg0: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  %0 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<16x16xf32, strided<[16, 1]>>
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = hivm.hir.vmul ins(%0, %0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %3 = tensor.empty() : tensor<16x16xf32>
  %4 = hivm.hir.store ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = tensor.empty() : tensor<16x16xf32>
  %6 = hivm.hir.load ins(%4 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  hivm.hir.store ins(%6 : tensor<16x16xf32>) outs(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1]>>)
  return
}

// -----
// CHECK-LABEL: @insert_load_store_between_vector_and_cube
// CHECK: %[[VAL1:.*]] = hivm.hir.vmul
// CHECK: %[[VAL2:.*]] = hivm.hir.store
// CHECK: %[[VAL3:.*]] = hivm.hir.load
// CHECK: %[[VAL4:.*]] = hivm.hir.store
// CHECK: %[[VAL5:.*]] = hivm.hir.load
// CHECK: %[[VAL6:.*]] = hivm.hir.mmadL1
func.func @insert_load_store_between_vector_and_cube(%arg0 : memref<?xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %3 = hivm.hir.vmul ins(%2, %cst_1 : tensor<16x16xf32>, f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = tensor.empty() : tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%3, %3, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  hivm.hir.store ins(%5 : tensor<16x16xf32>) outs(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1], offset: 0>>)
  %6 = tensor.empty() : tensor<16x16xf32>
  %7 = hivm.hir.vmul ins(%3, %cst_1 : tensor<16x16xf32>, f32) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [1024], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 1024>>
  hivm.hir.store ins(%7 : tensor<16x16xf32>) outs(%reinterpret_cast_1 : memref<16x16xf32, strided<[16, 1], offset: 1024>>)
  return
}



// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_madl1
// CHECK: %[[F_VAL:.*]] = hivm.hir.fixpipe
// CHECK: %[[VAL1:.*]] = tensor.empty
// CHECK: %[[VAL2:.*]] = hivm.hir.load
// CHECK: %[[VAL3:.*]] = tensor.empty
// CHECK: %[[VAL4:.*]] = hivm.hir.load
func.func @insert_load_between_fixpipe_and_madl1(%arg0: memref<?xf32>, %arg1: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %0 = tensor.empty() : tensor<16x16xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  %1 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<16x16xf32, strided<[16, 1]>>
  %2 = tensor.empty() : tensor<16x16xf32>
  %3 = hivm.hir.load ins(%1 : tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = hivm.hir.mmadL1 ins(%3, %3, %false, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
  %5 = bufferization.to_tensor %reinterpret_cast_0 restrict writable : memref<16x16xf16, strided<[16, 1]>>
  %6 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%4 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %7 = tensor.empty() : tensor<16x16xf16>
  %8 = hivm.hir.load ins(%6 : tensor<16x16xf16>) outs(%7 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %9 = tensor.empty() : tensor<16x16xf16>
  %10 = hivm.hir.load ins(%6 : tensor<16x16xf16>) outs(%9 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %11 = hivm.hir.mmadL1 ins(%10, %8, %false, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%11 : tensor<16x16xf32>) outs(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>>)
  return
}

// -----
// CHECK-LABEL: @fixpipe_with_loop
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x64xf32>, %[[ARG1:.*]]: tensor<128x64xf32>) -> tensor<128x64xf32>
module {
  func.func @fixpipe_with_loop(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 3.200000e+01 : f32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg0 : tensor<128x64xf32>) outs(%arg1 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c8_i32 iter_args(%arg3 = %0) -> (tensor<128x64xf32>)  : i32 {
      %2 = tensor.empty() : tensor<128x64xf32>
      %3 = hivm.hir.load ins(%arg3 : tensor<128x64xf32>) outs(%2 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %4 = tensor.empty() : tensor<128x64xf32>
      %5 = hivm.hir.load ins(%arg3 : tensor<128x64xf32>) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %6 = tensor.empty() : tensor<128x64xf32>
      %7 = hivm.hir.vadd ins(%5, %cst : tensor<128x64xf32>, f32) outs(%6 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %8 = hivm.hir.store ins(%7 : tensor<128x64xf32>) outs(%3 : tensor<128x64xf32>) -> tensor<128x64xf32>
      scf.yield %8 : tensor<128x64xf32>
    }
    return %1 : tensor<128x64xf32>
  }
}

// -----
func.func @fixpipe_with_multiple_user(%arg0 : memref<?xf32>, %arg1 : memref<?xf16>, %arg2 : memref<?xf16>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %3 = tensor.empty() : tensor<16x16xf32>
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%4, %4, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  %7 = bufferization.to_tensor %6 restrict writable :memref<16x16xf16, strided<[16, 1], offset: 0>>
  %8 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
        ins(%5 : tensor<16x16xf32>) outs(%7 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[F_VAL:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
  // CHECK: %[[USE1:.*]] = hivm.hir.load ins(%[[F_VAL]] : tensor<16x16xf16>)
  // CHECK: %[[USE2:.*]] = hivm.hir.load ins(%[[F_VAL]] : tensor<16x16xf16>)
  // CHECK: hivm.hir.store ins(%[[USE1]] : tensor<16x16xf16>)
  // CHECK: hivm.hir.mmadL1 ins(%[[USE2]]
  %9 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  hivm.hir.store ins(%8 : tensor<16x16xf16>) outs(%9 : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  %10 = tensor.empty() : tensor<16x16xf16>
  %11 = hivm.hir.mmadL1 ins(%8, %10, %init_condition, %c16, %c16, %c16 :
                                 tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
                           outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
       ins(%11 : tensor<16x16xf32>) outs(%6 : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  return
}

// -----
// CHECK-LABEL: @insert_load_between_discrete_load_and_madl1
// CHECK: %[[VAL1:.*]] = hivm.hir.store
// CHECK: %[[VAL2:.*]] = hivm.hir.load
// CHECK: hivm.hir.mmadL1
func.func @insert_load_between_discrete_load_and_madl1(%arg0: memref<?xf32>, %arg1: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xi64>
  %3 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %1) -> (tensor<16x16xf32>) {
    %10 = scf.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %arg3) -> (tensor<16x16xf32>) {
      %extracted = tensor.extract %2[%arg2, %arg4] : tensor<16x16xi64>
      %11 = arith.index_cast %extracted : i64 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %12 = memref.load %reinterpret_cast_0[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %13 = tensor.empty() : tensor<1x1xf32>
      %14 = hivm.hir.vbrc ins(%12 : f32) outs(%13 : tensor<1x1xf32>) -> tensor<1x1xf32>
      %inserted_slice = tensor.insert_slice %14 into %arg5[%arg2, %arg4] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
    }
    scf.yield %10 : tensor<16x16xf32>
  } {ExtractedLoadOrStore}
  %4 = tensor.empty() : tensor<16x16xf32>
  %5 = hivm.hir.store ins(%3 : tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = tensor.empty() : tensor<16x16xf32>
  %7 = hivm.hir.load ins(%5 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %8 = tensor.empty() : tensor<16x16xf32>
  %9 = hivm.hir.mmadL1 ins(%7, %8, %false, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%9 : tensor<16x16xf32>) outs(%reinterpret_cast : memref<16x16xf16, strided<[16, 1]>>)
  return
}

// -----
// CHECK-LABEL: @insert_store_load_between_implicit_transposeb_and_mmad
// CHECK: %[[TENSORB:.*]] = bufferization.to_tensor
// CHECK: annotation.mark
// CHECK: %[[RES:.*]] = hivm.hir.mmadL1
func.func @insert_store_load_between_implicit_transposeb_and_mmad(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  annotation.mark %1 {MayImplicitTransposeWithLastAxis} : tensor<16x16xf16>
  %2 = tensor.empty() : tensor<16x16xf32>
  %3 = hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_mmad
// CHECK: %[[EMPTY1:.*]] = tensor.empty
// CHECK: %[[LOAD:.*]] = hivm.hir.load
func.func @insert_load_between_fixpipe_and_mmad(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x16xf16>
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %5 = tensor.empty() : tensor<16x16xf32>
  %6 = hivm.hir.mmadL1 ins(%0, %4, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %6 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_vector
// CHECK: %[[EMPTY1:.*]] = tensor.empty
// CHECK: %[[LOAD:.*]] = hivm.hir.load
func.func @insert_load_between_fixpipe_and_vector(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x16xf16>
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %5 = tensor.empty() : tensor<16x16xf16>
  %6 = hivm.hir.vexp ins(%4 : tensor<16x16xf16>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %6 : tensor<16x16xf16>
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_tensor_extract
// CHECK: hivm.hir.fixpipe
// CHECK: tensor.extract
func.func @insert_load_between_fixpipe_and_tensor_extract(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> f16 attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %extracted = tensor.extract %2[%c0, %c0] : tensor<16x16xf16>
  return %extracted : f16
}

// -----
// CHECK-LABEL: @insert_load_between_store_and_vector
// CHECK: %[[EMPTY1:.*]] = tensor.empty
// CHECK: %[[LOAD:.*]] = hivm.hir.load
func.func @insert_load_between_store_and_vector(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.store ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x16xf16>
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %5 = tensor.empty() : tensor<16x16xf16>
  %6 = hivm.hir.vexp ins(%4 : tensor<16x16xf16>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %6 : tensor<16x16xf16>
}

// -----
// CHECK-LABEL: @insert_load_between_vector_and_load
// CHECK: %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
// CHECK: %1 = tensor.empty() : tensor<16x16xf16>
// CHECK: %2 = hivm.hir.vexp ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: %3 = tensor.empty() : tensor<16x16xf16>
// CHECK: %4 = hivm.hir.store ins(%2 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: %5 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
// CHECK: %6 = hivm.hir.load ins(%5 : tensor<16x16xf16>) outs(%4 : tensor<16x16xf16>) -> tensor<16x16xf16>
func.func @insert_load_between_vector_and_load(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = tensor.empty() : tensor<16x16xf16>
  %2 = hivm.hir.vexp ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK-NOT: hivm.hir.store
  %3 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %4 = hivm.hir.load ins(%3 : tensor<16x16xf16>) outs(%2 : tensor<16x16xf16>) init_out_buffer = false -> tensor<16x16xf16>
  return %4 : tensor<16x16xf16>
}

// -----
// C HECK-LABEL: func.func @test_no_store_on_load_outs
// func.func @test_no_store_on_load_outs(%src_memref: memref<64x8xf32, strided<[8, 1]>>, %dst_alloc: memref<64x32xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
//   %cst = arith.constant 0.000000e+00 : f32
//   %c0 = arith.constant 0 : index
//   %dst_tensor = bufferization.to_tensor %dst_alloc restrict writable : memref<64x32xf32>
//   %vec_res = hivm.hir.vbrc ins(%cst : f32) outs(%dst_tensor : tensor<64x32xf32>) -> tensor<64x32xf32>
//   %vec_memref = bufferization.to_memref %vec_res : memref<64x32xf32>
//   %dst_subview = memref.subview %vec_memref[0, 0] [64, 8] [1, 1] : memref<64x32xf32> to memref<64x8xf32, strided<[32, 1]>>
//   // C HECK-NOT: hivm.hir.store
//   // C HECK: hivm.hir.load ins(%arg0 : memref<64x8xf32, strided<[8, 1]>>) outs(%[[DST_SUBVIEW:.*]] : memref<64x8xf32, strided<[32, 1]>>)
//   hivm.hir.load ins(%src_memref : memref<64x8xf32, strided<[8, 1]>>) outs(%dst_subview : memref<64x8xf32, strided<[32, 1]>>) left_padding_num = %c0 : index
//   return
// }

// -----
// CHECK-LABEL: @collapse
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK: annotation.mark
// CHECK: %[[RES:.*]] = hivm.hir.mmadL1
func.func @collapse(%arg0: memref<2x8x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x8x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = tensor.empty() : tensor<16x16xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1], [2]] : tensor<2x8x16xf16> into tensor<16x16xf16>
  annotation.mark %collapsed {maybeUnCollapsibleReshape} : tensor<16x16xf16>
  %3 = hivm.hir.mmadL1 ins(%collapsed, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @insert_store_load_for_attr_parallel_loop
// CHECK: %[[TENSORB:.*]] = bufferization.to_tensor
// CHECK: %[[RES:.*]] = hivm.hir.mmadL1
func.func @insert_store_load_for_attr_parallel_loop(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable {gather_load} : memref<16x16xf16>
  %2 = tensor.empty() : tensor<16x16xf32>
  %3 = hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @gather_load_base_should_stay_gm(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf16>, %{{.*}}: tensor<16x16xf16>, %{{.*}}: tensor<16x16xi64>) -> tensor<16x16xf32>
// CHECK-NOT: hivm.hir.load ins(%[[BASE]] : memref<?xf16>)
// CHECK: %[[GATHER:.*]] = hivm.hir.gather_load ins(%[[BASE]] : memref<?xf16>, %{{.*}} : tensor<16x16xi64>, %{{.*}} : i32, %{{.*}} : tensor<16x16xi1>, %{{.*}} : tensor<16x16xf16>) outs(%{{.*}} : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<1x1x16x16xf16>) outs(%[[CAST]] : memref<1x1x16x16xf16>) {"inserted-copy"}
// CHECK: hivm.hir.mmadL1 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x1x16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @gather_load_base_should_stay_gm(%base: memref<?xf16>, %rhs: tensor<16x16xf16>, %indices: tensor<16x16xi64>) -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f16
    %init = tensor.empty() : tensor<16x16xf16>
    %other = hivm.hir.vbrc ins(%zero : f16) outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
    %maskInit = tensor.empty() : tensor<16x16xi1>
    %mask = hivm.hir.vcmp ins(%rhs, %zero : tensor<16x16xf16>, f16) outs(%maskInit : tensor<16x16xi1>) compare_mode = <ne> -> tensor<16x16xi1>
    %g = hivm.hir.gather_load ins(%base : memref<?xf16>, %indices : tensor<16x16xi64>, %c1_i32 : i32, %mask : tensor<16x16xi1>, %other : tensor<16x16xf16>) outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
    %out = tensor.empty() : tensor<16x16xf32>
    %res = hivm.hir.mmadL1 ins(%g, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %res : tensor<16x16xf32>
  }
}

// -----
// CHECK-LABEL: @scatter_store_base_should_stay_gm(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf32>, %{{.*}}: tensor<16x16xi64>, %{{.*}}: tensor<16x16xf32>)
// CHECK-NOT: hivm.hir.load ins(%[[BASE]] : memref<?xf32>)
// CHECK: hivm.hir.scatter_store ins(%{{.*}} : tensor<16x16xi64>, %{{.*}} : tensor<16x16xf32>, %{{.*}} : i32) outs(%[[BASE]] : memref<?xf32>)
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @scatter_store_base_should_stay_gm(%base: memref<?xf32>, %indices: tensor<16x16xi64>, %data: tensor<16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.scatter_store ins(%indices : tensor<16x16xi64>, %data : tensor<16x16xf32>, %c1_i32 : i32) outs(%base : memref<?xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>}
    return
  }
}

// -----
// CHECK-LABEL: @insert_load_store_between_cross_loop_vector_and_cube(
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x64xf32>, %[[ARG1:.*]]: tensor<64x64xf32>) -> tensor<128x64xf32>
module {
  func.func @insert_load_store_between_cross_loop_vector_and_cube(%arg0: tensor<128x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %cst = arith.constant 3.200000e+01 : f32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c8_i32 iter_args(%arg3 = %arg0) -> (tensor<128x64xf32>)  : i32 {
      %1 = tensor.empty() : tensor<128x64xf32>
      %2 = hivm.hir.mmadL1 ins(%arg3, %arg1, %true, %c0, %c0, %c0 : tensor<128x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%1 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %3 = tensor.empty() : tensor<128x64xf32>
      %4 = hivm.hir.vadd ins(%2, %cst : tensor<128x64xf32>, f32) outs(%3 : tensor<128x64xf32>) -> tensor<128x64xf32>
      scf.yield %4 : tensor<128x64xf32>
    }
    return %0 : tensor<128x64xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_insert_load_before_for
// CHECK: %[[FIXPIPE_RES:.*]] = hivm.hir.fixpipe
// CHECK: scf.for
module {
  func.func @test_insert_load_before_for(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = hivm.hir.fixpipe {enable_nz2nd} ins(%arg0 : tensor<32x32xf32>) outs(%arg1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %0) -> (tensor<32x32xf32>) {
      %extracted_slice = tensor.extract_slice %arg3[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
      %2 = tensor.empty() : tensor<1x32xf32>
      %3 = hivm.hir.load ins(%extracted_slice : tensor<1x32xf32>) outs(%2 : tensor<1x32xf32>) -> tensor<1x32xf32>
      %4 = tensor.empty() : tensor<1x32xf32>
      %5 = hivm.hir.load ins(%extracted_slice : tensor<1x32xf32>) outs(%4 : tensor<1x32xf32>) -> tensor<1x32xf32>
      %6 = tensor.empty() : tensor<1x32xf32>
      %7 = hivm.hir.load ins(%extracted_slice : tensor<1x32xf32>) outs(%6 : tensor<1x32xf32>) -> tensor<1x32xf32>
      %8 = hivm.hir.vadd ins(%7, %5 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%3 : tensor<1x32xf32>) -> tensor<1x32xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg3[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
      scf.yield %inserted_slice : tensor<32x32xf32>
    }
    return %1 : tensor<32x32xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_insert_load_before_while
// CHECK: %[[FIXPIPE_RES:.*]] = hivm.hir.fixpipe
// CHECK: scf.while
module {
  func.func @test_insert_load_before_while(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = hivm.hir.fixpipe {enable_nz2nd} ins(%arg0 : tensor<32x32xf32>) outs(%arg1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %1:2 = scf.while (%arg2 = %0, %arg3 = %c0_i32) : (tensor<32x32xf32>, i32) -> (tensor<32x32xf32>, i32) {
      %2 = arith.cmpi slt, %arg3, %c8_i32 : i32
      scf.condition(%2) %arg2, %arg3 : tensor<32x32xf32>, i32
    } do {
    ^bb0(%arg2: tensor<32x32xf32>, %arg3: i32):
      %extracted_slice = tensor.extract_slice %arg2[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
      %2 = hivm.hir.vadd ins(%extracted_slice, %extracted_slice : tensor<1x32xf32>, tensor<1x32xf32>) outs(%extracted_slice : tensor<1x32xf32>) -> tensor<1x32xf32>
      %inserted_slice = tensor.insert_slice %2 into %arg2[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
      %3 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %inserted_slice, %3 : tensor<32x32xf32>, i32
    }
    return %1#0 : tensor<32x32xf32>
  }
}

// -----
// CHECK-LABEL: @insert_load_store_ignore_insert_slice
module {
  func.func @insert_load_store_ignore_insert_slice(%arg2: memref<?xf32>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf32> , %arg6: i32, %arg7: i32, %arg8: i32) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
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
    %3 = tensor.empty() : tensor<3x16x16xf32>
    %4 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %3) -> (tensor<3x16x16xf32>) {
      %extracted_slice = tensor.extract_slice %0[%arg9, 0, 0] [1, 16, 32] [1, 1, 1] : tensor<3x16x32xf16> to tensor<16x32xf16>
      %extracted_slice_5 = tensor.extract_slice %1[%arg9, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<3x32x16xf16> to tensor<32x16xf16>
      %9 = tensor.empty() : tensor<16x16xf32>
      // CHECK: %[[mmadL1:.*]] = hivm.hir.mmadL1
      %10 = hivm.hir.mmadL1 ins(%extracted_slice, %extracted_slice_5, %true, %c16, %c32, %c16 : tensor<16x32xf16>, tensor<32x16xf16>, i1, index, index, index) outs(%9 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %extracted_slice_6 = tensor.extract_slice %arg10[%arg9, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<3x16x16xf32> to tensor<16x16xf32>
      %11 = hivm.hir.fixpipe {enable_nz2nd} ins(%10 : tensor<16x16xf32>) outs(%extracted_slice_6 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %inserted_slice = tensor.insert_slice %11 into %arg10[%arg9, 0, 0] [1, 16, 16] [1, 1, 1] {elide_after_bufferize} : tensor<16x16xf32> into tensor<3x16x16xf32>
      scf.yield %inserted_slice : tensor<3x16x16xf32>
    }
    // CHECK-NOT: %[[store:.*]] = hivm.hir.store
    %5 = tensor.empty() : tensor<3x16x16xf32>
    %6 = hivm.hir.load ins(%4 : tensor<3x16x16xf32>) outs(%5 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
    %7 = tensor.empty() : tensor<3x16x16xf32>
    %8 = hivm.hir.vadd ins(%6, %2 : tensor<3x16x16xf32>, tensor<3x16x16xf32>) outs(%7 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1] : memref<?xf32> to memref<3x16x16xf32, strided<[256, 16, 1]>>
    hivm.hir.store ins(%8 : tensor<3x16x16xf32>) outs(%reinterpret_cast_4 : memref<3x16x16xf32, strided<[256, 16, 1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @test_insert_store_scf_if
module {
  func.func @test_insert_store_scf_if(%arg0: tensor<32x32xf32>, %arg1: i1) -> tensor<32x32xf32> attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
    %r = scf.if %arg1 -> (tensor<32x32xf32>) {
      %0 = tensor.empty() : tensor<32x32xf32>
      %1 = hivm.hir.fixpipe {enable_nz2nd} ins(%0 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      // CHECK: hivm.hir.load
      scf.yield %1 : tensor<32x32xf32>
    } else {
      %2 = tensor.empty() : tensor<32x32xf32>
      %3 = tensor.empty() : tensor<32x32xf32>
      %4 = hivm.hir.vadd ins(%2, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %4 : tensor<32x32xf32>
    }
    %5 = tensor.empty() : tensor<32x32xf32>
    %6 = hivm.hir.vadd ins(%r, %r : tensor<32x32xf32>, tensor<32x32xf32>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %6 : tensor<32x32xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_insert_load_scf_for_yield
module {
  func.func @test_insert_load_scf_for_yield(%arg0: tensor<32x32xf32>, %arg1: i1) -> tensor<32x32xf32> attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = hivm.hir.vadd ins(%0, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: hivm.hir.store
    %r = scf.for %i = %c0 to %c32 step %c1 iter_args(%iter_arg = %2) -> (tensor<32x32xf32>) {
      %3 = tensor.empty() : tensor<32x32xf32>
      // CHECK: hivm.hir.load
      %4 = hivm.hir.mmadL1 ins(%iter_arg, %iter_arg, %true, %c32, %c32, %c32 : tensor<32x32xf32>, tensor<32x32xf32>, i1, index, index, index)
                           outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %5 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %6 = tensor.empty() : tensor<32x32xf32>
      %7 = tensor.empty() : tensor<32x32xf32>
      %8 = hivm.hir.vadd ins(%6, %6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %9 = hivm.hir.store ins(%8 : tensor<32x32xf32>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
      // CHECK: hivm.hir.store
      scf.yield %9 : tensor<32x32xf32>
    }
    return %r : tensor<32x32xf32>
  }
}

// -----
// CHECK-LABEL: @test_index_select_simd
// CHECK: %c4 = arith.constant 4 : index
// CHECK: %true = arith.constant true
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c3 = arith.constant 3 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: hivm.hir.set_mask_norm
// CHECK: %0 = arith.muli %arg8, %arg9 : i32
// CHECK: %1 = arith.muli %0, %arg10 : i32
// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [3], strides: [1]
// CHECK: %alloc = memref.alloc() : memref<3xi32>
// CHECK: hivm.hir.load ins(%reinterpret_cast : memref<3xi32, strided<[1]>>) outs(%alloc : memref<3xi32>)
// CHECK: %2 = bufferization.to_tensor %alloc restrict writable : memref<3xi32>
// CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [6, 4], strides: [4, 1]
// CHECK: %alloc_1 = memref.alloc() : memref<3x4xf32>
// CHECK: scf.for %arg11 = %c0 to %c3 step %c1 {
// CHECK: %extracted = tensor.extract %2[%arg11] : tensor<3xi32>
// CHECK: %7 = arith.index_cast %extracted : i32 to index
// CHECK: hivm.hir.load ins(%subview : memref<1x4xf32, strided<[4, 1], offset: ?>>) outs(%subview_5 : memref<1x4xf32, strided<[4, 1], offset: ?>>) left_padding_num = %c0 : index
// CHECK: } {hivm.parallel_loop}
// CHECK: %3 = bufferization.to_tensor %alloc_1 restrict writable {index_select_simd} : memref<3x4xf32>
// CHECK: %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 3], strides: [3, 1]
// CHECK: %alloc_3 = memref.alloc() : memref<4x3xf32>
// CHECK: hivm.hir.load ins(%reinterpret_cast_2 : memref<4x3xf32, strided<[3, 1]>>) outs(%alloc_3 : memref<4x3xf32>)
// CHECK: %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<4x3xf32>
// CHECK: %5 = tensor.empty() : tensor<3x3xf32>
// CHECK: %6 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%3, %4, %true, %c1, %c4, %c3 : tensor<3x4xf32>, tensor<4x3xf32>, i1, index, index, index) outs(%5 : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [3, 3], strides: [3, 1]
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6 : tensor<3x3xf32>) outs(%reinterpret_cast_4 : memref<3x3xf32, strided<[3, 1]>>)
// CHECK: return
func.func @test_index_select_simd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %c4 = arith.constant 4 : index
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [3], strides: [1] : memref<?xi32> to memref<3xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<3xi32>
  hivm.hir.load ins(%reinterpret_cast : memref<3xi32, strided<[1]>>) outs(%alloc : memref<3xi32>)
  %2 = bufferization.to_tensor %alloc restrict writable : memref<3xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [6, 4], strides: [4, 1] : memref<?xf32> to memref<6x4xf32, strided<[4, 1]>>
  %alloc_1 = memref.alloc() : memref<3x4xf32>
  scf.for %arg11 = %c0 to %c3 step %c1 {
    %extracted = tensor.extract %2[%arg11] : tensor<3xi32>
    %7 = arith.index_cast %extracted : i32 to index
    %subview = memref.subview %reinterpret_cast_0[%7, 0] [1, 4] [1, 1] : memref<6x4xf32, strided<[4, 1]>> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %subview_5 = memref.subview %alloc_1[%arg11, 0] [1, 4] [1, 1] : memref<3x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    annotation.mark %subview_5 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x4xf32, strided<[4, 1], offset: ?>>
    hivm.hir.load ins(%subview : memref<1x4xf32, strided<[4, 1], offset: ?>>) outs(%subview_5 : memref<1x4xf32, strided<[4, 1], offset: ?>>) left_padding_num = %c0 : index
  } {hivm.parallel_loop}
  %3 = bufferization.to_tensor %alloc_1 restrict writable {index_select_simd} : memref<3x4xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 3], strides: [3, 1] : memref<?xf32> to memref<4x3xf32, strided<[3, 1]>>
  %alloc_3 = memref.alloc() : memref<4x3xf32>
  hivm.hir.load ins(%reinterpret_cast_2 : memref<4x3xf32, strided<[3, 1]>>) outs(%alloc_3 : memref<4x3xf32>)
  %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<4x3xf32>
  %5 = tensor.empty() : tensor<3x3xf32>
  %6 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%3, %4, %true, %c1, %c4, %c3 : tensor<3x4xf32>, tensor<4x3xf32>, i1, index, index, index) outs(%5 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [3, 3], strides: [3, 1] : memref<?xf32> to memref<3x3xf32, strided<[3, 1]>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6 : tensor<3x3xf32>) outs(%reinterpret_cast_4 : memref<3x3xf32, strided<[3, 1]>>)
  return
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_hivm_bitcast
func.func @insert_load_between_fixpipe_and_hivm_bitcast(
    %src: tensor<16x16xf32>,
    %dst: tensor<16x16xf16>) -> tensor<16x16xi16> {
  %0 = hivm.hir.fixpipe {enable_nz2nd}
        ins(%src : tensor<16x16xf32>)
        outs(%dst : tensor<16x16xf16>)
        -> tensor<16x16xf16>

  %1 = hivm.hir.bitcast %0 : tensor<16x16xf16> -> tensor<16x16xi16>

  return %1 : tensor<16x16xi16>
}

// -----
// CHECK-LABEL: @insert_load_store_between_cross_loop_vector_and_cube_batchMmad
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %true = arith.constant true
// CHECK: %cst = arith.constant 3.200000e+01 : f32
// CHECK: %c8_i32 = arith.constant 8 : i32
// CHECK: %c32_i32 = arith.constant 32 : i32
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %0 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c8_i32 iter_args(%arg3 = %arg0) -> (tensor<2x128x64xf32>)  : i32 {
// CHECK: %1 = tensor.empty() : tensor<2x128x64xf32>
// CHECK: %2 = hivm.hir.batchMmadL1 ins(%arg3, %arg1, %true, %c0, %c0, %c0 : tensor<2x128x64xf32>, tensor<2x64x64xf32>, i1, index, index, index) outs(%1 : tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
// CHECK: %3 = tensor.empty() : tensor<2x128x64xf32>
// CHECK: %4 = hivm.hir.vadd ins(%2, %cst : tensor<2x128x64xf32>, f32) outs(%3 : tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
// CHECK: scf.yield %4 : tensor<2x128x64xf32>
// CHECK: }
// CHECK: return %0 : tensor<2x128x64xf32>
func.func @insert_load_store_between_cross_loop_vector_and_cube_batchMmad(%arg0: tensor<2x128x64xf32>, %arg1: tensor<2x64x64xf32>) -> tensor<2x128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %cst = arith.constant 3.200000e+01 : f32
  %c8_i32 = arith.constant 8 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c8_i32 iter_args(%arg3 = %arg0) -> (tensor<2x128x64xf32>)  : i32 {
    %1 = tensor.empty() : tensor<2x128x64xf32>
    %2 = hivm.hir.batchMmadL1 ins(%arg3, %arg1, %true, %c0, %c0, %c0 : tensor<2x128x64xf32>, tensor<2x64x64xf32>, i1, index, index, index) outs(%1 : tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
    %3 = tensor.empty() : tensor<2x128x64xf32>
    %4 = hivm.hir.vadd ins(%2, %cst : tensor<2x128x64xf32>, f32) outs(%3 : tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
    scf.yield %4 : tensor<2x128x64xf32>
  }
  return %0 : tensor<2x128x64xf32>
}

// -----
// CHECK-LABEL: @mix_cv_batch
// CHECK: %c32 = arith.constant 32 : index
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %true = arith.constant true
// CHECK: hivm.hir.set_mask_norm
// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [3, 16, 32], strides: [512, 32, 1]
// CHECK: %alloc = memref.alloc() : memref<3x16x32xf16>
// CHECK: hivm.hir.load ins(%reinterpret_cast : memref<3x16x32xf16, strided<[512, 32, 1]>>) outs(%alloc : memref<3x16x32xf16>)
// CHECK: %0 = bufferization.to_tensor %alloc restrict writable : memref<3x16x32xf16>
// CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [3, 32, 16], strides: [512, 16, 1]
// CHECK: %alloc_1 = memref.alloc() : memref<3x32x16xf16>
// CHECK: hivm.hir.load ins(%reinterpret_cast_0 : memref<3x32x16xf16, strided<[512, 16, 1]>>) outs(%alloc_1 : memref<3x32x16xf16>)
// CHECK: %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<3x32x16xf16>
// CHECK: %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1]
// CHECK: %alloc_3 = memref.alloc() : memref<3x16x16xf32>
// CHECK: hivm.hir.load ins(%reinterpret_cast_2 : memref<3x16x16xf32, strided<[256, 16, 1]>>) outs(%alloc_3 : memref<3x16x16xf32>)
// CHECK: %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<3x16x16xf32>
// CHECK: %3 = tensor.empty() : tensor<3x16x16xf32>
// CHECK: %4 = hivm.hir.batchMmadL1 ins(%0, %1, %true, %c16, %c32, %c16 : tensor<3x16x32xf16>, tensor<3x32x16xf16>, i1, index, index, index) outs(%3 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
// CHECK: %5 = tensor.empty() : tensor<3x16x16xf32>
// CHECK: %6 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<3x16x16xf32>) outs(%5 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
// CHECK: %7 = tensor.empty() : tensor<3x16x16xf32>
// CHECK: %8 = hivm.hir.load ins(%6 : tensor<3x16x16xf32>) outs(%7 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
// CHECK: %9 = tensor.empty() : tensor<3x16x16xf32>
// CHECK: %10 = hivm.hir.vadd ins(%8, %2 : tensor<3x16x16xf32>, tensor<3x16x16xf32>) outs(%9 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
// CHECK: %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1]
// CHECK: hivm.hir.store ins(%10 : tensor<3x16x16xf32>) outs(%reinterpret_cast_4 : memref<3x16x16xf32, strided<[256, 16, 1]>>)
// CHECK: return
func.func @mix_cv_batch(%arg0: memref<?xf32>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf32>, %arg4: i32, %arg5: i32, %arg6: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  hivm.hir.set_mask_norm
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [3, 16, 32], strides: [512, 32, 1] : memref<?xf16> to memref<3x16x32xf16, strided<[512, 32, 1]>>
  %alloc = memref.alloc() : memref<3x16x32xf16>
  hivm.hir.load ins(%reinterpret_cast : memref<3x16x32xf16, strided<[512, 32, 1]>>) outs(%alloc : memref<3x16x32xf16>)
  %0 = bufferization.to_tensor %alloc restrict writable : memref<3x16x32xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [3, 32, 16], strides: [512, 16, 1] : memref<?xf16> to memref<3x32x16xf16, strided<[512, 16, 1]>>
  %alloc_1 = memref.alloc() : memref<3x32x16xf16>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<3x32x16xf16, strided<[512, 16, 1]>>) outs(%alloc_1 : memref<3x32x16xf16>)
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<3x32x16xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1] : memref<?xf32> to memref<3x16x16xf32, strided<[256, 16, 1]>>
  %alloc_3 = memref.alloc() : memref<3x16x16xf32>
  hivm.hir.load ins(%reinterpret_cast_2 : memref<3x16x16xf32, strided<[256, 16, 1]>>) outs(%alloc_3 : memref<3x16x16xf32>)
  %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<3x16x16xf32>
  %3 = tensor.empty() : tensor<3x16x16xf32>
  %4 = hivm.hir.batchMmadL1 ins(%0, %1, %true, %c16, %c32, %c16 : tensor<3x16x32xf16>, tensor<3x32x16xf16>, i1, index, index, index) outs(%3 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
  %5 = tensor.empty() : tensor<3x16x16xf32>
  %6 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<3x16x16xf32>) outs(%5 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
  %7 = tensor.empty() : tensor<3x16x16xf32>
  %8 = hivm.hir.load ins(%6 : tensor<3x16x16xf32>) outs(%7 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
  %9 = tensor.empty() : tensor<3x16x16xf32>
  %10 = hivm.hir.vadd ins(%8, %2 : tensor<3x16x16xf32>, tensor<3x16x16xf32>) outs(%9 : tensor<3x16x16xf32>) -> tensor<3x16x16xf32>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [3, 16, 16], strides: [256, 16, 1] : memref<?xf32> to memref<3x16x16xf32, strided<[256, 16, 1]>>
  hivm.hir.store ins(%10 : tensor<3x16x16xf32>) outs(%reinterpret_cast_4 : memref<3x16x16xf32, strided<[256, 16, 1]>>)
  return
}

// -----
// CHECK-LABEL: @test_tensor_insert
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c2 = arith.constant 2 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %true = arith.constant true
// CHECK: %c3_i32 = arith.constant 3 : i32
// CHECK: %0 = tensor.empty() : tensor<2x2xi32>
// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 2], strides: [2, 1]
// CHECK: %alloc = memref.alloc() : memref<2x2xi32>
// CHECK: hivm.hir.load ins(%reinterpret_cast : memref<2x2xi32, strided<[2, 1]>>) outs(%alloc : memref<2x2xi32>)
// CHECK: %1 = bufferization.to_tensor %alloc restrict writable : memref<2x2xi32>
// CHECK: %inserted = tensor.insert %c3_i32 into %0[%c0, %c0] : tensor<2x2xi32>
// CHECK: %inserted_0 = tensor.insert %c3_i32 into %inserted[%c0, %c1] : tensor<2x2xi32>
// CHECK: %inserted_1 = tensor.insert %c3_i32 into %inserted_0[%c1, %c0] : tensor<2x2xi32>
// CHECK: %inserted_2 = tensor.insert %c3_i32 into %inserted_1[%c1, %c1] : tensor<2x2xi32>
// CHECK: %2 = tensor.empty() : tensor<2x2xi32>
// CHECK: %3 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%1, %inserted_2, %true, %c2, %c2, %c2 : tensor<2x2xi32>, tensor<2x2xi32>, i1, index, index, index) outs(%2 : tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK: %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [2, 2], strides: [2, 1]
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<2x2xi32>) outs(%reinterpret_cast_3
// CHECK: return
func.func @test_tensor_insert(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %c3_i32 = arith.constant 3 : i32
  %0 = tensor.empty() : tensor<2x2xi32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xi32> to memref<2x2xi32, strided<[2, 1]>>
  %alloc = memref.alloc() : memref<2x2xi32>
  hivm.hir.load ins(%reinterpret_cast : memref<2x2xi32, strided<[2, 1]>>) outs(%alloc : memref<2x2xi32>)
  %1 = bufferization.to_tensor %alloc restrict writable : memref<2x2xi32>
  %inserted = tensor.insert %c3_i32 into %0[%c0, %c0] : tensor<2x2xi32>
  %inserted_0 = tensor.insert %c3_i32 into %inserted[%c0, %c1] : tensor<2x2xi32>
  %inserted_1 = tensor.insert %c3_i32 into %inserted_0[%c1, %c0] : tensor<2x2xi32>
  %inserted_2 = tensor.insert %c3_i32 into %inserted_1[%c1, %c1] : tensor<2x2xi32>
  %2 = tensor.empty() : tensor<2x2xi32>
  %3 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%1, %inserted_2, %true, %c2, %c2, %c2 : tensor<2x2xi32>, tensor<2x2xi32>, i1, index, index, index) outs(%2 : tensor<2x2xi32>) -> tensor<2x2xi32>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xi32> to memref<2x2xi32, strided<[2, 1]>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<2x2xi32>) outs(%reinterpret_cast_3 : memref<2x2xi32, strided<[2, 1]>>)
  return
}

// -----
// CHECK-LABEL: @extract_i1
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %false = arith.constant false
// CHECK: %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
// CHECK: %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
// CHECK: %2 = tensor.empty() : tensor<16x16xi1>
// CHECK: %3 = hivm.hir.vcmp ins(%0, %1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%2 : tensor<16x16xi1>) compare_mode = <lt> -> tensor<16x16xi1>
// CHECK: %extracted = tensor.extract %3[%c0, %c0] : tensor<16x16xi1>
// CHECK: %4 = arith.select %extracted, %arg3, %arg4 : index
// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [16, 16], strides: [16, 1]
// CHECK: %5 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<16x16xf32, strided<[16, 1], offset: ?>>
// CHECK: %6 = tensor.empty() : tensor<16x16xf32>
// CHECK: %7 = hivm.hir.load ins(%5 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %8 = hivm.hir.mmadL1 ins(%7, %7, %false, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %9 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%8 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: return %9 : tensor<16x16xf32>
func.func @extract_i1(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<?xf32>, %arg3: index, %arg4: index) -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = tensor.empty() : tensor<16x16xi1>
  %3 = hivm.hir.vcmp ins(%0, %1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%2 : tensor<16x16xi1>) compare_mode = <lt> -> tensor<16x16xi1>
  %extracted = tensor.extract %3[%c0, %c0] : tensor<16x16xi1>
  %4 = arith.select %extracted, %arg3, %arg4 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
  %5 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<16x16xf32, strided<[16, 1], offset: ?>>
  %6 = tensor.empty() : tensor<16x16xf32>
  %7 = hivm.hir.load ins(%5 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %8 = hivm.hir.mmadL1 ins(%7, %7, %false, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %9 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%8 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %9 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @test_insert_load_scope
// CHECK: %0 = scope.scope : () -> tensor<32x32xf32> {
// CHECK: %3 = tensor.empty() : tensor<32x32xf32>
// CHECK: %4 = hivm.hir.fixpipe {enable_nz2nd} ins(%3 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: scope.return %4 : tensor<32x32xf32>
// CHECK: }
// CHECK: %1 = tensor.empty() : tensor<32x32xf32>
// CHECK: %2 = hivm.hir.vadd ins(%0, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: return %2 : tensor<32x32xf32>
func.func @test_insert_load_scope(%arg0: tensor<32x32xf32>, %arg1: i1) -> tensor<32x32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = scope.scope : () -> tensor<32x32xf32> {
    %3 = tensor.empty() : tensor<32x32xf32>
    %4 = hivm.hir.fixpipe {enable_nz2nd} ins(%3 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scope.return %4 : tensor<32x32xf32>
  }
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = hivm.hir.vadd ins(%0, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// -----
// CHECK-LABEL: @annotated_to_tensor_is_vector_source
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %true = arith.constant true
// CHECK: %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
// CHECK: annotation.mark %0 {MayImplicitTransposeWithLastAxis} : tensor<16x16xf16>
// CHECK: %1 = tensor.empty() : tensor<16x16xf32>
// CHECK: %2 = hivm.hir.mmadL1 ins(%0, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: hivm.hir.store ins(%2 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
// CHECK: return
func.func @annotated_to_tensor_is_vector_source(%arg0: memref<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: memref<16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  annotation.mark %0 {MayImplicitTransposeWithLastAxis} : tensor<16x16xf16>
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = hivm.hir.mmadL1 ins(%0, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%2 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
  return
}

// -----
// CHECK-LABEL: @plain_to_tensor_is_not_forced_vector
func.func @plain_to_tensor_is_not_forced_vector(
    %src: memref<16x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf32>) {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %tensor = bufferization.to_tensor %src restrict writable : memref<16x16xf16>
  // CHECK: bufferization.to_tensor
  // CHECK-NOT: "inserted-store"
  // CHECK: hivm.hir.mmadL1
  %empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%tensor, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  return
}

// -----
// CHECK-LABEL: @annotated_collapse_requires_boundary
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %true = arith.constant true
// CHECK: %collapsed = tensor.collapse_shape %arg0 {{\[}}[0, 1], [2]] : tensor<2x8x16xf16> into tensor<16x16xf16>
// CHECK: annotation.mark %collapsed {maybeUnCollapsibleReshape} : tensor<16x16xf16>
// CHECK: %0 = tensor.empty() : tensor<16x16xf32>
// CHECK: %1 = hivm.hir.mmadL1 ins(%collapsed, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: hivm.hir.store ins(%1 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
// CHECK: return
func.func @annotated_collapse_requires_boundary(%arg0: tensor<2x8x16xf16>, %arg1: tensor<16x16xf16>, %arg2: memref<16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x8x16xf16> into tensor<16x16xf16>
  annotation.mark %collapsed {maybeUnCollapsibleReshape} : tensor<16x16xf16>
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = hivm.hir.mmadL1 ins(%collapsed, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%1 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
  return
}

// -----
// CHECK-LABEL: @plain_collapse_is_not_forced_vector
func.func @plain_collapse_is_not_forced_vector(
    %src: tensor<2x8x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %collapsed = tensor.collapse_shape %src [[0, 1], [2]] : tensor<2x8x16xf16> into tensor<16x16xf16>
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: "inserted-store"
  // CHECK: hivm.hir.mmadL1
  %empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%collapsed, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  return
}

// -----
// CHECK-LABEL: @extract_from_inserted_store_keeps_gm_boundary
func.func @extract_from_inserted_store_keeps_gm_boundary(
    %src: tensor<16x16xf32>, %out: memref<16x16xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %stored = hivm.hir.store ins(%src : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) {"inserted-store"} -> tensor<16x16xf32>
  // CHECK: hivm.hir.store
  // CHECK-SAME: "inserted-store"
  // CHECK: tensor.extract
  // CHECK: hivm.hir.vmul
  %scalar = tensor.extract %stored[%c0, %c0] : tensor<16x16xf32>
  %vec_empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.vmul ins(%src, %scalar : tensor<16x16xf32>, f32) outs(%vec_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  return
}

// -----
// CHECK-LABEL: @extracted_load_or_store_loop_is_vector_boundary
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c32 = arith.constant 32 : index
// CHECK: %0 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %arg0) -> (tensor<32x32xf32>) {
// CHECK: %extracted_slice = tensor.extract_slice %arg3[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
// CHECK: %1 = tensor.empty() : tensor<1x32xf32>
// CHECK: %2 = hivm.hir.vadd ins(%extracted_slice, %extracted_slice : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1 : tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: %inserted_slice = tensor.insert_slice %2 into %arg3[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
// CHECK: scf.yield %inserted_slice : tensor<32x32xf32>
// CHECK: } {ExtractedLoadOrStore}
// CHECK: hivm.hir.store ins(%0 : tensor<32x32xf32>) outs(%arg1 : memref<32x32xf32>)
// CHECK: return
func.func @extracted_load_or_store_loop_is_vector_boundary(%arg0: tensor<32x32xf32>, %arg1: memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %0 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %arg0) -> (tensor<32x32xf32>) {
    %extracted_slice = tensor.extract_slice %arg3[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
    %1 = tensor.empty() : tensor<1x32xf32>
    %2 = hivm.hir.vadd ins(%extracted_slice, %extracted_slice : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %inserted_slice = tensor.insert_slice %2 into %arg3[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
    scf.yield %inserted_slice : tensor<32x32xf32>
  } {ExtractedLoadOrStore}
  hivm.hir.store ins(%0 : tensor<32x32xf32>) outs(%arg1 : memref<32x32xf32>)
  return
}

// -----
// CHECK-LABEL: @propagate_through_if_result
// CHECK: %0 = scf.if %arg1 -> (tensor<32x32xf32>) {
// CHECK: %7 = tensor.empty() : tensor<32x32xf32>
// CHECK: %8 = hivm.hir.fixpipe {enable_nz2nd} ins(%7 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %1 = tensor.empty() : tensor<32x32xf32>
// CHECK: %2 = hivm.hir.load ins(%0 : tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %3 = tensor.empty() : tensor<32x32xf32>
// CHECK: %4 = hivm.hir.load ins(%0 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %5 = tensor.empty() : tensor<32x32xf32>
// CHECK: %6 = hivm.hir.vadd ins(%4, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
func.func @propagate_through_if_result(
    %arg0: tensor<32x32xf32>, %pred: i1,
    %out: memref<32x32xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %if_res = scf.if %pred -> (tensor<32x32xf32>) {
    %empty = tensor.empty() : tensor<32x32xf32>
    %fix = hivm.hir.fixpipe {enable_nz2nd} ins(%empty : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %fix : tensor<32x32xf32>
  } else {
    scf.yield %arg0 : tensor<32x32xf32>
  }
  %vec_empty = tensor.empty() : tensor<32x32xf32>
  %res = hivm.hir.vadd ins(%if_res, %if_res : tensor<32x32xf32>, tensor<32x32xf32>) outs(%vec_empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  hivm.hir.store ins(%res : tensor<32x32xf32>) outs(%out : memref<32x32xf32>)
  return
}


// -----
// CHECK-LABEL: @propagate_through_while_result
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c2 = arith.constant 2 : index
// CHECK: %0 = tensor.empty() : tensor<32x32xf32>
// CHECK: %1 = hivm.hir.fixpipe {enable_nz2nd} ins(%0 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %5 = arith.cmpi slt, %arg3, %c2 : index
// CHECK: %extracted_slice = tensor.extract_slice %arg2[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
// CHECK: %5 = tensor.empty() : tensor<1x32xf32>
// CHECK: %6 = hivm.hir.vadd ins(%extracted_slice, %extracted_slice : tensor<1x32xf32>, tensor<1x32xf32>) outs(%5 : tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: %inserted_slice = tensor.insert_slice %6 into %arg2[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
// CHECK: %7 = arith.addi %arg3, %c1 : index
// CHECK: %3 = tensor.empty() : tensor<32x32xf32>
// CHECK: %4 = hivm.hir.vadd ins(%2#0, %2#0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
func.func @propagate_through_while_result(
    %arg0: tensor<32x32xf32>, %out: memref<32x32xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %empty = tensor.empty() : tensor<32x32xf32>
  %fix = hivm.hir.fixpipe {enable_nz2nd} ins(%empty : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %while_res:2 = scf.while (%iter = %fix, %i = %c0) : (tensor<32x32xf32>, index) -> (tensor<32x32xf32>, index) {
    %cond = arith.cmpi slt, %i, %c2 : index
    scf.condition(%cond) %iter, %i : tensor<32x32xf32>, index
  } do {
  ^bb0(%iter: tensor<32x32xf32>, %i: index):
    %slice = tensor.extract_slice %iter[0, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
    %out_slice = tensor.empty() : tensor<1x32xf32>
    %vec = hivm.hir.vadd ins(%slice, %slice : tensor<1x32xf32>, tensor<1x32xf32>) outs(%out_slice : tensor<1x32xf32>) -> tensor<1x32xf32>
    %next = tensor.insert_slice %vec into %iter[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<32x32xf32>
    %next_i = arith.addi %i, %c1 : index
    scf.yield %next, %next_i : tensor<32x32xf32>, index
  }
  %vec_empty = tensor.empty() : tensor<32x32xf32>
  %res = hivm.hir.vadd ins(%while_res#0, %while_res#0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%vec_empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  hivm.hir.store ins(%res : tensor<32x32xf32>) outs(%out : memref<32x32xf32>)
  return
}


// -----
// CHECK-LABEL: @propagate_through_scope_result
// CHECK: %0 = scope.scope : () -> tensor<32x32xf32> {
// CHECK: %3 = tensor.empty() : tensor<32x32xf32>
// CHECK: %4 = hivm.hir.fixpipe {enable_nz2nd} ins(%3 : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %1 = tensor.empty() : tensor<32x32xf32>
// CHECK: %2 = hivm.hir.vadd ins(%0, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
func.func @propagate_through_scope_result(
    %arg0: tensor<32x32xf32>, %out: memref<32x32xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %scope_res = scope.scope : () -> tensor<32x32xf32> {
    %empty = tensor.empty() : tensor<32x32xf32>
    %fix = hivm.hir.fixpipe {enable_nz2nd} ins(%empty : tensor<32x32xf32>) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scope.return %fix : tensor<32x32xf32>
  }
  %vec_empty = tensor.empty() : tensor<32x32xf32>
  %res = hivm.hir.vadd ins(%scope_res, %scope_res : tensor<32x32xf32>, tensor<32x32xf32>) outs(%vec_empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  hivm.hir.store ins(%res : tensor<32x32xf32>) outs(%out : memref<32x32xf32>)
  return
}

// -----
// CHECK-LABEL: @propagate_through_tensor_insert_slice
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %true = arith.constant true
// CHECK: %0 = tensor.empty() : tensor<16x16xf16>
// CHECK: %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 16] [1, 1] : tensor<16x16xf16> to tensor<1x16xf16>
// CHECK: %inserted_slice = tensor.insert_slice %extracted_slice into %arg1[0, 0] [1, 16] [1, 1] : tensor<1x16xf16> into tensor<16x16xf16>
// CHECK: %1 = hivm.hir.mmadL1 ins(%arg0, %inserted_slice, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: %2 = tensor.empty() : tensor<16x16xf16>
// CHECK: %3 = hivm.hir.fixpipe {enable_nz2nd} ins(%1 : tensor<16x16xf16>) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: %4 = tensor.empty() : tensor<16x16xf16>
// CHECK: %5 = hivm.hir.load ins(%3 : tensor<16x16xf16>) outs(%4 : tensor<16x16xf16>) -> tensor<16x16xf16>
func.func @propagate_through_tensor_insert_slice(
    %lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf16>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %cube_empty = tensor.empty() : tensor<16x16xf16>
  %slice = tensor.extract_slice %lhs[0, 0] [1, 16] [1, 1] : tensor<16x16xf16> to tensor<1x16xf16>
  %inserted = tensor.insert_slice %slice into %rhs[0, 0] [1, 16] [1, 1] : tensor<1x16xf16> into tensor<16x16xf16>
  %cube = hivm.hir.mmadL1 ins(%lhs, %inserted, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf16>) -> tensor<16x16xf16>
  %empty = tensor.empty() : tensor<16x16xf16>
  %fix = hivm.hir.fixpipe {enable_nz2nd} ins(%cube : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
  hivm.hir.store ins(%fix : tensor<16x16xf16>) outs(%out : memref<16x16xf16>)
  return
}

// -----
// CHECK-LABEL: @multi_scope_user_propagator
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c2 = arith.constant 2 : index
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %false = arith.constant false
// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
// CHECK: %subview = memref.subview %reinterpret_cast[0, 0] [%arg2, %arg3] [1, 1] : memref<16x16xf32, strided<[16, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
// CHECK: %alloc = memref.alloc() : memref<16x16xf32>
// CHECK: %subview_0 = memref.subview %alloc[0, 0] [%arg2, %arg3] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
// CHECK: %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
// CHECK: %1 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %0) -> (tensor<16x16xf32>) {
// CHECK: %10 = tensor.empty() : tensor<16x16xf32>
// CHECK: %11 = hivm.hir.vln ins(%arg7 : tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %2 = tensor.empty() : tensor<16x16xf32>
// CHECK: %3 = hivm.hir.mmadL1 ins(%0, %arg4, %false, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %4 = tensor.empty() : tensor<16x16xf32>
// CHECK: %5 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %6 = tensor.empty() : tensor<16x16xf32>
// CHECK: %7 = hivm.hir.load ins(%5 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %8 = tensor.empty() : tensor<16x16xf32>
// CHECK: %9 = hivm.hir.vadd ins(%1, %7 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%8 : tensor<16x16xf32>) -> tensor<16x16xf32>
func.func @multi_scope_user_propagator(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: tensor<16x16xf32>, %arg5: memref<16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, 0] [%arg2, %arg3] [1, 1] : memref<16x16xf32, strided<[16, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
  %alloc = memref.alloc() : memref<16x16xf32>
  %subview_0 = memref.subview %alloc[0, 0] [%arg2, %arg3] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  hivm.hir.load ins(%subview : memref<?x?xf32, strided<[16, 1], offset: ?>>) outs(%subview_0 : memref<?x?xf32, strided<[16, 1]>>)
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
  %1 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %0) -> (tensor<16x16xf32>) {
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = hivm.hir.vln ins(%arg7 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %6 : tensor<16x16xf32>
  }
  %5 = tensor.empty() : tensor<16x16xf32>
  %6 = hivm.hir.mmadL1 ins(%0, %arg4, %false, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %7 = tensor.empty() : tensor<16x16xf32>
  %8 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6 : tensor<16x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %9 = tensor.empty() : tensor<16x16xf32>
  %10 = hivm.hir.vadd ins(%1, %8 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%9 : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%10 : tensor<16x16xf32>) outs(%arg5 : memref<16x16xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_fixpipe_indirect_store(
// CHECK-SAME: %[[ARG0:.*]]: tensor<16x16xf32>,
// CHECK-SAME: %[[GM:.*]]: memref<?xf16>,
// CHECK-SAME: %[[ARG2:.*]]: tensor<16x16xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
// CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%[[ARG2]] : tensor<16x16xi64>) 
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[ARG0]] : tensor<16x16xf32>) outs(%[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>>)
// CHECK: hivm.hir.indirect_store ins(%[[TENSOR]] : tensor<16x16xf16>, %[[LOAD]] : tensor<16x16xi64>) outs(%[[GM]] : memref<?xf16>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_fixpipe_indirect_store(%arg0: tensor<16x16xf32>, %arg1: memref<?xf16>, %arg2: tensor<16x16xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    hivm.hir.indirect_store ins(%1 : tensor<16x16xf16>, %arg2 : tensor<16x16xi64>) outs(%arg1 : memref<?xf16>)
    return
  }
}

// -----

// CHECK-LABEL: @insert_tight_coupled_buffer_between_stride_load_and_mmad
// CHECK: %[[RHS_LOAD:.*]] = hivm.hir.load {{.*}} core_type = <CUBE>
// CHECK: %[[STRIDE_LOAD:.*]] = hivm.hir.stride_load
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[STRIDE_LOAD]]
// CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]]
// CHECK: %[[NZ:.*]] = tensor.expand_shape %[[TRANSPOSE]]
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]]
// CHECK: %[[L1_TENSOR:.*]] = bufferization.to_tensor %[[CAST]]
// CHECK: hivm.hir.copy ins(%[[NZ]]
// CHECK-SAME: "inserted-copy"
// CHECK: hivm.hir.mmadL1 ins(%[[L1_TENSOR]], %[[RHS_LOAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @insert_tight_coupled_buffer_between_stride_load_and_mmad(
      %src: memref<?xf16>, %rhs: tensor<16x16xf16>)
      -> tensor<16x16xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %off = arith.constant 0 : i32
    %s0 = arith.constant 32 : i32
    %s1 = arith.constant 3 : i32
    %n0 = arith.constant 16 : i32
    %n1 = arith.constant 16 : i32
    %other = arith.constant 0.000000e+00 : f16
    %dst = tensor.empty() : tensor<16x16xf16>
    %load = hivm.hir.stride_load
      ins(%src : memref<?xf16>)
      outs(%dst : tensor<16x16xf16>)
      offset(%off : i32)
      other(%other : f16)
      strides([%s0, %s1 : i32, i32])
      numels([%n0, %n1 : i32, i32])
      {hivm.vf_mode = #hivm.vf_mode<SIMT>} -> tensor<16x16xf16>
    %out = tensor.empty() : tensor<16x16xf32>
    %mm = hivm.hir.mmadL1
      ins(%load, %rhs, %true, %c16, %c16, %c16
          : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
      outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mm : tensor<16x16xf32>
  }
}

// -----
// CHECK: func.func @tensor_outs_multiple_scope(
// CHECK-NOT: {"inserted-store"}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @tensor_outs_multiple_scope(%arg0: tensor<128x1xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<128x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<128x64xf32> {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = tensor.empty() : tensor<4x8x16x16xf32>
    %2 = tensor.empty() : tensor<8x8x16x16xbf16>
    %3 = tensor.empty() : tensor<4x8x16x16xbf16>
    %4 = hivm.hir.vbrc ins(%arg0 : tensor<128x1xf32>) outs(%0 : tensor<128x64xf32>) broadcast_dims = [1] -> tensor<128x64xf32>
    %5 = hivm.hir.vmul ins(%arg1, %4 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %6 = hivm.hir.nd2nz {dst_continuous} ins(%arg2 : tensor<128x128xbf16>) outs(%2 : tensor<8x8x16x16xbf16>) -> tensor<8x8x16x16xbf16>
    %7 = hivm.hir.nd2nz {dst_continuous} ins(%arg3 : tensor<64x128xbf16>) outs(%3 : tensor<4x8x16x16xbf16>) -> tensor<4x8x16x16xbf16>
    %8 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%6, %7, %true, %c128, %c128, %c64 : tensor<8x8x16x16xbf16>, tensor<4x8x16x16xbf16>, i1, index, index, index) outs(%1 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
    %9 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%8 : tensor<4x8x16x16xf32>) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %10 = hivm.hir.load ins(%9 : tensor<128x64xf32>) outs(%0 : tensor<128x64xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<128x64xf32>
    %11 = hivm.hir.vadd ins(%10, %5 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %11 : tensor<128x64xf32>
  }
}

// -----

// CHECK-LABEL: @custom_gm_addr_arg_preserved(
// CHECK-SAME: %[[DST:.*]]: memref<?xf32>
// CHECK-NOT: hivm.hir.load ins(%[[DST]]
// CHECK: hivm.hir.custom
// CHECK-SAME: gm_addr_args_indices = array<i32: 0>
// CHECK-SAME: "__builtin_indirect_atomic" ins(%[[DST]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @custom_gm_addr_arg_preserved(
      %dst: memref<?xf32>, %offsets: tensor<256xi64>, %old: tensor<256xf32>,
      %updates: tensor<256xf32>) -> tensor<256xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = hivm.hir.custom {extra_attr = "operate=cas",
                           gm_addr_args_indices = array<i32: 0>,
                           hivm.pipe = #hivm.pipe<PIPE_V>,
                           hivm.tcore_type = #hivm.tcore_type<VECTOR>,
                           hivm.vf_mode = #hivm.vf_mode<SIMT>,
                           symbol = "__builtin_indirect_atomic"}
        "__builtin_indirect_atomic"
        ins(%dst, %offsets, %old, %updates
            : memref<?xf32>, tensor<256xi64>, tensor<256xf32>, tensor<256xf32>)
        outs(%old : tensor<256xf32>) -> tensor<256xf32>
    %empty = tensor.empty() : tensor<256xf32>
    %1 = hivm.hir.vmul ins(%0, %cst : tensor<256xf32>, f32)
        outs(%empty : tensor<256xf32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
}

// -----

// CHECK-LABEL: @cube_custom_gm_addr_arg_preserved(
// CHECK-SAME: %[[DST:.*]]: memref<?xf32>
// CHECK-NOT: hivm.hir.load ins(%[[DST]]
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.pipe = #hivm.pipe<PIPE_M>
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<CUBE>
// CHECK-SAME: ins(%[[DST]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cube_custom_gm_addr_arg_preserved(
      %dst: memref<?xf32>, %lhs: tensor<16x16xf16>,
      %out: tensor<16x16xf32>) -> tensor<16x16xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>,
                           hivm.pipe = #hivm.pipe<PIPE_M>,
                           hivm.tcore_type = #hivm.tcore_type<CUBE>,
                           symbol = "my_cube_custom"}
        "my_cube_custom"
        ins(%dst, %lhs : memref<?xf32>, tensor<16x16xf16>)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: @vbrc_mmad_rhs_no_convert_layout(
// CHECK-NOT: {"inserted-copy"}
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK: hivm.hir.mmadL1 {{.*}} ins({{.*}}, %[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vbrc_mmad_rhs_no_convert_layout(%lhs: tensor<16x128xbf16>, %rhs: tensor<16x16xf16>) -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %init = tensor.empty() : tensor<128x128xbf16>
    %vbrc = hivm.hir.vbrc ins(%zero : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%lhs, %vbrc, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
        outs(%out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vbrc_broadcast_between_fixpipe_and_vector(
// CHECK: hivm.hir.fixpipe
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK-NOT: hivm.hir.load ins(%[[VBRC]]
// CHECK: hivm.hir.vmul ins(%{{.*}}, %[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vbrc_broadcast_between_fixpipe_and_vector(%scale: tensor<16x1xf32>, %vec: tensor<16x128xf32>)
      -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %out = tensor.empty() : tensor<16x128xf32>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%vec : tensor<16x128xf32>) outs(%out : tensor<16x128xf32>) -> tensor<16x128xf32>
    %bcast_out = tensor.empty() : tensor<16x128xf32>
    %bcast = hivm.hir.vbrc ins(%scale : tensor<16x1xf32>) outs(%bcast_out : tensor<16x128xf32>) broadcast_dims = [1] -> tensor<16x128xf32>
    %mul_out = tensor.empty() : tensor<16x128xf32>
    %mul = hivm.hir.vmul ins(%fix, %bcast : tensor<16x128xf32>, tensor<16x128xf32>) outs(%mul_out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mul : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vbrc_init_for_insert_slice_mmad_rhs(
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK: %[[FOR:.*]]:2 = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[VBRC]]
// CHECK: hivm.hir.mmadL1 {{.*}} ins({{.*}}, %[[FOR]]#
// CHECK-NOT: hivm.hir.copy ins(%[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vbrc_init_for_insert_slice_mmad_rhs(%lhs: tensor<16x128xbf16>, %tile: tensor<32x128xbf16>)
      -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %init = tensor.empty() : tensor<128x128xbf16>
    %vbrc = hivm.hir.vbrc ins(%zero : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %for:2 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc0 = %vbrc, %acc1 = %vbrc) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) : i32 {
      %off = arith.muli %i, %c32_i32 : i32
      %off_idx = arith.index_cast %off : i32 to index
      %ins0 = tensor.insert_slice %tile into %acc0[%off_idx, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
      %ins1 = tensor.insert_slice %tile into %acc1[%off_idx, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
      scf.yield %ins0, %ins1 : tensor<128x128xbf16>, tensor<128x128xbf16>
    }
    %out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%lhs, %for#1, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
        outs(%out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vcast_mmad_lhs_convert_layout_vbrc_init_for_rhs(
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK: %[[FOR:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[VBRC]]
// CHECK: hivm.hir.vcast
// CHECK: {"inserted-copy"}
// CHECK: hivm.hir.mmadL1 {{.*}} ins({{.*}}, %[[FOR]]
// CHECK-NOT: hivm.hir.copy ins(%[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vcast_mmad_lhs_convert_layout_vbrc_init_for_rhs(%vec: tensor<16x128xf32>, %tile: tensor<32x128xbf16>)
      -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %init = tensor.empty() : tensor<128x128xbf16>
    %vbrc = hivm.hir.vbrc ins(%zero : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %for = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %vbrc) -> (tensor<128x128xbf16>) : i32 {
      %off = arith.muli %i, %c32_i32 : i32
      %off_idx = arith.index_cast %off : i32 to index
      %ins = tensor.insert_slice %tile into %acc[%off_idx, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
      scf.yield %ins : tensor<128x128xbf16>
    }
    %bf16_out = tensor.empty() : tensor<16x128xbf16>
    %vcast = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%vec : tensor<16x128xf32>) outs(%bf16_out : tensor<16x128xbf16>) -> tensor<16x128xbf16>
    %mmad_out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%vcast, %for, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
        outs(%mmad_out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vbrc_mmad_rhs_no_convert_layout(
// CHECK-NOT: {"inserted-copy"}
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK: hivm.hir.mmadL1 {{.*}} ins({{.*}}, %[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vbrc_mmad_rhs_no_convert_layout(%lhs: tensor<16x128xbf16>, %rhs: tensor<16x16xf16>) -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %init = tensor.empty() : tensor<128x128xbf16>
    %vbrc = hivm.hir.vbrc ins(%zero : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%lhs, %vbrc, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
        outs(%out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vbrc_broadcast_between_fixpipe_and_vector(
// CHECK: hivm.hir.fixpipe
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK-NOT: hivm.hir.load ins(%[[VBRC]]
// CHECK: hivm.hir.vmul ins(%{{.*}}, %[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vbrc_broadcast_between_fixpipe_and_vector(%scale: tensor<16x1xf32>, %vec: tensor<16x128xf32>)
      -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %out = tensor.empty() : tensor<16x128xf32>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%vec : tensor<16x128xf32>) outs(%out : tensor<16x128xf32>) -> tensor<16x128xf32>
    %bcast_out = tensor.empty() : tensor<16x128xf32>
    %bcast = hivm.hir.vbrc ins(%scale : tensor<16x1xf32>) outs(%bcast_out : tensor<16x128xf32>) broadcast_dims = [1] -> tensor<16x128xf32>
    %mul_out = tensor.empty() : tensor<16x128xf32>
    %mul = hivm.hir.vmul ins(%fix, %bcast : tensor<16x128xf32>, tensor<16x128xf32>) outs(%mul_out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mul : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vbrc_init_for_insert_slice_mmad_rhs(
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK: %[[FOR:.*]]:2 = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[VBRC]]
// CHECK: hivm.hir.mmadL1 {{.*}} ins({{.*}}, %[[FOR]]#
// CHECK-NOT: hivm.hir.copy ins(%[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vbrc_init_for_insert_slice_mmad_rhs(%lhs: tensor<16x128xbf16>, %tile: tensor<32x128xbf16>)
      -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %init = tensor.empty() : tensor<128x128xbf16>
    %vbrc = hivm.hir.vbrc ins(%zero : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %for:2 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc0 = %vbrc, %acc1 = %vbrc) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) : i32 {
      %off = arith.muli %i, %c32_i32 : i32
      %off_idx = arith.index_cast %off : i32 to index
      %ins0 = tensor.insert_slice %tile into %acc0[%off_idx, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
      %ins1 = tensor.insert_slice %tile into %acc1[%off_idx, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
      scf.yield %ins0, %ins1 : tensor<128x128xbf16>, tensor<128x128xbf16>
    }
    %out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%lhs, %for#1, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
        outs(%out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @vcast_mmad_lhs_convert_layout_vbrc_init_for_rhs(
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc
// CHECK: %[[FOR:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[VBRC]]
// CHECK: hivm.hir.vcast
// CHECK: {"inserted-copy"}
// CHECK: hivm.hir.mmadL1 {{.*}} ins({{.*}}, %[[FOR]]
// CHECK-NOT: hivm.hir.copy ins(%[[VBRC]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vcast_mmad_lhs_convert_layout_vbrc_init_for_rhs(%vec: tensor<16x128xf32>, %tile: tensor<32x128xbf16>)
      -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %init = tensor.empty() : tensor<128x128xbf16>
    %vbrc = hivm.hir.vbrc ins(%zero : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %for = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %vbrc) -> (tensor<128x128xbf16>) : i32 {
      %off = arith.muli %i, %c32_i32 : i32
      %off_idx = arith.index_cast %off : i32 to index
      %ins = tensor.insert_slice %tile into %acc[%off_idx, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
      scf.yield %ins : tensor<128x128xbf16>
    }
    %bf16_out = tensor.empty() : tensor<16x128xbf16>
    %vcast = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%vec : tensor<16x128xf32>) outs(%bf16_out : tensor<16x128xbf16>) -> tensor<16x128xbf16>
    %mmad_out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%vcast, %for, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
        outs(%mmad_out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @preserve_preset_cube_load_core_type_with_vtranspose_path(
// CHECK: hivm.hir.load {{.*}} core_type = <CUBE>
// CHECK-NOT: hivm.hir.load {{.*}} core_type = <VECTOR>
// CHECK: hivm.hir.vtranspose
// CHECK: {"inserted-copy"}
// CHECK: hivm.hir.mmadL1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preserve_preset_cube_load_core_type_with_vtranspose_path(
      %vec: tensor<16x128xf32>, %gm: memref<32x128xbf16, strided<[128, 1]>>,
      %rhs: tensor<8x8x16x16xbf16>) -> tensor<16x128xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : bf16
    %bf16_out = tensor.empty() : tensor<16x128xbf16>
    %vcast = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%vec : tensor<16x128xf32>) outs(%bf16_out : tensor<16x128xbf16>)
        -> tensor<16x128xbf16>
    %alloc = memref.alloc() : memref<32x128xbf16>
    %subview = memref.subview %gm[0, 0] [32, 128] [1, 1]
        : memref<32x128xbf16, strided<[128, 1]>> to memref<32x128xbf16, strided<[128, 1]>>
    %subview_out = memref.subview %alloc[0, 0] [32, 128] [1, 1]
        : memref<32x128xbf16> to memref<32x128xbf16, strided<[128, 1]>>
    hivm.hir.load ins(%subview : memref<32x128xbf16, strided<[128, 1]>>)
        outs(%subview_out : memref<32x128xbf16, strided<[128, 1]>>)
        pad_mode = <PadValue> pad_value = %zero : bf16
        init_out_buffer = true eviction_policy = <EvictFirst>
        core_type = <CUBE>
    %mmad_out = tensor.empty() : tensor<16x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%vcast, %rhs, %true, %c16, %c128, %c128
            : tensor<16x128xbf16>, tensor<8x8x16x16xbf16>, i1, index, index, index)
        outs(%mmad_out : tensor<16x128xf32>) -> tensor<16x128xf32>
    return %mmad : tensor<16x128xf32>
  }
}

// -----

// CHECK-LABEL: @preserve_preset_vector_load_core_type(
// CHECK: hivm.hir.load ins({{.*}} core_type = <VECTOR>
// CHECK: hivm.hir.vmul
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preserve_preset_vector_load_core_type(%gm: memref<16x16xf32, strided<[16, 1]>>)
      -> tensor<16x16xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%gm : memref<16x16xf32, strided<[16, 1]>>)
        outs(%alloc : memref<16x16xf32>) eviction_policy = <EvictFirst>
        core_type = <VECTOR>
    %loaded = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    %out = tensor.empty() : tensor<16x16xf32>
    %mul = hivm.hir.vmul ins(%loaded, %loaded : tensor<16x16xf32>, tensor<16x16xf32>)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mul : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: @preserve_preset_vector_inserted_load_core_type(
// CHECK: hivm.hir.fixpipe
// CHECK: {{"inserted-store"}}
// CHECK: %[[LOAD:.*]] = hivm.hir.load
// CHECK-SAME: {{"inserted-load"}}
// CHECK-SAME: core_type = <VECTOR>
// CHECK: hivm.hir.vadd ins(%[[LOAD]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preserve_preset_vector_inserted_load_core_type(
      %scale: tensor<128x1xf32>, %vec: tensor<128x64xf32>,
      %lhs: tensor<128x128xbf16>, %rhs: tensor<64x128xbf16>) -> tensor<128x64xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %out = tensor.empty() : tensor<128x64xf32>
    %mmad_tmp = tensor.empty() : tensor<4x8x16x16xf32>
    %lhs_nz = tensor.empty() : tensor<8x8x16x16xbf16>
    %rhs_nz = tensor.empty() : tensor<4x8x16x16xbf16>
    %bcast = hivm.hir.vbrc ins(%scale : tensor<128x1xf32>) outs(%out : tensor<128x64xf32>)
        broadcast_dims = [1] -> tensor<128x64xf32>
    %vmul = hivm.hir.vmul ins(%vec, %bcast : tensor<128x64xf32>, tensor<128x64xf32>)
        outs(%out : tensor<128x64xf32>) -> tensor<128x64xf32>
    %lhs_conv = hivm.hir.nd2nz {dst_continuous} ins(%lhs : tensor<128x128xbf16>)
        outs(%lhs_nz : tensor<8x8x16x16xbf16>) -> tensor<8x8x16x16xbf16>
    %rhs_conv = hivm.hir.nd2nz {dst_continuous} ins(%rhs : tensor<64x128xbf16>)
        outs(%rhs_nz : tensor<4x8x16x16xbf16>) -> tensor<4x8x16x16xbf16>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true,
                             normalized_in_L0C}
        ins(%lhs_conv, %rhs_conv, %true, %c128, %c128, %c64
            : tensor<8x8x16x16xbf16>, tensor<4x8x16x16xbf16>, i1, index, index, index)
        outs(%mmad_tmp : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad : tensor<4x8x16x16xf32>) outs(%out : tensor<128x64xf32>) -> tensor<128x64xf32>
    %loaded = hivm.hir.load ins(%fix : tensor<128x64xf32>) outs(%out : tensor<128x64xf32>)
        {"inserted-load"} core_type = <VECTOR> -> tensor<128x64xf32>
    %res = hivm.hir.vadd ins(%loaded, %vmul : tensor<128x64xf32>, tensor<128x64xf32>)
        outs(%out : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %res : tensor<128x64xf32>
  }
}
