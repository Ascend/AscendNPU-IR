// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

// IR taken from test_dot_exp_NZ.py before hivm-insert-fixpipe.
// float16 16x32x32, fractal A/B/C. The mmad result has one ND→Fractal
// convert_layout consumed by vexp. A5 L0C→L1 NZ2NZ is native, so the pass
// must fixpipe straight into the fractal tensor. An NZ2ND copy into a full
// ND UB tensor is the failing path.

// CHECK-LABEL: func.func @dot_exp_nz_f16_16x32x32
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<16x32xf32>
// CHECK-NOT: fixpipe_for_result_already_inserted
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe
// CHECK-SAME: ins(%[[MMAD]] : tensor<16x32xf32>) outs(%{{.*}} : tensor<2x1x16x16xf32>)
// CHECK-SAME: -> tensor<2x1x16x16xf32>
// CHECK-NOT: dma_mode = #hivm.dma_mode<nz2nd>
// CHECK: hivm.hir.vexp ins(%[[FIX]] : tensor<2x1x16x16xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @dot_exp_nz_f16_16x32x32(%arg2: memref<?xf32>, %arg3: memref<?xf16>, %arg4: memref<?xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 1, 16, 16], strides: [256, 256, 16, 1] : memref<?xf16> to memref<2x1x16x16xf16, strided<[256, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<2x1x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<2x1x16x16xf16, strided<[256, 256, 16, 1]>>) outs(%alloc : memref<2x1x16x16xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2x1x16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [2, 2, 16, 16], strides: [512, 256, 16, 1] : memref<?xf16> to memref<2x2x16x16xf16, strided<[512, 256, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<2x2x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<2x2x16x16xf16, strided<[512, 256, 16, 1]>>) outs(%alloc_1 : memref<2x2x16x16xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x2x16x16xf16>
    %4 = hivm.hir.convert_layout %2 output_shape [16, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<2x1x16x16xf16>) -> tensor<16x32xf16>
    %5 = hivm.hir.convert_layout %3 output_shape [32, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<2x2x16x16xf16>) -> tensor<32x32xf16>
    %6 = tensor.empty() : tensor<16x32xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%4, %5, %true, %c16, %c32, %c32 : tensor<16x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%6 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %8 = hivm.hir.convert_layout %7 output_shape [2, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x32xf32>) -> tensor<2x1x16x16xf32>
    %9 = tensor.empty() : tensor<2x1x16x16xf32>
    %10 = hivm.hir.vexp ins(%8 : tensor<2x1x16x16xf32>) outs(%9 : tensor<2x1x16x16xf32>) -> tensor<2x1x16x16xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 1, 16, 16], strides: [256, 256, 16, 1] : memref<?xf32> to memref<2x1x16x16xf32, strided<[256, 256, 16, 1]>>
    hivm.hir.store ins(%10 : tensor<2x1x16x16xf32>) outs(%reinterpret_cast_2 : memref<2x1x16x16xf32, strided<[256, 256, 16, 1]>>)
    return
  }
}

// -----

// float16 256x32x256, fractal A/B/C. The NZ2ND fallback allocates
// tensor<256x256xf32> in UB and overflows. Expect the fractal fixpipe.

// CHECK-LABEL: func.func @dot_exp_nz_f16_256x32x256
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<256x256xf32>
// CHECK-NOT: fixpipe_for_result_already_inserted
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe
// CHECK-SAME: ins(%[[MMAD]] : tensor<256x256xf32>) outs(%{{.*}} : tensor<16x16x16x16xf32>)
// CHECK-SAME: -> tensor<16x16x16x16xf32>
// CHECK-NOT: dma_mode = #hivm.dma_mode<nz2nd>
// CHECK: hivm.hir.vexp ins(%[[FIX]] : tensor<16x16x16x16xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @dot_exp_nz_f16_256x32x256(%arg2: memref<?xf32>, %arg3: memref<?xf16>, %arg4: memref<?xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 16, 16, 16], strides: [4096, 256, 16, 1] : memref<?xf16> to memref<2x16x16x16xf16, strided<[4096, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<2x16x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<2x16x16x16xf16, strided<[4096, 256, 16, 1]>>) outs(%alloc : memref<2x16x16x16xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2x16x16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 2, 16, 16], strides: [512, 256, 16, 1] : memref<?xf16> to memref<16x2x16x16xf16, strided<[512, 256, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x2x16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x2x16x16xf16, strided<[512, 256, 16, 1]>>) outs(%alloc_1 : memref<16x2x16x16xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x2x16x16xf16>
    %4 = hivm.hir.convert_layout %2 output_shape [256, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<2x16x16x16xf16>) -> tensor<256x32xf16>
    %5 = hivm.hir.convert_layout %3 output_shape [32, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<16x2x16x16xf16>) -> tensor<32x256xf16>
    %6 = tensor.empty() : tensor<256x256xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%4, %5, %true, %c256, %c32, %c256 : tensor<256x32xf16>, tensor<32x256xf16>, i1, index, index, index) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %8 = hivm.hir.convert_layout %7 output_shape [16, 16, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<256x256xf32>) -> tensor<16x16x16x16xf32>
    %9 = tensor.empty() : tensor<16x16x16x16xf32>
    %10 = hivm.hir.vexp ins(%8 : tensor<16x16x16x16xf32>) outs(%9 : tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16, 16, 16], strides: [4096, 256, 16, 1] : memref<?xf32> to memref<16x16x16x16xf32, strided<[4096, 256, 16, 1]>>
    hivm.hir.store ins(%10 : tensor<16x16x16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16x16x16xf32, strided<[4096, 256, 16, 1]>>)
    return
  }
}
