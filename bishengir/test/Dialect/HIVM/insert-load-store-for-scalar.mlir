// RUN: bishengir-opt -hivm-insert-load-store-for-scalar %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// CHECK-LABEL: @extract_i1_for_cube_init
// CHECK: tensor.extract
// CHECK-SAME: tensor<16x16xi1>
// CHECK: hivm.hir.store
// CHECK-SAME: tensor<1xi8>
// CHECK-SAME: "inserted-store"
// CHECK: DuplicateTensorExtractForCube::replacementLabel
// CHECK: hivm.hir.mmadL1
func.func @extract_i1_for_cube_init(
    %arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>,
    %lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %cmp_empty = tensor.empty() : tensor<16x16xi1>
  %cmp = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%cmp_empty : tensor<16x16xi1>) compare_mode = <lt> -> tensor<16x16xi1>
  %cond = tensor.extract %cmp[%c0, %c0] : tensor<16x16xi1>
  %cube_empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%lhs, %rhs, %cond, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  return
}
// -----

// CHECK-LABEL: @extract_for_index_use
// CHECK: %[[EXTRACT:.*]] = tensor.extract
// CHECK-SAME: tensor<16xi64>
// CHECK: %[[IDX:.*]] = arith.index_cast %[[EXTRACT]]
// CHECK-SAME: i64 to index
// CHECK-NOT: inserted-store
// CHECK: memref.load %{{.*}}[%[[IDX]]]
// CHECK-SAME: memref<16xf32>
func.func @extract_for_index_use(
    %indices: tensor<16xi64>, %data: memref<16xf32>,
    %lhs: tensor<16x16xf32>, %rhs: tensor<16x16xf32>,
    %out: memref<16x16xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %cube_empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  %raw_index = tensor.extract %indices[%c0] : tensor<16xi64>
  %idx = arith.index_cast %raw_index : i64 to index
  %value = memref.load %data[%idx] : memref<16xf32>
  return %value : f32
}

// -----

// CHECK-LABEL: func.func @test_extract_cube_tcore
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1xf32>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<1x1xf32>
// CHECK: hivm.hir.load ins(%[[_:.*]] : memref<1x1xf32, strided<[1, 1], offset: ?>>) outs(%[[ALLOC]] : memref<1x1xf32>) eviction_policy = <EvictFirst>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[TENSOR]]
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[COLLAPSED]]
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>} ins(%[[MMAD]] : tensor<128x128xf32>) outs(%[[_:.*]] : tensor<128x128xf32>) quant_scale = %[[EXTRACT]] : f32 -> tensor<128x128xf32>
func.func @test_extract_cube_tcore(%arg0: memref<?xf32>, %arg1: tensor<128x128xf8E4M3FN>, %arg2: tensor<128x128xf8E4M3FN>) -> tensor<128x128xf32> {
  %true = arith.constant true
  %c128 = arith.constant 128 : index
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true, normalized_in_L0C} ins(%arg1, %arg2, %true, %c128, %c128, %c128 : tensor<128x128xf8E4M3FN>, tensor<128x128xf8E4M3FN>, i1, index, index, index) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  
  %alloc = memref.alloc() : memref<1x1xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<1x1xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<1x1xf32, strided<[1, 1], offset: ?>>) outs(%alloc : memref<1x1xf32>) eviction_policy = <EvictFirst>
  %collapsed = tensor.collapse_shape %2 [] : tensor<1x1xf32> into tensor<f32>
  %extracted = tensor.extract %collapsed[] : tensor<f32>
  %3 = tensor.empty() : tensor<128x128xf32>
  %4 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>} ins(%1 : tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) quant_scale = %extracted : f32 -> tensor<128x128xf32>
  return %4 : tensor<128x128xf32>
}

// -----

// Cube user reachable only across an scf.for loop-carried dep: bridge inserted.
// CHECK-LABEL: @extract_cube_user_via_scf_for_iterarg
// CHECK: hivm.hir.vsel
// CHECK: hivm.hir.store
// CHECK-SAME: "inserted-store"
// CHECK: DuplicateTensorExtractForCube::replacementLabel
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
func.func @extract_cube_user_via_scf_for_iterarg(
    %cond: tensor<1xi1>, %val: tensor<1xi32>,
    %lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %c16_i32 = arith.constant 16 : i32
  %true = arith.constant true
  %empty = tensor.empty() : tensor<1xi32>
  %sel = hivm.hir.vsel ins(%cond, %val, %c0_i32 : tensor<1xi1>, tensor<1xi32>, i32) outs(%empty : tensor<1xi32>) -> tensor<1xi32>
  %extracted = tensor.extract %sel[%c0] : tensor<1xi32>
  %r = scf.for %i = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
    %sz_i = arith.maxsi %acc, %c0_i32 : i32
    %sz = arith.index_cast %sz_i : i32 to index
    %cube_empty = tensor.empty() : tensor<16x16xf32>
    %res = hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %sz, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
    %next = arith.addi %acc, %extracted : i32
    scf.yield %next : i32
  }
  return %r : i32
}