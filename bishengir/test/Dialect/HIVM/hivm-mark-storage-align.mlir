// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_mark_elementwise_unary
func.func @test_mark_elementwise_unary(%dim0 : index, %dim1: index) {
  // CHECK: %[[A:.*]] = memref.alloc
  // CHECK-NOT: annotation.mark %[[A]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  %a = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  // CHECK: %[[B:.*]] = memref.alloc
  // CHECK-NOT: annotation.mark %[[B]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  %b = memref.alloc(%dim0) : memref<?x15xf16, #hivm.address_space<ub>>
  // CHECK: %[[C:.*]] = memref.alloc
  // CHECK-NOT: annotation.mark %[[C]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  %c = memref.alloc(%dim1) : memref<2x?xf16, #hivm.address_space<ub>>
  // CHECK: %[[D:.*]] = memref.alloc
  // CHECK-NOT: annotation.mark %[[D]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  %d = memref.alloc(%dim0, %dim1) : memref<?x?xf16, #hivm.address_space<ub>>
  // COM: No more storage-align marks
  // CHECK-NOT: annotation.mark
  hivm.hir.vexp ins(%a : memref<2x15xf16, #hivm.address_space<ub>>) outs(%a : memref<2x15xf16, #hivm.address_space<ub>>)
  hivm.hir.vexp ins(%b : memref<?x15xf16, #hivm.address_space<ub>>) outs(%a : memref<2x15xf16, #hivm.address_space<ub>>)
  hivm.hir.vexp ins(%c : memref<2x?xf16, #hivm.address_space<ub>>) outs(%a : memref<2x15xf16, #hivm.address_space<ub>>)
  hivm.hir.vexp ins(%d : memref<?x?xf16, #hivm.address_space<ub>>) outs(%a : memref<2x15xf16, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: @test_mark_reduce_max_with_index_inner
func.func @test_mark_reduce_max_with_index_inner(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32, #hivm.address_space<gm>>, %arg4: i32) attributes {func_dyn_memref_args = dense<[true, true, true, true, false]> : vector<5xi1>, global_kernel = "local", hacc.dso_local = "", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.spir_kernel = "", hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c4096 = arith.constant 4096 : index
  %c16384_i32 = arith.constant 16384 : i32
  %c4096_i32 = arith.constant 4096 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %1, %c16384_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [4096, 4], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<4096x4xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC:.*]] {
  %4 = arith.addi %3, %c4096 : index
  %5 = arith.index_cast %arg4 : i32 to index
  %6 = arith.maxsi %3, %5 : index
  %7 = arith.minsi %4, %6 : index
  %8 = arith.subi %7, %3 : index
  %9 = arith.cmpi slt, %8, %c4096 : index
  scf.if %9 {
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<4096x4xf32, #hivm.address_space<ub>>)
  }
  %subview = memref.subview %reinterpret_cast[0, 0] [%8, 4] [1, 1] : memref<4096x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %alloc[0, 0] [%8, 4] [1, 1] : memref<4096x4xf32, #hivm.address_space<ub>> to memref<?x4xf32, strided<[4, 1]>, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[SUBVIEW0:.*]] {
  // CHECK: hivm.hir.load
  // CHECK-SAME: outs(%[[SUBVIEW0]]
  hivm.hir.load ins(%subview : memref<?x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x4xf32, strided<[4, 1]>, #hivm.address_space<ub>>)
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [4096, 4], strides: [1, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<4096x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc_2 = memref.alloc() : memref<4096x4xi32, #hivm.address_space<ub>>
  scf.if %9 {
    hivm.hir.vbrc ins(%c0_i32 : i32) outs(%alloc_2 : memref<4096x4xi32, #hivm.address_space<ub>>)
  }
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%8, 4] [1, 1] : memref<4096x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_4 = memref.subview %alloc_2[0, 0] [%8, 4] [1, 1] : memref<4096x4xi32, #hivm.address_space<ub>> to memref<?x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[SUBVIEW4:.*]] {
  // CHECK: hivm.hir.load
  // CHECK-SAME: outs(%[[SUBVIEW4]]
  hivm.hir.load ins(%subview_3 : memref<?x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>)
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32, #hivm.address_space<ub>>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4096xi32, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %alloc_5 [[0, 1]] output_shape [4096, 1] : memref<4096xf32, #hivm.address_space<ub>> into memref<4096x1xf32, #hivm.address_space<ub>>
  %expand_shape_7 = memref.expand_shape %alloc_6 [[0, 1]] output_shape [4096, 1] : memref<4096xi32, #hivm.address_space<ub>> into memref<4096x1xi32, #hivm.address_space<ub>>
  %alloc_8 = memref.alloc() : memref<16384xf32, #hivm.address_space<ub>>
  hivm.hir.vreduce <max_with_index_left> ins(%alloc : memref<4096x4xf32, #hivm.address_space<ub>>) outs(%expand_shape, %expand_shape_7 : memref<4096x1xf32, #hivm.address_space<ub>>, memref<4096x1xi32, #hivm.address_space<ub>>) temp_buffer(%alloc_8 : memref<16384xf32, #hivm.address_space<ub>>) reduce_dims = [1]
  %10 = arith.muli %1, %c4096_i32 : i32
  %11 = arith.index_cast %10 : i32 to index
  %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [4096], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %12 = arith.addi %11, %c4096 : index
  %13 = arith.maxsi %11, %5 : index
  %14 = arith.minsi %12, %13 : index
  %15 = arith.subi %14, %11 : index
  %subview_10 = memref.subview %alloc_5[0] [%15] [1] : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_11 = memref.subview %reinterpret_cast_9[0] [%15] [1] : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %reinterpret_cast_12 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [4096], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<4096xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_13 = memref.subview %alloc_6[0] [%15] [1] : memref<4096xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
  %subview_14 = memref.subview %reinterpret_cast_12[0] [%15] [1] : memref<4096xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_13 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_14 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: @test_slice_clone
func.func @test_slice_clone(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {func_dyn_memref_args = dense<[true, true, false]> : vector<3xi1>, global_kernel = "local", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.dso_local = "", hivm.entry = "", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.spir_kernel = ""} {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c16_i32 = arith.constant 16 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %1, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %3, %c32 : index
  %5 = arith.addi %4, %c1 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  %6 = arith.addi %3, %c16 : index
  %7 = arith.maxsi %3, %c16 : index
  %8 = arith.minsi %6, %7 : index
  %9 = arith.subi %8, %3 : index
  %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<16xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %alloc[0] [%9] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]]
  // CHECK-DAG: annotation.mark %[[SUBVIEW]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  hivm.hir.load ins(%subview : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>)
    outs(%subview_0 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_2 = memref.subview %reinterpret_cast_1[0] [%9] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_0 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    outs(%subview_2 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func @test_brc_last
// CHECK-NOT: annotation.mark
func.func @test_brc_last() {
  %src = memref.alloc() : memref<16x1xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x15xf32, #hivm.address_space<ub>>

  hivm.hir.vbrc ins(%src : memref<16x1xf32, #hivm.address_space<ub>>)
    outs(%dst : memref<16x15xf32, #hivm.address_space<ub>>)
    broadcast_dims = [1]
  return
}

// -----

// CHECK-LABEL: func @test_reduce_last_not_align
func.func @test_reduce_last_not_align() {
  %src0 = memref.alloc() : memref<32x4xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<32x1xf32, #hivm.address_space<ub>>

  // CHECK: %[[SRC0:.*]] = memref.alloc
  // CHECK-SAME: 32x4xf32
  // CHECK-NOT: annotation.mark %[[SRC0]]
  hivm.hir.vreduce <sum> ins(%src0 : memref<32x4xf32, #hivm.address_space<ub>>)
    outs(%dst : memref<32x1xf32, #hivm.address_space<ub>>)
    reduce_dims = [1]
  return
}

// -----

// CHECK-LABEL: func @test_reduce_last_align
func.func @test_reduce_last_align() {
  %dst = memref.alloc() : memref<32x1xf32, #hivm.address_space<ub>>
  %src1 = memref.alloc() : memref<32x5xf32, #hivm.address_space<ub>>
  
  // CHECK: %[[SRC1:.*]] = memref.alloc() : memref<32x5xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[SRC1]]
  hivm.hir.vreduce <sum> ins(%src1 : memref<32x5xf32, #hivm.address_space<ub>>)
      outs(%dst : memref<32x1xf32, #hivm.address_space<ub>>)
      reduce_dims = [1]
  return
}

// -----

// CHECK-LABEL: func @test_concat
func.func @test_concat() {
  %src0 = memref.alloc() : memref<32x4xf32, #hivm.address_space<ub>>
  %src1 = memref.alloc() : memref<32x4xf32, #hivm.address_space<ub>>
  %dst0 = memref.alloc() : memref<32x8xf32, #hivm.address_space<ub>>

  // CHECK: %[[SRC0:.*]] = memref.alloc
  // CHECK-SAME: 32x4xf32
  // CHECK: annotation.mark %[[SRC0]]
  // CHECK: %[[SRC1:.*]] = memref.alloc
  // CHECK-SAME: 32x4xf32
  // CHECK: annotation.mark %[[SRC1]]
  // CHECK: %[[SRC2:.*]] = memref.alloc
  // CHECK-SAME: 32x8xf32
  // CHECK-NOT: annotation.mark %[[SRC2]]
  hivm.hir.vconcat dim(1) ins(%src0, %src1: memref<32x4xf32, #hivm.address_space<ub>>, memref<32x4xf32, #hivm.address_space<ub>>)
                          outs(%dst0: memref<32x8xf32, #hivm.address_space<ub>>)

  %src2 = memref.alloc() : memref<32x4x4xf32, #hivm.address_space<ub>>
  %src3 = memref.alloc() : memref<32x4x4xf32, #hivm.address_space<ub>>
  %dst1 = memref.alloc() : memref<32x8x4xf32, #hivm.address_space<ub>>

  // CHECK: %[[SRC3:.*]] = memref.alloc
  // CHECK-SAME: 32x4x4xf32
  // CHECK: annotation.mark %[[SRC3]]
  // CHECK: %[[SRC4:.*]] = memref.alloc
  // CHECK-SAME: 32x4x4xf32
  // CHECK: annotation.mark %[[SRC4]]
  // CHECK: %[[SRC5:.*]] = memref.alloc
  // CHECK-SAME: 32x8x4xf32
  // CHECK: annotation.mark %[[SRC5]]
  hivm.hir.vconcat dim(1) ins(%src2, %src3: memref<32x4x4xf32, #hivm.address_space<ub>>, memref<32x4x4xf32, #hivm.address_space<ub>>)
                          outs(%dst1: memref<32x8x4xf32, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func @test_pad
func.func @test_pad(%pad_value : f32) {
  %src0 = memref.alloc() : memref<32x4xf32, #hivm.address_space<ub>>
  %dst0 = memref.alloc() : memref<32x6xf32, #hivm.address_space<ub>>

  // CHECK: %[[SRC0:.*]] = memref.alloc
  // CHECK-SAME: 32x4xf32
  // CHECK: annotation.mark %[[SRC0]]
  // CHECK: %[[DST0:.*]] = memref.alloc
  // CHECK-SAME: 32x6xf32
  // CHECK: annotation.mark %[[DST0]]
  hivm.hir.vpad ins(%src0: memref<32x4xf32, #hivm.address_space<ub>>)
                outs(%dst0: memref<32x6xf32, #hivm.address_space<ub>>)
                low[0, 2]
                high[0, 0]
                pad_value %pad_value : f32

  %src1 = memref.alloc() : memref<4x5xf32, #hivm.address_space<ub>>
  %dst1 = memref.alloc() : memref<6x5xf32, #hivm.address_space<ub>>

  // CHECK: %[[SRC1:.*]] = memref.alloc
  // CHECK-SAME: 4x5xf32
  // CHECK: annotation.mark %[[SRC1]]
  // CHECK: %[[DST1:.*]] = memref.alloc
  // CHECK-SAME: 6x5xf32
  // CHECK: annotation.mark %[[DST1]]
  hivm.hir.vpad ins(%src1: memref<4x5xf32, #hivm.address_space<ub>>)
                  outs(%dst1: memref<6x5xf32, #hivm.address_space<ub>>)
                  low[2, 0]
                  high[0, 0]
                  pad_value %pad_value : f32
  return
}

// -----

// CHECK-LABEL: func @single_cube
// CHECK-NOT: annotation.mark
func.func @single_cube(%arg0: memref<?x4096xf32, #hivm.address_space<gm>>, %arg1: memref<6144x4096xf32, #hivm.address_space<gm>>, %arg2: memref<?x6144xf32, #hivm.address_space<gm>>, %arg3: memref<12xi64, #hivm.address_space<gm>>) {
  hivm.hir.mix_matmul ins(%arg0, %arg1 : memref<?x4096xf32, #hivm.address_space<gm>>, memref<6144x4096xf32, #hivm.address_space<gm>>)
                  outs(%arg2 : memref<?x6144xf32, #hivm.address_space<gm>>)
                  tiling_params = %arg3 : memref<12xi64, #hivm.address_space<gm>>
                  b_transpose
  return
}

// -----

// CHECK-LABEL: func @test_deinterleave_op_2d_continuous
func.func @test_deinterleave_op_2d_continuous(
                       %src0: memref<2x3x2xf32, strided<[15, 4, 2], offset: ?>, #hivm.address_space<ub>>,
                       %dst0: memref<2x3x2xf32, strided<[15, 2, 1], offset: ?>, #hivm.address_space<ub>>) {

  // CHECK-SAME: %[[SRC0:.*]]: memref<2x3x2xf32, strided<[15, 4, 2], offset: ?>, #hivm.address_space<ub>>
  // CHECK-SAME: %[[DST0:.*]]: memref<2x3x2xf32, strided<[15, 2, 1], offset: ?>, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[DST0]]
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  // CHECK: annotation.mark %[[SRC0]]
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  hivm.hir.vdeinterleave ins(%src0: memref<2x3x2xf32, strided<[15, 4, 2], offset: ?>, #hivm.address_space<ub>>)
                         outs(%dst0: memref<2x3x2xf32, strided<[15, 2, 1], offset: ?>, #hivm.address_space<ub>>)
                         channel_num = 2
                         index_mode = <CHANNEL_0>
  return
}

// -----
// CHECK-LABEL: func @test_deinterleave_op_2d_uncontinuous
func.func @test_deinterleave_op_2d_uncontinuous(
                       %src0: memref<2x3x2xf32, strided<[31, 15, 2], offset: ?>, #hivm.address_space<ub>>,
                       %dst0: memref<2x3x2xf32, strided<[31, 15, 1], offset: ?>, #hivm.address_space<ub>>) {

  // CHECK-SAME: %[[SRC0:.*]]: memref<2x3x2xf32, strided<[31, 15, 2], offset: ?>, #hivm.address_space<ub>>
  // CHECK-SAME: %[[DST0:.*]]: memref<2x3x2xf32, strided<[31, 15, 1], offset: ?>, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[DST0]]
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
  // CHECK: annotation.mark %[[SRC0]]
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
  hivm.hir.vdeinterleave ins(%src0: memref<2x3x2xf32, strided<[31, 15, 2], offset: ?>, #hivm.address_space<ub>>)
                         outs(%dst0: memref<2x3x2xf32, strided<[31, 15, 1], offset: ?>, #hivm.address_space<ub>>)
                         channel_num = 2
                         index_mode = <CHANNEL_0>
  return
}

// -----

// CHECK-LABEL: func @test_interleave_op_continuous
func.func @test_interleave_op_continuous() {
  // CHECK-NOT: annotation.mark

  %alloc = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<2x30xf16, #hivm.address_space<ub>>
  %alloc_2 = memref.alloc() : memref<160xf16, #hivm.address_space<ub>>
  hivm.hir.vinterleave ins(%alloc, %alloc_0 : memref<2x15xf16, #hivm.address_space<ub>>, memref<2x15xf16, #hivm.address_space<ub>>)
                       outs(%alloc_1 : memref<2x30xf16, #hivm.address_space<ub>>)
                       interleave_channel_nums = 2
                       temp_buffer(%alloc_2 : memref<160xf16, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func @test_interleave_op_uncontinuous
func.func @test_interleave_op_uncontinuous(
                %src0 : memref<2x15xf16, strided<[17, 1], offset: ?>, #hivm.address_space<ub>>,
                %src1 : memref<2x15xf16, strided<[17, 1], offset: ?>, #hivm.address_space<ub>>,
                %dst0 : memref<2x30xf16, strided<[33, 1], offset: ?>, #hivm.address_space<ub>>) {
  // CHECK: annotation.mark
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  // CHECK: annotation.mark
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  // CHECK: annotation.mark
  // CHECK-SAME: {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  // CHECK: memref.alloc
  // CHECK-NOT: annotation.mark
  %tmp = memref.alloc() : memref<160xf16, #hivm.address_space<ub>>
  hivm.hir.vinterleave ins(%src0, %src1 : memref<2x15xf16, strided<[17, 1], offset: ?>, #hivm.address_space<ub>>, memref<2x15xf16, strided<[17, 1], offset: ?>, #hivm.address_space<ub>>)
                       outs(%dst0 : memref<2x30xf16, strided<[33, 1], offset: ?>, #hivm.address_space<ub>>)
                       interleave_channel_nums = 2
                       temp_buffer(%tmp : memref<160xf16, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func @vector_gather_unalign
func.func @vector_gather_unalign(%valueA: memref<7x10xf16, #hivm.address_space<gm>>,
          %valueB: memref<7x1xi32, #hivm.address_space<gm>>,
          %valueC: memref<7x1xf16, #hivm.address_space<gm>>)
          attributes {hacc.entry}
{
  %ubA = memref.alloc() : memref<7x10xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueA : memref<7x10xf16, #hivm.address_space<gm>>)
                outs(%ubA : memref<7x10xf16, #hivm.address_space<ub>>)

  %ubB = memref.alloc() : memref<7x1xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueB : memref<7x1xi32, #hivm.address_space<gm>>)
                outs(%ubB : memref<7x1xi32, #hivm.address_space<ub>>)

  %ubC = memref.alloc() : memref<7x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<7x10xf16,
  // CHECK: annotation.mark  %[[ALLOC]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}

  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<7x1xi32,
  // CHECK: annotation.mark  %[[ALLOC]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
  hivm.hir.vgather ins(%ubA: memref<7x10xf16, #hivm.address_space<ub>>)
                   indices(%ubB : memref<7x1xi32, #hivm.address_space<ub>>)
                   outs(%ubC : memref<7x1xf16, #hivm.address_space<ub>>)

  hivm.hir.store ins(%ubC : memref<7x1xf16, #hivm.address_space<ub>>)
                 outs(%valueC : memref<7x1xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: func @vector_gather_align
// CHECK-NOT: annotation.mark
func.func @vector_gather_align(%valueA: memref<7x32xf16, #hivm.address_space<gm>>,
          %valueB: memref<7x8xi32, #hivm.address_space<gm>>,
          %valueC: memref<7x16xf16, #hivm.address_space<gm>>)
          attributes {hacc.entry}
{
  %ubA = memref.alloc() : memref<7x32xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueA : memref<7x32xf16, #hivm.address_space<gm>>)
                outs(%ubA : memref<7x32xf16, #hivm.address_space<ub>>)

  %ubB = memref.alloc() : memref<7x8xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueB : memref<7x8xi32, #hivm.address_space<gm>>)
                outs(%ubB : memref<7x8xi32, #hivm.address_space<ub>>)

  %ubC = memref.alloc() : memref<7x16xf16, #hivm.address_space<ub>>

  hivm.hir.vgather ins(%ubA: memref<7x32xf16, #hivm.address_space<ub>>)
                   indices(%ubB : memref<7x8xi32, #hivm.address_space<ub>>)
                   outs(%ubC : memref<7x16xf16, #hivm.address_space<ub>>)

  hivm.hir.store ins(%ubC : memref<7x16xf16, #hivm.address_space<ub>>)
                 outs(%valueC : memref<7x16xf16, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func @test_cumsum_alignment
// CHECK-NOT: annotation.mark
func.func @test_cumsum_alignment(%arg0: memref<5x3x3x4x2x5xi32, #hivm.address_space<gm>>, %arg1: memref<5x3x3x4x2x5xi32, #hivm.address_space<gm>>, %arg2: memref<5x3x3x4x2x5xi32, #hivm.address_space<gm>>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %alloc = memref.alloc() : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<5x3x3x4x2x5xi32, #hivm.address_space<gm>>) outs(%alloc : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>)
  %alloc_0 = memref.alloc() : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>
  hivm.hir.vcumsum ins(%alloc : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>) cum_dims = [2] reverse = false
  %alloc_1 = memref.alloc() : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>
  hivm.hir.vcumprod ins(%alloc : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>) cum_dims = [2] reverse = false
  hivm.hir.store ins(%alloc_0 : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>) outs(%arg1 : memref<5x3x3x4x2x5xi32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%alloc_1 : memref<5x3x3x4x2x5xi32, #hivm.address_space<ub>>) outs(%arg2 : memref<5x3x3x4x2x5xi32, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func @test_cumsum_unalignment
// CHECK: annotation.mark
// CHECK-SAME: {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>}
func.func @test_cumsum_unalignment(%arg0: memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>, %arg1: memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>, %arg2: memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %alloc = memref.alloc() : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>) outs(%alloc : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>)
  %alloc_0 = memref.alloc() : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  hivm.hir.vcumsum ins(%alloc : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) cum_dims = [2] reverse = false
  %alloc_1 = memref.alloc() : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  hivm.hir.vcumprod ins(%alloc : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) cum_dims = [2] reverse = false
  hivm.hir.store ins(%alloc_0 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%arg1 : memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%alloc_1 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%arg2 : memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func @test_vtranspose_unalignment
// CHECK: annotation.mark
// CHECK-SAME: {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
func.func @test_vtranspose_unalignment(%arg0: memref<1x2x1x3xf32, #hivm.address_space<ub>>, %arg1: memref<1x2x1x3xf32, #hivm.address_space<ub>>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.vtranspose ins(%arg0 : memref<1x2x1x3xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<1x2x1x3xf32, #hivm.address_space<ub>>) permutation = [2, 1, 0, 3]
  return
}
