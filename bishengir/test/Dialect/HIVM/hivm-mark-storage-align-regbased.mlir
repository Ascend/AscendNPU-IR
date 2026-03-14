// RUN: bishengir-opt %s -hivm-mark-stride-align -split-input-file | FileCheck %s -check-prefix=A5

// -----

// A5-LABEL: func.func @test_non_i1_vf_no_align()
// A5-NOT: annotation.mark
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @vf_non_i1(%arg0: memref<3x741xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [3, 64] : vector<3x64xi1>
  %1 = vector.transfer_read %arg0[%c0, %c0], %cst, %0 {in_bounds = [true, true]} : memref<3x741xf32, #hivm.address_space<ub>>, vector<3x64xf32>
  vector.transfer_write %1, %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : vector<3x64xf32>, memref<3x741xf32, #hivm.address_space<ub>>
  return
}
func.func @test_non_i1_vf_no_align() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x741xf32, #hivm.address_space<ub>>
  func.call @vf_non_i1(%alloc) {hivm.vector_function, no_inline} : (memref<3x741xf32, #hivm.address_space<ub>>) -> ()
  return
}
}

// -----

// A5-LABEL: func @mark_fixpipe
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @mark_fixpipe() {
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<9x4x16x16xf32, #hivm.address_space<cc>>
  // A5-NOT: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>}
  %alloc_1 = memref.alloc() : memref<36x256xf32, #hivm.address_space<ub>>
  // A5-NOT: annotation.mark %alloc_1 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
  hivm.hir.fixpipe ins(%alloc_0 : memref<9x4x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_1 : memref<36x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
  return
}
}

// -----

// A5-LABEL: func.func @test_skip_aic_mark(
// A5-NOT: hivm.stride_align_dims
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_skip_aic_mark(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<?xf32, #hivm.address_space<gm>>, %arg2: i32, %arg3: i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c32 = arith.constant 32 : index
  %true = arith.constant true
  %false = arith.constant false
  %cst = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = arith.index_cast %arg2 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32], strides: [%0, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %1 = arith.index_cast %arg3 : i32 to index
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 16], strides: [%1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<16x32xf32, #hivm.address_space<cbuf>>
  %subview = memref.subview %reinterpret_cast[0, 0] [%c16, 32] [1, 1] : memref<16x32xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %alloc[0, 0] [%c16, 32] [1, 1] : memref<16x32xf32, #hivm.address_space<cbuf>> to memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
  hivm.hir.load ins(%subview : memref<?x32xf32, strided<[?, 1]>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %false : i1 eviction_policy = <EvictFirst>
  %alloc_1 = memref.alloc() : memref<32x16xf32, #hivm.address_space<cbuf>>
  %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [32, %c16] [1, 1] : memref<32x16xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<32x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %subview_2 = memref.subview %alloc_1[0, 0] [32, %c16] [1, 1] : memref<32x16xf32, #hivm.address_space<cbuf>> to memref<32x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>
  hivm.hir.load ins(%subview_1 : memref<32x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>) outs(%subview_2 : memref<32x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %false : i1 eviction_policy = <EvictFirst>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%alloc, %alloc_1, %true, %c16, %c32, %c16 : memref<16x32xf32, #hivm.address_space<cbuf>>, memref<32x16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_2 : memref<16x16xf32, #hivm.address_space<cc>>)
  %subview_3 = memref.subview %alloc_2[0, 0] [%c16, %c16] [1, 1] : memref<16x16xf32, #hivm.address_space<cc>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cc>>
  %alloc_3 = memref.alloc(%c16, %c16) : memref<?x?xf32, #hivm.address_space<ub>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%subview_3 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cc>>) outs(%alloc_3 : memref<?x?xf32, #hivm.address_space<ub>>)
  return
}
}

// -----

// A5-LABEL: func.func @test_cumsum_a5_last_dim_unaligned(
// A5-SAME: %[[SRC:.*]]: memref<2x3xi32, #hivm.address_space<ub>>,
// A5-SAME: %[[DST:.*]]: memref<2x3xi32, #hivm.address_space<ub>>)
// A5: annotation.mark %[[DST]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
// A5: annotation.mark %[[SRC]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_cumsum_a5_last_dim_unaligned(
    %src: memref<2x3xi32, #hivm.address_space<ub>>,
    %dst: memref<2x3xi32, #hivm.address_space<ub>>) {
  hivm.hir.vcumsum ins(%src : memref<2x3xi32, #hivm.address_space<ub>>)
    outs(%dst : memref<2x3xi32, #hivm.address_space<ub>>)
    cum_dims = [0] reverse = false
  return
}
}

// -----

// A5-LABEL: func.func @test_cumprod_a5_last_dim_unaligned(
// A5-SAME: %[[SRC:.*]]: memref<2x3xi32, #hivm.address_space<ub>>,
// A5-SAME: %[[DST:.*]]: memref<2x3xi32, #hivm.address_space<ub>>)
// A5: annotation.mark %[[DST]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
// A5: annotation.mark %[[SRC]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_cumprod_a5_last_dim_unaligned(
    %src: memref<2x3xi32, #hivm.address_space<ub>>,
    %dst: memref<2x3xi32, #hivm.address_space<ub>>) {
  hivm.hir.vcumprod ins(%src : memref<2x3xi32, #hivm.address_space<ub>>)
    outs(%dst : memref<2x3xi32, #hivm.address_space<ub>>)
    cum_dims = [0] reverse = false
  return
}
}
