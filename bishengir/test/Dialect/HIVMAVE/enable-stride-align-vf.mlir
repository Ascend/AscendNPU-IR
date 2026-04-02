// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-mark-stride-align -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL: @triton_kernel_outlined_vf_0
// CHECK: %[[ARG0:.*]]: memref<3x741xf32
// CHECK: %[[ARG1:.*]]: memref<3x741xf32
// CHECK: %[[ARG2:.*]]: memref<3x741xf32

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_kernel_outlined_vf_0(%arg0: memref<3x741xf32, #hivm.address_space<ub>>, %arg1: memref<3x741xf32, #hivm.address_space<ub>>, %arg2: memref<3x741xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c741 = arith.constant 741 : index
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c741 step %c64 {
        %0 = affine.min affine_map<(d0) -> (-d0 + 741, 64)>(%arg4)
        %subview = memref.subview %arg0[%arg3, %arg4] [1, %0] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg3, %arg4] [1, %0] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg3, %arg4] [1, %0] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
        %2 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %3 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %4 = arith.addf %2, %3 : vector<1x64xf32>
        vector.transfer_write %4, %subview_1[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @triton_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c741 = arith.constant 741 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg8, %arg7 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg6 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.index_cast %arg4 : i32 to index
    %8 = arith.maxsi %6, %c0 : index
    %9 = arith.minsi %8, %c741 : index
    %10 = arith.minsi %9, %c741 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x741xf32, #hivm.address_space<ub>>
    scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %11 = arith.muli %arg9, %c3_i32 : i32
      %12 = arith.addi %5, %11 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%13, %6]
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [3, 741], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_0 = memref.alloc() : memref<3x741xf32, #hivm.address_space<ub>>
      %15 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%13]
      %16 = arith.maxsi %13, %7 : index
      %17 = arith.minsi %15, %16 : index
      %18 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%17, %13]
      %19 = arith.minsi %18, %c3 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %alloc_0[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [3, 741], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_3 = memref.alloc() : memref<3x741xf32, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %alloc_3[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index
      func.call @triton_kernel_outlined_vf_0(%alloc_0, %alloc_3, %alloc) {hivm.vector_function} : (memref<3x741xf32, #hivm.address_space<ub>>, memref<3x741xf32, #hivm.address_space<ub>>, memref<3x741xf32, #hivm.address_space<ub>>) -> ()
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [3, 741], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %alloc[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>
      %subview_8 = memref.subview %reinterpret_cast_6[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%subview_7 : memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @dot_mask_exp_kernel_mix_aic
// CHECK: %[[DYN:.*]] = memref.alloc(%[[M:.*]], %[[N:.*]]) : memref<?x?xf32, #hivm.address_space<ub>>
// CHECK-NOT: memref<?x?x1xf32, #hivm.address_space<ub>>
// CHECK-NOT: annotation.mark %[[DYN]] {hivm.stride_align_dims = array<i32: 1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dot_mask_exp_kernel_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 32], strides: [%0, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %1 = arith.index_cast %arg8 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 16], strides: [%1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<16x32xf32, #hivm.address_space<cbuf>>
    %2 = arith.index_cast %arg5 : i32 to index
    %3 = arith.maxsi %2, %c0 : index
    %4 = arith.minsi %3, %c16 : index
    %5 = arith.cmpi slt, %4, %c16 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%4, 32] [1, 1] : memref<16x32xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %alloc[0, 0] [%4, 32] [1, 1] : memref<16x32xf32, #hivm.address_space<cbuf>> to memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%subview : memref<?x32xf32, strided<[?, 1]>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %5 : i1 eviction_policy = <EvictFirst>
    %alloc_1 = memref.alloc() : memref<32x16xf32, #hivm.address_space<cbuf>>
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.maxsi %6, %c0 : index
    %8 = arith.minsi %7, %c16 : index
    %9 = arith.cmpi slt, %8, %c16 : index
    %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [32, %8] [1, 1] : memref<32x16xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<32x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %alloc_1[0, 0] [32, %8] [1, 1] : memref<32x16xf32, #hivm.address_space<cbuf>> to memref<32x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%subview_1 : memref<32x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>) outs(%subview_2 : memref<32x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %9 : i1 eviction_policy = <EvictFirst>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%alloc, %alloc_1, %true, %c16, %c32, %c16 : memref<16x32xf32, #hivm.address_space<cbuf>>, memref<32x16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_2 : memref<16x16xf32, #hivm.address_space<cc>>)
    %10 = arith.minsi %4, %c16 : index
    %11 = arith.minsi %8, %c16 : index
    %subview_3 = memref.subview %alloc_2[0, 0] [%10, %11] [1, 1] : memref<16x16xf32, #hivm.address_space<cc>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cc>>
    %alloc_3 = memref.alloc(%10, %11) : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {buffer_size_in_byte = 1024 : i64} : memref<?x?xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%subview_3 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cc>>) outs(%alloc_3 : memref<?x?xf32, #hivm.address_space<ub>>)
    return
  }
}
