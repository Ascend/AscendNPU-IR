// Off-registry: the whitelist already rejects cross-WI tensor carry.
// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true set-depth-in-unroll-mode=2" -allow-unregistered-dialect -split-input-file %s | FileCheck %s
// Registered-heuristic path (3530 same-stage): every work item is LCD, so
// pipelining is still rejected — there is no independent stage to skew against.
// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true set-depth-in-unroll-mode=2 bypass-shape-registry=true" -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Regression test (extracted from a gated-delta-product kernel): the inner
// reduction loop carries %arg15 across iterations. The carried tensor is
// consumed at the top of the body (hivm.hir.store of an extract_slice and a
// vtranspose chain feeding both mmadL1 ops) and produced at the bottom
// (vadd). The transitive consumer chain therefore spans every candidate
// work item, leaving no LCD-free stage to skew against. Pipelining must be
// rejected: a skewed consumer stage would read the carried tensor several
// iterations stale and corrupt both matmuls.

// CHECK-LABEL: func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64
// CHECK-NOT: scope.scope
// CHECK: scf.for %{{.*}} = %c0_i32 to %13 step %c1_i32 iter_args(%{{.*}} = %35) -> (tensor<64x64xf32>)
// CHECK-NOT: scope.scope
// CHECK: hivm.hir.store
// CHECK-NOT: scope.scope
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: scope.scope
// CHECK: hivm.hir.vadd
// CHECK-NOT: scope.scope
// CHECK: scf.yield %{{.*}} : tensor<64x64xf32>
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i32 = arith.constant 0 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c63_i32 = arith.constant 63 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant -1.000000e+00 : f32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c28_i32 = arith.constant 28 : i32
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    scf.for %arg13 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg13, %arg10 : i32
      %5 = arith.divsi %arg13, %arg10 : i32
      %6 = arith.remsi %5, %arg11 : i32
      %7 = arith.extsi %6 : i32 to i64
      %8 = arith.divsi %7, %c2_i64 : i64
      %9 = arith.remsi %7, %c2_i64 : i64
      %10 = arith.extsi %arg9 : i32 to i64
      %11 = arith.muli %8, %10 : i64
      %12 = arith.addi %arg9, %c63_i32 : i32
      %13 = arith.divsi %12, %c64_i32 : i32
      %14 = arith.extsi %13 : i32 to i64
      %15 = arith.muli %8, %14 : i64
      %16 = arith.muli %15, %c2_i64 : i64
      %17 = arith.addi %16, %9 : i64
      %18 = arith.muli %17, %c4096_i64 : i64
      %19 = arith.muli %11, %c2_i64 : i64
      %20 = arith.addi %19, %9 : i64
      %21 = arith.muli %20, %c64_i64 : i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = arith.muli %7, %c4096_i64 : i64
      %24 = arith.muli %4, %c64_i32 : i32
      %25 = arith.index_cast %23 : i64 to index
      %26 = arith.index_cast %24 : i32 to index
      %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%27], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %28 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%26]
      %29 = arith.maxsi %26, %c64 : index
      %30 = arith.minsi %28, %29 : index
      %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %26]
      %32 = arith.minsi %31, %c64 : index
      %33 = arith.maxsi %32, %c0 : index
      %34 = arith.cmpi slt, %33, %c64 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [64, %33] [1, 1] : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x?xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %memspacecast[0, 0] [64, %33] [1, 1] : memref<64x64xf32> to memref<64x?xf32, strided<[64, 1]>>
      hivm.hir.load ins(%subview : memref<64x?xf32, strided<[64, 1], offset: ?>>) outs(%subview_1 : memref<64x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 init_out_buffer = true init_condition = %34 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %35 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
      %36 = scf.for %arg14 = %c0_i32 to %13 step %c1_i32 iter_args(%arg15 = %35) -> (tensor<64x64xf32>)  : i32 {
        %37 = arith.extsi %arg14 : i32 to i64
        %38 = arith.muli %37, %c8192_i64 : i64
        %39 = arith.addi %18, %38 : i64
        %40 = arith.index_cast %39 : i64 to index
        %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%40, %26]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        %extracted_slice_5 = tensor.extract_slice %arg15[0, 0] [64, %33] [1, 1] : tensor<64x64xf32> to tensor<64x?xf32>
        %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [64, %33] [1, 1] : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x?xf32, strided<[64, 1], offset: ?>>
        hivm.hir.store ins(%extracted_slice_5 : tensor<64x?xf32>) outs(%subview_6 : memref<64x?xf32, strided<[64, 1], offset: ?>>)
        %42 = arith.muli %37, %c64_i64 : i64
        %43 = arith.index_cast %42 : i64 to index
        %44 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%22, %43]
        %45 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%26, %22, %43]
        %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%45], sizes: [64, 64], strides: [128, 1] : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [64, 64], strides: [128, 1] : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        %46 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%43]
        %47 = arith.index_cast %arg9 : i32 to index
        %48 = arith.maxsi %43, %47 : index
        %49 = arith.minsi %46, %48 : index
        %50 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%49, %43]
        %51 = arith.minsi %50, %c64 : index
        %52 = arith.maxsi %51, %c0 : index
        %53 = arith.cmpi slt, %52, %c64 : index
        %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [%52, 64] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x64xf32, strided<[128, 1], offset: ?>>
        %54 = affine.apply affine_map<(d0) -> (d0 ceildiv 16)>(%52)
        %alloc_10 = memref.alloc() : memref<8x4x16x8xf32>
        %subview_11 = memref.subview %alloc_10[0, 0, 0, 0] [8, %54, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32> to memref<8x?x16x8xf32, strided<[512, 128, 8, 1]>>
        hivm.hir.nd2nz {dst_continuous} ins(%subview_9 : memref<?x64xf32, strided<[128, 1], offset: ?>>) outs(%subview_11 : memref<8x?x16x8xf32, strided<[512, 128, 8, 1]>>) init_out_buffer = true pad_value = %cst : f32 init_condition = %53 : i1
        %55 = bufferization.to_tensor %alloc_10 restrict writable : memref<8x4x16x8xf32>
        %expanded = tensor.expand_shape %arg15 [[0], [1, 2]] output_shape [64, 8, 8] : tensor<64x64xf32> into tensor<64x8x8xf32>
        %56 = tensor.empty() : tensor<8x64x8xf32>
        %57 = hivm.hir.vtranspose ins(%expanded : tensor<64x8x8xf32>) outs(%56 : tensor<8x64x8xf32>) permutation = [1, 0, 2] -> tensor<8x64x8xf32>
        %expanded_12 = tensor.expand_shape %57 [[0], [1, 2], [3]] output_shape [8, 4, 16, 8] : tensor<8x64x8xf32> into tensor<8x4x16x8xf32>
        %alloc_13 = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %58 = bufferization.to_tensor %alloc_13 restrict writable : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_12 : tensor<8x4x16x8xf32>) outs(%alloc_13 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %59 = tensor.empty() : tensor<4x4x16x16xf32>
        %60 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%55, %58, %true, %c64, %c64, %c64 : tensor<8x4x16x8xf32>, tensor<8x4x16x8xf32>, i1, index, index, index) outs(%59 : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
        %alloc_14 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x64xf32, #hivm.address_space<ub>>
        %memspacecast_15 = memref.memory_space_cast %alloc_14 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %61 = bufferization.to_tensor %memspacecast_15 restrict writable : memref<64x64xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%60 : tensor<4x4x16x16xf32>) outs(%alloc_14 : memref<64x64xf32, #hivm.address_space<ub>>)
        %62 = tensor.empty() : tensor<64x64xf32>
        %63 = hivm.hir.vmul ins(%61, %cst_0 : tensor<64x64xf32>, f32) outs(%62 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %alloc_16 = memref.alloc() : memref<64x64xf32>
        %64 = arith.ori %53, %34 : i1
        %subview_17 = memref.subview %reinterpret_cast_7[0, 0] [%52, %33] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        %subview_18 = memref.subview %alloc_16[0, 0] [%52, %33] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        hivm.hir.load ins(%subview_17 : memref<?x?xf32, strided<[128, 1], offset: ?>>) outs(%subview_18 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 init_out_buffer = true init_condition = %64 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %65 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x64xf32>
        %66 = hivm.hir.vadd ins(%63, %65 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%62 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %reinterpret_cast_19 = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [64, 64], strides: [128, 1] : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        %extracted_slice_20 = tensor.extract_slice %66[0, 0] [%52, %33] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
        %subview_21 = memref.subview %reinterpret_cast_19[0, 0] [%52, %33] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        hivm.hir.store ins(%extracted_slice_20 : tensor<?x?xf32>) outs(%subview_21 : memref<?x?xf32, strided<[128, 1], offset: ?>>)
        %reinterpret_cast_22 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [128, 1] : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        %subview_23 = memref.subview %reinterpret_cast_22[0, 0] [%52, 64] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x64xf32, strided<[128, 1], offset: ?>>
        %alloc_24 = memref.alloc() : memref<8x4x16x8xf32>
        %subview_25 = memref.subview %alloc_24[0, 0, 0, 0] [8, %54, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32> to memref<8x?x16x8xf32, strided<[512, 128, 8, 1]>>
        hivm.hir.nd2nz {dst_continuous} ins(%subview_23 : memref<?x64xf32, strided<[128, 1], offset: ?>>) outs(%subview_25 : memref<8x?x16x8xf32, strided<[512, 128, 8, 1]>>) init_out_buffer = true pad_value = %cst : f32 init_condition = %53 : i1
        %67 = bufferization.to_tensor %alloc_24 restrict writable : memref<8x4x16x8xf32>
        %expanded_26 = tensor.expand_shape %66 [[0], [1, 2]] output_shape [64, 8, 8] : tensor<64x64xf32> into tensor<64x8x8xf32>
        %68 = hivm.hir.vtranspose ins(%expanded_26 : tensor<64x8x8xf32>) outs(%56 : tensor<8x64x8xf32>) permutation = [1, 0, 2] -> tensor<8x64x8xf32>
        %expanded_27 = tensor.expand_shape %68 [[0], [1, 2], [3]] output_shape [8, 4, 16, 8] : tensor<8x64x8xf32> into tensor<8x4x16x8xf32>
        %alloc_28 = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %alloc_28 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %69 = bufferization.to_tensor %alloc_28 restrict writable : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_27 : tensor<8x4x16x8xf32>) outs(%alloc_28 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %70 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%67, %69, %true, %c64, %c64, %c64 : tensor<8x4x16x8xf32>, tensor<8x4x16x8xf32>, i1, index, index, index) outs(%59 : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
        %alloc_29 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_29 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x64xf32, #hivm.address_space<ub>>
        %memspacecast_30 = memref.memory_space_cast %alloc_29 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %71 = bufferization.to_tensor %memspacecast_30 restrict writable : memref<64x64xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true} ins(%70 : tensor<4x4x16x16xf32>) outs(%alloc_29 : memref<64x64xf32, #hivm.address_space<ub>>)
        %72 = hivm.hir.vadd ins(%71, %arg15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%62 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %72 : tensor<64x64xf32>
      } {fixpipe_for_mmad_result_already_inserted = true}
      %reinterpret_cast_2 = memref.reinterpret_cast %arg8 to offset: [%27], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %36[0, 0] [64, %33] [1, 1] : tensor<64x64xf32> to tensor<64x?xf32>
      %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [64, %33] [1, 1] : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x?xf32, strided<[64, 1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<64x?xf32>) outs(%subview_3 : memref<64x?xf32, strided<[64, 1], offset: ?>>)
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    return
  }
}
