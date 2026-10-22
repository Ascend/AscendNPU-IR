// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew bypass-shape-registry=true" -allow-unregistered-dialect %s | FileCheck %s


// CHECK-LABEL: func.func @parallel_nsa_compression_fwd_kernel(
// CHECK-LABEL:       scope.scope : () ->
// CHECK-LABEL:       } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 5 : i32, hivm.preload_num = 4 : i32, no_inline}
// CHECK-LABEL:       scope.scope : () ->
// CHECK-LABEL:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 5 : i32, hivm.preload_num = 3 : i32, no_inline}
// CHECK-LABEL:       scope.scope : () ->
// CHECK-LABEL:       } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 5 : i32, hivm.preload_num = 2 : i32, no_inline}
// CHECK-LABEL:       scope.scope : () ->
// CHECK-LABEL:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 5 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK-LABEL:       scope.scope : () ->
// CHECK-LABEL:       } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 5 : i32, hivm.preload_num = 0 : i32, no_inline}
// CHECK-LABEL:     } {fixpipe_for_mmad_result_already_inserted = true, hivm.cv_pipelined_loop}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @parallel_nsa_compression_fwd_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i64 = arith.constant 64 : i64
  %c32_i64 = arith.constant 32 : i64
  %c64_i32 = arith.constant 64 : i32
  %c16_i64 = arith.constant 16 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c1_i64 = arith.constant 1 : i64
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0_i32 = arith.constant 0 : i32
  %c2048_i64 = arith.constant 2048 : i64
  %c32_i32 = arith.constant 32 : i32
  %c31_i32 = arith.constant 31 : i32
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 0.000000e+00 : f16
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %c-2139095040_i32 = arith.constant -2139095040 : i32
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
    %7 = arith.divsi %arg13, %0 : i32
    %8 = arith.remsi %7, %arg12 : i32
    %9 = tensor.empty() : tensor<16x64xf32>
    %10 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %11 = tensor.empty() : tensor<16xf32>
    %12 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_0 : f32) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
    %13 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_0 : f32) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %14 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
    %15 = arith.extsi %4 : i32 to i64
    %16 = arith.extsi %8 : i32 to i64
    %17 = arith.divsi %16, %c2_i64 : i64
    %18 = arith.remsi %16, %c2_i64 : i64
    %19 = arith.extsi %arg8 : i32 to i64
    %20 = arith.muli %17, %19 : i64
    %21 = arith.addi %arg9, %c31_i32 : i32
    %22 = arith.divsi %21, %c32_i32 : i32
    %23 = arith.extsi %22 : i32 to i64
    %24 = arith.muli %17, %23 : i64
    %25 = arith.muli %18, %c16_i64 : i64
    %26 = arith.addi %20, %15 : i64
    %27 = arith.muli %26, %c2048_i64 : i64
    %28 = arith.index_cast %27 : i64 to index
    %29 = arith.index_cast %25 : i64 to index
    %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%28, %29]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1], offset: ?>>
    %31 = arith.subi %arg9, %arg8 : i32
    %alloc = memref.alloc() : memref<16x64xf16>
    %32 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%29]
    %33 = arith.maxsi %29, %c32 : index
    %34 = arith.minsi %32, %33 : index
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%34, %29]
    %36 = arith.minsi %35, %c16 : index
    %37 = arith.maxsi %36, %c0 : index
    %38 = arith.cmpi slt, %37, %c16 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%37, 64] [1, 1] : memref<16x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0] [%37, 64] [1, 1] : memref<16x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    hivm.hir.load ins(%subview : memref<?x64xf16, strided<[64, 1], offset: ?>>) outs(%subview_2 : memref<?x64xf16, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst_1 : f16 init_out_buffer = true init_condition = %38 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
    %39 = bufferization.to_tensor %alloc restrict writable : memref<16x64xf16>
    %40 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%39 : tensor<16x64xf16>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %41 = hivm.hir.vmul ins(%40, %arg7 : tensor<16x64xf32>, f32) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %42 = tensor.empty() : tensor<16x64xf16>
    %43 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%41 : tensor<16x64xf32>) outs(%42 : tensor<16x64xf16>) -> tensor<16x64xf16>
    %44 = arith.extsi %31 : i32 to i64
    %45 = arith.addi %15, %44 : i64
    %46 = arith.addi %45, %c1_i64 : i64
    %47 = arith.divsi %46, %c32_i64 : i64
    %48 = arith.muli %6, %c64_i32 : i32
    %49 = arith.index_cast %48 : i32 to index
    %50 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 64)>()[%49, %28, %29]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%50], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1], offset: ?>>
    %51 = arith.muli %24, %c2_i64 : i64
    %52 = arith.addi %51, %18 : i64
    %53 = arith.muli %52, %c64_i64 : i64
    %54 = arith.index_cast %53 : i64 to index
    %55:3 = scf.for %arg14 = %c0_i64 to %47 step %c64_i64 iter_args(%arg15 = %10, %arg16 = %12, %arg17 = %14) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i64 {
      %66 = arith.index_cast %arg14 : i64 to index
      %67 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%54, %66]
      %68 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%49, %54, %66]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [64, 64], strides: [128, 1] : memref<?xf16> to memref<64x64xf16, strided<[128, 1], offset: ?>>
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [64, 64], strides: [128, 1] : memref<?xf16> to memref<64x64xf16, strided<[128, 1], offset: ?>>
      %69 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%66]
      %70 = arith.index_cast %22 : i32 to index
      %71 = arith.maxsi %66, %70 : index
      %72 = arith.minsi %69, %71 : index
      %73 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%72, %66]
      %74 = arith.minsi %73, %c64 : index
      %75 = arith.maxsi %74, %c0 : index
      %76 = arith.cmpi slt, %75, %c64 : index
      %subview_7 = memref.subview %reinterpret_cast_6[0, 0] [%75, 64] [1, 1] : memref<64x64xf16, strided<[128, 1], offset: ?>> to memref<?x64xf16, strided<[128, 1], offset: ?>>
      %77 = affine.apply affine_map<(d0) -> (d0 ceildiv 16)>(%75)
      %alloc_8 = memref.alloc() : memref<4x4x16x16xf16>
      %subview_9 = memref.subview %alloc_8[0, 0, 0, 0] [4, %77, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf16> to memref<4x?x16x16xf16, strided<[1024, 256, 16, 1]>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<?x64xf16, strided<[128, 1], offset: ?>>) outs(%subview_9 : memref<4x?x16x16xf16, strided<[1024, 256, 16, 1]>>) init_out_buffer = true pad_value = %cst_1 : f16 init_condition = %76 : i1
      %78 = bufferization.to_tensor %alloc_8 restrict writable : memref<4x4x16x16xf16>
      %79 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%49]
      %80 = arith.maxsi %49, %c64 : index
      %81 = arith.minsi %79, %80 : index
      %82 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%81, %49]
      %83 = arith.minsi %82, %c64 : index
      %84 = arith.maxsi %83, %c0 : index
      %85 = arith.cmpi slt, %84, %c64 : index
      %86 = arith.ori %76, %85 : i1
      %subview_10 = memref.subview %reinterpret_cast_5[0, 0] [%75, %84] [1, 1] : memref<64x64xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
      %87 = affine.apply affine_map<(d0) -> (d0 ceildiv 16)>(%84)
      %alloc_11 = memref.alloc() : memref<4x4x16x16xf16>
      %subview_12 = memref.subview %alloc_11[0, 0, 0, 0] [%87, %77, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf16> to memref<?x?x16x16xf16, strided<[1024, 256, 16, 1]>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_10 : memref<?x?xf16, strided<[128, 1], offset: ?>>) outs(%subview_12 : memref<?x?x16x16xf16, strided<[1024, 256, 16, 1]>>) init_out_buffer = true pad_value = %cst_1 : f16 init_condition = %86 : i1
      %88 = bufferization.to_tensor %alloc_11 restrict writable : memref<4x4x16x16xf16>
      %expanded = tensor.expand_shape %43 [[0], [1, 2]] output_shape [16, 4, 16] : tensor<16x64xf16> into tensor<16x4x16xf16>
      %89 = tensor.empty() : tensor<4x16x16xf16>
      %90 = hivm.hir.vtranspose ins(%expanded : tensor<16x4x16xf16>) outs(%89 : tensor<4x16x16xf16>) permutation = [1, 0, 2] -> tensor<4x16x16xf16>
      %expanded_13 = tensor.expand_shape %90 [[0], [1, 2], [3]] output_shape [4, 1, 16, 16] : tensor<4x16x16xf16> into tensor<4x1x16x16xf16>
      %alloc_14 = memref.alloc() : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      %91 = bufferization.to_tensor %alloc_14 restrict writable : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%expanded_13 : tensor<4x1x16x16xf16>) outs(%alloc_14 : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %92 = tensor.empty() : tensor<4x1x16x16xf32>
      %93 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%91, %78, %true, %c16, %c64, %c64 : tensor<4x1x16x16xf16>, tensor<4x4x16x16xf16>, i1, index, index, index) outs(%92 : tensor<4x1x16x16xf32>) -> tensor<4x1x16x16xf32>
      %alloc_15 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_15 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_15 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
      %94 = bufferization.to_tensor %memspacecast restrict writable : memref<16x64xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%93 : tensor<4x1x16x16xf32>) outs(%alloc_15 : memref<16x64xf32, #hivm.address_space<ub>>)
      %95 = arith.index_cast %47 : i64 to index
      %96 = arith.maxsi %66, %95 : index
      %97 = arith.minsi %69, %96 : index
      %98 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%97, %66]
      %extracted_slice_16 = tensor.extract_slice %94[0, 0] [16, %98] [1, 1] : tensor<16x64xf32> to tensor<16x?xf32>
      %inserted_slice = tensor.insert_slice %extracted_slice_16 into %13[0, 0] [16, %98] [1, 1] : tensor<16x?xf32> into tensor<16x64xf32>
      %99 = hivm.hir.bitcast %inserted_slice : tensor<16x64xf32> -> tensor<16x64xi32>
      %100 = tensor.empty() : tensor<16x64xi32>
      %101 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%c2147483647_i32 : i32) outs(%100 : tensor<16x64xi32>) -> tensor<16x64xi32>
      %102 = hivm.hir.vand ins(%99, %101 : tensor<16x64xi32>, tensor<16x64xi32>) outs(%100 : tensor<16x64xi32>) -> tensor<16x64xi32>
      %103 = hivm.hir.vadd ins(%102, %c-2139095040_i32 : tensor<16x64xi32>, i32) outs(%100 : tensor<16x64xi32>) -> tensor<16x64xi32>
      %104 = hivm.hir.vmin ins(%103, %c1_i32 : tensor<16x64xi32>, i32) outs(%103 : tensor<16x64xi32>) -> tensor<16x64xi32>
      %105 = hivm.hir.vrelu ins(%104 : tensor<16x64xi32>) outs(%104 : tensor<16x64xi32>) -> tensor<16x64xi32>
      %106 = hivm.hir.vcast {enable_overflow = false, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%105 : tensor<16x64xi32>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
      %107 = tensor.empty() : tensor<16x64xi1>
      %108 = hivm.hir.vcmp ins(%106, %cst : tensor<16x64xf32>, f32) outs(%107 : tensor<16x64xi1>) compare_mode = <ne> -> tensor<16x64xi1>
      %109 = hivm.hir.vsel ins(%108, %cst_0, %inserted_slice : tensor<16x64xi1>, f32, tensor<16x64xf32>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
      %expanded_17 = tensor.expand_shape %12 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %110 = hivm.hir.vreduce <max> ins(%109 : tensor<16x64xf32>) outs(%expanded_17 : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %collapsed = tensor.collapse_shape %110 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
      %111 = hivm.hir.bitcast %arg16 : tensor<16xf32> -> tensor<16xi32>
      %112 = tensor.empty() : tensor<16xi32>
      %113 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%c2147483647_i32 : i32) outs(%112 : tensor<16xi32>) -> tensor<16xi32>
      %114 = hivm.hir.vand ins(%111, %113 : tensor<16xi32>, tensor<16xi32>) outs(%112 : tensor<16xi32>) -> tensor<16xi32>
      %115 = hivm.hir.vadd ins(%114, %c-2139095040_i32 : tensor<16xi32>, i32) outs(%112 : tensor<16xi32>) -> tensor<16xi32>
      %116 = hivm.hir.vmin ins(%115, %c1_i32 : tensor<16xi32>, i32) outs(%115 : tensor<16xi32>) -> tensor<16xi32>
      %117 = hivm.hir.vmax ins(%116, %c0_i32 : tensor<16xi32>, i32) outs(%116 : tensor<16xi32>) -> tensor<16xi32>
      %118 = hivm.hir.vcast {enable_overflow = false, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%117 : tensor<16xi32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %119 = tensor.empty() : tensor<16xi1>
      %120 = hivm.hir.vcmp ins(%118, %cst : tensor<16xf32>, f32) outs(%119 : tensor<16xi1>) compare_mode = <ne> -> tensor<16xi1>
      %121 = hivm.hir.vsel ins(%120, %cst_0, %arg16 : tensor<16xi1>, f32, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %122 = hivm.hir.bitcast %collapsed : tensor<16xf32> -> tensor<16xi32>
      %123 = hivm.hir.vand ins(%122, %113 : tensor<16xi32>, tensor<16xi32>) outs(%112 : tensor<16xi32>) -> tensor<16xi32>
      %124 = hivm.hir.vadd ins(%123, %c-2139095040_i32 : tensor<16xi32>, i32) outs(%112 : tensor<16xi32>) -> tensor<16xi32>
      %125 = hivm.hir.vmin ins(%124, %c1_i32 : tensor<16xi32>, i32) outs(%124 : tensor<16xi32>) -> tensor<16xi32>
      %126 = hivm.hir.vrelu ins(%125 : tensor<16xi32>) outs(%125 : tensor<16xi32>) -> tensor<16xi32>
      %127 = hivm.hir.vcast {enable_overflow = false, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%126 : tensor<16xi32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %128 = hivm.hir.vcmp ins(%127, %cst : tensor<16xf32>, f32) outs(%119 : tensor<16xi1>) compare_mode = <ne> -> tensor<16xi1>
      %129 = hivm.hir.vsel ins(%128, %cst_0, %collapsed : tensor<16xi1>, f32, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %130 = hivm.hir.vmax ins(%121, %129 : tensor<16xf32>, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %131 = hivm.hir.vsub ins(%arg16, %130 : tensor<16xf32>, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %132 = hivm.hir.vexp ins(%131 : tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %expanded_18 = tensor.expand_shape %130 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %133 = hivm.hir.vsub ins(%inserted_slice, %expanded_18 : tensor<16x64xf32>, tensor<16x1xf32>) outs(%9 : tensor<16x64xf32>) broadcast = [1] -> tensor<16x64xf32>
      %134 = hivm.hir.vexp ins(%133 : tensor<16x64xf32>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
      %135 = hivm.hir.vmul ins(%arg17, %132 : tensor<16xf32>, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %expanded_19 = tensor.expand_shape %14 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %136 = hivm.hir.vreduce <sum> ins(%134 : tensor<16x64xf32>) outs(%expanded_19 : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %collapsed_20 = tensor.collapse_shape %136 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
      %137 = hivm.hir.vadd ins(%135, %collapsed_20 : tensor<16xf32>, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %expanded_21 = tensor.expand_shape %132 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %138 = hivm.hir.vmul ins(%arg15, %expanded_21 : tensor<16x64xf32>, tensor<16x1xf32>) outs(%9 : tensor<16x64xf32>) broadcast = [1] -> tensor<16x64xf32>
      %139 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%134 : tensor<16x64xf32>) outs(%42 : tensor<16x64xf16>) -> tensor<16x64xf16>
      %expanded_22 = tensor.expand_shape %139 [[0], [1, 2]] output_shape [16, 4, 16] : tensor<16x64xf16> into tensor<16x4x16xf16>
      %140 = hivm.hir.vtranspose ins(%expanded_22 : tensor<16x4x16xf16>) outs(%89 : tensor<4x16x16xf16>) permutation = [1, 0, 2] -> tensor<4x16x16xf16>
      %expanded_23 = tensor.expand_shape %140 [[0], [1, 2], [3]] output_shape [4, 1, 16, 16] : tensor<4x16x16xf16> into tensor<4x1x16x16xf16>
      %alloc_24 = memref.alloc() : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_24 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      %141 = bufferization.to_tensor %alloc_24 restrict writable : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%expanded_23 : tensor<4x1x16x16xf16>) outs(%alloc_24 : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %142 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%141, %88, %true, %c16, %c64, %c64 : tensor<4x1x16x16xf16>, tensor<4x4x16x16xf16>, i1, index, index, index) outs(%92 : tensor<4x1x16x16xf32>) -> tensor<4x1x16x16xf32>
      %alloc_25 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_25 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x64xf32, #hivm.address_space<ub>>
      %memspacecast_26 = memref.memory_space_cast %alloc_25 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
      %143 = bufferization.to_tensor %memspacecast_26 restrict writable : memref<16x64xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%142 : tensor<4x1x16x16xf32>) outs(%alloc_25 : memref<16x64xf32, #hivm.address_space<ub>>)
      %144 = hivm.hir.vadd ins(%143, %138 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
      scf.yield %144, %130, %137 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
    } {fixpipe_for_mmad_result_already_inserted = true}
    %56 = arith.cmpi eq, %47, %c0_i64 : i64
    %57:2 = scf.if %56 -> (tensor<16x64xf32>, tensor<16xf32>) {
      scf.yield %55#0, %14 : tensor<16x64xf32>, tensor<16xf32>
    } else {
      %expanded = tensor.expand_shape %55#2 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %66 = hivm.hir.vdiv ins(%55#0, %expanded : tensor<16x64xf32>, tensor<16x1xf32>) outs(%9 : tensor<16x64xf32>) broadcast = [1] -> tensor<16x64xf32>
      %67 = hivm.hir.vln ins(%55#2 : tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      %68 = hivm.hir.vadd ins(%55#1, %67 : tensor<16xf32>, tensor<16xf32>) outs(%11 : tensor<16xf32>) -> tensor<16xf32>
      scf.yield %66, %68 : tensor<16x64xf32>, tensor<16xf32>
    }
    %58 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%57#0 : tensor<16x64xf32>) outs(%42 : tensor<16x64xf16>) -> tensor<16x64xf16>
    %59 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%49]
    %60 = arith.maxsi %49, %c64 : index
    %61 = arith.minsi %59, %60 : index
    %62 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%61, %49]
    %63 = arith.minsi %62, %c64 : index
    %64 = arith.maxsi %63, %c0 : index
    %extracted_slice = tensor.extract_slice %58[0, 0] [%37, %64] [1, 1] : tensor<16x64xf16> to tensor<?x?xf16>
    %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%37, %64] [1, 1] : memref<16x64xf16, strided<[64, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview_4 : memref<?x?xf16, strided<[64, 1], offset: ?>>)
    %65 = arith.cmpi eq, %6, %c0_i32 : i32
    scf.if %65 {
      %66 = arith.muli %26, %c32_i64 : i64
      %67 = arith.addi %66, %25 : i64
      %68 = arith.index_cast %67 : i64 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%68], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%57#1 : tensor<16xf32>) outs(%reinterpret_cast_5 : memref<16xf32, strided<[1], offset: ?>>)
    }
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  return
}
}
