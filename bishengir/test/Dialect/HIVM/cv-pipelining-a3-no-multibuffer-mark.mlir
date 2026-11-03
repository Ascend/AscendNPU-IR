// RUN: bishengir-opt -cv-pipelining -allow-unregistered-dialect %s 2>&1 | FileCheck %s

// CHECK-LABEL: @chunk_kda_bwd_kernel_dA_post_left
// CHECK-NOT:  hivm.cv_pipelined_loop
#map = affine_map<()[s0, s1] -> (s0 + s1 * 16)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 16)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 16, 0)>
#map3 = affine_map<()[s0] -> (16, s0)>
#map4 = affine_map<()[s0] -> (-s0 + (s0 floordiv 16) * 16 + 16, 0)>
#map5 = affine_map<()[s0, s1] -> (16, s1, s0)>
#map6 = affine_map<()[s0, s1] -> (s0 - s1)>
#map7 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map8 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map9 = affine_map<()[s0, s1] -> (s1, s0)>
#map10 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map11 = affine_map<()[s0] -> (0, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.0.0">, hacc.target = #hacc.target<"Ascend910_9382">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @chunk_kda_bwd_kernel_dA_post_left(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c16_i32 = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant -1.000000e+00 : f32
    %true = arith.constant true
    %c24_i32 = arith.constant 24 : i32
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    scf.for %arg10 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %4 = arith.remsi %arg10, %arg7 : i32
      %5 = arith.divsi %arg10, %arg7 : i32
      %6 = arith.remsi %5, %arg8 : i32
      %7 = arith.divsi %arg10, %0 : i32
      %8 = arith.remsi %7, %arg9 : i32
      %9 = tensor.empty() : tensor<16x16xf16>
      %10 = hivm.hir.vbrc ins(%cst : f16) outs(%9 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %11 = tensor.empty() : tensor<16x16xf32>
      %12 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %13 = arith.muli %6, %arg6 : i32
      %14 = arith.muli %13, %c16_i32 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %8, %c16_i32 : i32
      %17 = tensor.empty() : tensor<16xi32>
      %18 = hivm.hir.varange offset[%c0] strides[%c1] outs(%17 : tensor<16xi32>) -> tensor<16xi32>
      %19 = hivm.hir.vadd ins(%18, %16 : tensor<16xi32>, i32) outs(%17 : tensor<16xi32>) -> tensor<16xi32>
      %20 = arith.muli %4, %c16_i32 : i32
      %21 = hivm.hir.vadd ins(%19, %20 : tensor<16xi32>, i32) outs(%17 : tensor<16xi32>) -> tensor<16xi32>
      %22 = hivm.hir.vadd ins(%18, %20 : tensor<16xi32>, i32) outs(%17 : tensor<16xi32>) -> tensor<16xi32>
      %23 = arith.maxsi %20, %c0_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 16)>()[%15, %24]
      %26 = arith.index_cast %arg6 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
      %27 = arith.maxsi %16, %c0_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 16)>()[%28, %15, %24]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x16xf16>
      %30 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 16, 0)>()[%26, %28, %24]
      %31 = affine.min affine_map<()[s0] -> (16, s0)>()[%30]
      %32 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 16) * 16 + 16, 0)>()[%28]
      %33 = affine.min affine_map<()[s0] -> (16, s0)>()[%32]
      %34 = arith.subi %c0_i32, %20 : i32
      %35 = arith.maxsi %34, %c0_i32 : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%36, %30]
      %38 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%31, %37]
      %39 = arith.subi %c0_i32, %16 : i32
      %40 = arith.maxsi %39, %c0_i32 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%41, %32]
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%33, %42]
      %44 = arith.cmpi slt, %38, %c16 : index
      %45 = arith.cmpi slt, %43, %c16 : index
      %46 = arith.ori %44, %45 : i1
      %subview = memref.subview %reinterpret_cast_2[0, 0] [%38, %43] [1, 1] : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<?x?xf16, strided<[16, 1], offset: ?>>
      %subview_3 = memref.subview %alloc[%37, %42] [%38, %43] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1], offset: ?>>
      %47 = arith.remui %42, %c16 : index
      hivm.hir.load ins(%subview : memref<?x?xf16, strided<[16, 1], offset: ?>>) outs(%subview_3 : memref<?x?xf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %47 : index init_out_buffer = true init_condition = %46 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %48 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
      %49 = hivm.hir.vtranspose ins(%48 : tensor<16x16xf16>) outs(%9 : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
      %alloc_4 = memref.alloc() : memref<16x16xf32>
      %50 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%26, %24]
      %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
      %52 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%36, %50]
      %53 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%51, %52]
      %54 = arith.cmpi slt, %53, %c16 : index
      %subview_5 = memref.subview %reinterpret_cast[0, 0] [%53, 16] [1, 1] : memref<16x16xf32, strided<[16, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[%52, 0] [%53, 16] [1, 1] : memref<16x16xf32> to memref<?x16xf32, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%subview_5 : memref<?x16xf32, strided<[16, 1], offset: ?>>) outs(%subview_6 : memref<?x16xf32, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 init_out_buffer = true init_condition = %54 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %55 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x16xf32>
      %56 = arith.index_cast %16 : i32 to index
      %57 = affine.max affine_map<()[s0] -> (16, s0)>()[%56]
      %58 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%57, %56]
      %59 = arith.index_cast %20 : i32 to index
      %60 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%59, %26]
      %61 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%60, %59]
      %62 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%58, %56]
      %63 = affine.max affine_map<()[s0] -> (0, s0)>()[%62]
      %64 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%61, %59]
      %65 = affine.max affine_map<()[s0] -> (0, s0)>()[%64]
      %extracted_slice = tensor.extract_slice %49[0, 0] [%63, %65] [1, 1] : tensor<16x16xf16> to tensor<?x?xf16>
      %inserted_slice = tensor.insert_slice %extracted_slice into %10[0, 0] [%63, %65] [1, 1] : tensor<?x?xf16> into tensor<16x16xf16>
      %66 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<16x16xf16>
      %67 = bufferization.to_tensor %66 restrict writable : memref<16x16xf16>
      %68 = hivm.hir.store ins(%inserted_slice : tensor<16x16xf16>) outs(%67 : tensor<16x16xf16>) {"hivm.inserted-store"} -> tensor<16x16xf16>
      %69 = hivm.hir.load ins(%68 : tensor<16x16xf16>) outs(%9 : tensor<16x16xf16>) {"hivm.inserted-load"} core_type = <CUBE> -> tensor<16x16xf16>
      %extracted_slice_7 = tensor.extract_slice %55[0, 0] [%65, 16] [1, 1] : tensor<16x16xf32> to tensor<?x16xf32>
      %inserted_slice_8 = tensor.insert_slice %extracted_slice_7 into %12[0, 0] [%65, 16] [1, 1] : tensor<?x16xf32> into tensor<16x16xf32>
      %70 = hivm.hir.vcast ins(%inserted_slice_8 : tensor<16x16xf32>) outs(%9 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %71 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<16x16xf16>
      %72 = bufferization.to_tensor %71 restrict writable : memref<16x16xf16>
      %73 = hivm.hir.store ins(%70 : tensor<16x16xf16>) outs(%72 : tensor<16x16xf16>) {"hivm.inserted-store"} -> tensor<16x16xf16>
      %74 = hivm.hir.load ins(%73 : tensor<16x16xf16>) outs(%9 : tensor<16x16xf16>) {"hivm.inserted-load"} core_type = <CUBE> -> tensor<16x16xf16>
      %75 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%69, %74, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %76 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<16x16xf32>
      %77 = bufferization.to_tensor %76 restrict writable : memref<16x16xf32>
      %78 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%75 : tensor<16x16xf32>) outs(%77 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %79 = hivm.hir.load ins(%78 : tensor<16x16xf32>) outs(%11 : tensor<16x16xf32>) {"hivm.inserted-load"} core_type = <VECTOR> -> tensor<16x16xf32>
      %expanded = tensor.expand_shape %21 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
      %expanded_9 = tensor.expand_shape %22 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
      %80 = tensor.empty() : tensor<16x16xi32>
      %81 = hivm.hir.vbrc ins(%expanded_9 : tensor<1x16xi32>) outs(%80 : tensor<16x16xi32>) broadcast_dims = [0] -> tensor<16x16xi32>
      %82 = tensor.empty() : tensor<16x16xi1>
      %83 = hivm.hir.vmax ins(%expanded, %expanded_9 : tensor<16x1xi32>, tensor<1x16xi32>) outs(%80 : tensor<16x16xi32>) broadcast = [0, 1] -> tensor<16x16xi32>
      %84 = hivm.hir.vcmp ins(%83, %81 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %85 = hivm.hir.vnot ins(%84 : tensor<16x16xi1>) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %86 = tensor.empty() : tensor<16x1xi32>
      %87 = hivm.hir.vmax ins(%expanded, %arg6 : tensor<16x1xi32>, i32) outs(%86 : tensor<16x1xi32>) -> tensor<16x1xi32>
      %88 = tensor.empty() : tensor<16x1xi1>
      %89 = hivm.hir.vcmp ins(%87, %expanded : tensor<16x1xi32>, tensor<16x1xi32>) outs(%88 : tensor<16x1xi1>) -> tensor<16x1xi1>
      %90 = hivm.hir.vnot ins(%89 : tensor<16x1xi1>) outs(%88 : tensor<16x1xi1>) -> tensor<16x1xi1>
      %91 = tensor.empty() : tensor<16x1xf16>
      %92 = tensor.empty() : tensor<1x16xf16>
      %93 = hivm.hir.vcast ins(%90 : tensor<16x1xi1>) outs(%91 : tensor<16x1xf16>) round_mode = <trunc> -> tensor<16x1xf16>
      %94 = hivm.hir.vbrc ins(%93 : tensor<16x1xf16>) outs(%9 : tensor<16x16xf16>) broadcast_dims = [1] -> tensor<16x16xf16>
      %95 = hivm.hir.vcmp ins(%94, %cst : tensor<16x16xf16>, f16) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %96 = hivm.hir.vnot ins(%95 : tensor<16x16xi1>) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %97 = hivm.hir.vand ins(%85, %96 : tensor<16x16xi1>, tensor<16x16xi1>) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %98 = tensor.empty() : tensor<1x16xi32>
      %99 = hivm.hir.vmax ins(%expanded_9, %arg6 : tensor<1x16xi32>, i32) outs(%98 : tensor<1x16xi32>) -> tensor<1x16xi32>
      %100 = tensor.empty() : tensor<1x16xi1>
      %101 = hivm.hir.vcmp ins(%99, %expanded_9 : tensor<1x16xi32>, tensor<1x16xi32>) outs(%100 : tensor<1x16xi1>) -> tensor<1x16xi1>
      %102 = hivm.hir.vnot ins(%101 : tensor<1x16xi1>) outs(%100 : tensor<1x16xi1>) -> tensor<1x16xi1>
      %103 = hivm.hir.vcast ins(%102 : tensor<1x16xi1>) outs(%92 : tensor<1x16xf16>) round_mode = <trunc> -> tensor<1x16xf16>
      %104 = hivm.hir.vbrc ins(%103 : tensor<1x16xf16>) outs(%9 : tensor<16x16xf16>) broadcast_dims = [0] -> tensor<16x16xf16>
      %105 = hivm.hir.vcmp ins(%104, %cst : tensor<16x16xf16>, f16) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %106 = hivm.hir.vnot ins(%105 : tensor<16x16xi1>) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %107 = hivm.hir.vand ins(%97, %106 : tensor<16x16xi1>, tensor<16x16xi1>) outs(%82 : tensor<16x16xi1>) -> tensor<16x16xi1>
      %108 = hivm.hir.vmul ins(%79, %cst_1 : tensor<16x16xf32>, f32) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %109 = hivm.hir.vadd ins(%108, %cst_0 : tensor<16x16xf32>, f32) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %110 = hivm.hir.vsel ins(%107, %109, %cst_0 : tensor<16x16xi1>, tensor<16x16xf32>, f32) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %111 = arith.addi %20, %16 : i32
      %112 = arith.maxsi %111, %c0_i32 : i32
      %113 = arith.index_cast %112 : i32 to index
      %114 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 16)>()[%15, %113]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%114], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
      %115 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%26, %113]
      %116 = affine.min affine_map<()[s0] -> (16, s0)>()[%115]
      %117 = arith.subi %c0_i32, %111 : i32
      %118 = arith.maxsi %117, %c0_i32 : i32
      %119 = arith.index_cast %118 : i32 to index
      %120 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%119, %115]
      %121 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%116, %120]
      %extracted_slice_11 = tensor.extract_slice %110[%120, 0] [%121, 16] [1, 1] : tensor<16x16xf32> to tensor<?x16xf32>
      %subview_12 = memref.subview %reinterpret_cast_10[0, 0] [%121, 16] [1, 1] : memref<16x16xf32, strided<[16, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1], offset: ?>>
      hivm.hir.store ins(%extracted_slice_11 : tensor<?x16xf32>) outs(%subview_12 : memref<?x16xf32, strided<[16, 1], offset: ?>>) {autoblockify.dma}
    } {autoblockify.subloop}
    return
  }
}
