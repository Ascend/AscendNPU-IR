// RUN: bishengir-opt -test-hivm-dimension-analyzer %s -split-input-file | FileCheck %s

// CHECK: 1 succeedFunc - Function analyzed count
// CHECK: hivm_dimension_ok
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @hivm_dimension_ok(%arg0: memref<?xf32>, %arg1: tensor<16x128xf32>, %arg2: tensor<16xf32>, %arg3: memref<16x128xbf16, strided<[100, 1], offset: ?>>, %arg4: tensor<16x128xbf16>, %arg5: index, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = hivm.hir.vcast ins(%arg1 : tensor<16x128xf32>) outs(%arg4 : tensor<16x128xbf16>) -> tensor<16x128xbf16>
    hivm.hir.store ins(%0 : tensor<16x128xbf16>) outs(%arg3 : memref<16x128xbf16, strided<[100, 1], offset: ?>>)
    %1 = arith.cmpi eq, %arg7, %arg9 : i32
    scf.if %1 {
      %2 = arith.muli %arg6, %arg8 : i32
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.addi %3, %arg5 : index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%arg2 : tensor<16xf32>) outs(%reinterpret_cast : memref<16xf32, strided<[1], offset: ?>>)
    }
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
    return
  }
}

// -----

// CHECK: 1 succeedFunc - Function analyzed count
// CHECK: Tiling dim for {{.*}} is 0
// CHECK: Tiling dim for {{.*}} is 0
func.func @_attn_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 2.000000e+00 : f32
  %cst_0 = arith.constant 0.693147182 : f32
  %c16384 = arith.constant 16384 : index
  %c64 = arith.constant 64 : index
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c131072_i64 = arith.constant 131072 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c128_i32 = arith.constant 128 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c16_i32 = arith.constant 16 : i32
  %c20_i32 = arith.constant 20 : i32
  %cst_1 = arith.constant 0.72134751 : f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %cst_3 = arith.constant 0xFF800000 : f32
  %cst_4 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %cst_5 = arith.constant 1.44269502 : f32
  %true = arith.constant true
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg9 : i32
  %7 = tensor.empty() : tensor<1xf32>
  %8 = tensor.empty() : tensor<128xf32>
  %9 = hivm.hir.vbrc ins(%cst_4 : f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
  %10 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
  %11 = tensor.empty() : tensor<128x256xf32>
  %12 = tensor.empty() : tensor<128x64xf32>
  %13 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%12 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %14 = tensor.empty() : tensor<1xf32>
  %15 = hivm.hir.vbrc ins(%arg8 : f32) outs(%14 : tensor<1xf32>) -> tensor<1xf32>
  %16 = hivm.hir.vmul ins(%15, %cst_5 : tensor<1xf32>, f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %16[%c0] : tensor<1xf32>
  scf.for %arg12 = %6 to %c16_i32 step %c20_i32  : i32 {
    %17 = arith.divsi %arg12, %c8_i32 : i32
    %18 = arith.remsi %arg12, %c8_i32 : i32
    %19 = arith.divsi %17, %c2_i32 : i32
    %20 = arith.remsi %17, %c2_i32 : i32
    %21 = arith.extsi %19 : i32 to i64
    %22 = arith.muli %21, %c131072_i64 : i64
    %23 = arith.extsi %20 : i32 to i64
    %24 = arith.muli %23, %c65536_i64 : i64
    %25 = arith.addi %22, %24 : i64
    %26 = arith.index_cast %25 : i64 to index
    %27 = arith.muli %18, %c128_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29 = arith.muli %28, %c64 : index
    %30 = arith.addi %29, %26 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%30], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<128x64xf16>) init_out_buffer = false
    %31 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf16>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast_7 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
    %cast_9 = memref.cast %reinterpret_cast_8 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
    %32 = arith.index_cast %c256_i32 : i32 to index
    %33 = affine.apply affine_map<(d0) -> (d0 * 4)>(%32)
    %34 = arith.index_cast %33 : index to i32
    %35:9 = scf.for %arg13 = %c0_i32 to %c1024_i32 step %34 iter_args(%arg14 = %9, %arg15 = %13, %arg16 = %10, %arg17 = %cast, %arg18 = %cast_9, %arg19 = %26, %arg20 = %c0, %arg21 = %26, %arg22 = %c0) -> (tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, memref<256x64xf16, strided<[?, ?], offset: ?>>, memref<256x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %49 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<4x128x256xf32>
      %50 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<4x128x64xf32>
      %51 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<4x128x256xf16>
      %52 = arith.index_cast %arg13 : i32 to index
      %53 = arith.index_cast %c1024_i32 : i32 to index
      %54 = affine.min affine_map<(d0, d1, d2) -> (d0 + d1, d2)>(%52, %33, %53)
      %55 = affine.apply affine_map<(d0, d1)[s0] -> ((d0 - d1) floordiv s0)>(%54, %52)[%32]
      annotation.mark %49 : memref<4x128x256xf32>
      annotation.mark %51 : memref<4x128x256xf16>
      annotation.mark %50 : memref<4x128x64xf32>
      %c0_11 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %56:2 = scf.for %arg23 = %c0_11 to %55 step %c1 iter_args(%arg24 = %arg18, %arg25 = %arg21) -> (memref<256x64xf16, strided<[?, ?], offset: ?>>, index) {
        %64 = arith.index_cast %c256_i32 : i32 to index
        %65 = arith.index_cast %arg13 : i32 to index
        %66 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg23)[%64, %65]
        %67 = arith.index_cast %66 : index to i32
        %alloc_18 = memref.alloc() : memref<256x64xf16>
        hivm.hir.load ins(%arg24 : memref<256x64xf16, strided<[?, ?], offset: ?>>) outs(%alloc_18 : memref<256x64xf16>) {load_to_tile} init_out_buffer = false
        %68 = bufferization.to_tensor %alloc_18 restrict writable {cube_to_fuse} : memref<256x64xf16>
        %69 = tensor.empty() : tensor<128x256xf32>
        // cube m, n
        %70 = hivm.hir.mmadL1 {b_transpose, cube_to_fuse} ins(%31, %68, %true, %c128, %c64, %c256 : tensor<128x64xf16>, tensor<256x64xf16>, i1, index, index, index) outs(%69 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %subview = memref.subview %49[%arg23, 0, 0] [1, 128, 256] [1, 1, 1] : memref<4x128x256xf32> to memref<1x128x256xf32, strided<[32768, 256, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x256xf32, strided<[32768, 256, 1], offset: ?>> into memref<128x256xf32, strided<[256, 1], offset: ?>>
        // find fixpipe tile and fuse
        hivm.hir.fixpipe {enable_nz2nd, "fixpipe_1"} ins(%70 : tensor<128x256xf32>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>>)
        %71 = arith.addi %arg25, %c16384 : index
        %72 = arith.addi %71, %arg22 : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%72], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %cast_20 = memref.cast %reinterpret_cast_19 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
        scf.yield %cast_20, %72 : memref<256x64xf16, strided<[?, ?], offset: ?>>, index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      %57 = bufferization.to_tensor %49 restrict : memref<4x128x256xf32>
      %c0_12 = arith.constant 0 : index
      %c1_13 = arith.constant 1 : index
      %58 = tensor.empty() : tensor<4x128x64xf32>
      %59:3 = scf.for %arg23 = %c0_12 to %55 step %c1_13 iter_args(%arg24 = %arg16, %arg25 = %arg14, %arg26 = %58) -> (tensor<128xf32>, tensor<128xf32>, tensor<4x128x64xf32>) {
        %64 = arith.index_cast %c256_i32 : i32 to index
        %65 = arith.index_cast %arg13 : i32 to index
        %66 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg23)[%64, %65]
        %67 = arith.index_cast %66 : index to i32
        %68 = tensor.empty() : tensor<128x256xf32>
        %extracted_slice = tensor.extract_slice %57[%arg23, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<4x128x256xf32> to tensor<128x256xf32>
        %69 = hivm.hir.load ins(%extracted_slice : tensor<128x256xf32>) outs(%68 : tensor<128x256xf32>) {to_fuse} init_out_buffer = false -> tensor<128x256xf32>
        %70 = tensor.empty() : tensor<128x256xf32>
        %extracted_slice_18 = tensor.extract_slice %57[%arg23, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<4x128x256xf32> to tensor<128x256xf32>
        %71 = hivm.hir.load ins(%extracted_slice_18 : tensor<128x256xf32>) outs(%70 : tensor<128x256xf32>) {to_fuse} init_out_buffer = false -> tensor<128x256xf32>
        %expanded_19 = tensor.expand_shape %10 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %72 = hivm.hir.vreduce {to_fuse} <max> ins(%69 : tensor<128x256xf32>) outs(%expanded_19 : tensor<128x1xf32>) reduce_dims = [1]  -> tensor<128x1xf32> 
        %collapsed = tensor.collapse_shape %72 [[0, 1]] {to_fuse} : tensor<128x1xf32> into tensor<128xf32>
        %73 = hivm.hir.vmul {to_fuse} ins(%collapsed, %extracted : tensor<128xf32>, f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %74 = hivm.hir.vmax {to_fuse} ins(%arg24, %73 : tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %75 = hivm.hir.vmul {to_fuse} ins(%71, %extracted : tensor<128x256xf32>, f32) outs(%11 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %expanded_20 = tensor.expand_shape %74 [[0, 1]] output_shape [128, 1] {to_fuse} : tensor<128xf32> into tensor<128x1xf32>
        %76 = hivm.hir.vbrc {to_fuse} ins(%expanded_20 : tensor<128x1xf32>) outs(%11 : tensor<128x256xf32>) broadcast_dims = [1] -> tensor<128x256xf32>
        %77 = hivm.hir.vsub {to_fuse} ins(%75, %76 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%11 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %78 = hivm.hir.vmul {to_fuse} ins(%77, %cst_0 : tensor<128x256xf32>, f32) outs(%11 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %79 = hivm.hir.vexp {to_fuse} ins(%78 : tensor<128x256xf32>) outs(%11 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %80 = tensor.empty() : tensor<128x256xf16>
        %81 = hivm.hir.vcast {to_fuse} ins(%79 : tensor<128x256xf32>) outs(%80 : tensor<128x256xf16>) -> tensor<128x256xf16> 
        %subview = memref.subview %51[%arg23, 0, 0] [1, 128, 256] [1, 1, 1] : memref<4x128x256xf16> to memref<1x128x256xf16, strided<[32768, 256, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x256xf16, strided<[32768, 256, 1], offset: ?>> into memref<128x256xf16, strided<[256, 1], offset: ?>>
        hivm.hir.store ins(%81 : tensor<128x256xf16>) outs(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>>) {store_to_tile}
        %82 = hivm.hir.vbrc {to_fuse} ins(%cst_2 : f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32> 
        %expanded_21 = tensor.expand_shape %82 [[0, 1]] output_shape [128, 1] {to_fuse} : tensor<128xf32> into tensor<128x1xf32>
        %83 = hivm.hir.vreduce {to_fuse} <sum> ins(%79 : tensor<128x256xf32>) outs(%expanded_21 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed_22 = tensor.collapse_shape %83 [[0, 1]] {to_fuse} : tensor<128x1xf32> into tensor<128xf32>
        %84 = hivm.hir.vsub {to_fuse} ins(%arg24, %74 : tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %85 = hivm.hir.vmul {to_fuse} ins(%84, %cst_0 : tensor<128xf32>, f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %86 = hivm.hir.vexp {to_fuse} ins(%85 : tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %87 = hivm.hir.vmul {to_fuse} ins(%arg25, %86 : tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %88 = hivm.hir.vadd {vadd_to_tile} ins(%87, %collapsed_22 : tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_23 = tensor.expand_shape %86 [[0, 1]] output_shape [128, 1] {to_fuse} : tensor<128xf32> into tensor<128x1xf32>
        %extracted_slice_24 = tensor.extract_slice %arg26[%arg23, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<4x128x64xf32> to tensor<128x64xf32>
        %89 = hivm.hir.vbrc {vbrc_to_tile} ins(%expanded_23 : tensor<128x1xf32>) outs(%extracted_slice_24 : tensor<128x64xf32>) broadcast_dims = [1] -> tensor<128x64xf32>
        %90 = hivm.hir.vmul {to_fuse} ins(%74, %extracted : tensor<128xf32>, f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %91 = hivm.hir.vdiv {vdiv_to_tile} ins(%90, %cst_1 : tensor<128xf32>, f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
        %inserted_slice = tensor.insert_slice %89 into %arg26[%arg23, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<128x64xf32> into tensor<4x128x64xf32>
        scf.yield %91, %88, %inserted_slice : tensor<128xf32>, tensor<128xf32>, tensor<4x128x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      %60 = bufferization.to_tensor %51 restrict : memref<4x128x256xf16>
      %c0_14 = arith.constant 0 : index
      %c1_15 = arith.constant 1 : index
      %61:2 = scf.for %arg23 = %c0_14 to %55 step %c1_15 iter_args(%arg24 = %arg17, %arg25 = %arg19) -> (memref<256x64xf16, strided<[?, ?], offset: ?>>, index) {
        %64 = arith.index_cast %c256_i32 : i32 to index
        %65 = arith.index_cast %arg13 : i32 to index
        %66 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg23)[%64, %65]
        %67 = arith.index_cast %66 : index to i32
        %alloc_18 = memref.alloc() : memref<256x64xf16>
        hivm.hir.load ins(%arg24 : memref<256x64xf16, strided<[?, ?], offset: ?>>) outs(%alloc_18 : memref<256x64xf16>) init_out_buffer = false
        %68 = bufferization.to_tensor %alloc_18 restrict writable : memref<256x64xf16>
        %69 = tensor.empty() : tensor<128x256xf16>
        %extracted_slice = tensor.extract_slice %60[%arg23, 0, 0] [1, 128, 256] [1, 1, 1] : tensor<4x128x256xf16> to tensor<128x256xf16>
        %70 = hivm.hir.load ins(%extracted_slice : tensor<128x256xf16>) outs(%69 : tensor<128x256xf16>) init_out_buffer = false -> tensor<128x256xf16>
        %71 = tensor.empty() : tensor<128x64xf32>
        %72 = hivm.hir.mmadL1 ins(%70, %68, %true, %c128, %c256, %c64 : tensor<128x256xf16>, tensor<256x64xf16>, i1, index, index, index) outs(%71 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %subview = memref.subview %50[%arg23, 0, 0] [1, 128, 64] [1, 1, 1] : memref<4x128x64xf32> to memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>> into memref<128x64xf32, strided<[64, 1], offset: ?>>
        // find fixpipe tile and fuse
        hivm.hir.fixpipe {enable_nz2nd} ins(%72 : tensor<128x64xf32>) outs(%collapse_shape : memref<128x64xf32, strided<[64, 1], offset: ?>>)
        %73 = arith.addi %arg25, %c16384 : index
        %74 = arith.addi %73, %arg20 : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg5 to offset: [%74], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %cast_20 = memref.cast %reinterpret_cast_19 : memref<256x64xf16, strided<[64, 1], offset: ?>> to memref<256x64xf16, strided<[?, ?], offset: ?>>
        scf.yield %cast_20, %74 : memref<256x64xf16, strided<[?, ?], offset: ?>>, index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      %62 = bufferization.to_tensor %50 restrict : memref<4x128x64xf32>
      %c0_16 = arith.constant 0 : index
      %c1_17 = arith.constant 1 : index
      %63 = scf.for %arg23 = %c0_16 to %55 step %c1_17 iter_args(%arg24 = %arg15) -> (tensor<128x64xf32>) {
        %64 = arith.index_cast %c256_i32 : i32 to index
        %65 = arith.index_cast %arg13 : i32 to index
        %66 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg23)[%64, %65]
        %67 = arith.index_cast %66 : index to i32
        %extracted_slice = tensor.extract_slice %59#2[%arg23, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<4x128x64xf32> to tensor<128x64xf32>
        %68 = hivm.hir.vmul ins(%arg24, %extracted_slice : tensor<128x64xf32>, tensor<128x64xf32>) outs(%12 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %69 = tensor.empty() : tensor<128x64xf32>
        %extracted_slice_18 = tensor.extract_slice %62[%arg23, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<4x128x64xf32> to tensor<128x64xf32>
        %70 = hivm.hir.load ins(%extracted_slice_18 : tensor<128x64xf32>) outs(%69 : tensor<128x64xf32>) init_out_buffer = false -> tensor<128x64xf32>
        %71 = tensor.empty() : tensor<128x64xf32>
        %72 = hivm.hir.vadd ins(%70, %68 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%71 : tensor<128x64xf32>) -> tensor<128x64xf32>
        scf.yield %72 : tensor<128x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      scf.yield %59#1, %63, %59#0, %61#0, %56#0, %61#1, %c0, %56#1, %c0 : tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, memref<256x64xf16, strided<[?, ?], offset: ?>>, memref<256x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %36 = hivm.hir.vln ins(%35#0 : tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
    %37 = tensor.empty() : tensor<128xf32>
    %38 = hivm.hir.vbrc ins(%cst : f32) outs(%37 : tensor<128xf32>) -> tensor<128xf32>
    %39 = hivm.hir.vln ins(%38 : tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
    %40 = hivm.hir.vdiv ins(%36, %39 : tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
    %41 = hivm.hir.vadd ins(%35#2, %40 : tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
    %expanded = tensor.expand_shape %35#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %42 = hivm.hir.vbrc ins(%expanded : tensor<128x1xf32>) outs(%12 : tensor<128x64xf32>) broadcast_dims = [1] -> tensor<128x64xf32>
    %43 = hivm.hir.vdiv ins(%35#1, %42 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%12 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %44 = arith.muli %17, %c1024_i32 : i32
    %45 = arith.index_cast %44 : i32 to index
    %46 = arith.addi %45, %28 : index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%41 : tensor<128xf32>) outs(%reinterpret_cast_10 : memref<128xf32, strided<[1], offset: ?>>)
    %47 = tensor.empty() : tensor<128x64xf16>
    %48 = hivm.hir.vcast ins(%43 : tensor<128x64xf32>) outs(%47 : tensor<128x64xf16>) -> tensor<128x64xf16>
    hivm.hir.store ins(%48 : tensor<128x64xf16>) outs(%reinterpret_cast_6 : memref<128x64xf16, strided<[64, 1], offset: ?>>)
  }
  return
}

// -----

// CHECK: 1 succeedFunc - Function analyzed count
// CHECK: hivm_dimension_ok_with_yield
#map1 = affine_map<()[s0] -> (s0 * 15360 + 4096)>
#map2 = affine_map<()[s0] -> (s0 * 15360 + 6144)>
#map3 = affine_map<()[s0] -> (s0 * 15360 + 7168)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @hivm_dimension_ok_with_yield(%arg0: memref<?xi8>, %arg1: tensor<32xi32>, %arg2: i32, %arg3: index, %arg4: tensor<16x128xbf16>, %arg5: tensor<16xf32>, %arg6: tensor<16x32xf32>, %arg7: tensor<16x32xf32>, %arg8: tensor<16x128xf32>, %arg9: tensor<16x128xf32>, %arg10: tensor<16xf32>, %arg11: tensor<16xf32>, %arg12: f16, %arg13: index, %arg14: index, %arg15: i32, %arg16: i32) -> (tensor<16xf32>, tensor<16x128xbf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %0 = hivm.hir.varange offset[%arg14] strides[%arg13] outs(%arg1 : tensor<32xi32>) -> tensor<32xi32>
    %1:3 = scf.for %arg17 = %arg15 to %arg2 step %arg16 iter_args(%arg18 = %arg5, %arg19 = %arg10, %arg20 = %arg8) -> (tensor<16xf32>, tensor<16xf32>, tensor<16x128xf32>)  : i32 {
      %5 = hivm.hir.vadd ins(%0, %arg17 : tensor<32xi32>, i32) outs(%arg1 : tensor<32xi32>) -> tensor<32xi32>
      %6 = affine.apply #map1()[%arg3]
      %view = memref.view %arg0[%6][] : memref<?xi8> to memref<16x32xf32>
      %7 = bufferization.to_tensor %view restrict writable : memref<16x32xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
      %8 = hivm.hir.load ins(%7 : tensor<16x32xf32>) outs(%arg7 : tensor<16x32xf32>) init_out_buffer = false -> tensor<16x32xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
      %9 = tensor.empty() : tensor<32xi64>
      %10 = hivm.hir.vcast ins(%5 : tensor<32xi32>) outs(%9 : tensor<32xi64>) -> tensor<32xi64>
      %11 = arith.extsi %arg2 : i32 to i64
      %12 = hivm.hir.vbrc ins(%11 : i64) outs(%9 : tensor<32xi64>) -> tensor<32xi64>
      %13 = tensor.empty() : tensor<32xi1>
      %14 = hivm.hir.vcmp ins(%10, %12 : tensor<32xi64>, tensor<32xi64>) outs(%13 : tensor<32xi1>) compare_mode = <lt> -> tensor<32xi1>
      %15 = tensor.empty() : tensor<32xf16>
      %16 = hivm.hir.vcast ins(%14 : tensor<32xi1>) outs(%15 : tensor<32xf16>) round_mode = <trunc> -> tensor<32xf16>
      %17 = tensor.empty() : tensor<16x32xf16>
      %expanded = tensor.expand_shape %16 [[0, 1]] output_shape [1, 32] : tensor<32xf16> into tensor<1x32xf16>
      %18 = hivm.hir.vbrc ins(%expanded : tensor<1x32xf16>) outs(%17 : tensor<16x32xf16>) broadcast_dims = [0] -> tensor<16x32xf16>
      %19 = tensor.empty() : tensor<16x32xi1>
      %20 = hivm.hir.vcmp ins(%18, %arg12 : tensor<16x32xf16>, f16) outs(%19 : tensor<16x32xi1>) -> tensor<16x32xi1>
      %21 = hivm.hir.vnot ins(%20 : tensor<16x32xi1>) outs(%19 : tensor<16x32xi1>) -> tensor<16x32xi1>
      %22 = hivm.hir.vsel ins(%21, %8, %arg6 : tensor<16x32xi1>, tensor<16x32xf32>, tensor<16x32xf32>) outs(%arg7 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %23 = tensor.empty() : tensor<16x1xf32>
      %24 = hivm.hir.vreduce <max> ins(%22 : tensor<16x32xf32>) outs(%23 : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %collapsed = tensor.collapse_shape %24 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
      %25 = hivm.hir.vmax ins(%arg18, %collapsed : tensor<16xf32>, tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      %26 = hivm.hir.vsub ins(%arg18, %25 : tensor<16xf32>, tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      %27 = hivm.hir.vexp ins(%26 : tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      %expanded_0 = tensor.expand_shape %25 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %28 = hivm.hir.vbrc ins(%expanded_0 : tensor<16x1xf32>) outs(%arg7 : tensor<16x32xf32>) broadcast_dims = [1] -> tensor<16x32xf32>
      %29 = hivm.hir.vsub ins(%22, %28 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%arg7 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %30 = hivm.hir.vexp ins(%29 : tensor<16x32xf32>) outs(%arg7 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %31 = hivm.hir.vmul ins(%arg19, %27 : tensor<16xf32>, tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      %32 = tensor.empty() : tensor<16x1xf32>
      %33 = hivm.hir.vreduce <sum> ins(%30 : tensor<16x32xf32>) outs(%32 : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %collapsed_1 = tensor.collapse_shape %33 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
      %34 = hivm.hir.vadd ins(%31, %collapsed_1 : tensor<16xf32>, tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      %expanded_2 = tensor.expand_shape %27 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %35 = hivm.hir.vbrc ins(%expanded_2 : tensor<16x1xf32>) outs(%arg9 : tensor<16x128xf32>) broadcast_dims = [1] -> tensor<16x128xf32>
      %36 = hivm.hir.vmul ins(%arg20, %35 : tensor<16x128xf32>, tensor<16x128xf32>) outs(%arg9 : tensor<16x128xf32>) -> tensor<16x128xf32>
      %37 = tensor.empty() : tensor<16x32xbf16>
      %38 = hivm.hir.vcast ins(%30 : tensor<16x32xf32>) outs(%37 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
      %39 = affine.apply #map2()[%arg3]
      %view_3 = memref.view %arg0[%39][] : memref<?xi8> to memref<16x32xbf16>
      %40 = bufferization.to_tensor %view_3 restrict writable : memref<16x32xbf16>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
      %41 = hivm.hir.store ins(%38 : tensor<16x32xbf16>) outs(%40 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
      annotation.mark %41 : tensor<16x32xbf16>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 0
      %42 = affine.apply #map3()[%arg3]
      %view_4 = memref.view %arg0[%42][] : memref<?xi8> to memref<16x128xf32>
      %43 = bufferization.to_tensor %view_4 restrict writable : memref<16x128xf32>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
      %44 = hivm.hir.load ins(%43 : tensor<16x128xf32>) outs(%arg9 : tensor<16x128xf32>) init_out_buffer = false -> tensor<16x128xf32>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
      %45 = hivm.hir.vadd ins(%44, %36 : tensor<16x128xf32>, tensor<16x128xf32>) outs(%arg9 : tensor<16x128xf32>) -> tensor<16x128xf32>
      scf.yield %25, %34, %45 : tensor<16xf32>, tensor<16xf32>, tensor<16x128xf32>
    }
    %2 = arith.cmpi eq, %arg2, %arg15 : i32
    %3:2 = scf.if %2 -> (tensor<16x128xf32>, tensor<16xf32>) {
      scf.yield %1#2, %arg10 : tensor<16x128xf32>, tensor<16xf32>
    } else {
      %expanded = tensor.expand_shape %1#1 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
      %5 = hivm.hir.vbrc ins(%expanded : tensor<16x1xf32>) outs(%arg9 : tensor<16x128xf32>) broadcast_dims = [1] -> tensor<16x128xf32>
      %6 = hivm.hir.vdiv ins(%1#2, %5 : tensor<16x128xf32>, tensor<16x128xf32>) outs(%arg9 : tensor<16x128xf32>) -> tensor<16x128xf32>
      %7 = hivm.hir.vln ins(%1#1 : tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      %8 = hivm.hir.vadd ins(%1#0, %7 : tensor<16xf32>, tensor<16xf32>) outs(%arg11 : tensor<16xf32>) -> tensor<16xf32>
      scf.yield %6, %8 : tensor<16x128xf32>, tensor<16xf32>
    }
    %4 = hivm.hir.vcast ins(%3#0 : tensor<16x128xf32>) outs(%arg4 : tensor<16x128xbf16>) -> tensor<16x128xbf16>
    return %3#1, %4 : tensor<16xf32>, tensor<16x128xbf16>
  }
}