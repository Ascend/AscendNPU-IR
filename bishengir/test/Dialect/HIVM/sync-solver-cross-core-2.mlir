// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{force-is-reg-based=true}))" -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  func.func @test_block_sync_normal(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %arg3: memref<256xf32, #hivm.address_space<gm>>, %arg4: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg4
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
    %0 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%alloc, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %alloc_0 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg2 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%alloc_0, %alloc_1 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<256xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_block_sync_branch(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %arg3: memref<256xf32, #hivm.address_space<gm>>, %arg4: i1, %arg5: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg5
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
    %0 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%alloc, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %alloc_0 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    scf.if %arg4 {
      hivm.hir.load ins(%arg2 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_0, %alloc_1 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<256xf32, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_block_sync_loop(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %arg3: memref<256xf32, #hivm.address_space<gm>>, %arg4: i1, %arg5: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg5
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
    scf.for %arg6 = %c0 to %c16 step %c4 {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
      %0 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
      %1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
      hivm.hir.mmadL1 ins(%alloc, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
      // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
      hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
      // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
      %alloc_0 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
      // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
      hivm.hir.load ins(%arg2 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
      hivm.hir.vadd ins(%alloc_0, %alloc_1 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_0 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<256xf32, #hivm.address_space<gm>>)
    }
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
    return
  }
}

// -----
module {
  func.func @test_block_sync_loop(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: i1, %arg5: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg5
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<16xf32>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32>) outs(%alloc : memref<16xf32>)
    %alloc_0 = memref.alloc() : memref<16xf32>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32>) outs(%alloc_0 : memref<16xf32>)
    %alloc_1 = memref.alloc() : memref<256xf32>
    hivm.hir.mmadL1 ins(%alloc, %alloc_0, %true, %c16, %c256, %c16 : memref<16xf32>, memref<16xf32>, i1, index, index, index) outs(%alloc_1 : memref<256xf32>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_1 : memref<256xf32>) outs(%arg2 : memref<256xf32>)
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %0 = bufferization.to_tensor %arg2 restrict writable : memref<256xf32>
    %1 = tensor.empty() : tensor<256xf32>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %2 = hivm.hir.load ins(%0 : tensor<256xf32>) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %3 = tensor.empty() : tensor<256xf32>
    %4 = hivm.hir.vsub ins(%2, %2 : tensor<256xf32>, tensor<256xf32>) outs(%3 : tensor<256xf32>) -> tensor<256xf32>
    %5 = tensor.empty() : tensor<256xf32>
    %6 = hivm.hir.store ins(%4 : tensor<256xf32>) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
    return
  }
}

// -----
#map = affine_map<()[s0, s1, s2, s3] -> (s0 mod 2)>
module {
  func.func @_attn_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c40960 = arith.constant 40960 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32_i32 = arith.constant 32 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %cst = arith.constant 1.44269502 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 0.72134751 : f32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %cst_4 = arith.constant 0.693147182 : f32
    %cst_5 = arith.constant 2.000000e+00 : f32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg10, %arg9 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg8 : i32
    hivm.hir.set_mask_norm
    %5 = tensor.empty() : tensor<1xf32>
    %6 = tensor.empty() : tensor<64xf32>
    %7 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %8 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %9 = tensor.empty() : tensor<64x16xf32>
    %10 = tensor.empty() : tensor<64x64xf32>
    %11 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %12 = arith.muli %4, %c64_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.muli %13, %c64 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%14], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %15 = hivm.hir.vbrc ins(%arg7 : f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %16 = hivm.hir.vmul ins(%15, %cst : tensor<1xf32>, f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %16[%c0] : tensor<1xf32>
    %alloc = memref.alloc() : memref<64x64xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<64x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<64x64xf16>)
    %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1]>>
    %cast = memref.cast %reinterpret_cast_7 : memref<16x64xf16, strided<[64, 1]>> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1]>>
    %cast_9 = memref.cast %reinterpret_cast_8 : memref<16x64xf16, strided<[64, 1]>> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 2
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 2
    %18:9 = scf.for %arg11 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg12 = %7, %arg13 = %11, %arg14 = %8, %arg15 = %cast, %arg16 = %cast_9, %arg17 = %c0, %arg18 = %c0, %arg19 = %c0, %arg20 = %c0) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<16x64xf16, strided<[?, ?], offset: ?>>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %28 = memref_ext.alloc_workspace() from %arg1 offset = [%c32768] : from memref<?xi8> to memref<2x64x16xf32>
      %29 = bufferization.to_tensor %28 restrict writable : memref<2x64x16xf32>
      %30 = memref_ext.alloc_workspace() from %arg1 offset = [%c40960] : from memref<?xi8> to memref<2x64x16xf16>
      %31 = bufferization.to_tensor %30 restrict writable : memref<2x64x16xf16>
      %32 = memref_ext.alloc_workspace() from %arg1 offset = [%c0] : from memref<?xi8> to memref<2x64x64xf32>
      %33 = bufferization.to_tensor %32 restrict writable : memref<2x64x64xf32>
      annotation.mark %28 : memref<2x64x16xf32>
      annotation.mark %30 : memref<2x64x16xf16>
      annotation.mark %32 : memref<2x64x64xf32>
      %34:3 = scf.for %arg21 = %c0 to %c2 step %c1 iter_args(%arg22 = %arg16, %arg23 = %arg19, %arg24 = %29) -> (memref<16x64xf16, strided<[?, ?], offset: ?>>, index, tensor<2x64x16xf32>) {
        %39 = arith.index_cast %arg11 : i32 to index
        %40 = arith.index_cast %c0_i32 : i32 to index
        %41 = arith.index_cast %c64_i32 : i32 to index
        %42 = arith.index_cast %c32_i32 : i32 to index
        %43 = affine.apply #map()[%arg21, %39, %40, %42]
        %44 = arith.index_cast %43 : index to i1
        %c5_i64 = arith.constant 5 : i64
        %c6_i64 = arith.constant 6 : i64
        %45 = arith.select %44, %c5_i64, %c6_i64 : i64
        %alloc_11 = memref.alloc() : memref<16x64xf16>
        hivm.hir.load ins(%arg22 : memref<16x64xf16, strided<[?, ?], offset: ?>>) outs(%alloc_11 : memref<16x64xf16>)
        %46 = bufferization.to_tensor %alloc_11 restrict writable : memref<16x64xf16>
        %47 = hivm.hir.mmadL1 {b_transpose} ins(%17, %46, %true, %c64, %c64, %c16 : tensor<64x64xf16>, tensor<16x64xf16>, i1, index, index, index) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
        %extracted_slice = tensor.extract_slice %arg24[%arg21, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<2x64x16xf32> to tensor<64x16xf32>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = [[EVENT0:[0-9]+]]
        %48 = hivm.hir.fixpipe {enable_nz2nd} ins(%47 : tensor<64x16xf32>) outs(%extracted_slice : tensor<64x16xf32>) -> tensor<64x16xf32>
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = [[EVENT1:[0-9]+]]
        %49 = arith.addi %arg23, %c1024 : index
        %50 = arith.addi %49, %arg20 : index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%50], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1], offset: ?>>
        %cast_13 = memref.cast %reinterpret_cast_12 : memref<16x64xf16, strided<[64, 1], offset: ?>> to memref<16x64xf16, strided<[?, ?], offset: ?>>
        %inserted_slice = tensor.insert_slice %48 into %arg24[%arg21, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<64x16xf32> into tensor<2x64x16xf32>
        scf.yield %cast_13, %50, %inserted_slice : memref<16x64xf16, strided<[?, ?], offset: ?>>, index, tensor<2x64x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %35 = tensor.empty() : tensor<2x64x64xf32>
      %36:4 = scf.for %arg21 = %c0 to %c2 step %c1 iter_args(%arg22 = %arg14, %arg23 = %arg12, %arg24 = %35, %arg25 = %31) -> (tensor<64xf32>, tensor<64xf32>, tensor<2x64x64xf32>, tensor<2x64x16xf16>) {
        %39 = arith.index_cast %arg11 : i32 to index
        %40 = arith.index_cast %c0_i32 : i32 to index
        %41 = arith.index_cast %c64_i32 : i32 to index
        %42 = arith.index_cast %c32_i32 : i32 to index
        %43 = affine.apply #map()[%arg21, %39, %40, %42]
        %44 = arith.index_cast %43 : index to i1
        %c5_i64 = arith.constant 5 : i64
        %c6_i64 = arith.constant 6 : i64
        %45 = arith.select %44, %c5_i64, %c6_i64 : i64
        %c3_i64 = arith.constant 3 : i64
        %c4_i64 = arith.constant 4 : i64
        %46 = arith.select %44, %c3_i64, %c4_i64 : i64
        %47 = tensor.empty() : tensor<64x16xf16>
        %extracted_slice = tensor.extract_slice %34#2[%arg21, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<2x64x16xf32> to tensor<64x16xf32>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = [[EVENT1]]
        %48 = hivm.hir.load ins(%extracted_slice : tensor<64x16xf32>) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
        // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = [[EVENT0]]
        %expanded_11 = tensor.expand_shape %6 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %49 = hivm.hir.vreduce <max> ins(%48 : tensor<64x16xf32>) outs(%expanded_11 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
        %collapsed = tensor.collapse_shape %49 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %50 = hivm.hir.vmul ins(%collapsed, %extracted : tensor<64xf32>, f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %51 = hivm.hir.vmax ins(%arg22, %50 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %52 = hivm.hir.vmul ins(%48, %extracted : tensor<64x16xf32>, f32) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
        %expanded_12 = tensor.expand_shape %51 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %53 = hivm.hir.vbrc ins(%expanded_12 : tensor<64x1xf32>) outs(%9 : tensor<64x16xf32>) broadcast_dims = [1] -> tensor<64x16xf32>
        %54 = hivm.hir.vsub ins(%52, %53 : tensor<64x16xf32>, tensor<64x16xf32>) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
        %55 = hivm.hir.vmul ins(%54, %cst_4 : tensor<64x16xf32>, f32) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
        %56 = hivm.hir.vexp ins(%55 : tensor<64x16xf32>) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
        %57 = hivm.hir.vreduce <sum> ins(%56 : tensor<64x16xf32>) outs(%expanded_11 : tensor<64x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<64x1xf32>
        %collapsed_13 = tensor.collapse_shape %57 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %58 = hivm.hir.vsub ins(%arg22, %51 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %59 = hivm.hir.vmul ins(%58, %cst_4 : tensor<64xf32>, f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %60 = hivm.hir.vexp ins(%59 : tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %61 = hivm.hir.vmul ins(%arg23, %60 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %62 = hivm.hir.vadd ins(%61, %collapsed_13 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %expanded_14 = tensor.expand_shape %60 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %63 = hivm.hir.vbrc ins(%expanded_14 : tensor<64x1xf32>) outs(%10 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
        %extracted_slice_15 = tensor.extract_slice %arg24[%arg21, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<2x64x64xf32> to tensor<64x64xf32>
        %64 = hivm.hir.vmul ins(%arg13, %63 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%extracted_slice_15 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %65 = hivm.hir.vcast ins(%56 : tensor<64x16xf32>) outs(%47 : tensor<64x16xf16>) -> tensor<64x16xf16>
        %extracted_slice_16 = tensor.extract_slice %arg25[%arg21, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<2x64x16xf16> to tensor<64x16xf16>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = [[EVENT2:[0-9]+]]
        %66 = hivm.hir.store ins(%65 : tensor<64x16xf16>) outs(%extracted_slice_16 : tensor<64x16xf16>) -> tensor<64x16xf16>
        // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = [[EVENT3:[0-9]+]]
        %67 = hivm.hir.vmul ins(%51, %extracted : tensor<64xf32>, f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %68 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %69 = hivm.hir.vdiv ins(%67, %68 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %inserted_slice = tensor.insert_slice %64 into %arg24[%arg21, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64xf32> into tensor<2x64x64xf32>
        %inserted_slice_17 = tensor.insert_slice %66 into %arg25[%arg21, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<64x16xf16> into tensor<2x64x16xf16>
        scf.yield %69, %62, %inserted_slice, %inserted_slice_17 : tensor<64xf32>, tensor<64xf32>, tensor<2x64x64xf32>, tensor<2x64x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      %37:3 = scf.for %arg21 = %c0 to %c2 step %c1 iter_args(%arg22 = %arg15, %arg23 = %arg17, %arg24 = %33) -> (memref<16x64xf16, strided<[?, ?], offset: ?>>, index, tensor<2x64x64xf32>) {
        %39 = arith.index_cast %arg11 : i32 to index
        %40 = arith.index_cast %c0_i32 : i32 to index
        %41 = arith.index_cast %c64_i32 : i32 to index
        %42 = arith.index_cast %c32_i32 : i32 to index
        %43 = affine.apply #map()[%arg21, %39, %40, %42]
        %44 = arith.index_cast %43 : index to i1
        %c3_i64 = arith.constant 3 : i64
        %c4_i64 = arith.constant 4 : i64
        %45 = arith.select %44, %c3_i64, %c4_i64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c2_i64 = arith.constant 2 : i64
        %46 = arith.select %44, %c1_i64, %c2_i64 : i64
        %alloc_11 = memref.alloc() : memref<16x64xf16>
        hivm.hir.load ins(%arg22 : memref<16x64xf16, strided<[?, ?], offset: ?>>) outs(%alloc_11 : memref<16x64xf16>)
        %47 = bufferization.to_tensor %alloc_11 restrict writable : memref<16x64xf16>
        %48 = tensor.empty() : tensor<64x16xf16>
        %extracted_slice = tensor.extract_slice %36#3[%arg21, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<2x64x16xf16> to tensor<64x16xf16>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = [[EVENT3]]
        %49 = hivm.hir.load ins(%extracted_slice : tensor<64x16xf16>) outs(%48 : tensor<64x16xf16>) -> tensor<64x16xf16>
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = [[EVENT2]]
        %50 = hivm.hir.mmadL1 ins(%49, %47, %true, %c64, %c16, %c64 : tensor<64x16xf16>, tensor<16x64xf16>, i1, index, index, index) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %extracted_slice_12 = tensor.extract_slice %arg24[%arg21, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<2x64x64xf32> to tensor<64x64xf32>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = [[EVENT4:[0-9]+]]
        %51 = hivm.hir.fixpipe {enable_nz2nd} ins(%50 : tensor<64x64xf32>) outs(%extracted_slice_12 : tensor<64x64xf32>) -> tensor<64x64xf32>
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = [[EVENT5:[0-9]+]]
        %52 = arith.addi %arg23, %c1024 : index
        %53 = arith.addi %52, %arg18 : index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1], offset: ?>>
        %cast_14 = memref.cast %reinterpret_cast_13 : memref<16x64xf16, strided<[64, 1], offset: ?>> to memref<16x64xf16, strided<[?, ?], offset: ?>>
        %inserted_slice = tensor.insert_slice %51 into %arg24[%arg21, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64xf32> into tensor<2x64x64xf32>
        scf.yield %cast_14, %53, %inserted_slice : memref<16x64xf16, strided<[?, ?], offset: ?>>, index, tensor<2x64x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %38 = scf.for %arg21 = %c0 to %c2 step %c1 iter_args(%arg22 = %arg13) -> (tensor<64x64xf32>) {
        %39 = arith.index_cast %arg11 : i32 to index
        %40 = arith.index_cast %c0_i32 : i32 to index
        %41 = arith.index_cast %c64_i32 : i32 to index
        %42 = arith.index_cast %c32_i32 : i32 to index
        %43 = affine.apply #map()[%arg21, %39, %40, %42]
        %44 = arith.index_cast %43 : index to i1
        %c1_i64 = arith.constant 1 : i64
        %c2_i64 = arith.constant 2 : i64
        %45 = arith.select %44, %c1_i64, %c2_i64 : i64
        %extracted_slice = tensor.extract_slice %37#2[%arg21, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<2x64x64xf32> to tensor<64x64xf32>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = [[EVENT5]]
        %46 = hivm.hir.load ins(%extracted_slice : tensor<64x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
        // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = [[EVENT4]]
        %extracted_slice_11 = tensor.extract_slice %36#2[%arg21, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<2x64x64xf32> to tensor<64x64xf32>
        %47 = hivm.hir.vadd ins(%46, %extracted_slice_11 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %47 : tensor<64x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      scf.yield %36#1, %38, %36#0, %37#0, %34#0, %37#1, %c0, %34#1, %c0 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<16x64xf16, strided<[?, ?], offset: ?>>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 2
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 2
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
    %19 = hivm.hir.vln ins(%18#0 : tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %20 = hivm.hir.vbrc ins(%cst_5 : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %21 = hivm.hir.vln ins(%20 : tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %22 = hivm.hir.vdiv ins(%19, %21 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %23 = hivm.hir.vadd ins(%18#2, %22 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %expanded = tensor.expand_shape %18#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %24 = hivm.hir.vbrc ins(%expanded : tensor<64x1xf32>) outs(%10 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
    %25 = hivm.hir.vdiv ins(%18#1, %24 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%13], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%23 : tensor<64xf32>) outs(%reinterpret_cast_10 : memref<64xf32, strided<[1], offset: ?>>)
    %26 = tensor.empty() : tensor<64x64xf16>
    %27 = hivm.hir.vcast ins(%25 : tensor<64x64xf32>) outs(%26 : tensor<64x64xf16>) -> tensor<64x64xf16>
    hivm.hir.store ins(%27 : tensor<64x64xf16>) outs(%reinterpret_cast_6 : memref<64x64xf16, strided<[64, 1], offset: ?>>)
    return
  }
}

// -----
module {
  func.func @matmul_x_w_bias_down_up_fused_layer_1_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32, %arg19: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<20xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.divsi %1, %arg19 : i32
    %3 = arith.remsi %2, %arg18 : i32
    %4 = arith.muli %arg19, %arg18 : i32
    %5 = arith.divsi %1, %4 : i32
    %6 = arith.remsi %5, %arg17 : i32
    hivm.hir.set_mask_norm
    %7 = arith.muli %6, %c32_i32 : i32
    %8 = arith.muli %3, %c32_i32 : i32
    %9 = arith.index_cast %7 : i32 to index
    %10 = arith.index_cast %arg12 : i32 to index
    %11 = arith.muli %9, %10 : index
    %12 = arith.index_cast %8 : i32 to index
    %13 = arith.index_cast %arg14 : i32 to index
    %14 = arith.addi %arg11, %c31_i32 : i32
    %15 = arith.divsi %14, %c32_i32 : i32
    %16 = arith.muli %arg13, %c32_i32 : i32
    %17 = arith.muli %arg14, %c32_i32 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [%10, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32, 64], strides: [%13, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1]>>
    %cast_1 = memref.cast %reinterpret_cast_0 : memref<32x64xf16, strided<[?, 1]>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %18 = tensor.empty() : tensor<32x64xf32>
    %19:9 = scf.for %arg20 = %c0_i32 to %15 step %c1_i32 iter_args(%arg21 = %18, %arg22 = %cast, %arg23 = %cast_1, %arg24 = %11, %arg25 = %c0, %arg26 = %12, %arg27 = %c0, %arg28 = %c0, %arg29 = %c0) -> (tensor<32x64xf32>, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index, index, index)  : i32 {
      %47 = arith.muli %arg20, %c32_i32 : i32
      %alloc_7 = memref.alloc() : memref<32x32xf16>
      %48 = arith.addi %9, %c32 : index
      %49 = arith.index_cast %arg9 : i32 to index
      %50 = arith.maxsi %9, %49 : index
      %51 = arith.minsi %48, %50 : index
      %52 = arith.subi %51, %9 : index
      %53 = arith.index_cast %47 : i32 to index
      %54 = arith.addi %53, %c32 : index
      %55 = arith.index_cast %arg11 : i32 to index
      %56 = arith.maxsi %53, %55 : index
      %57 = arith.minsi %54, %56 : index
      %58 = arith.subi %57, %53 : index
      %59 = arith.minsi %52, %c32 : index
      %60 = arith.minsi %58, %c32 : index
      %subview_8 = memref.subview %arg22[0, 0] [%59, %60] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_9 = memref.subview %alloc_7[0, 0] [%59, %60] [1, 1] : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
      hivm.hir.load ins(%subview_8 : memref<?x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_9 : memref<?x?xf16, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
      %61 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x32xf16>
      %alloc_10 = memref.alloc() : memref<32x64xf16>
      %subview_11 = memref.subview %arg23[0, 0] [%58, 64] [1, 1] : memref<32x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
      %subview_12 = memref.subview %alloc_10[0, 0] [%58, 64] [1, 1] : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
      hivm.hir.load ins(%subview_11 : memref<?x64xf16, strided<[?, ?], offset: ?>>) outs(%subview_12 : memref<?x64xf16, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
      %62 = bufferization.to_tensor %alloc_10 restrict writable : memref<32x64xf16>
      %63 = arith.cmpi eq, %arg20, %c0_i32 : i32
      %64 = hivm.hir.mmadL1 ins(%61, %62, %63, %59, %60, %c64 : tensor<32x32xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%arg21 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %65 = arith.addi %arg24, %c32 : index
      %66 = arith.addi %65, %arg25 : index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%66], sizes: [32, 32], strides: [%10, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
      %cast_14 = memref.cast %reinterpret_cast_13 : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %67 = arith.index_cast %16 : i32 to index
      %68 = arith.addi %arg26, %67 : index
      %69 = arith.addi %68, %arg27 : index
      %70 = arith.index_cast %17 : i32 to index
      %71 = arith.addi %arg28, %70 : index
      %72 = arith.addi %71, %arg29 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg5 to offset: [%72], sizes: [32, 64], strides: [%13, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
      %cast_16 = memref.cast %reinterpret_cast_15 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %64, %cast_14, %cast_16, %66, %c0, %69, %c0, %72, %c0 : tensor<32x64xf32>, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index, index, index
    }
    %20 = arith.muli %9, %c64 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg8 to offset: [%20], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    %21 = memref_ext.alloc_workspace() from %arg1 offset = [%c0] : from memref<?xi8> to memref<32x64xf16>
    %22 = bufferization.to_tensor %21 restrict writable : memref<32x64xf16>
    %23 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%19#0 : tensor<32x64xf32>) outs(%22 : tensor<32x64xf16>) -> tensor<32x64xf16>
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %24 = tensor.empty() : tensor<32x64xf16>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %25 = hivm.hir.load ins(%23 : tensor<32x64xf16>) outs(%24 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %26 = tensor.empty() : tensor<32x64xf16>
    // CHECK-NO: hivm.hir.sync_block[<BARRIER_CUBE>]
    %27 = hivm.hir.load ins(%23 : tensor<32x64xf16>) outs(%26 : tensor<32x64xf16>) -> tensor<32x64xf16>
    hivm.hir.store ins(%25 : tensor<32x64xf16>) outs(%reinterpret_cast_2 : memref<32x64xf16, strided<[64, 1], offset: ?>>)
    %28 = arith.index_cast %arg15 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%12], sizes: [64, 32], strides: [%28, 1] : memref<?xf16> to memref<64x32xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x32xf16>
    %29 = arith.addi %12, %c32 : index
    %30 = arith.index_cast %arg10 : i32 to index
    %31 = arith.maxsi %12, %30 : index
    %32 = arith.minsi %29, %31 : index
    %33 = arith.subi %32, %12 : index
    %subview = memref.subview %reinterpret_cast_3[0, 0] [64, %33] [1, 1] : memref<64x32xf16, strided<[?, 1], offset: ?>> to memref<64x?xf16, strided<[?, 1], offset: ?>>
    %subview_4 = memref.subview %alloc[0, 0] [64, %33] [1, 1] : memref<64x32xf16> to memref<64x?xf16, strided<[32, 1]>>
    hivm.hir.load ins(%subview : memref<64x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_4 : memref<64x?xf16, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
    %34 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf16>
    %35 = tensor.empty() : tensor<32x32xf32>
    %36 = hivm.hir.mmadL1 ins(%27, %34, %true, %c32, %c64, %33 : tensor<32x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%35 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %37 = arith.index_cast %arg16 : i32 to index
    %38 = arith.muli %9, %37 : index
    %39 = arith.addi %38, %12 : index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [32, 32], strides: [%37, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
    %40 = arith.addi %9, %c32 : index
    %41 = arith.index_cast %arg9 : i32 to index
    %42 = arith.maxsi %9, %41 : index
    %43 = arith.minsi %40, %42 : index
    %44 = arith.subi %43, %9 : index
    %45 = arith.minsi %44, %c32 : index
    %46 = arith.minsi %33, %c32 : index
    %extracted_slice = tensor.extract_slice %36[0, 0] [%45, %46] [1, 1] : tensor<32x32xf32> to tensor<?x?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%45, %46] [1, 1] : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%extracted_slice : tensor<?x?xf32>) outs(%subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----
module {
  func.func @matmul_x_w_bias_down_up_fused_layer_1_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<19xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg0
    %c8192 = arith.constant 8192 : index
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.divsi %1, %arg18 : i32
    %3 = arith.remsi %2, %arg17 : i32
    %4 = arith.muli %arg18, %arg17 : i32
    %5 = arith.divsi %1, %4 : i32
    %6 = arith.remsi %5, %arg16 : i32
    hivm.hir.set_mask_norm
    %7 = tensor.empty() : tensor<32x32xf32>
    %8 = arith.muli %6, %c32_i32 : i32
    %9 = arith.muli %3, %c32_i32 : i32
    %10 = arith.index_cast %8 : i32 to index
    %11 = arith.index_cast %arg11 : i32 to index
    %12 = arith.muli %10, %11 : index
    %13 = arith.index_cast %arg12 : i32 to index
    %14 = arith.index_cast %9 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1], offset: ?>>
    %15 = arith.index_cast %arg13 : i32 to index
    %16 = arith.addi %arg10, %c31_i32 : i32
    %17 = arith.divsi %16, %c32_i32 : i32
    %18 = arith.muli %arg12, %c32_i32 : i32
    %19 = arith.muli %arg13, %c32_i32 : i32
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%12], sizes: [32, 32], strides: [%11, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast_0 : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [32, 32], strides: [%13, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32, 64], strides: [%15, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1]>>
    %cast_4 = memref.cast %reinterpret_cast_3 : memref<32x64xf16, strided<[?, 1]>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %20 = tensor.empty() : tensor<32x32xf32>
    %21 = tensor.empty() : tensor<32x64xf32>
    %22:11 = scf.for %arg19 = %c0_i32 to %17 step %c1_i32 iter_args(%arg20 = %20, %arg21 = %21, %arg22 = %cast, %arg23 = %cast_2, %arg24 = %cast_4, %arg25 = %12, %arg26 = %c0, %arg27 = %14, %arg28 = %c0, %arg29 = %c0, %arg30 = %c0) -> (tensor<32x32xf32>, tensor<32x64xf32>, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index, index, index)  : i32 {
      %67 = arith.muli %arg19, %c32_i32 : i32
      %alloc_12 = memref.alloc() : memref<32x32xf16>
      %68 = arith.addi %10, %c32 : index
      %69 = arith.index_cast %arg8 : i32 to index
      %70 = arith.maxsi %10, %69 : index
      %71 = arith.minsi %68, %70 : index
      %72 = arith.subi %71, %10 : index
      %73 = arith.index_cast %67 : i32 to index
      %74 = arith.addi %73, %c32 : index
      %75 = arith.index_cast %arg10 : i32 to index
      %76 = arith.maxsi %73, %75 : index
      %77 = arith.minsi %74, %76 : index
      %78 = arith.subi %77, %73 : index
      %79 = arith.minsi %72, %c32 : index
      %80 = arith.minsi %78, %c32 : index
      %subview_13 = memref.subview %arg22[0, 0] [%79, %80] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_14 = memref.subview %alloc_12[0, 0] [%79, %80] [1, 1] : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
      hivm.hir.load ins(%subview_13 : memref<?x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_14 : memref<?x?xf16, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
      %81 = bufferization.to_tensor %alloc_12 restrict writable : memref<32x32xf16>
      %alloc_15 = memref.alloc() : memref<32x32xf16>
      %82 = arith.addi %14, %c32 : index
      %83 = arith.index_cast %arg9 : i32 to index
      %84 = arith.maxsi %14, %83 : index
      %85 = arith.minsi %82, %84 : index
      %86 = arith.subi %85, %14 : index
      %87 = arith.minsi %86, %c32 : index
      %subview_16 = memref.subview %arg23[0, 0] [%80, %87] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0, 0] [%80, %87] [1, 1] : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
      hivm.hir.load ins(%subview_16 : memref<?x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_17 : memref<?x?xf16, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
      %88 = bufferization.to_tensor %alloc_15 restrict writable : memref<32x32xf16>
      %alloc_18 = memref.alloc() : memref<32x64xf16>
      %subview_19 = memref.subview %arg24[0, 0] [%78, 64] [1, 1] : memref<32x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
      %subview_20 = memref.subview %alloc_18[0, 0] [%78, 64] [1, 1] : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
      hivm.hir.load ins(%subview_19 : memref<?x64xf16, strided<[?, ?], offset: ?>>) outs(%subview_20 : memref<?x64xf16, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
      %89 = bufferization.to_tensor %alloc_18 restrict writable : memref<32x64xf16>
      %90 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %91 = hivm.hir.mmadL1 ins(%81, %88, %90, %79, %80, %87 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%arg20 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %92 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %93 = hivm.hir.mmadL1 ins(%81, %89, %92, %79, %80, %c64 : tensor<32x32xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%arg21 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %94 = arith.addi %arg25, %c32 : index
      %95 = arith.addi %94, %arg26 : index
      %reinterpret_cast_21 = memref.reinterpret_cast %arg2 to offset: [%95], sizes: [32, 32], strides: [%11, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
      %cast_22 = memref.cast %reinterpret_cast_21 : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %96 = arith.index_cast %18 : i32 to index
      %97 = arith.addi %arg27, %96 : index
      %98 = arith.addi %97, %arg28 : index
      %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [32, 32], strides: [%13, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
      %cast_24 = memref.cast %reinterpret_cast_23 : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %99 = arith.index_cast %19 : i32 to index
      %100 = arith.addi %arg29, %99 : index
      %101 = arith.addi %100, %arg30 : index
      %reinterpret_cast_25 = memref.reinterpret_cast %arg5 to offset: [%101], sizes: [32, 64], strides: [%15, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
      %cast_26 = memref.cast %reinterpret_cast_25 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %91, %93, %cast_22, %cast_24, %cast_26, %95, %c0, %98, %c0, %101, %c0 : tensor<32x32xf32>, tensor<32x64xf32>, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index, index, index
    }
    %23 = memref_ext.alloc_workspace() from %arg1 offset = [%c0] : from memref<?xi8> to memref<32x32xf32>
    %24 = bufferization.to_tensor %23 restrict writable : memref<32x32xf32>
    %25 = hivm.hir.fixpipe {enable_nz2nd} ins(%22#0 : tensor<32x32xf32>) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %26 = tensor.empty() : tensor<32x32xf32>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %27 = hivm.hir.load ins(%25 : tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %alloc = memref.alloc() : memref<1x32xf16>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<1x32xf16> into memref<32xf16>
    %28 = arith.addi %14, %c32 : index
    %29 = arith.index_cast %arg9 : i32 to index
    %30 = arith.maxsi %14, %29 : index
    %31 = arith.minsi %28, %30 : index
    %32 = arith.subi %31, %14 : index
    %33 = arith.cmpi slt, %32, %c32 : index
    scf.if %33 {
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<32xf16>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%32] [1] : memref<32xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    %subview_5 = memref.subview %collapse_shape[0] [%32] [1] : memref<32xf16> to memref<?xf16, strided<[1]>>
    hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>>) outs(%subview_5 : memref<?xf16, strided<[1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
    %34 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf16>
    %35 = tensor.empty() : tensor<1x32xf32>
    %36 = hivm.hir.vcast ins(%34 : tensor<1x32xf16>) outs(%35 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %37 = hivm.hir.vbrc ins(%36 : tensor<1x32xf32>) outs(%7 : tensor<32x32xf32>) broadcast_dims = [0] -> tensor<32x32xf32>
    %38 = hivm.hir.vadd ins(%27, %37 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %39 = arith.index_cast %arg14 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%14], sizes: [64, 32], strides: [%39, 1] : memref<?xf16> to memref<64x32xf16, strided<[?, 1], offset: ?>>
    %alloc_7 = memref.alloc() : memref<64x32xf16>
    %subview_8 = memref.subview %reinterpret_cast_6[0, 0] [64, %32] [1, 1] : memref<64x32xf16, strided<[?, 1], offset: ?>> to memref<64x?xf16, strided<[?, 1], offset: ?>>
    %subview_9 = memref.subview %alloc_7[0, 0] [64, %32] [1, 1] : memref<64x32xf16> to memref<64x?xf16, strided<[32, 1]>>
    hivm.hir.load ins(%subview_8 : memref<64x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_9 : memref<64x?xf16, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
    %40 = bufferization.to_tensor %alloc_7 restrict writable : memref<64x32xf16>
    %41 = memref_ext.alloc_workspace() from %arg1 offset = [%c4096] : from memref<?xi8> to memref<32x64xf16>
    %42 = bufferization.to_tensor %41 restrict writable : memref<32x64xf16>
    %43 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%22#1 : tensor<32x64xf32>) outs(%42 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %44 = tensor.empty() : tensor<32x64xf16>
    // CHECK-NO: hivm.hir.sync_block[<BARRIER_CUBE>]
    %45 = hivm.hir.load ins(%43 : tensor<32x64xf16>) outs(%44 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %46 = tensor.empty() : tensor<32x32xf32>
    %47 = hivm.hir.mmadL1 ins(%45, %40, %true, %c32, %c64, %32 : tensor<32x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%46 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %48 = memref_ext.alloc_workspace() from %arg1 offset = [%c8192] : from memref<?xi8> to memref<32x32xf32>
    %49 = bufferization.to_tensor %48 restrict writable : memref<32x32xf32>
    %50 = hivm.hir.fixpipe {enable_nz2nd} ins(%47 : tensor<32x32xf32>) outs(%49 : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %51 = tensor.empty() : tensor<32x32xf32>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    %52 = hivm.hir.load ins(%50 : tensor<32x32xf32>) outs(%51 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %53 = tensor.empty() : tensor<32x32xf32>
    %54 = hivm.hir.vadd ins(%52, %38 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%53 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %55 = arith.index_cast %arg15 : i32 to index
    %56 = arith.muli %10, %55 : index
    %57 = arith.addi %56, %14 : index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%57], sizes: [32, 32], strides: [%55, 1] : memref<?xf16> to memref<32x32xf16, strided<[?, 1], offset: ?>>
    %58 = tensor.empty() : tensor<32x32xf16>
    %59 = hivm.hir.vcast ins(%54 : tensor<32x32xf32>) outs(%58 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %60 = arith.addi %10, %c32 : index
    %61 = arith.index_cast %arg8 : i32 to index
    %62 = arith.maxsi %10, %61 : index
    %63 = arith.minsi %60, %62 : index
    %64 = arith.subi %63, %10 : index
    %65 = arith.minsi %64, %c32 : index
    %66 = arith.minsi %32, %c32 : index
    %extracted_slice = tensor.extract_slice %59[0, 0] [%65, %66] [1, 1] : tensor<32x32xf16> to tensor<?x?xf16>
    %subview_11 = memref.subview %reinterpret_cast_10[0, 0] [%65, %66] [1, 1] : memref<32x32xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview_11 : memref<?x?xf16, strided<[?, 1], offset: ?>>)
    return
  }
}

