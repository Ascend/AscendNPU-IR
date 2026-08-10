// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{force-is-reg-based=true}))" -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  // CHECK: test_block_sync_loop
  func.func @test_block_sync_loop(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<?xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<256xf32, #hivm.address_space<gm>>, %arg3: memref<256xf32, #hivm.address_space<gm>>, %arg4: memref<256xf32, #hivm.address_space<gm>>, %arg5: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg5
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %true_0 = arith.constant true
    %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    %alloc_1 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    %alloc_3 = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    %alloc_4 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1024_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c1024_i64) : memref<256xf32, #hivm.address_space<cc>>
    %alloc_6 = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    %alloc_7 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c2048_i64) : memref<256xf32, #hivm.address_space<cc>>
    %alloc_workspace_0 = memref_ext.alloc_workspace() from %arg1 offset = [%c0] : from memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, #hivm.address_space<gm>>
    %alloc_workspace_1 = memref_ext.alloc_workspace() from %arg1 offset = [%c1024] : from memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, #hivm.address_space<gm>>
    %alloc_workspace_2 = memref_ext.alloc_workspace() from %arg1 offset = [%c2048] : from memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, #hivm.address_space<gm>>
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: scf.for
    scf.for %arg8 = %c0 to %c16 step %c4 {
      scf.for %arg9 = %c0 to %c2 step %c1 {
        hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
        hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
        hivm.hir.mmadL1 ins(%alloc, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.fixpipe
        hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%alloc_workspace_1 : memref<256xf32, #hivm.address_space<gm>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_6 : memref<16xf32, #hivm.address_space<cbuf>>)
        hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%4 : memref<16xf32, #hivm.address_space<cbuf>>)
        hivm.hir.mmadL1 ins(%alloc_6, %4, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%5 : memref<256xf32, #hivm.address_space<cc>>)
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.fixpipe
        hivm.hir.fixpipe {enable_nz2nd} ins(%5 : memref<256xf32, #hivm.address_space<cc>>) outs(%alloc_workspace_0 : memref<256xf32, #hivm.address_space<gm>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
      } {multibuffer_unroll_factor = 2 : i32}
      // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
      // CHECK-NEXT: scf.if
      scf.if %true_0 {
        hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_3 : memref<16xf32, #hivm.address_space<cbuf>>)
        hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%2 : memref<16xf32, #hivm.address_space<cbuf>>)
        hivm.hir.mmadL1 ins(%alloc_3, %2, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%3 : memref<256xf32, #hivm.address_space<cc>>)
        hivm.hir.fixpipe {enable_nz2nd} ins(%3 : memref<256xf32, #hivm.address_space<cc>>) outs(%alloc_workspace_2 : memref<256xf32, #hivm.address_space<gm>>)
      }
      // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
      // CHECK-NEXT: scf.for
      scf.for %arg9 = %c0 to %c2 step %c1 {
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%alloc_workspace_1 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<256xf32, #hivm.address_space<ub>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        hivm.hir.vadd ins(%alloc_1, %alloc_2 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.store ins(%alloc_1 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
      } {multibuffer_unroll_factor = 2 : i32}
      scf.for %arg9 = %c0 to %c2 step %c1 {
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%alloc_workspace_0 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_7 : memref<256xf32, #hivm.address_space<ub>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        hivm.hir.vadd ins(%alloc_7, %alloc_8 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.store ins(%alloc_7 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<256xf32, #hivm.address_space<gm>>)
      } {multibuffer_unroll_factor = 2 : i32}
      // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
      // CHECK-NEXT: scf.if
      scf.if %true_0 {
        hivm.hir.load ins(%alloc_workspace_2 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_4 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_4, %alloc_5 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%alloc_4 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.store ins(%alloc_4 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg4 : memref<256xf32, #hivm.address_space<gm>>)
      }
      // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    } {cv_unrolled_loop}
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: return
    return
  }
}

// -----
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (d0 + d1, 1024)>
#map3 = affine_map<(d0, d1)[s0] -> ((d0 - d1) ceildiv s0)>
module {
  func.func @_attn_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %c131072 = arith.constant 131072 : index
    %c393216 = arith.constant 393216 : index
    %c1024 = arith.constant 1024 : index
    %c512_i32 = arith.constant 512 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %c2_i32 = arith.constant 2 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg8, %arg9 : i32
    %1 = arith.muli %0, %arg10 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg10 : i32
    %5 = arith.remsi %4, %arg9 : i32
    %6 = arith.muli %arg10, %arg9 : i32
    %7 = arith.divsi %3, %6 : i32
    %8 = arith.remsi %7, %arg8 : i32
    %9 = tensor.empty() : tensor<128x64xf32>
    %10 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_2 : f32) outs(%9 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %11 = tensor.empty() : tensor<128x128xf32>
    %12 = tensor.empty() : tensor<128xf32>
    %13 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_0 : f32) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
    %14 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
    %15 = arith.divsi %5, %c2_i32 : i32
    %16 = arith.remsi %5, %c2_i32 : i32
    %17 = arith.extsi %15 : i32 to i64
    %18 = arith.muli %17, %c131072_i64 : i64
    %19 = arith.extsi %16 : i32 to i64
    %20 = arith.muli %19, %c65536_i64 : i64
    %21 = arith.addi %18, %20 : i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = arith.muli %8, %c128_i32 : i32
    %24 = arith.maxsi %23, %c0_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = affine.apply #map()[%25]
    %27 = affine.apply #map1()[%26, %22]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [128, 64], strides: [64, 1] : memref<?xbf16> to memref<128x64xbf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%27], sizes: [128, 64], strides: [64, 1] : memref<?xbf16> to memref<128x64xbf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x64xbf16>
    %28 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<128x64xbf16, strided<[64, 1], offset: ?>>
    %29 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
    %30 = hivm.hir.load ins(%28 : tensor<128x64xbf16>) outs(%29 : tensor<128x64xbf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} -> tensor<128x64xbf16>
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: {{.*}}scf.for{{.*}}
    %31:5 = scf.for %arg11 = %c0_i32 to %c1024_i32 step %c512_i32 iter_args(%arg12 = %14, %arg13 = %10, %arg14 = %13, %arg15 = %c0_i32, %arg16 = %c0_i32) -> (tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, i32, i32)  : i32 {
      %35 = memref_ext.alloc_workspace() from %arg2 offset = [%c393216] : from memref<?xi8> to memref<4x128x128xbf16>
      %36 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<4x128x64xf32>
      %37 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8> to memref<4x128x128xf32>
      %38 = arith.index_cast %arg11 : i32 to index
      %39 = affine.min #map2(%38, %c512, %c1024)
      %40 = affine.apply #map3(%39, %38)[%c128]
      annotation.mark %37 : memref<4x128x128xf32>
      annotation.mark %35 : memref<4x128x128xbf16>
      annotation.mark %36 : memref<4x128x64xf32>
      %41 = scf.for %arg17 = %c0 to %40 step %c1 iter_args(%arg18 = %arg16) -> (i32) {
        %49 = arith.index_cast %arg18 : i32 to index
        %50 = affine.apply #map()[%49]
        %51 = affine.apply #map1()[%50, %22]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [128, 64], strides: [64, 1] : memref<?xbf16> to memref<128x64xbf16, strided<[64, 1], offset: ?>>
        %alloc_5 = memref.alloc() : memref<128x64xbf16>
        %52 = bufferization.to_tensor %reinterpret_cast_4 restrict writable : memref<128x64xbf16, strided<[64, 1], offset: ?>>
        %53 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x64xbf16>
        %54 = hivm.hir.load ins(%52 : tensor<128x64xbf16>) outs(%53 : tensor<128x64xbf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} -> tensor<128x64xbf16>
        %55 = tensor.empty() : tensor<128x128xf32>
        %56 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%30, %54, %true, %c128, %c64, %c128 : tensor<128x64xbf16>, tensor<128x64xbf16>, i1, index, index, index) outs(%55 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %subview = memref.subview %37[%arg17, 0, 0] [1, 128, 128] [1, 1, 1] : memref<4x128x128xf32> to memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>> into memref<128x128xf32, strided<[128, 1], offset: ?>>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} {{.*}}
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%56 : tensor<128x128xf32>) outs(%collapse_shape : memref<128x128xf32, strided<[128, 1], offset: ?>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        %57 = arith.addi %arg18, %c128_i32 : i32
        scf.yield %57 : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      %42 = bufferization.to_tensor %37 restrict : memref<4x128x128xf32>
      %43 = tensor.empty() : tensor<4x128xf32>
      %44:3 = scf.for %arg17 = %c0 to %40 step %c1 iter_args(%arg18 = %arg14, %arg19 = %arg12, %arg20 = %43) -> (tensor<128xf32>, tensor<128xf32>, tensor<4x128xf32>) {
        %49 = tensor.empty() : tensor<128x128xf32>
        %extracted_slice = tensor.extract_slice %42[%arg17, 0, 0] [1, 128, 128] [1, 1, 1] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<4x128x128xf32> to tensor<128x128xf32>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: {{.*}}hivm.hir.load{{.*}}
        %50 = hivm.hir.load ins(%extracted_slice : tensor<128x128xf32>) outs(%49 : tensor<128x128xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} -> tensor<128x128xf32>
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        %51 = hivm.hir.vmul ins(%50, %cst_1 : tensor<128x128xf32>, f32) outs(%11 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %52 = tensor.empty() : tensor<128x1xf32>
        %53 = hivm.hir.vreduce <max> ins(%51 : tensor<128x128xf32>) outs(%52 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed = tensor.collapse_shape %53 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %54 = tensor.empty() : tensor<128xi1>
        %55 = hivm.hir.vcmp ins(%arg18, %arg18 : tensor<128xf32>, tensor<128xf32>) outs(%54 : tensor<128xi1>) -> tensor<128xi1>
        %56 = hivm.hir.vnot ins(%55 : tensor<128xi1>) outs(%54 : tensor<128xi1>) -> tensor<128xi1>
        %57 = hivm.hir.vcmp ins(%collapsed, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%54 : tensor<128xi1>) -> tensor<128xi1>
        %58 = hivm.hir.vnot ins(%57 : tensor<128xi1>) outs(%54 : tensor<128xi1>) -> tensor<128xi1>
        %59 = hivm.hir.vmax ins(%arg18, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
        %60 = hivm.hir.vsel ins(%56, %collapsed, %59 : tensor<128xi1>, tensor<128xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
        %61 = hivm.hir.vsel ins(%58, %arg18, %60 : tensor<128xi1>, tensor<128xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_4 = tensor.expand_shape %61 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %62 = hivm.hir.vsub ins(%51, %expanded_4 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%11 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %63 = hivm.hir.vexp ins(%62 : tensor<128x128xf32>) outs(%11 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %64 = tensor.empty() : tensor<128x128xbf16>
        %65 = hivm.hir.vcast ins(%63 : tensor<128x128xf32>) outs(%64 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %subview = memref.subview %35[%arg17, 0, 0] [1, 128, 128] [1, 1, 1] : memref<4x128x128xbf16> to memref<1x128x128xbf16, strided<[16384, 128, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x128xbf16, strided<[16384, 128, 1], offset: ?>> into memref<128x128xbf16, strided<[128, 1], offset: ?>>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
        // CHECK-NEXT: {{.*}}hivm.hir.store{{.*}}
        hivm.hir.store ins(%65 : tensor<128x128xbf16>) outs(%collapse_shape : memref<128x128xbf16, strided<[128, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = {{.*}}
        %66 = tensor.empty() : tensor<128x1xf32>
        %67 = hivm.hir.vreduce <sum> ins(%63 : tensor<128x128xf32>) outs(%66 : tensor<128x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed_5 = tensor.collapse_shape %67 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %68 = hivm.hir.vsub ins(%arg18, %61 : tensor<128xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
        %extracted_slice_6 = tensor.extract_slice %arg20[%arg17, 0] [1, 128] [1, 1] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<4x128xf32> to tensor<128xf32>
        %69 = hivm.hir.vexp ins(%68 : tensor<128xf32>) outs(%extracted_slice_6 : tensor<128xf32>) -> tensor<128xf32>
        %70 = hivm.hir.vmul ins(%arg19, %69 : tensor<128xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
        %71 = hivm.hir.vadd ins(%70, %collapsed_5 : tensor<128xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
        %inserted_slice = tensor.insert_slice %69 into %arg20[%arg17, 0] [1, 128] [1, 1] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<128xf32> into tensor<4x128xf32>
        scf.yield %61, %71, %inserted_slice : tensor<128xf32>, tensor<128xf32>, tensor<4x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      %45 = bufferization.to_tensor %35 restrict : memref<4x128x128xbf16>
      %46 = scf.for %arg17 = %c0 to %40 step %c1 iter_args(%arg18 = %arg15) -> (i32) {
        %49 = arith.index_cast %arg18 : i32 to index
        %50 = affine.apply #map()[%49]
        %51 = affine.apply #map1()[%50, %22]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [128, 64], strides: [64, 1] : memref<?xbf16> to memref<128x64xbf16, strided<[64, 1], offset: ?>>
        %52 = tensor.empty() : tensor<128x128xbf16>
        %extracted_slice = tensor.extract_slice %45[%arg17, 0, 0] [1, 128, 128] [1, 1, 1] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : tensor<4x128x128xbf16> to tensor<128x128xbf16>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: {{.*}}hivm.hir.load{{.*}}
        %53 = hivm.hir.load ins(%extracted_slice : tensor<128x128xbf16>) outs(%52 : tensor<128x128xbf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} -> tensor<128x128xbf16>
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
        %alloc_5 = memref.alloc() : memref<128x64xbf16>
        %54 = bufferization.to_tensor %reinterpret_cast_4 restrict writable : memref<128x64xbf16, strided<[64, 1], offset: ?>>
        %55 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x64xbf16>
        %56 = hivm.hir.load ins(%54 : tensor<128x64xbf16>) outs(%55 : tensor<128x64xbf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} -> tensor<128x64xbf16>
        %57 = tensor.empty() : tensor<128x64xf32>
        %58 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%53, %56, %true, %c128, %c128, %c64 : tensor<128x128xbf16>, tensor<128x64xbf16>, i1, index, index, index) outs(%57 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %subview = memref.subview %36[%arg17, 0, 0] [1, 128, 64] [1, 1, 1] : memref<4x128x64xf32> to memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf32, strided<[8192, 64, 1], offset: ?>> into memref<128x64xf32, strided<[64, 1], offset: ?>>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} {{.*}}
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%58 : tensor<128x64xf32>) outs(%collapse_shape : memref<128x64xf32, strided<[64, 1], offset: ?>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        %59 = arith.addi %arg18, %c128_i32 : i32
        scf.yield %59 : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      %47 = bufferization.to_tensor %36 restrict : memref<4x128x64xf32>
      %48 = scf.for %arg17 = %c0 to %40 step %c1 iter_args(%arg18 = %arg13) -> (tensor<128x64xf32>) {
        %extracted_slice = tensor.extract_slice %44#2[%arg17, 0] [1, 128] [1, 1] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<4x128xf32> to tensor<128xf32>
        %expanded_4 = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %49 = hivm.hir.vmul ins(%arg18, %expanded_4 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%9 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
        %50 = tensor.empty() : tensor<128x64xf32>
        %extracted_slice_5 = tensor.extract_slice %47[%arg17, 0, 0] [1, 128, 64] [1, 1, 1] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<4x128x64xf32> to tensor<128x64xf32>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = {{.*}}
        // CHECK-NEXT: {{.*}}hivm.hir.load{{.*}}
        %51 = hivm.hir.load ins(%extracted_slice_5 : tensor<128x64xf32>) outs(%50 : tensor<128x64xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} -> tensor<128x64xf32>
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
        %52 = tensor.empty() : tensor<128x64xf32>
        %53 = hivm.hir.vadd ins(%51, %49 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%52 : tensor<128x64xf32>) -> tensor<128x64xf32>
        scf.yield %53 : tensor<128x64xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      scf.yield %44#1, %48, %44#0, %46, %41 : tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, i32, i32
    } {cv_unrolled_loop}
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = {{.*}}
    %expanded = tensor.expand_shape %31#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %32 = hivm.hir.vdiv ins(%31#1, %expanded : tensor<128x64xf32>, tensor<128x1xf32>) outs(%9 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
    %33 = tensor.empty() : tensor<128x64xbf16>
    %34 = hivm.hir.vcast ins(%32 : tensor<128x64xf32>) outs(%33 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
    hivm.hir.store ins(%34 : tensor<128x64xbf16>) outs(%reinterpret_cast_3 : memref<128x64xbf16, strided<[64, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
