// UNSUPPORTED: bishengir_published
// RUN: bishengir-opt -pass-pipeline="builtin.module(hivm-insert-anchors-and-backup{insert-anchor-only-before-cube-ops=false insert-anchor-only-before-vector-ops=false insert-anchor-before-cube-and-vector-ops=true})" -split-input-file -verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0 - d1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK: func.func @triton_dot_2
  func.func @triton_dot_2(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    // CHECK-NEXT: hivm.hir.anchor {id = 0 : i64}
    %c2_i32 = arith.constant 2 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = tensor.empty() : tensor<7x8xf32>
    // CHECK: hivm.hir.anchor {id = 1 : i64}
    // CHECK-NEXT: hivm.hir.vbrc
    %3 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%2 : tensor<7x8xf32>) -> tensor<7x8xf32>
    // CHECK-NEXT: hivm.hir.anchor {id = 2 : i64}
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8, 9], strides: [9, 1] : memref<?xf32> to memref<8x9xf32, strided<[9, 1]>>
    %alloc = memref.alloc() : memref<2x1x16x8xf32>
    // CHECK: hivm.hir.anchor {id = 3 : i64}
    // CHECK-NEXT: hivm.hir.nd2nz
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<8x9xf32, strided<[9, 1]>>) outs(%alloc : memref<2x1x16x8xf32>)
    // CHECK-NEXT: hivm.hir.anchor {id = 4 : i64}
    %4 = bufferization.to_tensor %alloc restrict writable : memref<2x1x16x8xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xf32> to memref<7x8xf32, strided<[8, 1]>>
    // CHECK: memref.alloca
    %alloca = memref.alloca() {normalize_matmul_counter} : memref<i32>
    // CHECK-NEXT: hivm.hir.anchor {id = 5 : i64}
    // CHECK: memref.store
    memref.store %c0_i32, %alloca[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
    // CHECK-NEXT: hivm.hir.anchor {id = 6 : i64}
    %5 = tensor.empty() : tensor<1x1x16x16xf32>
    // CHECK: scf.for
    %6:2 = scf.for %arg10 = %c0_i32 to %arg6 step %c2_i32 iter_args(%arg11 = %3, %arg12 = %5) -> (tensor<7x8xf32>, tensor<1x1x16x16xf32>)  : i32 {
      // CHECK-NEXT: hivm.hir.anchor {id = 7 : i64}
      %11 = arith.index_cast %arg6 : i32 to index
      %12 = arith.index_cast %arg10 : i32 to index
      %13 = affine.apply #map(%11, %12)
      %14 = arith.minui %13, %c2 : index
      // CHECK: hivm.hir.anchor {id = 8 : i64}
      // CHECK-NEXT: scf.for
      %15 = scf.for %arg13 = %c0 to %14 step %c1 iter_args(%arg14 = %arg11) -> (tensor<7x8xf32>) {
        // CHECK-NEXT: hivm.hir.anchor {id = 9 : i64}
        %alloc_4 = memref.alloc() : memref<7x8xf32>
        // CHECK: hivm.hir.anchor {id = 10 : i64}
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%reinterpret_cast_0 : memref<7x8xf32, strided<[8, 1]>>) outs(%alloc_4 : memref<7x8xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} eviction_policy = <EvictFirst> core_type = <VECTOR>
        // CHECK-NEXT: hivm.hir.anchor {id = 11 : i64}
        %17 = bufferization.to_tensor %alloc_4 restrict writable : memref<7x8xf32>
        annotation.mark %17 {cv_pipeline_lazy_load = true} : tensor<7x8xf32>
        // CHECK: hivm.hir.anchor {id = 12 : i64}
        // CHECK-NEXT: hivm.hir.vabs
        %18 = hivm.hir.vabs ins(%17 : tensor<7x8xf32>) outs(%2 : tensor<7x8xf32>) -> tensor<7x8xf32>
        // CHECK: hivm.hir.anchor {id = 13 : i64}
        // CHECK-NEXT: scf.yield
        scf.yield %18 : tensor<7x8xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      // CHECK: hivm.hir.anchor {id = 14 : i64}
      // CHECK-NEXT: scf.for
      %16 = scf.for %arg13 = %c0 to %14 step %c1 iter_args(%arg14 = %arg12) -> (tensor<1x1x16x16xf32>) {
        // CHECK-NEXT: hivm.hir.anchor {id = 15 : i64}
        %alloc_4 = memref.alloc() : memref<1x1x16x8xf32>
        // CHECK: hivm.hir.anchor {id = 16 : i64}
        // CHECK-NEXT: hivm.hir.nd2nz
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<7x8xf32, strided<[8, 1]>>) outs(%alloc_4 : memref<1x1x16x8xf32>)
        // CHECK-NEXT: hivm.hir.anchor {id = 17 : i64}
        %17 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x1x16x8xf32>
        annotation.mark %17 {cv_pipeline_lazy_load = true} : tensor<1x1x16x8xf32>
        // CHECK: hivm.hir.anchor {id = 18 : i64}
        // CHECK-NEXT: memref.load
        %18 = memref.load %alloca[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
        // CHECK-NEXT: hivm.hir.anchor {id = 19 : i64}
        %19 = arith.cmpi eq, %18, %c0_i32 : i32
        // CHECK: hivm.hir.mmadL1
        %20 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C} ins(%17, %4, %19, %c7, %c8, %c9 : tensor<1x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index) outs(%arg14 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        // CHECK-NEXT: hivm.hir.anchor {id = 20 : i64}
        %21 = arith.addi %18, %c1_i32 : i32
        memref.store %21, %alloca[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
        // CHECK: hivm.hir.anchor {id = 21 : i64}
        // CHECK-NEXT: scf.yield
        scf.yield %20 : tensor<1x1x16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      // CHECK: hivm.hir.anchor {id = 22 : i64}
      // CHECK-NEXT: scf.yield
      scf.yield %15, %16 : tensor<7x8xf32>, tensor<1x1x16x16xf32>
    } {cv_unrolled_loop}
    %alloc_1 = memref.alloc() : memref<7x9xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_1 : memref<7x9xf32, #hivm.address_space<ub>> to memref<7x9xf32>
    %7 = bufferization.to_tensor %memspacecast restrict writable : memref<7x9xf32>
    // CHECK: hivm.hir.anchor {id = 23 : i64}
    // CHECK-NEXT: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6#1 : tensor<1x1x16x16xf32>) outs(%alloc_1 : memref<7x9xf32, #hivm.address_space<ub>>)
    // CHECK-NEXT: hivm.hir.anchor {id = 24 : i64}
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
    // CHECK: memref.load
    %8 = memref.load %alloca[] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<i32>
    // CHECK-NEXT: hivm.hir.anchor {id = 25 : i64}
    %9 = arith.cmpi eq, %8, %c0_i32 : i32
    // CHECK: scf.if
    %10 = scf.if %9 -> (tensor<7x9xf32>) {
      // CHECK-NEXT: hivm.hir.anchor {id = 26 : i64}
      %11 = tensor.empty() : tensor<7x9xf32>
      // CHECK: hivm.hir.anchor {id = 27 : i64}
      // CHECK-NEXT: hivm.hir.vbrc
      %12 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%11 : tensor<7x9xf32>) -> tensor<7x9xf32>
      // CHECK-NEXT: hivm.hir.anchor {id = 28 : i64}
      // CHECK-NEXT: scf.yield
      scf.yield %12 : tensor<7x9xf32>
    } else {
      // CHECK: hivm.hir.anchor {id = 29 : i64}
      // CHECK-NEXT: scf.yield
      scf.yield %7 : tensor<7x9xf32>
    }
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xf32> to memref<7x8xf32, strided<[8, 1]>>
    // CHECK: hivm.hir.anchor {id = 30 : i64}
    // CHECK-NEXT: hivm.hir.store
    hivm.hir.store ins(%6#0 : tensor<7x8xf32>) outs(%reinterpret_cast_2 : memref<7x8xf32, strided<[8, 1]>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    // CHECK-NEXT: hivm.hir.anchor {id = 31 : i64}
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [7, 9], strides: [9, 1] : memref<?xf32> to memref<7x9xf32, strided<[9, 1]>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%10 : tensor<7x9xf32>) outs(%reinterpret_cast_3 : memref<7x9xf32, strided<[9, 1]>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    // CHECK-NEXT: hivm.hir.anchor {id = 32 : i64}
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}

