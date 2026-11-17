// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{solver-version=v1}))" %s | FileCheck %s
// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{solver-version=v2}))" %s | FileCheck %s

// cv-preloading set/wait placement on the non-delayed cross-core path, flat
// preload scope vs. one extra scope level in between. This is the same pair of
// shapes as hivm-dccgss-preload-sink-into-scope.mlir, written as a single mix
// kernel so the solver sees both cores in one function instead of through the
// anchor skeleton of a backup kernel.
//
// Both kernels carry the same single cross-core preload edge: %ubA (ub,
// multi_buffer=2) is written by CUBE at preload_num=1 and read by VECTOR at
// preload_num=0, so diff=1 and the window is 2.
//
// The sync must sit next to the op it guards in both shapes: the set right
// after the producing fixpipe, the wait right before the consuming copy. The
// unrelated mmadL1 has to stay ahead of the wait, which is what pins the
// placement to the op rather than to the preload scope body.

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // ---- flat: the conflicting op sits directly in the preload scope body ----
  // CHECK-LABEL: func.func @k_flat
  func.func @k_flat(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0_i64 = arith.constant 0 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c69632_i64 = arith.constant 69632 : i64
    %c73728_i64 = arith.constant 73728 : i64
    %c77824_i64 = arith.constant 77824 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %cbIn = hivm.hir.pointer_cast(%c0_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    %cbB = hivm.hir.pointer_cast(%c65536_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ccA = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccA {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ccB = hivm.hir.pointer_cast(%c131072_i64, %c196608_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccB {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ubC = hivm.hir.pointer_cast(%c73728_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1 = hivm.hir.pointer_cast(%c65536_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1b = hivm.hir.pointer_cast(%c69632_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV2 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      scope.scope : () -> () {
        // CHECK:      hivm.hir.mmadL1
        hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
        // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
        // CHECK-NEXT: hivm.hir.fixpipe
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubA : memref<32x32xf32, #hivm.address_space<ub>>)
        // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>]
      scope.return
      // CHECK:      hivm.preload_num = 1
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      scope.scope : () -> () {
        hivm.hir.copy ins(%ubV1 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV1b : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      scope.scope : () -> () {
        hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubC : memref<32x32xf32, #hivm.address_space<ub>>)
      scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scope.scope : () -> () {
        // CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>]
        // CHECK-NEXT: hivm.hir.copy
        hivm.hir.copy ins(%ubA : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV2 : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]
      scope.return
      // CHECK:      hivm.preload_num = 0
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    }
    return
  }

  // ---- nested: an inner core scope sits between preload scope and the op ----
  // The sync has to follow the op into that inner scope, so it still appears
  // before the inner scope's own terminator and core-type attribute.
  // CHECK-LABEL: func.func @k_nested
  func.func @k_nested(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0_i64 = arith.constant 0 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c69632_i64 = arith.constant 69632 : i64
    %c73728_i64 = arith.constant 73728 : i64
    %c77824_i64 = arith.constant 77824 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %cbIn = hivm.hir.pointer_cast(%c0_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    %cbB = hivm.hir.pointer_cast(%c65536_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ccA = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccA {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ccB = hivm.hir.pointer_cast(%c131072_i64, %c196608_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccB {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ubC = hivm.hir.pointer_cast(%c73728_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1 = hivm.hir.pointer_cast(%c65536_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1b = hivm.hir.pointer_cast(%c69632_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV2 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      scope.scope : () -> () {
        scope.scope : () -> () {
          // CHECK:      hivm.hir.mmadL1
          hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
          // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
          // CHECK-NEXT: hivm.hir.fixpipe
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubA : memref<32x32xf32, #hivm.address_space<ub>>)
          // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>]
        // CHECK-NEXT: scope.return
        scope.return
        // CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        scope.scope : () -> () {
          hivm.hir.copy ins(%ubV1 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV1b : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      scope.scope : () -> () {
        scope.scope : () -> () {
          hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubC : memref<32x32xf32, #hivm.address_space<ub>>)
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        scope.scope : () -> () {
          // CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>]
          // CHECK-NEXT: hivm.hir.copy
          hivm.hir.copy ins(%ubA : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV2 : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]
        // CHECK-NEXT: scope.return
        scope.return
        // CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    }
    return
  }
}
