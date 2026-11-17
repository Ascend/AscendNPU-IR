// RUN: bishengir-opt %s -hivm-delayed-cross-core-gss | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(hivm-delayed-cross-core-gss{solver-version=v1})" -o /dev/null

// cv-preloading stages that are not adjacent rounds. The loop is sliced into
// max_preload_num = 3 stages and the conflicting pair sits in stages 2 and 0,
// so the stage distance is 2 and the flag rotates over a window of 3 - one
// more than the usual back-to-back pair.
//
// The producing fixpipe and the consuming copy each sit one scope deeper than
// the preload scope, so the sync has to follow them in there: the set right
// after the producer, the wait right before the consumer. Nothing about the
// window size changes that.
//
// The expected sync is checked where it lands, next to the op it guards; the
// backup kernel below carries the anchor skeleton the solver runs on.

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @nested_preload_kernel_backup(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-merge-same-preload-scopes">, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          hivm.hir.anchor {id = 5 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 6 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 7 : i64}
          hivm.hir.anchor {id = 8 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 9 : i64}
      scope.return
      } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
      hivm.hir.anchor {id = 10 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 11 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 12 : i64}
          hivm.hir.anchor {id = 13 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 14 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.anchor {id = 16 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 17 : i64}
      scope.return
      } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 18 : i64}
    }
    hivm.hir.anchor {id = 19 : i64}
    return
  }
  // ---- producer side, stage 2 ----
  // CHECK-LABEL: func.func @nested_preload_kernel_mix_aic
  func.func @nested_preload_kernel_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    %c102400_i64 = arith.constant 102400 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c327680_i64 = arith.constant 327680 : i64
    %cbIn = hivm.hir.pointer_cast(%c0_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
    %cbB = hivm.hir.pointer_cast(%c65536_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64, %c65536_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ccA = hivm.hir.pointer_cast(%c0_i64, %c65536_i64, %c131072_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccA {hivm.multi_buffer = 3 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ccB = hivm.hir.pointer_cast(%c196608_i64, %c262144_i64, %c327680_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccB {hivm.multi_buffer = 3 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ubC = hivm.hir.pointer_cast(%c73728_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          // CHECK:      hivm.hir.mmadL1
          hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
          // CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
          // CHECK-NEXT: hivm.hir.fixpipe
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubA : memref<32x32xf32, #hivm.address_space<ub>>)
          // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>]
          hivm.hir.anchor {id = 5 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 6 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 7 : i64}
          hivm.hir.anchor {id = 8 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 9 : i64}
      scope.return
      // CHECK:      hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32
      } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
      hivm.hir.anchor {id = 10 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 11 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 12 : i64}
          hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubC : memref<32x32xf32, #hivm.address_space<ub>>)
          hivm.hir.anchor {id = 13 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 14 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.anchor {id = 16 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 17 : i64}
      scope.return
      } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 18 : i64}
    }
    hivm.hir.anchor {id = 19 : i64}
    return
  }
  // ---- consumer side, stage 0 ----
  // CHECK-LABEL: func.func @nested_preload_kernel_mix_aiv
  func.func @nested_preload_kernel_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    %c102400_i64 = arith.constant 102400 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c327680_i64 = arith.constant 327680 : i64
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64, %c65536_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1b = hivm.hir.pointer_cast(%c69632_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV2 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          hivm.hir.anchor {id = 5 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 6 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 7 : i64}
          hivm.hir.copy ins(%ubV1 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV1b : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 8 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 9 : i64}
      scope.return
      } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
      hivm.hir.anchor {id = 10 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 11 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 12 : i64}
          hivm.hir.anchor {id = 13 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 14 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 15 : i64}
          // CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>]
          // CHECK-NEXT: hivm.hir.copy
          hivm.hir.copy ins(%ubA : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV2 : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]
          hivm.hir.anchor {id = 16 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 17 : i64}
      scope.return
      // CHECK:      hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32
      } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 18 : i64}
    }
    hivm.hir.anchor {id = 19 : i64}
    return
  }
}
