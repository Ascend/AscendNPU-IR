// RUN: bishengir-opt %s -hivm-delayed-cross-core-gss | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(hivm-delayed-cross-core-gss{solver-version=v1})" -o /dev/null

// Limits on how deep a cv-preloading set/wait may sink. The flag has to fire
// exactly once per iteration of the preload scope, so two kinds of nesting
// between that scope and the conflicting op stop the descent:
//
//  - an inner loop, because sinking in would repeat the flag on every inner
//    iteration with nothing to balance it;
//  - a condition whose else branch holds no position to carry the twin flag,
//    as with an `scf.if` that has no else in the source: sinking into the then
//    branch alone would skip the flag whenever that branch is not taken.
//
// In both kernels the producing fixpipe is wrapped, so the sync sinks into the
// inner core scope but brackets the wrapper instead of entering it.
//
// The expected sync is checked where it lands, next to the op it guards; the
// backup kernel below carries the anchor skeleton the solver runs on.

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @k_for_backup(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-merge-same-preload-scopes">, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          scf.for %argw = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
            hivm.hir.anchor {id = 5 : i64}
            hivm.hir.anchor {id = 6 : i64}
          }
          hivm.hir.anchor {id = 7 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 8 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 11 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      hivm.hir.anchor {id = 12 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 13 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 14 : i64}
          scf.for %argw = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
          }
          hivm.hir.anchor {id = 17 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 18 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.anchor {id = 20 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 21 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 22 : i64}
    }
    hivm.hir.anchor {id = 23 : i64}
    return
  }
  // ---- inner loop: sync brackets the scf.for ----
  // CHECK-LABEL: func.func @k_for_mix_aic
  func.func @k_for_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ccA = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccA {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ccB = hivm.hir.pointer_cast(%c131072_i64, %c196608_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccB {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ubC = hivm.hir.pointer_cast(%c73728_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          // Sinking into the loop would repeat the flag per inner iteration,
          // so the pair brackets it.
          // CHECK:      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
          // CHECK-NEXT: scf.for
          scf.for %argw = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
            hivm.hir.anchor {id = 5 : i64}
            hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
            // CHECK:        hivm.hir.fixpipe
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubA : memref<32x32xf32, #hivm.address_space<ub>>)
            // CHECK-NEXT:   hivm.hir.anchor
            hivm.hir.anchor {id = 6 : i64}
          // CHECK-NEXT: {{^ *\}$}}
          }
          // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>]
          hivm.hir.anchor {id = 7 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 8 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 11 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      hivm.hir.anchor {id = 12 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 13 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 14 : i64}
          scf.for %argw = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubC : memref<32x32xf32, #hivm.address_space<ub>>)
            hivm.hir.anchor {id = 16 : i64}
          }
          hivm.hir.anchor {id = 17 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 18 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.anchor {id = 20 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 21 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 22 : i64}
    }
    hivm.hir.anchor {id = 23 : i64}
    return
  }
  func.func @k_for_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1 = hivm.hir.pointer_cast(%c65536_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1b = hivm.hir.pointer_cast(%c69632_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV2 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          scf.for %argw = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
            hivm.hir.anchor {id = 5 : i64}
            hivm.hir.anchor {id = 6 : i64}
          }
          hivm.hir.anchor {id = 7 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 8 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.copy ins(%ubV1 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV1b : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 10 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 11 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      hivm.hir.anchor {id = 12 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 13 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 14 : i64}
          scf.for %argw = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
          }
          hivm.hir.anchor {id = 17 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 18 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.copy ins(%ubA : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV2 : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 20 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 21 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 22 : i64}
    }
    hivm.hir.anchor {id = 23 : i64}
    return
  }
  func.func @k_if_backup(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-merge-same-preload-scopes">, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          %cnd = arith.cmpi sgt, %arg9, %c0_i32 : i32
          scf.if %cnd {
            hivm.hir.anchor {id = 5 : i64}
            hivm.hir.anchor {id = 6 : i64}
          }
          hivm.hir.anchor {id = 7 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 8 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 11 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      hivm.hir.anchor {id = 12 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 13 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 14 : i64}
          %cnd = arith.cmpi sgt, %arg9, %c0_i32 : i32
          scf.if %cnd {
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
          }
          hivm.hir.anchor {id = 17 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 18 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.anchor {id = 20 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 21 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 22 : i64}
    }
    hivm.hir.anchor {id = 23 : i64}
    return
  }
  // ---- condition without a usable else branch: sync brackets the scf.if ----
  // CHECK-LABEL: func.func @k_if_mix_aic
  func.func @k_if_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ccA = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccA {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ccB = hivm.hir.pointer_cast(%c131072_i64, %c196608_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      annotation.mark %ccB {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %ubC = hivm.hir.pointer_cast(%c73728_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          %cnd = arith.cmpi sgt, %arg9, %c0_i32 : i32
          // No else branch in the source, so there is nowhere to put a twin
          // flag and the pair brackets the condition.
          // CHECK:      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
          // CHECK-NEXT: scf.if
          scf.if %cnd {
            hivm.hir.anchor {id = 5 : i64}
            hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
            // CHECK:        hivm.hir.fixpipe
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccA : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubA : memref<32x32xf32, #hivm.address_space<ub>>)
            // CHECK-NEXT:   hivm.hir.anchor
            hivm.hir.anchor {id = 6 : i64}
          // CHECK-NEXT: {{^ *\}$}}
          }
          // CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>]
          hivm.hir.anchor {id = 7 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 8 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 11 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      hivm.hir.anchor {id = 12 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 13 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 14 : i64}
          %cnd = arith.cmpi sgt, %arg9, %c0_i32 : i32
          scf.if %cnd {
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%cbIn, %cbB, %true, %c64, %c32, %c32 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ccB : memref<2x4x16x16xf32, #hivm.address_space<cc>>) outs(%ubC : memref<32x32xf32, #hivm.address_space<ub>>)
            hivm.hir.anchor {id = 16 : i64}
          }
          hivm.hir.anchor {id = 17 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 18 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.anchor {id = 20 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 21 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 22 : i64}
    }
    hivm.hir.anchor {id = 23 : i64}
    return
  }
  func.func @k_if_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg9 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %ubA = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %ubA {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1 = hivm.hir.pointer_cast(%c65536_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV1b = hivm.hir.pointer_cast(%c69632_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %ubV2 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 2 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 3 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 4 : i64}
          %cnd = arith.cmpi sgt, %arg9, %c0_i32 : i32
          scf.if %cnd {
            hivm.hir.anchor {id = 5 : i64}
            hivm.hir.anchor {id = 6 : i64}
          }
          hivm.hir.anchor {id = 7 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 8 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.copy ins(%ubV1 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV1b : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 10 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 11 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      hivm.hir.anchor {id = 12 : i64}
      scope.scope : () -> () {
        hivm.hir.anchor {id = 13 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 14 : i64}
          %cnd = arith.cmpi sgt, %arg9, %c0_i32 : i32
          scf.if %cnd {
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
          }
          hivm.hir.anchor {id = 17 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
        hivm.hir.anchor {id = 18 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 19 : i64}
          hivm.hir.copy ins(%ubA : memref<32x32xf32, #hivm.address_space<ub>>) outs(%ubV2 : memref<32x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 20 : i64}
        scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
        hivm.hir.anchor {id = 21 : i64}
      scope.return
      } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      hivm.hir.anchor {id = 22 : i64}
    }
    hivm.hir.anchor {id = 23 : i64}
    return
  }
}
