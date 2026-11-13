// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" | FileCheck %s --check-prefix=GSS --implicit-check-not=hivm.hir.sync_block
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 enable-hivm-inject-barrier-all-sync=true})" | FileCheck %s --check-prefix=BARRIER --implicit-check-not=hivm.hir.set_flag --implicit-check-not=hivm.hir.wait_flag --implicit-check-not=hivm.hir.sync_block
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 enable-hivm-inject-block-all-sync=true disable-hivm-auto-inject-sync=true})" | FileCheck %s --check-prefix=BLOCK --implicit-check-not=hivm.hir.set_ffts_base_addr --implicit-check-not=hivm.hir.set_flag --implicit-check-not=hivm.hir.wait_flag
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 enable-hivm-inject-barrier-all-sync=true enable-hivm-inject-block-all-sync=true})" | FileCheck %s --check-prefix=COMBINED --implicit-check-not=hivm.hir.set_ffts_base_addr --implicit-check-not=hivm.hir.set_flag --implicit-check-not=hivm.hir.wait_flag
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend950PR_9579 enable-hivm-inject-barrier-all-sync=true enable-hivm-inject-block-all-sync=true})" | FileCheck %s --check-prefix=COMBINED --implicit-check-not=hivm.hir.set_ffts_base_addr --implicit-check-not=hivm.hir.set_flag --implicit-check-not=hivm.hir.wait_flag
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 enable-hivm-inject-barrier-all-sync=true enable-hivm-inject-block-all-sync=true disable-hivm-auto-inject-sync=true disable-auto-inject-block-sync=true})" | FileCheck %s --check-prefix=DISABLED --implicit-check-not=hivm.hir.pipe_barrier --implicit-check-not=hivm.hir.sync_block --implicit-check-not=hivm.hir.set_flag --implicit-check-not=hivm.hir.wait_flag

// Exercise real memory operations through the full v2 pipeline. The MIX has no
// FFTS argument: reg-based block-all synchronization must work without one.
// GSS orders the load/store dependency with flags by default.
// GSS-LABEL: func.func @copy_mix_aiv(
// GSS:      hivm.hir.load
// GSS-NEXT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <[[EVENT:EVENT_ID[0-9]+]]>]
// GSS:      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <[[EVENT]]>]
// GSS-NEXT: hivm.hir.store

// Barrier-all replaces the dependency flags with PIPE_ALL barriers.
// BARRIER-LABEL: func.func @copy_mix_aiv(
// BARRIER:      hivm.hir.pointer_cast
// BARRIER-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// BARRIER-NEXT: hivm.hir.load
// BARRIER:      hivm.hir.pipe_barrier[<PIPE_ALL>]
// BARRIER-NEXT: hivm.hir.store
// BARRIER-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]

// GSS block-all inserts matching handshakes on both split kernels before
// each memory operation, even when intra-core auto-sync is disabled.
// BLOCK-LABEL: func.func @copy_mix_aic(
// BLOCK:      hivm.hir.pipe_barrier[<PIPE_ALL>]
// BLOCK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// BLOCK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
// BLOCK-LABEL: func.func @copy_mix_aiv(
// BLOCK:      hivm.hir.pipe_barrier[<PIPE_ALL>]
// BLOCK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// BLOCK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
// BLOCK-NEXT: hivm.hir.load
// BLOCK:      hivm.hir.pipe_barrier[<PIPE_ALL>]
// BLOCK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// BLOCK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
// BLOCK-NEXT: hivm.hir.store

// Both all-sync options can be enabled together on A5 and A6.
// COMBINED-LABEL: func.func @copy_mix_aic(
// COMBINED:      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// COMBINED-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// COMBINED-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
// COMBINED-LABEL: func.func @copy_mix_aiv(
// COMBINED:      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// COMBINED-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// COMBINED-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
// COMBINED-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// COMBINED-NEXT: hivm.hir.load
// COMBINED:      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// COMBINED-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// COMBINED-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
// COMBINED-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// COMBINED-NEXT: hivm.hir.store
// COMBINED-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]

// The disable flags take precedence over the all-sync options while preserving
// the actual load and store.
// DISABLED-LABEL: func.func @copy_mix_aiv(
// DISABLED: hivm.hir.load
// DISABLED: hivm.hir.store
// DISABLED: return
module {
  func.func @copy(%src: memref<16x16xf32, #hivm.address_space<gm>>,
                  %dst: memref<16x16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %buf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16x16xf32, #hivm.address_space<gm>>)
                  outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
                   outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
    return
  }
}
