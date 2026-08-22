// UNSUPPORTED: bishengir_published
// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{force-is-mem-based=true}))" %s | FileCheck %s --check-prefix=DEFAULT
// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{force-is-mem-based=true round-robin-event-ids=false}))" %s | FileCheck %s --check-prefix=ORIGINAL
// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{force-is-reg-based=true}))" %s | FileCheck %s --check-prefix=REGBASE

module {
  func.func @round_robin_cross_core_event_ids(
      %ffts_base_addr: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg0: memref<256xf32, #hivm.address_space<cc>>,
      %arg1: memref<256xf32, #hivm.address_space<cc>>,
      %arg2: memref<256xf32, #hivm.address_space<cc>>,
      %arg3: memref<256xf32, #hivm.address_space<cc>>,
      %arg4: memref<256xf32, #hivm.address_space<gm>>,
      %arg5: memref<256xf32, #hivm.address_space<gm>>,
      %arg6: memref<256xf32, #hivm.address_space<gm>>,
      %arg7: memref<256xf32, #hivm.address_space<gm>>) attributes {
      hivm.func_core_type = #hivm.func_core_type<MIX>
    } {
    %alloc0 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc1 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc2 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc3 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg0 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg4 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // ORIGINAL: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // ORIGINAL: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg4 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc0 : memref<256xf32, #hivm.address_space<ub>>)

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg5 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
    // ORIGINAL: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // ORIGINAL: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg5 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc1 : memref<256xf32, #hivm.address_space<ub>>)

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg2 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg6 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 2
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 2
    // ORIGINAL: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // ORIGINAL: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg6 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc2 : memref<256xf32, #hivm.address_space<ub>>)

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg3 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg7 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 3
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 3
    // ORIGINAL: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // ORIGINAL: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // REGBASE: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg7 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc3 : memref<256xf32, #hivm.address_space<ub>>)
    return
  }

  // Three repeated allocations do not reach the fallback threshold.
  // DEFAULT-LABEL: func.func @keep_default_event_ids_for_three_repeats
  func.func @keep_default_event_ids_for_three_repeats(
      %ffts_base_addr: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg0: memref<256xf32, #hivm.address_space<cc>>,
      %arg1: memref<256xf32, #hivm.address_space<cc>>,
      %arg2: memref<256xf32, #hivm.address_space<cc>>,
      %arg3: memref<256xf32, #hivm.address_space<gm>>,
      %arg4: memref<256xf32, #hivm.address_space<gm>>,
      %arg5: memref<256xf32, #hivm.address_space<gm>>) attributes {
      hivm.func_core_type = #hivm.func_core_type<MIX>
    } {
    %alloc0 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc1 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %alloc2 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg0 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg3 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg3 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc0 : memref<256xf32, #hivm.address_space<ub>>)

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg4 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg4 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc1 : memref<256xf32, #hivm.address_space<ub>>)

    hivm.hir.fixpipe {enable_nz2nd} ins(%arg2 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg5 : memref<256xf32, #hivm.address_space<gm>>)
    // DEFAULT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    // DEFAULT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    hivm.hir.load ins(%arg5 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc2 : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}
