// RUN: bishengir-opt %s -canonicalize -cse -hivm-mark-real-core-type -hivm-inject-block-sync | FileCheck %s

// Verify that canonicalization before mark-real-core-type cleans up dead ops
// so they don't get marked with ambiguous core types (CUBE_OR_VECTOR) that
// would trigger assertion failures in inject-block-sync.
// This tests the pipeline fix that adds canonicalize before mark-real-core-type
// in hivmCrossCoreSyncPipeline.

module {
  // CHECK-LABEL: func.func @test_canon_before_mark_core_type
  func.func @test_canon_before_mark_core_type(
      %arg0: memref<16xf32, #hivm.address_space<gm>>,
      %arg1: memref<256xf32, #hivm.address_space<gm>>,
      %arg2: memref<256xf32, #hivm.address_space<gm>>,
      %arg3: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>})
      attributes {hacc.always_inline,
      hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
      hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64

    // CUBE side: matmul writing to gm via fixpipe
    %0 = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%1 : memref<16xf32, #hivm.address_space<cbuf>>)
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>,
                        memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index)
                        outs(%2 : memref<256xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%2 : memref<256xf32, #hivm.address_space<cc>>)
                     outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>)

    // CHECK: hivm.hir.sync_block_set{{\[}}<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    // CHECK: hivm.hir.sync_block_wait{{\[}}<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0

    // VECTOR side: load from gm + vector compute + store
    %5 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %4 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<256xf32, #hivm.address_space<gm>>)
                  outs(%5 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%5, %4 : memref<256xf32, #hivm.address_space<ub>>,
                  memref<256xf32, #hivm.address_space<ub>>)
                  outs(%5 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%5 : memref<256xf32, #hivm.address_space<ub>>)
                  outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)

    // Dead scalar ops: a memref.load/store to a local alloc never read back.
    // Canonicalize removes these; without canonicalize, mark-real-core-type
    // marks them as CUBE_OR_VECTOR which crashes inject-block-sync.
    %dead = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %c0 = arith.constant 0 : index
    %val = memref.load %5[%c0] : memref<256xf32, #hivm.address_space<ub>>
    memref.store %val, %dead[%c0] : memref<256xf32, #hivm.address_space<ub>>

    // CHECK-NOT: memref.load
    // CHECK-NOT: memref.store {{.*}} memref<256xf32, #hivm.address_space<ub>>
    // CHECK: return
    return
  }
}
