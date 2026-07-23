// RUN: bishengir-opt -hivm-graph-sync-solver -hivm-lower-multi-buffer-counter -split-input-file %s | FileCheck %s

// ============================================================================
// GraphSyncSolver MmadMxL1Op sync injection tests
// ============================================================================

// -----
// TC-01: Scale operand independent event ID allocation (core feature)
module {
  // CHECK-LABEL: func.func @test_mmad_mx_scale_independent_sync
  func.func @test_mmad_mx_scale_independent_sync(
      %gmA     : memref<256x128xf8E5M2, #hivm.address_space<gm>>,
      %gmScaleA: memref<256x4xui8,    #hivm.address_space<gm>>,
      %gmB     : memref<128x256xf8E5M2, #hivm.address_space<gm>>,
      %gmScaleB: memref<256x4xui8,    #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64    = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c128      = arith.constant 128 : index
    %c256      = arith.constant 256 : index
    %c1        = arith.constant 1 : i1

    %bufA      = hivm.hir.pointer_cast(%c0_i64)
        : memref<256x128xf8E5M2, #hivm.address_space<cbuf>>
    %bufScaleA = hivm.hir.pointer_cast(%c65536_i64)
        : memref<256x4xui8,    #hivm.address_space<cbuf>>
    %bufB      = hivm.hir.pointer_cast(%c131072_i64)
        : memref<128x256xf8E5M2, #hivm.address_space<cbuf>>
    %bufScaleB = hivm.hir.pointer_cast(%c196608_i64)
        : memref<256x4xui8,    #hivm.address_space<cbuf>>
    %bufC      = memref.alloc() : memref<256x256xf32>

    // MTE2 DMA: LoadOp transfers A
    hivm.hir.load ins(%gmA : memref<256x128xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufA : memref<256x128xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    // MTE2 DMA: LoadMXScaleOp transfers ScaleA
    hivm.hir.load_scale ins(%gmScaleA : memref<256x4xui8, #hivm.address_space<gm>>)
        outs(%bufScaleA : memref<256x4xui8, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]

    // MTE2 DMA: LoadOp transfers B
    hivm.hir.load ins(%gmB : memref<128x256xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufB : memref<128x256xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]

    // MTE2 DMA: LoadMXScaleOp transfers ScaleB
    hivm.hir.load_scale ins(%gmScaleB : memref<256x4xui8, #hivm.address_space<gm>>)
        outs(%bufScaleB : memref<256x4xui8, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]

    // M→MTE1 Pre-Set before the first Mmad
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]

    // MmadMxL1Op: 8 sync_related_args
    // [0]=ID0:Wait A, [1]=ID1:Wait ScaleA, [2]=ID2:Wait B, [3]=ID3:Wait ScaleB
    // [4..7]=-1: no downstream MTE2 consumers
    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %bufScaleA, %bufScaleB, %c1, %c256, %c128, %c256 :
            memref<256x128xf8E5M2, #hivm.address_space<cbuf>>,
            memref<128x256xf8E5M2, #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC : memref<256x256xf32>)
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: sync_related_args(
    // CHECK-SAME:   %c0_i64, %c1_i64, %c2_i64, %c3_i64
    // CHECK-SAME:   : i64, i64, i64, i64, i64, i64, i64, i64)

    // M→MTE1 Post-Wait after the last Mmad
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    return
  }
}

// -----
// TC-02: Partial conflict — only ScaleA has a DMA producer
module {
  // CHECK-LABEL: func.func @test_mmad_mx_scale_only_conflict
  func.func @test_mmad_mx_scale_only_conflict(
      %gmScaleA: memref<256x4xui8, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64    = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c128      = arith.constant 128 : index
    %c256      = arith.constant 256 : index
    %c1        = arith.constant 1 : i1

    // L1 buffers for A and B have no DMA producer (no upstream MTE2 conflict)
    %bufA      = hivm.hir.pointer_cast(%c0_i64)
        : memref<256x128xf8E5M2, #hivm.address_space<cbuf>>
    %bufB      = hivm.hir.pointer_cast(%c131072_i64)
        : memref<128x256xf8E5M2, #hivm.address_space<cbuf>>
    %bufC      = memref.alloc() : memref<256x256xf32>

    // ScaleA has a DMA producer → RAW conflict with LoadL0ScaleA
    %bufScaleA = hivm.hir.pointer_cast(%c65536_i64)
        : memref<256x4xui8, #hivm.address_space<cbuf>>
    hivm.hir.load_scale ins(%gmScaleA : memref<256x4xui8, #hivm.address_space<gm>>)
        outs(%bufScaleA : memref<256x4xui8, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    // ScaleB has no DMA producer
    %bufScaleB = hivm.hir.pointer_cast(%c196608_i64)
        : memref<256x4xui8, #hivm.address_space<cbuf>>

    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %bufScaleA, %bufScaleB, %c1, %c256, %c128, %c256 :
            memref<256x128xf8E5M2, #hivm.address_space<cbuf>>,
            memref<128x256xf8E5M2, #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC : memref<256x256xf32>)
    // [0]=-1:Wait A, [1]=ID0:Wait ScaleA, [2]=-1:Wait B, [3]=-1:Wait ScaleB
    // CHECK: sync_related_args(
    // CHECK-SAME: %c-1_i64, %c0_i64, %c-1_i64, %c-1_i64
    return
  }
}

// -----
// TC-03: M→MTE1 unified Pre-Set/Post-Wait for standalone MmadMxL1Op
module {
  // CHECK-LABEL: func.func @test_mmad_mx_mte1_pingpong
  func.func @test_mmad_mx_mte1_pingpong(
      %gmA     : memref<16xf8E5M2, #hivm.address_space<gm>>,
      %gmB     : memref<16xf8E5M2, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64  = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c192_i64 = arith.constant 192 : i64
    %true    = arith.constant true
    %c16     = arith.constant 16 : index
    %c256    = arith.constant 256 : index

    %bufA      = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf8E5M2, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%gmA : memref<16xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    %bufB      = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf8E5M2, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%gmB : memref<16xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]

    %scaleA = hivm.hir.pointer_cast(%c128_i64)
        : memref<1xui8, #hivm.address_space<cbuf>>
    %scaleB = hivm.hir.pointer_cast(%c192_i64)
        : memref<1xui8, #hivm.address_space<cbuf>>
    %bufC = memref.alloc() : memref<256xf32>

    // Pre-Set(M→MTE1) before the first (and only) Mmad
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]

    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %scaleA, %scaleB, %true, %c16, %c256, %c16 :
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<1xui8, #hivm.address_space<cbuf>>,
            memref<1xui8, #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC : memref<256xf32>)
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: sync_related_args(
    // CHECK-SAME: %c0_i64, %c-1_i64, %c1_i64, %c-1_i64
    // CHECK-SAME: : i64, i64, i64, i64, i64, i64, i64, i64)

    // Post-Wait(M→MTE1) after the last (and only) Mmad
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    return
  }
}

// -----
// TC-04: MmadL1Op + MmadMxL1Op interleaved — shared M→MTE1 event IDs
module {
  // CHECK-LABEL: func.func @test_mmad_l1_and_mx_interleaved
  func.func @test_mmad_l1_and_mx_interleaved(
      %gmA : memref<16xf8E5M2, #hivm.address_space<gm>>,
      %gmB : memref<16xf8E5M2, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64   = arith.constant 0 : i64
    %c64_i64  = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c192_i64 = arith.constant 192 : i64
    %true     = arith.constant true
    %c16      = arith.constant 16 : index
    %c256     = arith.constant 256 : index

    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf8E5M2, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%gmA : memref<16xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf8E5M2, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%gmB : memref<16xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]

    %bufC1 = memref.alloc() : memref<256xf32>
    %bufC2 = memref.alloc() : memref<256xf32>

    // Pre-Set(M→MTE1) before the FIRST Mmad (MmadMxL1Op)
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]

    // First Mmad: MmadMxL1Op
    %scaleA = hivm.hir.pointer_cast(%c128_i64)
        : memref<1xui8, #hivm.address_space<cbuf>>
    %scaleB = hivm.hir.pointer_cast(%c192_i64)
        : memref<1xui8, #hivm.address_space<cbuf>>
    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %scaleA, %scaleB, %true, %c16, %c256, %c16 :
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<1xui8, #hivm.address_space<cbuf>>,
            memref<1xui8, #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC1 : memref<256xf32>)
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: sync_related_args(

    // No extra Set/Wait between Mmad ops
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID{{[0-1]}}>]

    // Second Mmad: MmadL1Op (reads same L1 buffers)
    // CHECK: hivm.hir.mmadL1
    hivm.hir.mmadL1
        ins(%bufA, %bufB, %true, %c16, %c256, %c16 :
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC2 : memref<256xf32>)

    // Post-Wait(M→MTE1) after the LAST Mmad (MmadL1Op)
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    return
  }
}

// -----
// TC-05: MmadMxL1Op inside a loop — backward sync with onFirstIter/onLastIter
module {
  // CHECK-LABEL: func.func @test_mmad_mx_in_loop
  func.func @test_mmad_mx_in_loop(
      %gmA     : memref<16xf8E5M2, #hivm.address_space<gm>>,
      %gmB     : memref<16xf8E5M2, #hivm.address_space<gm>>,
      %gmScaleA: memref<1xui8, #hivm.address_space<gm>>,
      %gmScaleB: memref<1xui8, #hivm.address_space<gm>>,
      %gmOut   : memref<256xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64   = arith.constant 0 : i64
    %c64_i64  = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c192_i64 = arith.constant 192 : i64
    %true     = arith.constant true
    %c16      = arith.constant 16 : index
    %c256     = arith.constant 256 : index
    %c0       = arith.constant 0 : index
    %c4       = arith.constant 4 : index
    %c1       = arith.constant 1 : index

    // Pre-Set for backward sync (MTE1→MTE2 + M→MTE1) outside the loop
    // CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    scf.for %iv = %c0 to %c4 step %c1 {
      %bufA = hivm.hir.pointer_cast(%c0_i64)
          : memref<16xf8E5M2, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>
      hivm.hir.load ins(%gmA : memref<16xf8E5M2, #hivm.address_space<gm>>)
          outs(%bufA : memref<16xf8E5M2, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>

      %bufB = hivm.hir.pointer_cast(%c64_i64)
          : memref<16xf8E5M2, #hivm.address_space<cbuf>>
      hivm.hir.load ins(%gmB : memref<16xf8E5M2, #hivm.address_space<gm>>)
          outs(%bufB : memref<16xf8E5M2, #hivm.address_space<cbuf>>)

      %scaleA = hivm.hir.pointer_cast(%c128_i64)
          : memref<1xui8, #hivm.address_space<cbuf>>
      hivm.hir.load_scale ins(%gmScaleA : memref<1xui8, #hivm.address_space<gm>>)
          outs(%scaleA : memref<1xui8, #hivm.address_space<cbuf>>)

      %scaleB = hivm.hir.pointer_cast(%c192_i64)
          : memref<1xui8, #hivm.address_space<cbuf>>
      hivm.hir.load_scale ins(%gmScaleB : memref<1xui8, #hivm.address_space<gm>>)
          outs(%scaleB : memref<1xui8, #hivm.address_space<cbuf>>)

      %bufC = memref.alloc() : memref<256xf32>

      // MmadMxL1Op inside the loop: must have sync_related_args
      // CHECK: hivm.hir.mmadmxL1
      // CHECK-SAME: sync_related_args(
      hivm.hir.mmadmxL1
          ins(%bufA, %bufB, %scaleA, %scaleB, %true, %c16, %c256, %c16 :
              memref<16xf8E5M2, #hivm.address_space<cbuf>>,
              memref<16xf8E5M2, #hivm.address_space<cbuf>>,
              memref<1xui8, #hivm.address_space<cbuf>>,
              memref<1xui8, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%bufC : memref<256xf32>)
    }
    // Post-Wait after the loop
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>
    return
  }
}

// -----
// TC-06: No sync conflicts — M→MTE1 sync only, no sync_related_args
module {
  // CHECK-LABEL: func.func @test_mmad_mx_no_conflict_default_args
  func.func @test_mmad_mx_no_conflict_default_args()
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64   = arith.constant 0 : i64
    %c64_i64  = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c192_i64 = arith.constant 192 : i64
    %true     = arith.constant true
    %c16      = arith.constant 16 : index
    %c256     = arith.constant 256 : index

    // All buffers are pointer_cast only — no DMA producers
    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf8E5M2, #hivm.address_space<cbuf>>
    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf8E5M2, #hivm.address_space<cbuf>>
    %scaleA = hivm.hir.pointer_cast(%c128_i64)
        : memref<1xui8, #hivm.address_space<cbuf>>
    %scaleB = hivm.hir.pointer_cast(%c192_i64)
        : memref<1xui8, #hivm.address_space<cbuf>>
    %bufC = memref.alloc() : memref<256xf32>

    // M→MTE1 Pre-Set is still inserted
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]

    // MmadMxL1Op without any MTE2 conflicts: no sync_related_args
    // CHECK: hivm.hir.mmadmxL1
    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %scaleA, %scaleB, %true, %c16, %c256, %c16 :
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<16xf8E5M2, #hivm.address_space<cbuf>>,
            memref<1xui8, #hivm.address_space<cbuf>>,
            memref<1xui8, #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC : memref<256xf32>)
    // CHECK-NOT: sync_related_args

    // M→MTE1 Post-Wait
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    return
  }
}

// -----
// TC-07: LoadMXScaleOp participates as ordinary MTE2 DMA
module {
  // CHECK-LABEL: func.func @test_load_scale_sync
  func.func @test_load_scale_sync(
      %gmScaleA: memref<256x4xui8, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64     = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c128       = arith.constant 128 : index
    %c256       = arith.constant 256 : index
    %c1         = arith.constant 1 : i1

    %bufA      = hivm.hir.pointer_cast(%c0_i64)
        : memref<256x128xf8E5M2, #hivm.address_space<cbuf>>
    %bufScaleA = hivm.hir.pointer_cast(%c65536_i64)
        : memref<256x4xui8, #hivm.address_space<cbuf>>
    %bufB      = hivm.hir.pointer_cast(%c131072_i64)
        : memref<128x256xf8E5M2, #hivm.address_space<cbuf>>
    %bufScaleB = hivm.hir.pointer_cast(%c196608_i64)
        : memref<256x4xui8, #hivm.address_space<cbuf>>
    %bufC      = memref.alloc() : memref<256x256xf32>

    // Only ScaleA has a DMA producer
    hivm.hir.load_scale ins(%gmScaleA : memref<256x4xui8, #hivm.address_space<gm>>)
        outs(%bufScaleA : memref<256x4xui8, #hivm.address_space<cbuf>>)
    // Verify load_scale produces a SetFlag with a valid event ID
    // CHECK: hivm.hir.load_scale
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %bufScaleA, %bufScaleB, %c1, %c256, %c128, %c256 :
            memref<256x128xf8E5M2, #hivm.address_space<cbuf>>,
            memref<128x256xf8E5M2, #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC : memref<256x256xf32>)
    // ScaleA wait (index [1]) should be ID0, A/B/ScaleB should be -1
    // CHECK: sync_related_args(
    // CHECK-SAME: %c-1_i64, %c0_i64, %c-1_i64, %c-1_i64
    return
  }
}

// -----
// TC-08: Two MmadMxL1Op instances sharing same L1 buffers (event ID reuse)
module {
  // CHECK-LABEL: func.func @test_mmad_mx_shared_buffer_reuse
  func.func @test_mmad_mx_shared_buffer_reuse(
      %gmA : memref<256x128xf8E5M2, #hivm.address_space<gm>>,
      %gmB : memref<128x256xf8E5M2, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i64     = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c128       = arith.constant 128 : index
    %c256       = arith.constant 256 : index
    %c1         = arith.constant 1 : i1

    %bufA      = hivm.hir.pointer_cast(%c0_i64)
        : memref<256x128xf8E5M2, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%gmA : memref<256x128xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufA : memref<256x128xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    %bufB      = hivm.hir.pointer_cast(%c131072_i64)
        : memref<128x256xf8E5M2, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%gmB : memref<128x256xf8E5M2, #hivm.address_space<gm>>)
        outs(%bufB : memref<128x256xf8E5M2, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]

    %bufScaleA = hivm.hir.pointer_cast(%c65536_i64)
        : memref<256x4xui8, #hivm.address_space<cbuf>>
    %bufScaleB = hivm.hir.pointer_cast(%c196608_i64)
        : memref<256x4xui8, #hivm.address_space<cbuf>>
    %bufC1 = memref.alloc() : memref<256x256xf32>
    %bufC2 = memref.alloc() : memref<256x256xf32>

    // First MmadMxL1Op — sync_related_args with ID0 (A) and ID1 (B)
    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %bufScaleA, %bufScaleB, %c1, %c256, %c128, %c256 :
            memref<256x128xf8E5M2, #hivm.address_space<cbuf>>,
            memref<128x256xf8E5M2, #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC1 : memref<256x256xf32>)
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: sync_related_args(
    // CHECK-SAME:   %c0_i64, %c-1_i64, %c1_i64, %c-1_i64

    // Second MmadMxL1Op (reads same L1 buffers, different output)
    // Should reuse the same wait event IDs (ID0 for A, ID1 for B)
    hivm.hir.mmadmxL1
        ins(%bufA, %bufB, %bufScaleA, %bufScaleB, %c1, %c256, %c128, %c256 :
            memref<256x128xf8E5M2, #hivm.address_space<cbuf>>,
            memref<128x256xf8E5M2, #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            memref<256x4xui8,     #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%bufC2 : memref<256x256xf32>)
    // The second MmadMxL1Op reuses the same event IDs from the first
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-NOT: sync_related_args
    return
  }
}
