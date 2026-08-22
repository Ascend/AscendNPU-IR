// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-unit-flag=true ignore-workspace-func-args=true}))" | FileCheck %s

// =============================================================================
// Category A: Positive — UF should be applied
// =============================================================================

// =============================================================================
// A0: Block{M→F} no loop — UF replaces M→FIX SetWait, unconditional [EWU]
// =============================================================================

// CHECK-LABEL: @same_block_mf_uf
func.func @same_block_mf_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  // CHECK: unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  // CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
  // CHECK: unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// =============================================================================
// A1: ForOp M→F — UF replaces M→FIX SetWait for both forward and backward sync
// =============================================================================

// CHECK-LABEL: @same_for_op_basic_uf
func.func @same_for_op_basic_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c-1_i64 = arith.constant -1 : i64
  // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    // CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  }
  // CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
  return
}

// -----

// =============================================================================
// A2: If{true: M→F} — UF replaces M→FIX SetWait within If true branch
// =============================================================================

// CHECK-LABEL: @if_true_branch_mf_uf
func.func @if_true_branch_mf_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %cond: i1) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c-1_i64 = arith.constant -1 : i64
  %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // M and F in same If branch → UF applies
  scf.if %cond {
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----

// =============================================================================
// A3: Block M1→M2→F (no loop) — closest M2 gets UF, M1 falls back
// =============================================================================

// CHECK-LABEL: @same_block_mm_f_chain_uf
func.func @same_block_mm_f_chain_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %buf_a1 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a1 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b1 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // M1: writes CC, should NOT get UF (distant producer, SetWait)
  // CHECK-NOT: unit_flag_mode
  hivm.hir.mmadL1 ins(%buf_a1, %buf_b1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  %buf_a2 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a2 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b2 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b2 : memref<16xf32, #hivm.address_space<cbuf>>)
  // M2: writes same CC, closest to F — SHOULD get UF
  // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.mmadL1 ins(%buf_a2, %buf_b2, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  // F: reads CC — SHOULD get UF
  // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// =============================================================================
// A4: ForOp M1→M2→F — closest M2 gets UF within loop, M1 falls back
// =============================================================================

// CHECK-LABEL: @for_op_mm_f_chain_uf
func.func @for_op_mm_f_chain_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a1 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a1 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b1 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // M1: cross-iteration → UF with 3-mode [DISABLED, EWU_WO, EWU_WO] + conds
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<DISABLED>, #hivm.unit_flag<ENABLED_WITHOUT_UPDATE>, #hivm.unit_flag<ENABLED_WITHOUT_UPDATE>]) unit_flag_cond(%{{.*}}, %{{.*}}, %true{{.*}})
    hivm.hir.mmadL1 ins(%buf_a1, %buf_b1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_a2 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a2 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b2 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b2 : memref<16xf32, #hivm.address_space<cbuf>>)
    // M2: same-iteration, closest to F → UF with [EWU]
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 ins(%buf_a2, %buf_b2, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    // F: closest consumer → UF with [EWU]
    // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----

// =============================================================================
// A5: Block M1→F1→M2→F2 (no loop) — Ping-Pong chain, both pairs get UF
// =============================================================================

// CHECK-LABEL: @same_block_mfmf_chain_uf
func.func @same_block_mfmf_chain_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  // M1→F1: first pair, writes to CC
  %buf_a1 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a1 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b1 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.mmadL1 ins(%buf_a1, %buf_b1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc1 : memref<256xf32, #hivm.address_space<cc>>)
  // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  // M2→F2: second pair, uses separate CC buffer
  %buf_a2 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a2 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b2 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b2 : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.mmadL1 ins(%buf_a2, %buf_b2, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc2 : memref<256xf32, #hivm.address_space<cc>>)
  // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc2 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// =============================================================================
// A6: ForOp M1→F1→M2→F2 Ping-Pong chain — all pairs get UF within loop
// =============================================================================

// CHECK-LABEL: @for_op_mfmf_chain_uf
func.func @for_op_mfmf_chain_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    // M1→F1: first pair
    %buf_a1 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a1 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b1 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 ins(%buf_a1, %buf_b1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc1 : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    // M2→F2: second pair, uses separate CC buffer
    %buf_a2 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a2 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b2 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b2 : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 ins(%buf_a2, %buf_b2, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc2 : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc2 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----

// =============================================================================
// B8: ForOp{If{true:M→F}} — UF rejected, SetWait fallback
//     M and F inside If, not direct children of ForOp → forward UF blocked
//     to avoid deadlock from mixed UF (forward) + SetWait (backward)
// =============================================================================

// CHECK-LABEL: @for_op_if_true_mf_setwait
func.func @for_op_if_true_mf_setwait(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %cond: i1) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    scf.if %cond {
      // CHECK-NOT: unit_flag_mode
      hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      // CHECK-NOT: unit_flag_mode
      hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
  }
  return
}

// -----

// =============================================================================
// A8: ForOp{M} → F (outer) — forward gets UF, backward falls to SetWait
// =============================================================================

// CHECK-LABEL: @for_op_m_outer_f_uf
func.func @for_op_m_outer_f_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
    // M inside ForOp → UF with [DISABLED, EWU] + conds
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<DISABLED>, #hivm.unit_flag<ENABLED_WITH_UPDATE>]) unit_flag_cond(%{{.*}}, %{{.*}})
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  }
  // F outside ForOp → UF with [EWU] + cond
  // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>]) unit_flag_cond(%{{.*}})
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// =============================================================================
// Category B: Negative — UF should fall back to SetWait
// =============================================================================

// =============================================================================
// B1: Nested ForOp forA{forB{M} F} — UF rejected, SetWait fallback
// =============================================================================

// CHECK-LABEL: @nested_for_m_inner_f_outer
func.func @nested_for_m_inner_f_outer(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK-NOT: unit_flag_mode
    scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----

// =============================================================================
// =============================================================================
// B2: If{true:M} → F — UF rejected, different Blocks
// =============================================================================

// CHECK-LABEL: @if_branch_m_outer_f_setwait
func.func @if_branch_m_outer_f_setwait(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %cond: i1) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c-1_i64 = arith.constant -1 : i64
  %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // M inside If, F outside — different Blocks, no ForOp mode → SetWait
  scf.if %cond {
    // CHECK-NOT: unit_flag_mode
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  }
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// =============================================================================
// B3: Sibling ForOp for1{M} for2{F} — UF rejected, SetWait fallback
//     Test result: triggers llvm_unreachable — see analysis below
// =============================================================================

// CHECK-LABEL: @sibling_for_ops_setwait_fallback
func.func @sibling_for_ops_setwait_fallback(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // Expected: set_flag M→FIX for forward, wait_flag for backward
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK-NOT: unit_flag_mode
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  }
  scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
    // CHECK-NOT: unit_flag_mode
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----

// =============================================================================
// B4: OuterForOp{ForOp1{M}, ForOp2{F}} — UF rejected, SetWait fallback
// =============================================================================

// CHECK-LABEL: @outer_for_sibling_for_setwait
func.func @outer_for_sibling_for_setwait(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    // CHECK-NOT: unit_flag_mode
    scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
      %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
    scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      // CHECK-NOT: unit_flag_mode
      hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    }
  }
  return
}

// -----

// =============================================================================
// B5: OuterForOp{ForOp{M}} → F — UF rejected, SetWait fallback
// =============================================================================

// CHECK-LABEL: @outer_for_nested_m_outer_f_setwait
func.func @outer_for_nested_m_outer_f_setwait(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    // CHECK-NOT: unit_flag_mode
    scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
      %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    }
  }
  // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
  hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// =============================================================================
// B6: WhileOp M→F — forward gets UF, backward falls to SetWait
// =============================================================================

// CHECK-LABEL: @while_op_same_iter_mf_uf
func.func @while_op_same_iter_mf_uf(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
  %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
  // Backward: cross-iteration signal before while loop
  // CHECK: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>
  %r = scf.while (%cond = %true) : (i1) -> i1 {
    scf.condition(%cond) %cond : i1
  } do {
  ^bb0(%cin: i1):
    // Forward: same-iteration M→F within WhileOp body → UF applies
    // CHECK: hivm.hir.mmadL1 {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
    hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    scf.yield %cin : i1
  }
  // Backward: cross-iteration wait after while loop
  // CHECK: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>
  return
}

// -----

// =============================================================================
// B7: ForOp{If{true: ForOp{M}, F}} — UF rejected, SetWait fallback
// =============================================================================

// CHECK-LABEL: @for_if_for_m_f_setwait
func.func @for_if_for_m_f_setwait(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %cond: i1) {
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
  scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
    %buf_a = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_a : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_b = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%buf_b : memref<16xf32, #hivm.address_space<cbuf>>)
    %buf_cc = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    scf.if %cond {
      // CHECK-NOT: unit_flag_mode
      scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
        hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      }
      // CHECK-NOT: unit_flag_mode
      hivm.hir.fixpipe {enable_nz2nd} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
  }
  return
}

// -----
