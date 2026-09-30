// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{solver-version=v1 enable-unit-flag=true ignore-workspace-func-args=true}))" | FileCheck %s
// RUN: bishengir-opt -split-input-file %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{solver-version=v2 enable-unit-flag=true ignore-workspace-func-args=true}))" | FileCheck %s

// Tests for isSoftwareSplitFixpipe() in SyncSolverIRTranslator.cpp.
//
// GraphSyncSolver enables the unit-flag (UF) feature on fixpipe ops that
// implement UnitFlagEnabledInterface. However, a dual-dst fixpipe that would
// fall back to a software split (two copy_matrix_cc_to_ubuf intrinsics) must
// NOT receive unit-flag, because unit-flag semantics assume a single atomic
// instruction.
//
// Each test case sets up a minimal mmadL1→fixpipe pair sharing a CC buffer.
//   - When the fixpipe does NOT split: both ops get unit_flag_mode; no
//     set_flag/wait_flag for PIPE_M→PIPE_FIX is emitted (CHECK-NOT).
//   - When the fixpipe WOULD split: neither op gets unit_flag_mode
//     (CHECK-NOT); traditional set_flag/wait_flag sync is used instead.
//
// Split trigger conditions (mirroring the template library's canEnableHWDualDst):
//   1. dma_mode == NZ2DN  (allow_hw_dual_for_transform is hardcoded false)
//   2. channel_split == true
//   3. pre_quant != NO_QUANT  (explicit quantization)
//   4. pre_relu != NO_RELU  (pre-stage ReLU fusion)

// =============================================================================
// Group 1: UF enabled — fixpipe does NOT split
// =============================================================================

// 1.1: NO_DUAL (default) — not a dual-dst fixpipe; UF applies normally.
//      Writes to GM; no module target needed.
// CHECK-LABEL: @uf_nodual_default
// CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode
func.func @uf_nodual_default(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
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
  hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
  return
}

// -----

// 1.2: ROW_SPLIT, NZ2ND, f32→f32, no quant, no channel_split — hardware
//      dual-dst is available; isSoftwareSplitFixpipe returns false.
// CHECK-LABEL: @uf_dual_row_split_nz2nd_f32_to_f32
// CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @uf_dual_row_split_nz2nd_f32_to_f32(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 1.3: COLUMN_SPLIT, normal (NZ2NZ default), f32→f32 — hardware dual-dst
//      available; isSoftwareSplitFixpipe returns false.
// CHECK-LABEL: @uf_dual_column_split_normal_f32_to_f32
// CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.fixpipe {{.*}} unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @uf_dual_column_split_normal_f32_to_f32(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dual_dst_mode = #hivm.fixpipe_dual_dst_mode<COLUMN_SPLIT>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// =============================================================================
// Group 2: UF disabled — fixpipe WOULD split
// =============================================================================

// 2.1: ROW_SPLIT + NZ2DN — NZ2DN always splits (allow_hw_dual_for_transform
//      is hardcoded to false in the template library).
// CHECK-LABEL: @no_uf_dual_nz2dn
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_nz2dn(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.2: ROW_SPLIT + channel_split — splits due to channel_split being true.
// CHECK-LABEL: @no_uf_dual_channel_split
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_channel_split(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, channel_split = true}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.3: ROW_SPLIT + explicit pre_quant (F322F16) — splits due to quantization.
// CHECK-LABEL: @no_uf_dual_explicit_quant
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_explicit_quant(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.4: ROW_SPLIT + explicit F322F16 quantization causes a split.
// CHECK-LABEL: @no_uf_dual_f322f16
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_f322f16(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.5: ROW_SPLIT + explicit F322BF16 quantization causes a split.
// CHECK-LABEL: @no_uf_dual_f322bf16
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_f322bf16(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xbf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322BF16>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xbf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.6: COLUMN_SPLIT + NZ2DN — COLUMN_SPLIT with NZ2DN also always splits.
// CHECK-LABEL: @no_uf_dual_column_split_nz2dn
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_column_split_nz2dn(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<COLUMN_SPLIT>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.7: ROW_SPLIT + pre_relu=NORMAL_RELU — splits due to pre-stage ReLU fusion.
// CHECK-LABEL: @no_uf_dual_pre_relu_normal
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_pre_relu_normal(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, pre_relu = #hivm.fixpipe_pre_relu_mode<NORMAL_RELU>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 2.8: ROW_SPLIT + pre_relu=LEAKY_RELU — splits due to pre-stage ReLU fusion.
// CHECK-LABEL: @no_uf_dual_pre_relu_leaky
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>
// CHECK-NOT: hivm.hir.fixpipe {{.*}}unit_flag_mode
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_uf_dual_pre_relu_leaky(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>) {
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
    hivm.hir.mmadL1 ins(%buf_a, %buf_b, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
    %buf_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, pre_relu = #hivm.fixpipe_pre_relu_mode<LEAKY_RELU>}
      ins(%buf_cc : memref<256xf32, #hivm.address_space<cc>>)
      outs(%buf_ub : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}
