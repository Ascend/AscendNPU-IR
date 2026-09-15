// RUN: bishengir-opt %s -split-input-file -convert-hivm-to-llvm | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: llvm.func @cube_11(
  // CHECK: %[[CONFIG:.*]] = llvm.mlir.constant(801 : i64)
  // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[CONFIG]]) <{pipe = 10 : i64}>
  // CHECK-NEXT: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 4 : i64, pipe = 2 : i64}>
  func.func @cube_11() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_M>] flag = 4
    // Other scopes keep their original FFTS encoding under CV11.
    // CHECK: %[[INTER:.*]] = llvm.mlir.constant(3841 : i64)
    // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[INTER]]) <{pipe = 10 : i64}>
    // CHECK-NEXT: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 14 : i64, pipe = 0 : i64}>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 15 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 14 sync_instr_mode = <INTER_SUBBLOCK_SYNCHRONIZATION>
    // CHECK-NEXT: llvm.return
    return
  }

  // CHECK-LABEL: llvm.func @vector_11(
  // CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 3 : i64, pipe = 1 : i64}>
  // CHECK-NEXT: %[[CONFIG:.*]] = llvm.mlir.constant(1057 : i64)
  // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[CONFIG]]) <{pipe = 5 : i64}>
  func.func @vector_11() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_M>] flag = 4
    // A scalar receiver does not change the sender's pipe or add a barrier.
    // CHECK-NEXT: %[[VECTOR:.*]] = llvm.mlir.constant(289 : i64)
    // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[VECTOR]]) <{pipe = 1 : i64}>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 1
    // Current scalar SET fallback: PIPE_ALL barrier, then SET on MTE3.
    // CHECK-NEXT: "hivm.intr.hivm.BARRIER"() <{pipe = 6 : i64}>
    // CHECK-NEXT: %[[SCALAR:.*]] = llvm.mlir.constant(33 : i64)
    // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[SCALAR]]) <{pipe = 5 : i64}>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = 0
    // CHECK-NEXT: llvm.return
    return
  }

  // CHECK-LABEL: llvm.func @dynamic_11(
  // CHECK-SAME: %[[ID:.*]]: i64)
  // CHECK: %[[MODE:.*]] = llvm.mlir.constant(33 : i64)
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(8 : i64)
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.shl %[[ID]], %[[SHIFT]]
  // CHECK-NEXT: %[[CONFIG:.*]] = llvm.or %[[SHIFTED]], %[[MODE]]
  // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[CONFIG]]) <{pipe = 10 : i64}>
  // CHECK-NEXT: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.REG"(%[[ID]]) <{pipe = 0 : i64}>
  // CHECK-NEXT: llvm.return
  func.func @dynamic_11(%id: i64) attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %id
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = %id
    return
  }

  // Current scalar SET fallback: PIPE_ALL barrier, then SET on FIX.
  // A scalar WAIT retains PIPE_S without an additional barrier.
  // CHECK-LABEL: llvm.func @scalar_cube_11(
  // CHECK: "hivm.intr.hivm.BARRIER"() <{pipe = 6 : i64}>
  // CHECK-NEXT: %[[CONFIG:.*]] = llvm.mlir.constant(3873 : i64)
  // CHECK-NEXT: "hivm.intr.hivm.SET.CROSS.CORE"(%[[CONFIG]]) <{pipe = 10 : i64}>
  // CHECK-NEXT: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 15 : i64, pipe = 0 : i64}>
  // CHECK-NEXT: llvm.return
  func.func @scalar_cube_11() attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    return
  }

  // CHECK-LABEL: llvm.func @cube_12(
  // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 16 : i64}>
  // CHECK-NEXT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 0 : i64}>
  // CHECK-NEXT: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 0 : i64, sync_id = 17 : i64}>
  // CHECK-NEXT: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 0 : i64, sync_id = 1 : i64}>
  // CHECK-NEXT: llvm.return
  func.func @cube_12() attributes {hivm.core_ratio = #hivm.core_ratio<1, 2>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    return
  }
}

// -----

// A 1:1 ratio on a memory-based architecture still uses the original FFTS
// implementation, including scalar ST_DEV and the non-pipe WAIT variant.
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  // CHECK-LABEL: llvm.func @intra_scalar(
  // CHECK-SAME: %[[BASE:.*]]: i64)
  // CHECK: %[[CONFIG:.*]] = llvm.mlir.constant(801 : i64)
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[BASE]] : i64 to !llvm.ptr<1>
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK-NEXT: "hivm.intr.hivm.ST.DEV.s64"(%[[CONFIG]], %[[PTR]], %[[ZERO]])
  // CHECK-NEXT: %[[ID:.*]] = llvm.mlir.constant(3 : i64)
  // CHECK-NEXT: "hivm.intr.hivm.WAIT.FLAG.DEV.REG"(%[[ID]])
  // CHECK-NEXT: llvm.return
  func.func @intra_scalar(%base: i64) attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 3 ffts_base_addr = %base
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 3
    return
  }
}
