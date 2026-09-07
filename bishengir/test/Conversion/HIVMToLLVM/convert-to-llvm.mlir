// RUN: bishengir-opt %s -split-input-file -convert-hivm-to-llvm | FileCheck %s

// Test GetBlockIdxInstrOp
// CHECK-LABEL: test_get_block_idx
func.func @test_get_block_idx() {
  // CHECK: %[[IDX:.*]] = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
  %block = hivm.hir.get_block_idx -> i64
  return
}

// -----

// Test GetBlockNumInstrOp
// CHECK-LABEL: test_get_block_num
func.func @test_get_block_num() {
  // CHECK: %[[IDX:.*]] = "hivm.intr.hivm.GET.BLOCK.NUM"() : () -> i64
  %block = hivm.hir.get_block_num -> i64
  return
}

// -----

// Test GetSubBlockIdxInstrOp
// CHECK-LABEL: test_get_sub_block_idx
func.func @test_get_sub_block_idx() {
  // CHECK: %[[IDX:.*]] = "hivm.intr.hivm.GET.SUBBLOCKID"() : () -> i64
  %block = hivm.hir.get_sub_block_idx -> i64
  return
}

// -----

// Test GetSubBlockNumInstrOp
// CHECK-LABEL: test_get_sub_block_num
func.func @test_get_sub_block_num() {
  // CHECK: %[[IDX:.*]] = "hivm.intr.hivm.GET.SUBBLOCKDIM"() : () -> i64
  %block = hivm.hir.get_sub_block_num -> i64
  return
}

// -----

// Test AddressSpaceAttr lowering
func.func @test_addrspaceattr_lowering() {
  // CHECK: llvm.alloca {{.*}} x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr<6>
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<f32, #hivm.address_space<ub>>
  %val = arith.constant 4.5 : f32
  memref.store %val, %alloca[] : memref<f32, #hivm.address_space<ub>>
  return
}

// -----

// Test annotation-only AddressSpace lowering: FixBUF -> LLVM ptr<7>
func.func @test_addrspaceattr_lowering_fixbuf() {
  // CHECK: llvm.alloca {{.*}} x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr<7>
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<f32, #hivm.address_space<fixbuf>>
  return
}

// -----

// Test annotation-only AddressSpace lowering: BiasBUF -> LLVM ptr<12>
func.func @test_addrspaceattr_lowering_biasbuf() {
  // CHECK: llvm.alloca {{.*}} x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr<12>
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<f32, #hivm.address_space<biasbuf>>
  return
}

// -----

// Test dependent dialects lowering
// CHECK-NOT: #hivm.address_space
module {
  func.func private @dummy_func(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>,
                                memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {llvm.emit_c_interface}
  func.func @main(%arg0 : i1) {
      %alloca   = memref.alloca() {alignment = 64 : i64} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %selected = arith.select %arg0, %alloca, %alloca_1 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      cf.br ^bb1(%alloca : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
    ^bb1(%1 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>): // pred: ^bb0
      return
  }
}

// Test DynamicSetFlagInstrOp
// CHECK-LABEL: llvm.func @test_dyn_set_flag
func.func @test_dyn_set_flag(%eventId : i64) {
  // CHECK: "hivm.intr.hivm.SET.FLAG.REG"(%{{.*}})
  // CHECK-SAME-DAG: set_flag
  // CHECK-SAME-DAG: wait_flag
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
  return
}

// Test DynamicWaitFlagInstrOp
// CHECK-LABEL: llvm.func @test_dyn_wait_flag
func.func @test_dyn_wait_flag(%eventId : i64) {
  // CHECK: "hivm.intr.hivm.WAIT.FLAG.REG"(%{{.*}})
  // CHECK-SAME-DAG: set_flag
  // CHECK-SAME-DAG: wait_flag
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
  return
}

// Test StaticSetFlagInstrOp
// CHECK-LABEL: llvm.func @test_const_set_flag()
func.func @test_const_set_flag() {
  // CHECK: "hivm.intr.hivm.SET.FLAG.IMM"()
  // CHECK-SAME-DAG: set_flag
  // CHECK-SAME-DAG: wait_flag
  // CHECK-SAME-DAG: event_id = 0
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
  %eventId = arith.constant 1 : i64
  // CHECK: "hivm.intr.hivm.SET.FLAG.IMM"()
  // CHECK-SAME-DAG: set_flag
  // CHECK-SAME-DAG: wait_flag
  // CHECK-SAME-DAG: event_id = 1
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
  return
}

// Test StaticWaitFlagInstrOp
// CHECK-LABEL: llvm.func @test_const_wait_flag()
func.func @test_const_wait_flag() {
  // CHECK: "hivm.intr.hivm.WAIT.FLAG.IMM"()
  // CHECK-SAME-DAG: set_flag
  // CHECK-SAME-DAG: wait_flag
  // CHECK-SAME-DAG: event_id = 0
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
  %eventId = arith.constant 1 : i64
  // CHECK: "hivm.intr.hivm.WAIT.FLAG.IMM"()
  // CHECK-SAME-DAG: set_flag
  // CHECK-SAME-DAG: wait_flag
  // CHECK-SAME-DAG: event_id = 1
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
  return
}

// Test PipeBarrierInstrOp
// CHECK-LABEL: llvm.func @test_pipe_barrier
func.func @test_pipe_barrier() {
  // CHECK: "hivm.intr.hivm.BARRIER"() <{pipe =
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// Test SyncBlockSetOp
// CHECK-LABEL: llvm.func @test_sync_block_set
func.func @test_sync_block_set() {
  // CHECK-DAG: %[[CONST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CONST2:.*]] = llvm.mlir.constant(289 : i64) : i64
  // CHECK : "hivm.intr.hivm.SET.FFTS.BASE.ADDR"(%[[CONST0]]) : (i64) -> ()
  %ffts_base_addr = arith.constant 0 : i64
  // CHECK: "hivm.intr.hivm.SET.CROSS.CORE"(%[[CONST2]]) <{pipe =
   hivm.hir.sync_block_set[#hivm.tcore_type<CUBE_OR_VECTOR>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_MTE3>]
     flag = 1
     ffts_base_addr = %ffts_base_addr
  return
}

func.func @test_set_ffts_base_addr() {
  %ffts_base_addr = arith.constant 0 : i64
  // CHECK: hivm.intr.hivm.SET.FFTS.BASE.ADDR
  hivm.hir.set_ffts_base_addr %ffts_base_addr
  return
}

// Test SyncBlockWaitOp
// CHECK-LABEL: llvm.func @test_sync_block_wait
func.func @test_sync_block_wait() {
  // CHECK: %[[CONST0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK : "hivm.intr.hivm.WAIT.FLAG.DEV"(%[[CONST0]]) : (i64) -> ()
  hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>] flag = 1
  return
}

// Test SetMaskNormOp
// CHECK-LABEL: llvm.func @test_set_mask_norm
func.func @test_set_mask_norm() {
  // CHECK: %[[CTRL:.*]] = "hivm.intr.hivm.GET.CTRL"() : () -> i64
  // CHECK: %[[CST:.*]] = llvm.mlir.constant(56 : i64) : i64
  // CHECK: %[[SBSET:.*]] = "hivm.intr.hivm.SBITSET0"(%[[CTRL]], %[[CST]]) : (i64, i64) -> i64
  // CHECK: "hivm.intr.hivm.SET.CTRL"(%[[SBSET]]) : (i64) -> ()
  hivm.hir.set_mask_norm
  return
}

// Test dcci
// CHECK-LABEL: llvm.func @test_dcci
func.func @test_dcci() {
  // CHECK: "hivm.intr.hivm.DCCI.DST"
  hivm.hir.dcci (#hivm.DCCIMode<all_cache_lines>, #hivm.DataCacheKind<out>)
  return
}
// -----
// CHECK-LABEL: @test_bitcast_f32_i32_memref
func.func @test_bitcast_f32_i32_memref(%src : memref<6xi32>) -> memref<6xf32>{
  // CHECK-NOT: hivm.hir.bitcast
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %dst = hivm.hir.bitcast %src : memref<6xi32> -> memref<6xf32>
  return %dst : memref<6xf32>
}

// Test SetCtrlOp
// CHECK-LABEL: llvm.func @test_set_ctrl
func.func @test_set_ctrl() {
  // CHECK: %[[CTRL:.*]] = "hivm.intr.hivm.GET.CTRL"() : () -> i64
  // CHECK: %[[CST:.*]] = llvm.mlir.constant(60 : i64) : i64
  // CHECK: %[[SBSET:.*]] = "hivm.intr.hivm.SBITSET0"(%[[CTRL]], %[[CST]]) : (i64, i64) -> i64
  // CHECK: "hivm.intr.hivm.SET.CTRL"(%[[SBSET]]) : (i64) -> ()
  hivm.hir.set_ctrl false at ctrl[60]
  return
}

// -----
// Test SyncBlockSet Intra Cube
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_set_intra_cube() {
        // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 17 : i64}> : () -> ()
        // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 1 : i64}> : () -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_S>] flag = 1
        return
    }
}

// -----
// Test SyncBlockSet Intra Vector
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_set_intra_vector() {
        // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 1 : i64, sync_id = 1 : i64}> : () -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = 1
        return
    }
}

// -----
// Test SyncBlockSet Inter Cube
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_set_inter_cube() {
        // CHECK: "hivm.intr.hivm.SET.CROSS.CORE"(%0) <{pipe = 10 : i64}> : (i64) -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_S>] flag = 1 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
        return
    }
}
// -----
// Test SyncBlockSet Inter Vector
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_set_inter_vector() {
        // CHECK: "hivm.intr.hivm.SET.CROSS.CORE"(%0) <{pipe = 1 : i64}> : (i64) -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = 1 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
        return
    }
}


// -----
// Test SyncBlockWait Intra Cube
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_intra_cube() {
        // CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 0 : i64, sync_id = 17 : i64}> : () -> ()
        // CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 0 : i64, sync_id = 1 : i64}> : () -> ()
        hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_S>] flag = 1
        return
    }
}
// -----
// Test SyncBlockWait Intra Vector
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_intra_vector() {
        // CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 0 : i64, sync_id = 1 : i64}> : () -> ()
        hivm.hir.sync_block_wait[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = 1
        return
    }
}

// -----
// Test SyncBlockWait Inter Cube
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_inter_cube() {
        // CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 1 : i64, pipe = 0 : i64}> : () -> ()
        hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_S>] flag = 1 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
        return
    }
}
// -----
// Test SyncBlockWait Inter Vector
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_inter_vector() {
        // CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 1 : i64, pipe = 0 : i64}> : () -> ()
        hivm.hir.sync_block_wait[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = 1 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
        return
    }
}

// -----
// Test SyncBlockWait Inter Vector
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_set_intra_cube_reg(%arg0 : i64) {
        // CHECK:  %[[CONSTANT16:.*]] = llvm.mlir.constant(16 : i64) : i64
        // CHECK:  %1 = llvm.add %arg0, %[[CONSTANT16:.*]] : i64
        // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(%[[FLAG:.*]]) <{pipe = 10 : i64}> : (i64) -> ()
        // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(%arg0) <{pipe = 10 : i64}> : (i64) -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_S>] flag = %arg0
        return
    }
}

// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_set_intra_vector_reg(%arg0 : i64) {
        // CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(%arg0) <{pipe = 1 : i64}> : (i64) -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = %arg0
        return
    }
}

// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_intra_cube_reg(%arg0 : i64) {
        // CHECK:  %[[CONSTANT16:.*]] = llvm.mlir.constant(16 : i64) : i64
        // CHECK:  %1 = llvm.add %arg0, %[[CONSTANT16:.*]] : i64
        // CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCK.mode"(%[[FLAG:.*]]) <{pipe = 0 : i64}> : (i64) -> ()
        // CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCK.mode"(%arg0) <{pipe = 0 : i64}> : (i64) -> ()
        hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_S>] flag = %arg0
        return
    }
}

// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_intra_vector_reg(%arg0 : i64) {
         // CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCK.mode"(%arg0) <{pipe = 0 : i64}> : (i64) -> ()
        hivm.hir.sync_block_wait[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = %arg0
        return
    }
}

// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
    func.func @test_sync_block_wait_inter_vector_reg(%arg0 : i64) {
        // CHECK: "hivm.intr.hivm.SET.CROSS.CORE"(%[[CONFIG:.*]]) <{pipe = 1 : i64}> : (i64) -> ()
        hivm.hir.sync_block_set[#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_S>] flag = %arg0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
        return
    }
}

// -----
func.func private @use(memref<32x128xf32>)
// CHECK-LABEL: llvm.func @fold_marked
// CHECK:         llvm.getelementptr
func.func @fold_marked(%arg0: memref<32x128xf32, strided<[128, 1], offset: ?>>) {
  %0 = memref.cast %arg0 {fold_offset_into_ptr}
      : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<32x128xf32>
  func.call @use(%0) : (memref<32x128xf32>) -> ()
  return
}

// -----
func.func private @use(memref<32x128xf32>)
// CHECK-LABEL: llvm.func @no_fold_unmarked
// CHECK-NOT:     llvm.getelementptr
func.func @no_fold_unmarked(%arg0: memref<32x128xf32, strided<[128, 1], offset: ?>>) {
  %0 = memref.cast %arg0
      : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<32x128xf32>
  func.call @use(%0) : (memref<32x128xf32>) -> ()
  return
}

// -----
module {
  // CHECK-LABEL: llvm.func @test_ssbuf_to_llvm
  // CHECK-SAME: (%[[ARG0:.*]]: i64)
  func.func @test_ssbuf_to_llvm(%arg0: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {

    // CHECK: %[[RAW_PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr<11>
    %2 = hivm.hir.pointer_cast(%arg0) : memref<i8, #hivm.address_space<ssbuf>>

    // CHECK: %[[DESC0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<11>, ptr<11>, i64)>
    // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[RAW_PTR]], %[[DESC0]][0] : !llvm.struct<(ptr<11>, ptr<11>, i64)>
    // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[RAW_PTR]], %[[DESC1]][1] : !llvm.struct<(ptr<11>, ptr<11>, i64)>
    // CHECK: %[[FINAL_DESC:.*]] = llvm.insertvalue {{.*}}, %[[DESC2]][2] : !llvm.struct<(ptr<11>, ptr<11>, i64)>


    // CHECK: %[[LOAD_ADDR:.*]] = llvm.extractvalue %[[FINAL_DESC]][1] : !llvm.struct<(ptr<11>, ptr<11>, i64)>
    // CHECK: %[[VAL:.*]] = llvm.load volatile %[[LOAD_ADDR]] : !llvm.ptr<11> -> i8
    %3 = memref.load %2[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<i8, #hivm.address_space<ssbuf>>
	annotation.mark %3 {memref_ext.volatile} : i8

    %0 = llvm.mlir.constant(0 : i8) : i8
    %4 = llvm.icmp "sgt" %3, %0 : i8
    cf.cond_br %4, ^bb1, ^bb2

  ^bb1:
    %1 = llvm.mlir.constant(1 : i8) : i8
    %5 = llvm.sub %3, %1 : i8

    // CHECK: %[[STORE_ADDR:.*]] = llvm.extractvalue %[[FINAL_DESC]][1] : !llvm.struct<(ptr<11>, ptr<11>, i64)>
    // CHECK: llvm.store volatile %[[SUB_RES:.*]], %[[STORE_ADDR]] : i8, !llvm.ptr<11>
    memref.store %5, %2[] : memref<i8, #hivm.address_space<ssbuf>>
    cf.br ^bb2

  ^bb2:
    return
  }
}
