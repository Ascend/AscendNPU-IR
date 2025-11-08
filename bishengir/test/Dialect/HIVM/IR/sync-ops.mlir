// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.


// CHECK-LABEL: set_flag
func.func @set_flag() {
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, <EVENT_ID0>]
  return
}

// -----

// CHECK-LABEL: set_flag
func.func @set_flag() {
  %eventId = arith.constant 1 : i64
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
  return
}

// -----

// CHECK-LABEL: wait_flag
func.func @wait_flag() {
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, <EVENT_ID0>]
  return
}

// -----

// CHECK-LABEL: wait_flag
func.func @wait_flag() {
  %eventId = arith.constant 1 : i64
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
  return
}

// -----
// CHECK-LABEL: @test_sync_block_set_flag_attr
func.func @test_sync_block_set_flag_attr() {
  %ffts_base_addr = arith.constant 0 : i64
  hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_FIX>]
    flag = 1
    ffts_base_addr = %ffts_base_addr
    syn_instr_mode = #hivm.sync_block_instr_mode<INTER_BLOCK_SYNCHRONIZATION>
  return
}

// -----
// CHECK-LABEL: @test_sync_block_set_flag_value
func.func @test_sync_block_set_flag_value() {
  %ffts_base_addr = arith.constant 0 : i64
  %flag_id = arith.constant 0 : i64
  hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_FIX>]
    flag = %flag_id
    ffts_base_addr = %ffts_base_addr
    syn_instr_mode = #hivm.sync_block_instr_mode<INTER_BLOCK_SYNCHRONIZATION>
  return
}

// -----
// CHECK-LABEL: @test_sync_block_wait_flag_attr
func.func @test_sync_block_wait_flag_attr() {
  hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>] flag = 1
  return
}

// -----
// CHECK-LABEL: @test_sync_block_wait_flag_value
func.func @test_sync_block_wait_flag_value() {
  %flag_id = arith.constant 0 : i64
  hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>] flag = %flag_id
  return
}