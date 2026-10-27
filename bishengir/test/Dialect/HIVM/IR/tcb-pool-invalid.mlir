// RUN: bishengir-opt %s -verify-diagnostics -split-input-file
func.func @bad_range(%meta: memref<4xi32, #hivm.address_space<ssbuf>>, %seq: i32) {
  // expected-error @+1 {{queue entry range must fit metadata after the bitmap}}
  hivm.hir.tcb_pool_release %meta seq(%seq) {entry_offset = 3 : i32, queue_depth = 2 : i32, slots = 1 : i32} : memref<4xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @overlap(%meta: memref<8xi32, #hivm.address_space<ssbuf>>, %seq: i32) {
  // expected-error @+1 {{queue entry ranges must be identical or disjoint}}
  hivm.hir.tcb_pool_release %meta seq(%seq) {entry_offset = 1 : i32, queue_depth = 3 : i32, slots = 1 : i32} : memref<8xi32, #hivm.address_space<ssbuf>>
  hivm.hir.tcb_pool_release %meta seq(%seq) {entry_offset = 3 : i32, queue_depth = 2 : i32, slots = 1 : i32} : memref<8xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @bad_slots(%meta: memref<4xi32, #hivm.address_space<ssbuf>>, %seq: i32) {
  // expected-error @+1 {{slots must be in [1,32]}}
  hivm.hir.tcb_pool_release %meta seq(%seq) {entry_offset = 1 : i32, queue_depth = 3 : i32, slots = 33 : i32} : memref<4xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @different_slot_sizes(%meta: memref<3xi32, #hivm.address_space<ssbuf>>, %seq: i32) {
  %base = arith.constant 0 : i64
  %pool = hivm.hir.pointer_cast(%base) : memref<128xi8, #hivm.address_space<ub>>
  // expected-error @+1 {{shared metadata pool geometry must remain consistent}}
  %a = hivm.hir.tcb_pool_alloc %meta, %pool seq(%seq) {entry_offset = 1 : i32, queue_depth = 1 : i32, slot_bytes = 64 : i32, slots = 1 : i32} : memref<3xi32, #hivm.address_space<ssbuf>>, memref<128xi8, #hivm.address_space<ub>> -> i64
  %b = hivm.hir.tcb_pool_alloc %meta, %pool seq(%seq) {entry_offset = 2 : i32, queue_depth = 1 : i32, slot_bytes = 32 : i32, slots = 1 : i32} : memref<3xi32, #hivm.address_space<ssbuf>>, memref<128xi8, #hivm.address_space<ub>> -> i64
  return
}
// -----
func.func @different_get_pools(%meta: memref<3xi32, #hivm.address_space<ssbuf>>, %seq: i32) {
  %base = arith.constant 0 : i64
  %other_base = arith.constant 128 : i64
  %pool = hivm.hir.pointer_cast(%base) : memref<128xi8, #hivm.address_space<ub>>
  %other = hivm.hir.pointer_cast(%other_base) : memref<128xi8, #hivm.address_space<ub>>
  // expected-error @+1 {{shared metadata pool geometry must remain consistent}}
  %a = hivm.hir.tcb_pool_get %meta, %pool seq(%seq) {entry_offset = 1 : i32, queue_depth = 1 : i32, slot_bytes = 32 : i32} : memref<3xi32, #hivm.address_space<ssbuf>>, memref<128xi8, #hivm.address_space<ub>> -> i64
  %b = hivm.hir.tcb_pool_get %meta, %other seq(%seq) {entry_offset = 2 : i32, queue_depth = 1 : i32, slot_bytes = 32 : i32} : memref<3xi32, #hivm.address_space<ssbuf>>, memref<128xi8, #hivm.address_space<ub>> -> i64
  return
}
