// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivm-to-std | FileCheck %s
module {
  // CHECK-LABEL: func.func @cube
  func.func @cube(%meta: memref<8xi32, #hivm.address_space<ssbuf>>, %seq: i32) -> i64 attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<160xi8, #hivm.address_space<ub>>
    // CHECK: call @tcb_pool_init(
    hivm.hir.tcb_pool_init %meta : memref<8xi32, #hivm.address_space<ssbuf>>
    // Two queues with independently configurable depths.
    // CHECK: call @tcb_pool_alloc(
    %a = hivm.hir.tcb_pool_alloc %meta, %pool seq(%seq) {entry_offset = 1 : i32, queue_depth = 2 : i32, slot_bytes = 32 : i32, slots = 2 : i32} : memref<8xi32, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> i64
    // CHECK: call @tcb_pool_alloc(
    %b = hivm.hir.tcb_pool_alloc %meta, %pool seq(%seq) {entry_offset = 3 : i32, queue_depth = 4 : i32, slot_bytes = 32 : i32, slots = 1 : i32} : memref<8xi32, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> i64
    // CHECK: call @tcb_pool_release(
    hivm.hir.tcb_pool_release %meta seq(%seq) {entry_offset = 3 : i32, queue_depth = 4 : i32, slots = 1 : i32} : memref<8xi32, #hivm.address_space<ssbuf>>
    return %a : i64
  }
  // CHECK-LABEL: func.func @vector
  func.func @vector(%meta: memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ssbuf>>, %seq: i32) -> i64 attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %base = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%base) : memref<160xi8, #hivm.address_space<ub>>
    // CHECK: call @tcb_pool_get_address(
    %a = hivm.hir.tcb_pool_get %meta, %pool seq(%seq) {entry_offset = 3 : i32, queue_depth = 4 : i32, slot_bytes = 32 : i32} : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> i64
    return %a : i64
  }
}
