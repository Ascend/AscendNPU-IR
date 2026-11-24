// RUN: bishengir-opt %s -create-preload -verify-each | FileCheck %s

// GQA carries a counter and UB pointer aliases through the same loop. Removing
// a preload-local pointer must not index the compacted loop arguments with the
// original yield operand number. The unmarked state pointer remains available
// after the loop, while only the counter needs a guarded loop-carried value.
// CHECK-LABEL: func.func @gqa_filtered_loop_args
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[COUNTER:.*]] = {{.*}}) -> (i64)
// CHECK: %[[NEXT:.*]] = scf.if {{.*}} -> (i64)
// CHECK: scf.yield {{.*}} : i64
// CHECK: } else {
// CHECK: scf.yield %[[COUNTER]] : i64
// CHECK: scf.yield %[[NEXT]] : i64
// CHECK: return %[[LOOP]], {{.*}} : i64, memref<8xf32, #hivm.address_space<ub>>
func.func @gqa_filtered_loop_args(%ub: i32)
    -> (i64, memref<8xf32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %addr0 = arith.constant 0 : i64
  %addr1 = arith.constant 256 : i64
  %state_addr = arith.constant 512 : i64
  %buffer = hivm.hir.pointer_cast(%addr0, %addr1)
      : memref<8xf32, #hivm.address_space<ub>>
  %state = hivm.hir.pointer_cast(%state_addr)
      : memref<8xf32, #hivm.address_space<ub>>
  %result:3 = scf.for %i = %c0 to %ub step %c1
      iter_args(%counter = %zero, %carried_buffer = %buffer,
                %carried_state = %state)
      -> (i64, memref<8xf32, #hivm.address_space<ub>>,
          memref<8xf32, #hivm.address_space<ub>>) : i32 {
    annotation.mark %buffer {hivm.multi_buffer = 2 : i32,
                            hivm.preload_local_buffer}
        : memref<8xf32, #hivm.address_space<ub>>
    %updated:2 = scope.scope : () ->
        (i64, memref<8xf32, #hivm.address_space<ub>>) {
      %next = arith.addi %counter, %one : i64
      scope.return %next, %state : i64, memref<8xf32, #hivm.address_space<ub>>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %updated#0, %buffer, %updated#1
        : i64, memref<8xf32, #hivm.address_space<ub>>,
          memref<8xf32, #hivm.address_space<ub>>
  } {hivm.cv_pipelined_loop}
  return %result#0, %result#2 : i64, memref<8xf32, #hivm.address_space<ub>>
}
