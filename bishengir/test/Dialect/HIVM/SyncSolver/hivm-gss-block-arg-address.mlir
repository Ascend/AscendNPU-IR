// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v1 -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v2 -split-input-file %s | FileCheck %s

// Addresses supplied as block arguments have no defining operation.

// CHECK-LABEL: func.func @function_arg_address(
func.func @function_arg_address(%address: i64, %out: memref<16xf16, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %src = hivm.hir.pointer_cast(%address) : memref<16xf16, #hivm.address_space<gm>>
  %buf = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load
  hivm.hir.load ins(%src : memref<16xf16, #hivm.address_space<gm>>) outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <[[EVENT:EVENT_ID[0-9]+]]>]
  // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <[[EVENT]]>]
  // CHECK: hivm.hir.store
  hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>) outs(%out : memref<16xf16, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @loop_carried_address(
func.func @loop_carried_address(%initial_address: i64, %out: memref<16xf16, #hivm.address_space<gm>>, %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i64 = arith.constant 0 : i64
  %c32_i64 = arith.constant 32 : i64
  %buf = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
  // CHECK: scf.for {{.*}} iter_args(%[[ADDRESS:[a-zA-Z0-9_]+]] = {{.*}}) -> (i64) {
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%address = %initial_address) -> (i64) {
    // CHECK: hivm.hir.pointer_cast(%[[ADDRESS]])
    %src = hivm.hir.pointer_cast(%address) : memref<16xf16, #hivm.address_space<gm>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%src : memref<16xf16, #hivm.address_space<gm>>) outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <[[EVENT:EVENT_ID[0-9]+]]>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <[[EVENT]]>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>) outs(%out : memref<16xf16, #hivm.address_space<gm>>)
    %next_address = arith.addi %address, %c32_i64 : i64
    // CHECK: scf.yield {{.*}} : i64
    scf.yield %next_address : i64
  }
  return
}
