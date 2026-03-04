// RUN: bishengir-opt %s -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: test_sync_block_set
func.func @test_sync_block_set() {
  %0 = arith.constant 0 : i64
  "hivm.intr.hivm.SET.FFTS.BASE.ADDR"(%0) : (i64) -> ()
  %1 = arith.constant 298: i64
  "hivm.intr.hivm.SET.CROSS.CORE"(%1) <{pipe = 1 : i64}> : (i64) -> ()
  return
}

// -----
// CHECK-LABEL: test_sync_block_wait
func.func @test_sync_block_wait() {
  %0 = arith.constant 0 : i64
  "hivm.intr.hivm.WAIT.FLAG.DEV.REG" (%0) : (i64) -> ()
  return
}
