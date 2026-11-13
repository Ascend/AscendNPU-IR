// RUN: bishengir-opt %s -hivm-enable-multi-buffer -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @separate_accesses
// CHECK-SAME: %[[L:arg[0-9]+]]: index, %[[C:arg[0-9]+]]: index
func.func @separate_accesses(%load: index, %compute: index,
    %src: memref<32xf32, #hivm.address_space<gm>>,
    %dst: memref<32xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : i64
  %c128 = arith.constant 128 : i64
  %c256 = arith.constant 256 : i64
  %root = hivm.hir.pointer_cast(%c0, %c128, %c256) {hivm.experimental_buffer_slots} : memref<32xf32, #hivm.address_space<ub>>
  annotation.mark %root {hivm.mempool_id = 0 : i32, hivm.mempool_size = 3 : i32, hivm.multi_buffer = 3 : i32} : memref<32xf32, #hivm.address_space<ub>>
  // CHECK: arith.cmpi eq, %[[L]],
  // CHECK: arith.select
  // CHECK: arith.cmpi eq, %[[L]],
  // CHECK: %[[LB:.*]] = arith.select
  %lb = hivm.hir.experimental.pool_slot %root[%load] : memref<32xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load {{.*}} outs(%[[LB]]
  hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%lb : memref<32xf32, #hivm.address_space<ub>>)
  // CHECK: arith.cmpi eq, %[[C]],
  // CHECK: arith.select
  // CHECK: arith.cmpi eq, %[[C]],
  // CHECK: %[[CB:.*]] = arith.select
  %cb = hivm.hir.experimental.pool_slot %root[%compute] : memref<32xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.store ins(%[[CB]]
  hivm.hir.store ins(%cb : memref<32xf32, #hivm.address_space<ub>>) outs(%dst : memref<32xf32, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @one_slot
func.func @one_slot(%slot: index, %src: memref<32xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : i64
  // CHECK: %[[B:.*]] = hivm.hir.pointer_cast
  %root = hivm.hir.pointer_cast(%c0) : memref<32xf32, #hivm.address_space<ub>>
  // CHECK-NOT: arith.select
  // CHECK-NOT: pool_slot
  %buf = hivm.hir.experimental.pool_slot %root[%slot] : memref<32xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load {{.*}} outs(%[[B]]
  hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%buf : memref<32xf32, #hivm.address_space<ub>>)
  return
}
