// RUN: split-file %s %t
// RUN: bishengir-opt %t/no-spec.mlir -hivm-mark-buffer-pool | FileCheck %t/no-spec.mlir
// RUN: bishengir-opt %t/stages.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-mark-buffer-pool -split-input-file | FileCheck %t/stages.mlir
// RUN: bishengir-opt %t/stages.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-mark-buffer-pool -hivm-mark-buffer-pool -split-input-file | FileCheck %t/stages.mlir
// RUN: bishengir-opt %t/ssbuffer.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-mark-buffer-pool | FileCheck %t/ssbuffer.mlir
// RUN: bishengir-opt %t/unseparated.mlir -o %t/unseparated-before.mlir
// RUN: bishengir-opt %t/unseparated.mlir -hivm-mark-buffer-pool -o %t/unseparated-after.mlir
// RUN: diff %t/unseparated-before.mlir %t/unseparated-after.mlir

//--- stages.mlir
// One remaining event cannot hold the two inputs of one computation.
// With Q reduced to two buffers, both inputs share a pool and consume distinct
// consecutive slots through the same cursor.
// INPUTS-LABEL: func.func @insufficient_simultaneous_inputs
// CHECK-LABEL: func.func @insufficient_simultaneous_inputs
func.func @insufficient_simultaneous_inputs(%src: memref<32xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %q = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  %out = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  annotation.mark %q {hivm.multi_buffer = 6 : i32} : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%q : memref<32xf32, #hivm.address_space<ub>>)
  hivm.hir.vadd ins(%q, %q : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%out : memref<32xf32, #hivm.address_space<ub>>)
  // INPUTS: scf.for {{.*}}iter_args({{.*}}, %[[READ:arg[0-9]+]] = %{{.*}})
  scf.for %i = %c0 to %c8 step %c1 {
    %b0 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark {{.*}} {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
    annotation.mark %b0 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%b0 : memref<32xf32, #hivm.address_space<ub>>)
      scope.return
    // CHECK: hivm.load_preload = 1 : i32
    } {hivm.load_preload = 1 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %b1 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark {{.*}} {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
    annotation.mark %b1 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%b1 : memref<32xf32, #hivm.address_space<ub>>)
      scope.return
    // CHECK: hivm.load_preload = 1 : i32
    } {hivm.load_preload = 1 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    // INPUTS: %[[S0:.*]] = arith.remui %[[READ]], %c4
    // INPUTS: %[[R0:.*]] = hivm.hir.experimental.pool_slot {{.*}}[%[[S0]]]
    // INPUTS: %[[NEXT:.*]] = arith.addi %[[READ]], %c1
    // INPUTS: %[[S1:.*]] = arith.remui %[[NEXT]], %c4
    // INPUTS: %[[R1:.*]] = hivm.hir.experimental.pool_slot {{.*}}[%[[S1]]]
    // INPUTS: %[[END:.*]] = arith.addi %[[NEXT]], %c1
    scope.scope : () -> () {
      // INPUTS: hivm.hir.vadd ins(%[[R0]], %[[R1]]
      hivm.hir.vadd ins(%b0, %b1 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%out : memref<32xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.preload_num = 1 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Q reserves two events. All three L1 paddings share one additional event.
// The outside P select must not prevent the compatible K/V loads pooling.
// CHECK-LABEL: func.func @l1_padding_and_select
func.func @l1_padding_and_select(%src: memref<16x16xf16, #hivm.address_space<gm>>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %zero = arith.constant 0.0 : f16
  %q = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  annotation.mark %q {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  %p0 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  %p1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  %out = memref.alloc() : memref<1x1x16x16xf32, #hivm.address_space<cc>>
  %p = arith.select %pad, %p0, %p1 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  scf.if %pad {
    hivm.hir.vbrc ins(%zero : f16) outs(%q : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
  } {hivm.unlikely_condition}
  hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%q : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
  scf.for %i = %c0 to %c8 step %c1 {
    %k = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark {{.*}}hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32, hivm.multi_buffer = 5 : i32
    annotation.mark %k {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      scf.if %pad {
        hivm.hir.vbrc ins(%zero : f16) outs(%k : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%k : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      scope.return
    // CHECK: hivm.load_preload = 3 : i32{{.*}}hivm.mempool_id = 0 : i32
    } {hivm.load_preload = 3 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      hivm.hir.mmadL1 ins(%q, %k, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      scope.return
    } {no_inline}
    %v = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark {{.*}}hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32, hivm.multi_buffer = 5 : i32
    annotation.mark %v {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      scf.if %pad {
        hivm.hir.vbrc ins(%zero : f16) outs(%v : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%v : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      scope.return
    // CHECK: hivm.load_preload = 1 : i32{{.*}}hivm.mempool_id = 0 : i32
    } {hivm.load_preload = 1 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      hivm.hir.mmadL1 ins(%p, %v, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      scope.return
    } {no_inline}
  } {hivm.cv_pipelined_loop}
  return
}

//--- ssbuffer.mlir
// The module-wide driver must not stop at a function with no load scopes.
func.func private @external()
func.func @no_load() {
  return
}

// C2 checks C1's entry count, not the result of this iteration's C1. Two
// completed C1 tasks therefore yield K0, K1, K2, V0, K3, V1, ... .
// No CV loop marker is required for this SSBuffer scheduling block.
// ssbuffer.if's numbers deliberately have no relationship with these stages.
// CHECK-LABEL: func.func @startup_distance
// UNEQUAL-LABEL: func.func @startup_distance
// ENABLE-LABEL: func.func @startup_distance
func.func @startup_distance(%src: memref<?xf32, #hivm.address_space<gm>>,
                           %dst: memref<32xf32, #hivm.address_space<gm>>) {
  // ENABLE: %[[CURSOR:.*]] = memref.alloca() : memref<1xi64>
  // ENABLE: memref.store %c0_i64, %[[CURSOR]][%c0] : memref<1xi64>
  // ENABLE: %[[READ_CURSOR:.*]] = memref.alloca() : memref<1xi64>
  // ENABLE: memref.store %c0_i64, %[[READ_CURSOR]][%c0] : memref<1xi64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index
  // ENABLE: %[[SAVED:.*]] = memref.load %[[CURSOR]][%c0] : memref<1xi64>
  // ENABLE: %[[BASE:.*]] = arith.index_cast %[[SAVED]] : i64 to index
  // ENABLE: %[[READ_SAVED:.*]] = memref.load %[[READ_CURSOR]][%c0] : memref<1xi64>
  // ENABLE: %[[READ_BASE:.*]] = arith.index_cast %[[READ_SAVED]] : i64 to index
  // ENABLE: scf.for {{.*}} iter_args(%[[CK:arg[0-9]+]] = %c0, %[[CV:arg[0-9]+]] = %c0, %[[LK:arg[0-9]+]] = %c0, %[[LV:arg[0-9]+]] = %c0, %[[ISSUED:arg[0-9]+]] = %[[BASE]], %[[FREE:arg[0-9]+]] = %c15, %[[CONSUMED:arg[0-9]+]] = %[[READ_BASE]])
  // UNEQUAL: scf.for {{.*}} iter_args(%[[CK:arg[0-9]+]] = %c0, %[[CV:arg[0-9]+]] = %c0, %[[LK:arg[0-9]+]] = %c0, %[[LV:arg[0-9]+]] = %c0, %[[ISSUED:arg[0-9]+]] = %{{.*}}, %[[FREE:arg[0-9]+]] = %c15, %{{.*}} = %{{.*}})
  %counts:2 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%k = %c0, %v = %c0) -> (index, index) {
    %a = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark {{.*}}hivm.mempool_size = 4 : i32
    annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark {{.*}}hivm.mempool_size = 4 : i32
    annotation.mark %b {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    // ENABLE: arith.cmpi slt, %[[LK]], %c8
    // ENABLE-NOT: arith.cmpi sge, %[[LV]],
    // ENABLE: arith.cmpi sle, %[[LK]],
    // ENABLE-NOT: arith.cmpi sge, %[[LV]],
    // UNEQUAL: %[[V_DONE:.*]] = arith.cmpi sge, %[[LV]], %c2
    // UNEQUAL: arith.ori %[[V_DONE]],
    // ENABLE: %[[SLOT:.*]] = arith.remui %[[ISSUED]], %c4
    // ENABLE: scf.if
    scope.scope : () -> () {
      // ENABLE: %[[LOAD:.*]] = hivm.hir.experimental.pool_slot {{.*}}[%[[SLOT]]]
      // ENABLE: memref.subview %arg0[%[[LK]]]
      %s = memref.subview %src[%k] [32] [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      // ENABLE: hivm.hir.load {{.*}} outs(%[[LOAD]]
      hivm.hir.load ins(%s : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%a : memref<32xf32, #hivm.address_space<ub>>)
      // ENABLE: annotation.mark %[[LOAD]] {hivm.experimental_pool_ready}
      scope.return
    // CHECK: hivm.load_preload = 3 : i32
    // ENABLE: } else {
    // ENABLE-NEXT: scf.yield %[[LK]], %[[ISSUED]], %[[FREE]]
    } {hivm.load_preload = 3 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    // ENABLE: arith.cmpi slt, %[[CK]], %c8
    %validK = arith.cmpi slt, %k, %c8 : index
    // ENABLE: %[[K_COMPUTE:.*]]:6 = scf.if
    %nextK = scf.if %validK -> index {
      // ENABLE-NOT: arith.maxsi
      // ENABLE-NOT: arith.minsi
      // ENABLE: %[[READ_SLOT:.*]] = arith.remui %[[CONSUMED]], %c4
      // ENABLE: %[[READ:.*]] = hivm.hir.experimental.pool_slot {{.*}}[%[[READ_SLOT]]]
      // ENABLE: %[[NEXT_READ:.*]] = arith.addi %[[CONSUMED]], %c1
      // ENABLE: hivm.hir.store ins(%[[READ]]
      hivm.hir.store ins(%a : memref<32xf32, #hivm.address_space<ub>>) outs(%dst : memref<32xf32, #hivm.address_space<gm>>)
      // ENABLE: annotation.mark %[[READ]] {hivm.experimental_pool_release}
      %next = arith.addi %k, %c1 : index
      // ENABLE: scf.yield {{.*}}, %[[NEXT_READ]] :
      scf.yield %next : index
    } else {
      // ENABLE: scf.yield %[[CK]], {{.*}}, %[[CONSUMED]] :
      scf.yield %k : index
    // ENABLE: ssbuffer.if = 99 : i32
    } {ssbuffer.if = 99 : i32}
    // ENABLE: %[[V_ORDER:.*]] = arith.cmpi slt, %{{.*}}, %[[K_COMPUTE]]#1
    // ENABLE: %[[K_DONE:.*]] = arith.cmpi sge, %[[K_COMPUTE]]#1, %c8
    // ENABLE: arith.ori %[[K_DONE]], %[[V_ORDER]]
    scope.scope : () -> () {
      %s = memref.subview %src[%v] [32] [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%s : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%b : memref<32xf32, #hivm.address_space<ub>>)
      // ENABLE: hivm.experimental_pool_ready
      scope.return
    // CHECK: hivm.load_preload = 1 : i32
    } {hivm.load_preload = 1 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    // ENABLE: arith.cmpi sge, %[[CK]], %c8
    %doneK = arith.cmpi sge, %k, %c8 : index
    // ENABLE: arith.cmpi sge, %[[CK]], %c2
    %warm = arith.cmpi sge, %k, %c2 : index
    %started = arith.ori %doneK, %warm : i1
    // ENABLE: arith.cmpi slt, %[[CV]], %c8
    %validV = arith.cmpi slt, %v, %c8 : index
    %runV = arith.andi %started, %validV : i1
    // ENABLE: %[[V_COMPUTE:.*]]:6 = scf.if
    %nextV = scf.if %runV -> index {
      // ENABLE: %[[V_SLOT:.*]] = arith.remui %[[K_COMPUTE]]#5, %c4
      // ENABLE: hivm.hir.experimental.pool_slot {{.*}}[%[[V_SLOT]]]
      // ENABLE: %[[NEXT_V_READ:.*]] = arith.addi %[[K_COMPUTE]]#5, %c1
      hivm.hir.store ins(%b : memref<32xf32, #hivm.address_space<ub>>) outs(%dst : memref<32xf32, #hivm.address_space<gm>>)
      // ENABLE: hivm.experimental_pool_release
      %next = arith.addi %v, %c1 : index
      // ENABLE: scf.yield {{.*}}, %[[NEXT_V_READ]] :
      scf.yield %next : index
    } else {
      // ENABLE: scf.yield %[[CV]], {{.*}}, %[[K_COMPUTE]]#5 :
      scf.yield %v : index
    // ENABLE: ssbuffer.if = 4 : i32
    } {ssbuffer.if = 4 : i32}
    scf.yield %nextK, %nextV : index, index
  }
  // ENABLE-NOT: scf.for
  // ENABLE: %[[FINAL_SLOT:.*]] = arith.remui %{{.*}}#4, %c4
  // ENABLE: %[[FINAL_CURSOR:.*]] = arith.index_cast %[[FINAL_SLOT]] : index to i64
  // ENABLE: memref.store %[[FINAL_CURSOR]], %[[CURSOR]][%c0] : memref<1xi64>
  // ENABLE: %[[FINAL_READ_SLOT:.*]] = arith.remui %{{.*}}#6, %c4
  // ENABLE: %[[FINAL_READ:.*]] = arith.index_cast %[[FINAL_READ_SLOT]] : index to i64
  // ENABLE: memref.store %[[FINAL_READ]], %[[READ_CURSOR]][%c0] : memref<1xi64>
  // ENABLE: return
  return
}

//--- unseparated.mlir
// A normal compute scope is not an independent Load candidate. No target
// specification is needed when Mark has no work.
func.func @unseparated(%src: memref<16xf32, #hivm.address_space<gm>>,
                       %dst: memref<16xf32, #hivm.address_space<ub>>) {
  scope.scope : () -> () {
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%dst : memref<16xf32, #hivm.address_space<ub>>)
    scope.return
  } {hivm.preload_num = 2 : i32}
  // Core splitting may leave a marked scope without a transfer.
  scope.scope : () -> () {
    scope.return
  } {hivm.load_preload = 2 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
  return
}

//--- no-spec.mlir
// Separate owns eligibility. Mark needs no original schedule marker or spec
// to retain independent buffers and add their preload lifetime protection.
// CHECK-LABEL: func.func @no_target_spec
func.func @no_target_spec(%src: memref<32xf32, #hivm.address_space<gm>>, %dst: memref<32xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK-NOT: hivm.mempool
    %b = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark {{.*}}hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32
    annotation.mark %b {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%b : memref<32xf32, #hivm.address_space<ub>>)
      scope.return
    // CHECK: hivm.load_preload = 3 : i32
    } {hivm.load_preload = 3 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    hivm.hir.store ins(%b : memref<32xf32, #hivm.address_space<ub>>) outs(%dst : memref<32xf32, #hivm.address_space<gm>>)
  }
  return
}
