// RUN: split-file %s %t
// RUN: bishengir-opt %t/pipeline.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-mark-buffer-pool -hivm-plan-memory -hivm-enable-buffer-pool -cse -hivm-graph-sync-solver=solver-version=v1 | FileCheck %t/pipeline.mlir --implicit-check-not=hivm.hir.pipe_barrier
// RUN: bishengir-opt %t/pipeline.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-mark-buffer-pool -hivm-plan-memory -hivm-enable-buffer-pool -cse -hivm-graph-sync-solver=solver-version=v2 | FileCheck %t/pipeline.mlir --implicit-check-not=hivm.hir.pipe_barrier
// RUN: bishengir-opt %t/mmad.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-graph-sync-solver=solver-version=v1 | FileCheck %t/mmad.mlir
// RUN: bishengir-opt %t/mmad.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-graph-sync-solver=solver-version=v2 | FileCheck %t/mmad.mlir


//--- pipeline.mlir
module attributes {hivm.load_buffer_pool = true} {
// The ready bank reserves IDs 1, 3 and 5; the free bank reserves ID 0.
// Both streams share exactly five events per direction. Numbering need not
// match between the two directions or be contiguous.

// CHECK-LABEL: func.func @shared_five_slots
func.func @shared_five_slots(%src: memref<?x16xf16, #hivm.address_space<gm>>,
                           %dst: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  // CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  // CHECK: scf.for
  %counts:2 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%k = %c0, %v = %c0) -> (index, index) {
    %a = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %b = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK-COUNT-2: hivm.mempool_size = 5 : i32
    annotation.mark %a {hivm.multi_buffer = 3 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %b {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[WS:.*]] = arith.remui
    // CHECK-DAG: arith.addi %[[WS]],
    // CHECK-DAG: arith.cmpi eq, %[[WS]],
    // CHECK-DAG: arith.select
    // CHECK: %[[WRITE:.*]] = hivm.hir.experimental.pool_slot {{.*}}[%[[WS]]]
    scope.scope : () -> () {
      %s = memref.subview %src[%k, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %{{.*}}]
      // CHECK: hivm.hir.nd2nz {{.*}} outs(%[[WRITE]]
      hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%a : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %{{.*}}]
      scope.return
    } {hivm.load_preload = 3 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %validK = arith.cmpi slt, %k, %c8 : index
    %nextK = scf.if %validK -> index {
      // CHECK: %[[RS:.*]] = arith.remui
      // CHECK-DAG: arith.addi %[[RS]],
      // CHECK-DAG: arith.cmpi eq, %[[RS]],
      // CHECK-DAG: arith.select
      // CHECK: %[[READ:.*]] = hivm.hir.experimental.pool_slot {{.*}}[%[[RS]]]
      // CHECK: hivm.hir.mmadL1 ins(%[[READ]], %[[READ]]
      // CHECK-SAME: sync_related_args(
      hivm.hir.mmadL1 ins(%a, %a, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      %next = arith.addi %k, %c1 : index
      scf.yield %next : index
    } else {
      scf.yield %k : index
    } {ssbuffer.if = 99 : i32}
    scope.scope : () -> () {
      %s = memref.subview %src[%v, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%b : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      scope.return
    } {hivm.load_preload = 1 : i32, no_inline, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %doneK = arith.cmpi sge, %k, %c8 : index
    %warm = arith.cmpi sge, %k, %c2 : index
    %started = arith.ori %doneK, %warm : i1
    %validV = arith.cmpi slt, %v, %c8 : index
    %runV = arith.andi %started, %validV : i1
    %nextV = scf.if %runV -> index {
      hivm.hir.mmadL1 ins(%b, %b, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      %next = arith.addi %v, %c1 : index
      scf.yield %next : index
    } else {
      scf.yield %v : index
    } {ssbuffer.if = 4 : i32}
    scf.yield %nextK, %nextV : index, index
  // CHECK: hivm.cv_pipelined_loop
  } {hivm.cv_pipelined_loop}
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  // CHECK-NEXT: %[[FINAL_SLOT:.*]] = arith.remui
  // CHECK-NEXT: %[[FINAL_CURSOR:.*]] = arith.index_cast %[[FINAL_SLOT]] : index to i64
  // CHECK-NEXT: memref.store %[[FINAL_CURSOR]], {{.*}} : memref<1xi64>
  // CHECK-NEXT: %[[FINAL_READ_SLOT:.*]] = arith.remui
  // CHECK-NEXT: %[[FINAL_READ_CURSOR:.*]] = arith.index_cast %[[FINAL_READ_SLOT]] : index to i64
  // CHECK-NEXT: memref.store %[[FINAL_READ_CURSOR]], {{.*}} : memref<1xi64>
  // CHECK-NEXT: return
  return
}

}

//--- mmad.mlir
module attributes {hivm.load_buffer_pool = true} {
// One event is reused by Q/K/V padding, separate from the five pool slots.
// Ordinary GSS protects padding; slot waits must remain immediately before DMA.
// CHECK-LABEL: func.func @shared_l1_padding
func.func @shared_l1_padding(%src: memref<16x16xf16, #hivm.address_space<gm>>, %pad: i1, %event: i64) {
  // Unrelated event banks retain their ordinary handling, including dynamic IDs.
  // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, %{{.*}}]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, %event]
  // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, %{{.*}}]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, %event]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %zero = arith.constant 0.0 : f16
  %a0 = arith.constant 0 : i64
  %a1 = arith.constant 512 : i64
  %a2 = arith.constant 1024 : i64
  %a3 = arith.constant 1536 : i64
  %a4 = arith.constant 2048 : i64
  %a5 = arith.constant 2560 : i64
  %a6 = arith.constant 3072 : i64
  %out = hivm.hir.pointer_cast(%a0) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
  scf.for %outer = %c0 to %c5 step %c1 {
    %q = hivm.hir.pointer_cast(%a5, %a6) : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %q {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: scf.if
    scf.if %pad {
      // CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID[[PAD:[0-7]]]>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID[[PAD]]>]
      // CHECK-NEXT: hivm.hir.vbrc
      hivm.hir.vbrc ins(%zero : f16) outs(%q : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%q : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
    scf.for %i = %c0 to %c5 step %c1 {
      %k = hivm.hir.pointer_cast(%a0, %a1, %a2, %a3, %a4) {hivm.experimental_buffer_slots} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %k {hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.experimental.pool_slot
      %kw = hivm.hir.experimental.pool_slot %k[%i] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // CHECK-NOT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
      // CHECK: scf.if
      scf.if %pad {
        // CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID[[PAD]]>]
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID[[PAD]]>]
        // CHECK-NEXT: hivm.hir.vbrc
        %padding = memref.collapse_shape %kw [[0, 1, 2, 3]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> into memref<256xf16, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%zero : f16) outs(%padding : memref<256xf16, #hivm.address_space<cbuf>>)
        // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
      // CHECK-NEXT: } {hivm.unlikely_condition}
      } {hivm.unlikely_condition}
      // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %{{.*}}]
      // CHECK-NEXT: hivm.hir.nd2nz
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%kw : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      annotation.mark %kw {hivm.experimental_pool_ready} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %kr = hivm.hir.experimental.pool_slot %k[%i] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.mmadL1 ins(%q, %kr, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      annotation.mark %kr {hivm.experimental_pool_release} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %v = hivm.hir.pointer_cast(%a0, %a1, %a2, %a3, %a4) {hivm.experimental_buffer_slots} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %v {hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.experimental.pool_slot
      %vw = hivm.hir.experimental.pool_slot %v[%i] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // The prior K write is covered by the pool; padding's own barrier remains.
      // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
      // CHECK-NOT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
      // CHECK: scf.if
      scf.if %pad {
        // CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID[[PAD]]>]
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID[[PAD]]>]
        // CHECK-NEXT: hivm.hir.vbrc
        hivm.hir.vbrc ins(%zero : f16) outs(%vw : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
      // CHECK-NEXT: } {hivm.unlikely_condition}
      } {hivm.unlikely_condition}
      // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %{{.*}}]
      // CHECK-NEXT: hivm.hir.nd2nz
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%vw : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      annotation.mark %vw {hivm.experimental_pool_ready} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %vr = hivm.hir.experimental.pool_slot %v[%i] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.mmadL1 ins(%q, %vr, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      annotation.mark %vr {hivm.experimental_pool_release} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    } {hivm.cv_pipelined_loop}
  }
  return
}

// CHECK-LABEL: func.func @last_mmad_releases
func.func @last_mmad_releases(%src: memref<16x16xf16, #hivm.address_space<gm>>, %layout: i1) {
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  %a0 = arith.constant 0 : i64
  %a1 = arith.constant 512 : i64
  %a2 = arith.constant 1024 : i64
  %a3 = arith.constant 1536 : i64
  %a4 = arith.constant 2048 : i64
  %a5 = arith.constant 2560 : i64
  %a6 = arith.constant 3072 : i64
  %true = arith.constant true
  %out = hivm.hir.pointer_cast(%a0) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
  // CHECK: scf.for
  scf.for %i = %c0 to %c7 step %c1 {
    %root = hivm.hir.pointer_cast(%a0, %a1, %a2, %a3, %a4, %a5, %a6) {hivm.experimental_buffer_slots} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %w = hivm.hir.experimental.pool_slot %root[%i] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // Both layout branches submit the same task. Keep ready at the write,
    // including when the selected endpoint comes from the else branch.
    // CHECK: scf.if
    scf.if %layout {
      // CHECK: hivm.hir.nd2nz
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%w : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
    // CHECK: } else {
    } else {
      // CHECK: hivm.hir.nd2nz
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%w : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
    }
    annotation.mark %w {hivm.experimental_pool_ready} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %r = hivm.hir.experimental.pool_slot %root[%i] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %view = memref.subview %r[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // Seven events still fit with ID 1 reserved. Both view readers retain
    // one binding; only the final MMAD releases the slot.
    // CHECK: hivm.hir.mmadL1
    // CHECK-SAME: sync_related_args(%[[READY:[a-zA-Z0-9_]+]], %[[NEG:c-1_i64[a-zA-Z0-9_]*]], %[[NEG]], %[[NEG]],
    hivm.hir.mmadL1 ins(%view, %view, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.mmadL1
    // CHECK-SAME: sync_related_args(%[[LASTNEG:c-1_i64[a-zA-Z0-9_]*]], %[[LASTNEG]], %[[LASTNEG]], %[[FREE:[a-zA-Z0-9_]+]],
    hivm.hir.mmadL1 ins(%view, %view, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
    annotation.mark %r {hivm.experimental_pool_release} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
  }
  return
}
}
