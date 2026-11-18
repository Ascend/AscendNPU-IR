// Both for and while inputs below run under these three configurations.
// Partitioned pool: hint 0 assigns slots [0, 4) to Q0 and [4, 5) to Q1.
// RUN: bishengir-opt %s -split-input-file -hivm-tcb-pool-op-insertion -verify-each \
// RUN:   | FileCheck %s --check-prefixes=CHECK,PARTITION \
// RUN:       --implicit-check-not=hivm.tcb_pool_plan \
// RUN:       --implicit-check-not=hivm.tightly_coupled_buffer \
// RUN:       --implicit-check-not=arith.select

// Shared pool: hint -1 allows both queues to search all five slots.
// RUN: sed '/^module /s/hivm.tcb_pool_hint = 0/hivm.tcb_pool_hint = -1/' %s \
// RUN:   | bishengir-opt -split-input-file -hivm-tcb-pool-op-insertion -verify-each \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SHARED

// Without legacy FIX/V sync, insertion must still allocate IDs and emit sync.
// RUN: sed '/^[[:space:]]*hivm.hir.sync_block_.*PIPE_FIX/d' %s \
// RUN:   | bishengir-opt -split-input-file -hivm-tcb-pool-op-insertion -verify-each \
// RUN:   | FileCheck %s --check-prefix=CHECK

// One pool lifetime spans both work items. Each Q has one logical issue
// stage per core, with ping-pong FIX branches and selected Vector addresses.
// Exercise for scheduling loops with partitioned/shared pools.
// CHECK-LABEL: func.func @cube
// CHECK: hivm.hir.set_ctrl false at ctrl[60]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[48]
// CHECK: %[[CM:.*]] = hivm.hir.pointer_cast{{.*}}memref<{{(44|47)}}xi32, #hivm.address_space<ssbuf>>
// PARTITION: hivm.hir.tcb_pool_init {{.*}}flag_mask = 76 : i32{{.*}}search_begin = array<i32: 0, 4>{{.*}}search_end = array<i32: 4, 5>
// SHARED: hivm.hir.tcb_pool_init {{.*}}flag_mask = 108 : i32{{.*}}search_begin = array<i32: 0, 0>{{.*}}search_end = array<i32: 5, 5>
// CHECK: scf.for
// CHECK: %[[CP:.*]] = hivm.hir.pointer_cast{{.*}}memref<163840xi8, #hivm.address_space<ub>>
// CHECK: scf.for
// CHECK: %[[A0:.*]], %[[ID0:.*]] = hivm.hir.tcb_pool_alloc %[[CM]], %[[CP]] {segment = 0 : i32}
// CHECK: %[[F0:.*]] = arith.extui %[[ID0]] : i32 to i64
// CHECK: scf.if
// CHECK: hivm.hir.pointer_cast(%[[A0]])
// CHECK: hivm.hir.fixpipe
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %[[F0]]
// CHECK: } else {
// CHECK: hivm.hir.pointer_cast(%[[A0]])
// CHECK: hivm.hir.fixpipe
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %[[F0]]
// CHECK: %[[A1:.*]], %[[ID1:.*]] = hivm.hir.tcb_pool_alloc %[[CM]], %[[CP]] {segment = 1 : i32}
// CHECK-NEXT: %[[F1:.*]] = arith.extui %[[ID1]] : i32 to i64
// CHECK: scf.if
// CHECK: hivm.hir.pointer_cast(%[[A1]])
// CHECK: hivm.hir.fixpipe
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %[[F1]]
// CHECK: } else {
// CHECK: hivm.hir.pointer_cast(%[[A1]])
// CHECK: hivm.hir.fixpipe
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %[[F1]]
// CHECK: hivm.hir.tcb_pool_release %[[CM]]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[60]
// CHECK-NEXT: return
// CHECK-LABEL: func.func @vector
// CHECK: %[[VM:.*]] = hivm.hir.pointer_cast{{.*}}memref<{{(44|47)}}xi32, #hivm.address_space<ssbuf>>
// CHECK-NOT: hivm.hir.tcb_pool_init
// CHECK: scf.for
// CHECK: %[[VP:.*]] = hivm.hir.pointer_cast{{.*}}memref<163840xi8, #hivm.address_space<ub>>
// CHECK: scf.for
// CHECK: %[[G0:.*]], %[[VID0:.*]] = hivm.hir.tcb_pool_get %[[VM]], %[[VP]] {segment = 0 : i32}
// CHECK-NEXT: %[[VF0:.*]] = arith.extui %[[VID0]] : i32 to i64
// CHECK: %[[D0:.*]] = hivm.hir.pointer_cast(%[[G0]])
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = %[[VF0]]
// CHECK-NEXT: call @consume0(%[[D0]])
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %[[VF0]]
// CHECK: %[[G1:.*]], %[[VID1:.*]] = hivm.hir.tcb_pool_get %[[VM]], %[[VP]] {segment = 1 : i32}
// CHECK-NEXT: %[[VF1:.*]] = arith.extui %[[VID1]] : i32 to i64
// CHECK: %[[D1:.*]] = hivm.hir.pointer_cast(%[[G1]])
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = %[[VF1]]
// CHECK-NEXT: call @consume1(%[[D1]])
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %[[VF1]]
// CHECK-NOT: hivm.hir.tcb_pool_release

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.tcb_pool_hint = 0 : i32, hivm.tcb_pool_plan = {pool_bytes = 163840 : i64, slot_bytes = 32768 : i64, queues = [array<i32: 1, 5>, array<i32: 2, 4>]}} {
  func.func private @consume0(memref<64x256xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function}
  func.func private @consume1(memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function}
  func.func @cube(%a: memref<16x8x16x16xf32, #hivm.address_space<cc>>, %b: memref<8x8x16x16xf32, #hivm.address_space<cc>>, %ready: i1, %even: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %base = arith.constant 4096 : i64
    %zero = arith.constant 0 : i64
    %ss = hivm.hir.pointer_cast(%zero) : memref<i32, #hivm.address_space<ssbuf>>
    %zero_i32 = arith.constant 0 : i32
    memref.store %zero_i32, %ss[] : memref<i32, #hivm.address_space<ssbuf>>
    scf.for %work = %c0 to %c2 step %c1 {
      %q0_0 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q0_1 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q1_0 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
      %q1_1 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<64x128xf32, #hivm.address_space<ub>>
      scf.for %poll = %c0 to %c2 step %c1 {
        scf.if %ready {
          scf.if %even {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%a : memref<16x8x16x16xf32, #hivm.address_space<cc>>) outs(%q0_0 : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          } else {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%a : memref<16x8x16x16xf32, #hivm.address_space<cc>>) outs(%q0_1 : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
          }
        } else {
        }
        scf.if %ready {
          scf.if %even {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%b : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%q1_0 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          } else {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%b : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%q1_1 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
          }
        } else {
        }
      }
    }
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
  func.func @vector(%ready: i1, %even: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %base = arith.constant 4096 : i64
    %zero = arith.constant 0 : i64
    %sub = hivm.hir.get_sub_block_idx -> i64
    %stride = arith.constant 1024 : i64
    %offset = arith.muli %sub, %stride : i64
    %ss = hivm.hir.pointer_cast(%offset) : memref<i32, #hivm.address_space<ssbuf>>
    %zero_i32 = arith.constant 0 : i32
    memref.store %zero_i32, %ss[] : memref<i32, #hivm.address_space<ssbuf>>
    scf.for %work = %c0 to %c2 step %c1 {
      %q0_0 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q0_1 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q1_0 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
      %q1_1 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<64x128xf32, #hivm.address_space<ub>>
      scf.for %poll = %c0 to %c2 step %c1 {
        scf.if %ready {
          %selected0 = scf.if %even -> (memref<64x256xf32, #hivm.address_space<ub>>) {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
            scf.yield %q0_0 : memref<64x256xf32, #hivm.address_space<ub>>
          } else {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
            scf.yield %q0_1 : memref<64x256xf32, #hivm.address_space<ub>>
          }
          func.call @consume0(%selected0) {hivm.vector_function} : (memref<64x256xf32, #hivm.address_space<ub>>) -> ()
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        } else {
        }
        scf.if %ready {
          %selected1 = scf.if %even -> (memref<64x128xf32, #hivm.address_space<ub>>) {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
            scf.yield %q1_0 : memref<64x128xf32, #hivm.address_space<ub>>
          } else {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
            scf.yield %q1_1 : memref<64x128xf32, #hivm.address_space<ub>>
          }
          func.call @consume1(%selected1) {hivm.vector_function} : (memref<64x128xf32, #hivm.address_space<ub>>) -> ()
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        } else {
        }
      }
    }
    return
  }
}

// -----

// The outer for distributes work items. Only the inner scheduling loop is a
// while: each queue advances independently when its SSBUF counter permits a
// transfer. The before region tests completion; the after region issues FIX
// writes or Vector consumption and carries both progress counters forward.
// Pool initialization/release must span the outer loop, and alloc/get must
// stay inside the corresponding ready branch in the while's after region.

// CHECK-LABEL: func.func @cube_while
// CHECK: %[[CM:.*]] = hivm.hir.pointer_cast{{.*}}memref<{{(44|47)}}xi32, #hivm.address_space<ssbuf>>
// CHECK: hivm.hir.tcb_pool_init %[[CM]]
// CHECK: scf.for
// CHECK: %[[CP:.*]] = hivm.hir.pointer_cast{{.*}}memref<163840xi8, #hivm.address_space<ub>>
// CHECK: scf.while
// CHECK-NOT: hivm.hir.tcb_pool_alloc
// CHECK: scf.condition
// CHECK: } do {
// CHECK: ^bb0(%[[C0:.*]]: index, %[[C1:.*]]: index):
// CHECK: %[[NEXT0:.*]] = scf.if {{.*}} -> (index) {
// CHECK: %[[A0:.*]], %[[ID0:.*]] = hivm.hir.tcb_pool_alloc %[[CM]], %[[CP]] {segment = 0 : i32}
// CHECK-NEXT: %[[F0:.*]] = arith.extui %[[ID0]] : i32 to i64
// CHECK-NEXT: %[[D0:.*]] = hivm.hir.pointer_cast(%[[A0]])
// CHECK-NEXT: hivm.hir.fixpipe {{.*}}outs(%[[D0]] :
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %[[F0]]
// CHECK: scf.yield {{.*}} : index
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[C0]] : index
// CHECK: %[[NEXT1:.*]] = scf.if {{.*}} -> (index) {
// CHECK: %[[A1:.*]], %[[ID1:.*]] = hivm.hir.tcb_pool_alloc %[[CM]], %[[CP]] {segment = 1 : i32}
// CHECK-NEXT: %[[F1:.*]] = arith.extui %[[ID1]] : i32 to i64
// CHECK-NEXT: %[[D1:.*]] = hivm.hir.pointer_cast(%[[A1]])
// CHECK-NEXT: hivm.hir.fixpipe {{.*}}outs(%[[D1]] :
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %[[F1]]
// CHECK: scf.yield {{.*}} : index
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[C1]] : index
// CHECK: scf.yield %[[NEXT0]], %[[NEXT1]] : index, index
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: hivm.hir.tcb_pool_release %[[CM]]
// CHECK-NEXT: return

// CHECK-LABEL: func.func @vector_while
// CHECK: %[[VM:.*]] = hivm.hir.pointer_cast{{.*}}memref<{{(44|47)}}xi32, #hivm.address_space<ssbuf>>
// CHECK-NOT: hivm.hir.tcb_pool_init
// CHECK: scf.for
// CHECK: %[[VP:.*]] = hivm.hir.pointer_cast{{.*}}memref<163840xi8, #hivm.address_space<ub>>
// CHECK: scf.while
// CHECK-NOT: hivm.hir.tcb_pool_get
// CHECK: scf.condition
// CHECK: } do {
// CHECK: ^bb0(%[[V0:.*]]: index, %[[V1:.*]]: index):
// CHECK: %[[VNEXT0:.*]] = scf.if {{.*}} -> (index) {
// CHECK: %[[G0:.*]], %[[VID0:.*]] = hivm.hir.tcb_pool_get %[[VM]], %[[VP]] {segment = 0 : i32}
// CHECK-NEXT: %[[VF0:.*]] = arith.extui %[[VID0]] : i32 to i64
// CHECK-NEXT: %[[VD0:.*]] = hivm.hir.pointer_cast(%[[G0]])
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = %[[VF0]]
// CHECK-NEXT: call @consume0(%[[VD0]])
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %[[VF0]]
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[V0]] : index
// CHECK: %[[VNEXT1:.*]] = scf.if {{.*}} -> (index) {
// CHECK: %[[G1:.*]], %[[VID1:.*]] = hivm.hir.tcb_pool_get %[[VM]], %[[VP]] {segment = 1 : i32}
// CHECK-NEXT: %[[VF1:.*]] = arith.extui %[[VID1]] : i32 to i64
// CHECK-NEXT: %[[VD1:.*]] = hivm.hir.pointer_cast(%[[G1]])
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = %[[VF1]]
// CHECK-NEXT: call @consume1(%[[VD1]])
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %[[VF1]]
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[V1]] : index
// CHECK: scf.yield %[[VNEXT0]], %[[VNEXT1]] : index, index
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NOT: hivm.hir.tcb_pool_release
// CHECK-NEXT: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.tcb_pool_hint = 0 : i32, hivm.tcb_pool_plan = {pool_bytes = 163840 : i64, slot_bytes = 32768 : i64, queues = [array<i32: 1, 5>, array<i32: 2, 4>]}} {
  func.func private @consume0(memref<64x256xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function}
  func.func private @consume1(memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function}

  func.func @cube_while(%a: memref<16x8x16x16xf32, #hivm.address_space<cc>>, %b: memref<8x8x16x16xf32, #hivm.address_space<cc>>, %tiles: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %four = arith.constant 4 : i64
    %zero_i32 = arith.constant 0 : i32
    %one_i32 = arith.constant 1 : i32
    %two_i32 = arith.constant 2 : i32
    %base = arith.constant 4096 : i64
    %ss0 = hivm.hir.pointer_cast(%zero) : memref<i32, #hivm.address_space<ssbuf>>
    %ss1 = hivm.hir.pointer_cast(%four) : memref<i32, #hivm.address_space<ssbuf>>
    memref.store %zero_i32, %ss0[] : memref<i32, #hivm.address_space<ssbuf>>
    memref.store %zero_i32, %ss1[] : memref<i32, #hivm.address_space<ssbuf>>
    scf.for %work = %c0 to %tiles step %c1 {
      %q0_0 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q0_1 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q1_0 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
      %q1_1 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<64x128xf32, #hivm.address_space<ub>>
      %issued:2 = scf.while (%issued0 = %c0, %issued1 = %c0) : (index, index) -> (index, index) {
        %more0 = arith.cmpi ult, %issued0, %c2 : index
        %more1 = arith.cmpi ult, %issued1, %c2 : index
        %continue = arith.ori %more0, %more1 : i1
        scf.condition(%continue) %issued0, %issued1 : index, index
      } do {
      ^bb0(%issued0: index, %issued1: index):
        %pending0 = memref.load %ss0[] : memref<i32, #hivm.address_space<ssbuf>>
        annotation.mark %pending0 {memref_ext.volatile} : i32
        %space0 = arith.cmpi slt, %pending0, %two_i32 : i32
        %more0 = arith.cmpi ult, %issued0, %c2 : index
        %ready0 = arith.andi %space0, %more0 : i1
        %next0 = scf.if %ready0 -> (index) {
          %parity = arith.remui %issued0, %c2 : index
          %even = arith.cmpi eq, %parity, %c0 : index
          %selected = arith.select %even, %q0_0, %q0_1 : memref<64x256xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%a : memref<16x8x16x16xf32, #hivm.address_space<cc>>) outs(%selected : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %pending = arith.addi %pending0, %one_i32 : i32
          memref.store %pending, %ss0[] : memref<i32, #hivm.address_space<ssbuf>>
          %next = arith.addi %issued0, %c1 : index
          scf.yield %next : index
        } else {
          scf.yield %issued0 : index
        }
        %pending1 = memref.load %ss1[] : memref<i32, #hivm.address_space<ssbuf>>
        annotation.mark %pending1 {memref_ext.volatile} : i32
        %space1 = arith.cmpi slt, %pending1, %two_i32 : i32
        %more1 = arith.cmpi ult, %issued1, %c2 : index
        %ready1 = arith.andi %space1, %more1 : i1
        %next1 = scf.if %ready1 -> (index) {
          %parity = arith.remui %issued1, %c2 : index
          %even = arith.cmpi eq, %parity, %c0 : index
          %selected = arith.select %even, %q1_0, %q1_1 : memref<64x128xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%b : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%selected : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %pending = arith.addi %pending1, %one_i32 : i32
          memref.store %pending, %ss1[] : memref<i32, #hivm.address_space<ssbuf>>
          %next = arith.addi %issued1, %c1 : index
          scf.yield %next : index
        } else {
          scf.yield %issued1 : index
        }
        scf.yield %next0, %next1 : index, index
      }
    }
    return
  }

  func.func @vector_while(%tiles: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero_i32 = arith.constant 0 : i32
    %one_i32 = arith.constant 1 : i32
    %four = arith.constant 4 : i64
    %stride = arith.constant 1024 : i64
    %base = arith.constant 4096 : i64
    %sub = hivm.hir.get_sub_block_idx -> i64
    %offset0 = arith.muli %sub, %stride : i64
    %offset1 = arith.addi %offset0, %four : i64
    %ss0 = hivm.hir.pointer_cast(%offset0) : memref<i32, #hivm.address_space<ssbuf>>
    %ss1 = hivm.hir.pointer_cast(%offset1) : memref<i32, #hivm.address_space<ssbuf>>
    scf.for %work = %c0 to %tiles step %c1 {
      %q0_0 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q0_1 = hivm.hir.pointer_cast(%base) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %q0_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<64x256xf32, #hivm.address_space<ub>>
      %q1_0 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_0 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x128xf32, #hivm.address_space<ub>>
      %q1_1 = hivm.hir.pointer_cast(%base) : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %q1_1 {effects = ["read", "write"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<64x128xf32, #hivm.address_space<ub>>
      %consumed:2 = scf.while (%consumed0 = %c0, %consumed1 = %c0) : (index, index) -> (index, index) {
        %more0 = arith.cmpi ult, %consumed0, %c2 : index
        %more1 = arith.cmpi ult, %consumed1, %c2 : index
        %continue = arith.ori %more0, %more1 : i1
        scf.condition(%continue) %consumed0, %consumed1 : index, index
      } do {
      ^bb0(%consumed0: index, %consumed1: index):
        %pending0 = memref.load %ss0[] : memref<i32, #hivm.address_space<ssbuf>>
        annotation.mark %pending0 {memref_ext.volatile} : i32
        %available0 = arith.cmpi sgt, %pending0, %zero_i32 : i32
        %more0 = arith.cmpi ult, %consumed0, %c2 : index
        %ready0 = arith.andi %available0, %more0 : i1
        %next0 = scf.if %ready0 -> (index) {
          %parity = arith.remui %consumed0, %c2 : index
          %even = arith.cmpi eq, %parity, %c0 : index
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %selected = arith.select %even, %q0_0, %q0_1 : memref<64x256xf32, #hivm.address_space<ub>>
          func.call @consume0(%selected) {hivm.vector_function} : (memref<64x256xf32, #hivm.address_space<ub>>) -> ()
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
          %pending = arith.subi %pending0, %one_i32 : i32
          memref.store %pending, %ss0[] : memref<i32, #hivm.address_space<ssbuf>>
          %next = arith.addi %consumed0, %c1 : index
          scf.yield %next : index
        } else {
          scf.yield %consumed0 : index
        }
        %pending1 = memref.load %ss1[] : memref<i32, #hivm.address_space<ssbuf>>
        annotation.mark %pending1 {memref_ext.volatile} : i32
        %available1 = arith.cmpi sgt, %pending1, %zero_i32 : i32
        %more1 = arith.cmpi ult, %consumed1, %c2 : index
        %ready1 = arith.andi %available1, %more1 : i1
        %next1 = scf.if %ready1 -> (index) {
          %parity = arith.remui %consumed1, %c2 : index
          %even = arith.cmpi eq, %parity, %c0 : index
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %selected = arith.select %even, %q1_0, %q1_1 : memref<64x128xf32, #hivm.address_space<ub>>
          func.call @consume1(%selected) {hivm.vector_function} : (memref<64x128xf32, #hivm.address_space<ub>>) -> ()
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          %pending = arith.subi %pending1, %one_i32 : i32
          memref.store %pending, %ss1[] : memref<i32, #hivm.address_space<ssbuf>>
          %next = arith.addi %consumed1, %c1 : index
          scf.yield %next : index
        } else {
          scf.yield %consumed1 : index
        }
        scf.yield %next0, %next1 : index, index
      }
    }
    return
  }
}
