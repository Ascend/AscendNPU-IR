// RUN: split-file %s %t
// RUN: bishengir-opt %t/candidates.mlir -split-input-file -pass-pipeline='builtin.module(hivm-separate-load-compute)' | FileCheck %t/candidates.mlir
// RUN: bishengir-opt %t/candidates.mlir -hivm-separate-load-compute -hivm-separate-load-compute -split-input-file | FileCheck %t/candidates.mlir
// RUN: bishengir-opt %t/unchanged.mlir -o %t/before.mlir
// RUN: bishengir-opt %t/unchanged.mlir -hivm-separate-load-compute -hivm-mark-buffer-pool -o %t/after.mlir
// RUN: diff %t/before.mlir %t/after.mlir


// RUN: bishengir-opt %t/ssbuffer-stages.mlir -hivm-separate-load-compute -o %t/separated.mlir
// RUN: FileCheck %t/ssbuffer-stages.mlir --check-prefix=STAGE --implicit-check-not=hivm.mempool < %t/separated.mlir

//--- candidates.mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// A module pass must continue past declarations and functions with no work.
func.func private @external()
func.func @no_load() {
  return
}
// CHECK-LABEL: func.func @nested_staged
func.func @nested_staged(%src: memref<16x16xf16, #hivm.address_space<gm>>, %rhs: tensor<1x1x16x16xf32>) {
  %tile_size = arith.constant 16 : index
  %init = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: %[[BUF:.*]] = memref.alloc
    // CHECK: annotation.mark %[[BUF]] {hivm.multi_buffer = 2 : i32}
    // CHECK: scope.scope : () -> () {
    // CHECK: hivm.hir.nd2nz {{.*}} outs(%[[BUF]]
    // CHECK-NEXT: scope.return
    // CHECK-NEXT: } {hivm.load_preload = 1 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
    // CHECK: scope.scope : () -> tensor<1x1x16x16xf32>
    %result = scope.scope : () -> tensor<1x1x16x16xf32> {
      // CHECK: scope.scope : () -> tensor<1x1x16x16xf32>
      %inner = scope.scope : () -> tensor<1x1x16x16xf32> {
        %buf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %buf {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        // CHECK-NOT: hivm.hir.nd2nz
        hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%buf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        // CHECK: bufferization.to_tensor %[[BUF]]
        %t = bufferization.to_tensor %buf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        // CHECK: hivm.hir.mmadL1
        %sum = hivm.hir.mmadL1 ins(%t, %t, %init, %tile_size, %tile_size, %tile_size : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%rhs : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        scope.return %sum : tensor<1x1x16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.return %inner : tensor<1x1x16x16xf32>
    // CHECK: } {cv_merged_scope, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    } {cv_merged_scope, hivm.preload_num = 2 : i32, hivm.max_preload_num = 3 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}
}
// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @hoisted_buffer
func.func @hoisted_buffer(%src: memref<16x16xf16, #hivm.address_space<gm>>, %rhs: tensor<1x1x16x16xf32>) {
  %tile_size = arith.constant 16 : index
  %init = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: %[[BUF:.*]] = memref.alloc
    %buf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK-NEXT: scope.scope : () -> () {
    // CHECK: hivm.hir.nd2nz {{.*}} outs(%[[BUF]]
    // CHECK-NEXT: scope.return
    // CHECK-NEXT: } {hivm.load_preload = 1 : i32,
    // CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[BUF]]
    %t = bufferization.to_tensor %buf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK-NEXT: {{.*}}scope.scope : () -> tensor<1x1x16x16xf32>
    %r = scope.scope : () -> tensor<1x1x16x16xf32> {
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%buf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.mmadL1 ins(%[[T]]
      %sum = hivm.hir.mmadL1 ins(%t, %t, %init, %tile_size, %tile_size, %tile_size : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%rhs : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      scope.return %sum : tensor<1x1x16x16xf32>
    // CHECK: hivm.preload_num = 0 : i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, hivm.max_preload_num = 1 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}
}
// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @ssbuffer_padding
func.func @ssbuffer_padding(%src: memref<?x16xf16, #hivm.address_space<gm>>, %out: memref<1x1x16x16xf32, #hivm.address_space<cc>>, %pad: i1, %head: i32, %headStride: index) {
  %tile_size = arith.constant 16 : index
  %init = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %limit = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %z = arith.constant 0.0 : f16
  %c3 = arith.constant 3 : i32
  // CHECK: %[[KV_HEAD:.*]] = arith.divsi
  %kvHead = arith.divsi %head, %c3 : i32
  // CHECK: %[[HEAD_INDEX:.*]] = arith.index_cast %[[KV_HEAD]]
  %headIndex = arith.index_cast %kvHead : i32 to index
  // CHECK: %[[HEAD_OFFSET:.*]] = arith.muli %[[HEAD_INDEX]],
  %headOffset = arith.muli %headIndex, %headStride : index
  // CHECK: scf.for
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%count = %zero) -> i32 {
    %cond = arith.cmpi slt, %count, %limit : i32
    // CHECK: %[[BUF:.*]] = memref.alloc
    // CHECK: scope.scope : () -> () {
    // CHECK-NOT: arith.divsi
    // CHECK: arith.addi %{{.*}}, %[[HEAD_OFFSET]]
    // CHECK: scf.if %{{.*}} {
    // CHECK: hivm.hir.vbrc {{.*}} outs(%[[BUF]]
    // CHECK: hivm.hir.nd2nz {{.*}} outs(%[[BUF]]
    // CHECK-NEXT: scope.return
    // CHECK-NEXT: } {hivm.load_preload = 1 : i32,
    // CHECK: scf.if %{{.*}} -> (i32) {
    %next = scf.if %cond -> i32 {
      %buf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      scf.if %pad {
        hivm.hir.vbrc ins(%z : f16) outs(%buf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      } {hivm.unlikely_condition}
      %task = arith.index_cast %count : i32 to index
      %offset = arith.muli %task, %c16 : index
      %address = arith.addi %offset, %headOffset : index
      %tile = memref.reinterpret_cast %src to offset: [%address], sizes: [16, 16], strides: [16, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      // CHECK-NOT: hivm.hir.nd2nz
      hivm.hir.nd2nz ins(%tile : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%buf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.mmadL1
      hivm.hir.mmadL1 ins(%buf, %buf, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%out : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      %inc = arith.addi %count, %one : i32
      scf.yield %inc : i32
    // CHECK: } else {
    } else {
      // CHECK: scf.yield
      scf.yield %count : i32
    // CHECK: ssbuffer.if = 99 : i32
    } {ssbuffer.if = 99 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scf.yield %next : i32
  }
  return
}
}

//--- unchanged.mlir
// Unsupported candidates retain all original guards, allocations and writes.
// No load marker is created, so Mark has no work and needs no target spec.
func.func private @opaque(memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)

func.func @unknown_consumer(%src: memref<16x16xf16, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scope.scope : () -> () {
      %buf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%buf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      func.call @opaque(%buf) : (memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}

// Vector scopes retain their loads and hoisted buffer/tensor inputs.
func.func @vector_unchanged(%src: memref<16xf32, #hivm.address_space<gm>>, %rhs: tensor<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    %t = bufferization.to_tensor %buf restrict writable : memref<16xf32, #hivm.address_space<ub>>
    %r = scope.scope : () -> tensor<16xf32> {
      hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%buf : memref<16xf32, #hivm.address_space<ub>>)
      %sum = hivm.hir.vadd ins(%t, %rhs : tensor<16xf32>, tensor<16xf32>) outs(%rhs : tensor<16xf32>) -> tensor<16xf32>
      scope.return %sum : tensor<16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, hivm.max_preload_num = 1 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}

//--- while.mlir
// Both regions schedule two streams. They must form separate pools and
// retain separate state, including when only before executes (limit = 0).
// WHILEMARK-LABEL: func.func @ssbuffer_while_regions
// WHILE-LABEL: func.func @ssbuffer_while_regions
// WHILESYNC-LABEL: func.func @ssbuffer_while_regions
func.func @ssbuffer_while_regions(%src: memref<?x16xf16, #hivm.address_space<gm>>, %dst: memref<1x1x16x16xf32, #hivm.address_space<cc>>, %limit: index) {
  %tile_size = arith.constant 16 : index
  %init = arith.constant true
  // WHILE: %[[ZERO:.*]] = arith.constant 0 : i64
  // WHILE: memref.store %[[ZERO]], %[[AC:[a-zA-Z0-9_]+]][
  // WHILE: memref.store %[[ZERO]], %[[AR:[a-zA-Z0-9_]+]][
  // WHILE: memref.store %[[ZERO]], %[[BC:[a-zA-Z0-9_]+]][
  // WHILE: memref.store %[[ZERO]], %[[BR:[a-zA-Z0-9_]+]][
  // WHILESYNC-COUNT-8: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // WHILE: scf.for
  // WHILESYNC: scf.for
  scf.for %outer = %c0 to %c2 step %c1 {
    // WHILEMARK: scf.while
    // WHILE: memref.load %[[BC]][
    // WHILE: memref.load %[[BR]][
    // WHILE: memref.load %[[AC]][
    // WHILE: memref.load %[[AR]][
    // WHILE: %[[DONE:.*]]:14 = scf.while {{.*}} : (index, index, index, index, index, index, index, index, index, index, index, index, index, index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index)
    // WHILESYNC: scf.while
    %done:4 = scf.while (%kb = %c0, %vb = %c0, %ka = %c0, %va = %c0) : (index, index, index, index) -> (index, index, index, index) {
      %kbbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // WHILEMARK: hivm.mempool_id = 0 : i32, hivm.mempool_size = 4 : i32
      annotation.mark %kbbuf {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %kbvalid = arith.cmpi slt, %kb, %limit : index
      // WHILEMARK: hivm.load_preload = 3 : i32
      %kbnext = scf.if %kbvalid -> index {
        %s = memref.subview %src[%kb, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        // WHILE: hivm.hir.experimental.pool_slot
        // WHILESYNC: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
        // WHILESYNC: hivm.hir.nd2nz
        hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%kbbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        // WHILE: hivm.experimental_pool_ready
        // WHILESYNC: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %
        hivm.hir.mmadL1 ins(%kbbuf, %kbbuf, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
        // WHILE: hivm.experimental_pool_release
        %inc = arith.addi %kb, %c1 : index
        scf.yield %inc : index
      } else {
        scf.yield %kb : index
      } {ssbuffer.if = 7 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %vbbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // WHILEMARK: hivm.mempool_id = 0 : i32, hivm.mempool_size = 4 : i32
      annotation.mark %vbbuf {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %vbvalid = arith.cmpi slt, %vb, %limit : index
      %vbwarm = arith.cmpi sge, %kb, %c2 : index
      %vbdone = arith.cmpi sge, %kb, %limit : index
      %vbready = arith.ori %vbwarm, %vbdone : i1
      %vbrun = arith.andi %vbvalid, %vbready : i1
      // WHILEMARK: hivm.load_preload = 1 : i32
      // WHILEMARK-NEXT: {{.*}}arith.cmpi slt,
      // WHILEMARK-NEXT: {{.*}}scf.if
      %vbnext = scf.if %vbrun -> index {
        %s = memref.subview %src[%vb, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%vbbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.mmadL1 ins(%vbbuf, %vbbuf, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
        %inc = arith.addi %vb, %c1 : index
        scf.yield %inc : index
      } else {
        scf.yield %vb : index
      } {ssbuffer.if = 7 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %kbmore = arith.cmpi slt, %kbnext, %limit : index
      %vbmore = arith.cmpi slt, %vbnext, %limit : index
      %kamore = arith.cmpi slt, %ka, %limit : index
      %vamore = arith.cmpi slt, %va, %limit : index
      %bmore = arith.ori %kbmore, %vbmore : i1
      %amore = arith.ori %kamore, %vamore : i1
      %more = arith.ori %bmore, %amore : i1
      // WHILE: scf.condition({{.*}}) {{.*}} : index, index, index, index, index, index, index, index, index, index, index, index, index, index
      scf.condition(%more) %kbnext, %vbnext, %ka, %va : index, index, index, index
    // WHILEMARK: } do {
    // WHILE-NEXT: } do {
    // WHILESYNC: } do {
    } do {
    ^bb0(%kb: index, %vb: index, %ka: index, %va: index):
      %kabuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // WHILEMARK: hivm.mempool_id = 1 : i32, hivm.mempool_size = 4 : i32
      annotation.mark %kabuf {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %kavalid = arith.cmpi slt, %ka, %limit : index
      // WHILEMARK: hivm.load_preload = 3 : i32
      %kanext = scf.if %kavalid -> index {
        %s = memref.subview %src[%ka, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        // WHILE: hivm.hir.experimental.pool_slot
        // WHILESYNC: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
        // WHILESYNC: hivm.hir.nd2nz
        hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%kabuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        // WHILE: hivm.experimental_pool_ready
        // WHILESYNC: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %
        hivm.hir.mmadL1 ins(%kabuf, %kabuf, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
        // WHILE: hivm.experimental_pool_release
        %inc = arith.addi %ka, %c1 : index
        scf.yield %inc : index
      } else {
        scf.yield %ka : index
      } {ssbuffer.if = 7 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %vabuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      // WHILEMARK: hivm.mempool_id = 1 : i32, hivm.mempool_size = 4 : i32
      annotation.mark %vabuf {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %vavalid = arith.cmpi slt, %va, %limit : index
      %vawarm = arith.cmpi sge, %ka, %c2 : index
      %vadone = arith.cmpi sge, %ka, %limit : index
      %vaready = arith.ori %vawarm, %vadone : i1
      %varun = arith.andi %vavalid, %vaready : i1
      // WHILEMARK: hivm.load_preload = 1 : i32
      // WHILEMARK-NEXT: {{.*}}arith.cmpi slt,
      // WHILEMARK-NEXT: {{.*}}scf.if
      %vanext = scf.if %varun -> index {
        %s = memref.subview %src[%va, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%vabuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.mmadL1 ins(%vabuf, %vabuf, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
        %inc = arith.addi %va, %c1 : index
        scf.yield %inc : index
      } else {
        scf.yield %va : index
      } {ssbuffer.if = 7 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      // WHILE: scf.yield {{.*}} : index, index, index, index, index, index, index, index, index, index, index, index, index, index
      scf.yield %kb, %vb, %kanext, %vanext : index, index, index, index
    }
    // WHILE: arith.remui %[[DONE]]#11
    // WHILE: memref.store {{.*}}, %[[AC]][
    // WHILE: arith.remui %[[DONE]]#13
    // WHILE: memref.store {{.*}}, %[[AR]][
    // WHILE: arith.remui %[[DONE]]#6
    // WHILE: memref.store {{.*}}, %[[BC]][
    // WHILE: arith.remui %[[DONE]]#8
    // WHILE: memref.store {{.*}}, %[[BR]][
  }
  // WHILE-NOT: scf.while
  // WHILE: return
  // WHILESYNC-COUNT-8: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID
  // WHILESYNC: return
  return
}

//--- ssbuffer-stages.mlir
// SSBuffer numbers do not encode stages. Mark uses the stages even after
// the original scheduling attributes disappear.
// STAGE-LABEL: func.func @startup_distance
func.func @startup_distance(%src: memref<?x16xf16, #hivm.address_space<gm>>,
                           %dst: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  %tile_size = arith.constant 16 : index
  %init = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index
  // STAGE: scf.for
  %counts:2 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%k = %c0, %v = %c0) -> (index, index) {
    // Both loads must precede the computation handshake and scalar guards.
    // STAGE-NEXT: %[[KBUF:.*]] = memref.alloc
    // POOL: hivm.mempool_size = 8 : i32
    // STAGE: scope.scope : () -> () {
    // STAGE: hivm.hir.nd2nz {{.*}} outs(%[[KBUF]]
    // STAGE-NEXT: scope.return
    // STAGE-NEXT: } {hivm.load_preload = 3 : i32,
    // STAGE-NEXT: %[[VBUF:.*]] = memref.alloc
    // POOL: hivm.mempool_size = 8 : i32
    // STAGE: scope.scope : () -> () {
    // STAGE: hivm.hir.nd2nz {{.*}} outs(%[[VBUF]]
    // STAGE-NEXT: scope.return
    // STAGE-NEXT: } {hivm.load_preload = 1 : i32,
    // STAGE-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    %a = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %b = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %b {hivm.multi_buffer = 2 : i32} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // STAGE-NEXT: %[[VALID:.*]] = arith.cmpi slt,
    %validK = arith.cmpi slt, %k, %c8 : index
    // STAGE-NEXT: {{.*}}scf.if %[[VALID]]
    %nextK = scf.if %validK -> index {
      %s = memref.subview %src[%k, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%a : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%a, %a, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      %next = arith.addi %k, %c1 : index
      scf.yield %next : index
    } else {
      scf.yield %k : index
    } {ssbuffer.if = 99 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %doneK = arith.cmpi sge, %k, %c8 : index
    %warm = arith.cmpi sge, %k, %c2 : index
    %started = arith.ori %doneK, %warm : i1
    %validV = arith.cmpi slt, %v, %c8 : index
    %runV = arith.andi %started, %validV : i1
    %nextV = scf.if %runV -> index {
      %s = memref.subview %src[%v, 0] [16, 16] [1, 1] : memref<?x16xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz ins(%s : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%b : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%b, %b, %init, %tile_size, %tile_size, %tile_size : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%dst : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
      %next = arith.addi %v, %c1 : index
      scf.yield %next : index
    } else {
      scf.yield %v : index
    } {ssbuffer.if = 4 : i32, hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    scf.yield %nextK, %nextV : index, index
  }
  return
}
