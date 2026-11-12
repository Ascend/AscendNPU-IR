// RUN: bishengir-opt -hivm-inject-sync -hivm-enable-multi-buffer -split-input-file %s | FileCheck %s
// clang-format off

// The unlocalised half of a deliberate trade:
// InjectSync picks no counter block, and EnableMultiBuffer opens none.
// => slot select and flag ids read the loop counter: consistent, not per block.
// Do not restore symmetry by letting EnableMultiBuffer create the counter here:
// the slot and the flags splitting apart is the failure this file pins, whichever
// side moves first.
// Red case: GraphSyncSolver moves the counter into the guard on this same
// input, so these checks fail against it.

// -----
module {
  // The guarded pair's handshake does not move the counter into the guard.
  // CHECK-LABEL: func.func @guard_keeps_loop_counter(
  func.func @guard_keeps_loop_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
    // CHECK: arith.remui %[[CTR]], %{{.*}} : i64
    // Slot select, from the same counter.
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
    // Flag id, from the same counter.
    // CHECK: arith.remui %[[CTR]], %{{.*}} : i64
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i64
    scf.for %iv = %c0 to %c8 step %c1 {
      %a = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: scf.if
      // CHECK-NOT: hivm.hir.multi_buffer_counter
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
      scf.if %cond {
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%a : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%a : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
    }
    return
  }
}

// -----
module {
  // producer unconditional, consumer inside an scf.if
  // => cannot rotate on the guard: both ends must read one value
  // CHECK-LABEL: func.func @outer_pair_keeps_loop_counter(
  func.func @outer_pair_keeps_loop_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
    scf.for %iv = %c0 to %c8 step %c1 {
      %buf = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %buf {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
      scf.if %cond {
        hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK: scf.if
      // CHECK-NOT: hivm.hir.multi_buffer_counter
      // CHECK: return
    }
    return
  }
}

// clang-format on
