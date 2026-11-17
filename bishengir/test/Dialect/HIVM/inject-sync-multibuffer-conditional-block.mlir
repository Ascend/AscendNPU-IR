// RUN: bishengir-opt -hivm-inject-sync -hivm-enable-multi-buffer -split-input-file %s | FileCheck %s
// clang-format off

// The unlocalised half of a deliberate trade: InjectSync does not pick a counter
// block, and EnableMultiBuffer no longer opens one of its own. Both the slot
// select and the flag ids therefore read the loop counter -- consistent, just not
// per block. Do not "restore symmetry" by letting EnableMultiBuffer create the
// counter here: the two cadences splitting apart is the 507015 class, whichever
// side moves first. The measured instance splits as slot on the loop counter,
// `FIX->M` id on the guard's own counter (`error_report/README.md` section 6).
// Red case: the GraphSyncSolver pipeline on this same input moves the counter
// into the guard, so these checks fail against it.

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
  // A pair whose producer stays unconditional cannot rotate on the guard: both
  // ends must read one value, and the consumer sits inside an `scf.if`.
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
