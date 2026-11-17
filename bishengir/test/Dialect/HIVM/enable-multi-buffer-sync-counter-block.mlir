// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" -hivm-enable-multi-buffer -split-input-file %s | FileCheck %s
// clang-format off

// One counter per handshake, chosen by one rule: the innermost block enclosing
// every non-mark user of the protected buffers.
// The sync pass places that counter and EnableMultiBuffer adopts it, so the slot
// select and the flags cannot rotate on different cadences.
// GraphSyncSolver only: the InjectSync path keeps the loop counter and is pinned
// by inject-sync-multibuffer-conditional-block.mlir.

// -----
module {
  // The only handshake of this loop lands inside the conditional block, so the
  // sync pass put the counter there and the slot select reuses it.
  // CHECK-LABEL: func.func @guarded_pair_owns_block_counter(
  func.func @guarded_pair_owns_block_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      %a = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: scf.if
      scf.if %cond {
        // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
        // CHECK: %[[MOD:.*]] = arith.remui %[[CTR]], %{{.*}} : i64
        // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
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
  // The guarded pair's handshake is static, so no counter was placed in its
  // guard. Adoption only: the slot select stays at the definition and reads the
  // loop counter, which is also what the body handshake reads.
  // CHECK-LABEL: func.func @foreign_counter_keeps_loop_counter(
  func.func @foreign_counter_keeps_loop_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>,
      %arg2: memref<16xf16, #hivm.address_space<gm>>,
      %arg3: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
      %a = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %b = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %b {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%b : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%b : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg3 : memref<16xf16, #hivm.address_space<gm>>)
      // CHECK: scf.if
      scf.if %cond {
        // CHECK-NOT: hivm.hir.multi_buffer_counter
        // CHECK-NOT: arith.select
        // CHECK: hivm.hir.load
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%a : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%a : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK: return
    }
    return
  }
}

// -----
module {
  // Users spread over the guard and a guard-nested `scf.if` still share one
  // cadence: every entry runs both. Requiring the users to sit in the literal
  // same block would leave the select on the loop counter while the ids rotate
  // per entry -- the desynchronization this file guards against.
  // CHECK-LABEL: func.func @nested_if_users_share_guard_counter(
  func.func @nested_if_users_share_guard_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>,
      %cond: i1, %inner: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK-NOT: hivm.hir.multi_buffer_counter
      %a = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: scf.if
      scf.if %cond {
        // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
        // CHECK: %[[MOD:.*]] = arith.remui %[[CTR]], %{{.*}} : i64
        // CHECK: %[[SLOT:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%a : memref<16xf16, #hivm.address_space<ub>>)
        // The nested reader reuses the slot, so both blocks read one counter.
        // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i64
        // CHECK: scf.if
        scf.if %inner {
          hivm.hir.store ins(%a : memref<16xf16, #hivm.address_space<ub>>)
                         outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        }
      }
      // CHECK: }
    }
    return
  }
}

// -----
module {
  // Two sibling guards each touch the same pair, so no single block encloses
  // both entries: the rotation stays on the loop counter, which both read.
  // CHECK-LABEL: func.func @pair_in_two_sibling_guards(
  func.func @pair_in_two_sibling_guards(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>,
      %p: i1, %q: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      %a = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
      scf.if %p {
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%a : memref<16xf16, #hivm.address_space<ub>>)
      }
      scf.if %q {
        hivm.hir.store ins(%a : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK: }
    }
    return
  }
}

// -----
module {
  // A reader inside a guard-nested loop runs on that loop's trip count, not on
  // the guard's entries, so no block encloses both cadences.
  // CHECK-LABEL: func.func @user_in_nested_loop_keeps_loop_counter(
  func.func @user_in_nested_loop_keeps_loop_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
      %a = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: scf.if
      scf.if %cond {
        // CHECK-NOT: hivm.hir.multi_buffer_counter
        // CHECK: hivm.hir.load
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%a : memref<16xf16, #hivm.address_space<ub>>)
        // CHECK: scf.for
        scf.for %jv = %c0 to %c8 step %c1 {
          hivm.hir.store ins(%a : memref<16xf16, #hivm.address_space<ub>>)
                         outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        }
      }
      // CHECK: }
    }
    return
  }
}
