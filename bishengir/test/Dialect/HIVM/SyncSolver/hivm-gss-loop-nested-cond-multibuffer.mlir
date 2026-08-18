// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" -split-input-file %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" -split-input-file %s | FileCheck %s

// Regression for getMultiBufferEventIdInfo: a pair with eventIdNum > 2 must
// not get a multi-buffer event-id set when an scf.if sits on the path to the
// parent backward (multi-buffer) loop.
//
// The first load uses a 3-slot pointer_cast ([0, 3, 1]), so a candidate
// backward pair can report eventIdNum > 2. Those ops live under `scf.if`
// inside the loop, so a Condition is a proper ancestor of both RW ops but
// not of the multi-buffer loop scope. The solver then returns no
// EventIdInfo for that pair (SyncSolverBase.cpp: if eventIdNum > 2, walk
// parent Conditions of rwOp1 and bail out when the Condition also
// ancestors rwOp2 but does not ancestor the multi-buffer scope).
//
// Depth-2 rotating waits around the loop are still allowed; the 3-slot
// pair under the if is not.

module {
  // CHECK: test_sync_solver_loop_nested_cond_multibuffer
  func.func @test_sync_solver_loop_nested_cond_multibuffer(%n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64

    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    scf.for %arg0 = %c0 to %n step %c1 {
      scf.if %true {
        // RWOperation [PIPE_MTE2] read:[0, 3, 1] write:[1, 0]
        %rd0 = hivm.hir.pointer_cast(%c0_i64, %c3_i64, %c1_i64) : memref<1xi1, #hivm.address_space<gm>>
        %wr0 = hivm.hir.pointer_cast(%c1_i64, %c0_i64) : memref<1xi1, #hivm.address_space<ub>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
        // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%rd0 : memref<1xi1, #hivm.address_space<gm>>) outs(%wr0 : memref<1xi1, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE3] read:[2, 1] write:[1]
        %rd1 = hivm.hir.pointer_cast(%c2_i64, %c1_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr1 = hivm.hir.pointer_cast(%c1_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd1 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr1 : memref<1xi1, #hivm.address_space<gm>>)

        // RWOperation [PIPE_MTE3] read:[1, 4] write:[1, 3, 0]
        %rd2 = hivm.hir.pointer_cast(%c1_i64, %c4_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr2 = hivm.hir.pointer_cast(%c1_i64, %c3_i64, %c0_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd2 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr2 : memref<1xi1, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE2] read:[1, 3, 2] write:[0, 1]
        %rd3 = hivm.hir.pointer_cast(%c1_i64, %c3_i64, %c2_i64) : memref<1xi1, #hivm.address_space<gm>>
        %wr3 = hivm.hir.pointer_cast(%c0_i64, %c1_i64) : memref<1xi1, #hivm.address_space<ub>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%rd3 : memref<1xi1, #hivm.address_space<gm>>) outs(%wr3 : memref<1xi1, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE3] read:[1] write:[0, 4]
        %rd4 = hivm.hir.pointer_cast(%c1_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr4 = hivm.hir.pointer_cast(%c0_i64, %c4_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd4 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr4 : memref<1xi1, #hivm.address_space<gm>>)

        // RWOperation [PIPE_MTE3] read:[2, 4] write:[2, 3, 0]
        %rd5 = hivm.hir.pointer_cast(%c2_i64, %c4_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr5 = hivm.hir.pointer_cast(%c2_i64, %c3_i64, %c0_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd5 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr5 : memref<1xi1, #hivm.address_space<gm>>)

        // RWOperation [PIPE_MTE3] read:[2, 0, 4] write:[1, 3, 0]
        %rd6 = hivm.hir.pointer_cast(%c2_i64, %c0_i64, %c4_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr6 = hivm.hir.pointer_cast(%c1_i64, %c3_i64, %c0_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd6 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr6 : memref<1xi1, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE2] read:[1, 0, 4] write:[0, 3]
        %rd7 = hivm.hir.pointer_cast(%c1_i64, %c0_i64, %c4_i64) : memref<1xi1, #hivm.address_space<gm>>
        %wr7 = hivm.hir.pointer_cast(%c0_i64, %c3_i64) : memref<1xi1, #hivm.address_space<ub>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%rd7 : memref<1xi1, #hivm.address_space<gm>>) outs(%wr7 : memref<1xi1, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE3] read:[2, 3, 4] write:[3, 4]
        %rd8 = hivm.hir.pointer_cast(%c2_i64, %c3_i64, %c4_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr8 = hivm.hir.pointer_cast(%c3_i64, %c4_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd8 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr8 : memref<1xi1, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE2] read:[3] write:[2, 1, 4]
        %rd9 = hivm.hir.pointer_cast(%c3_i64) : memref<1xi1, #hivm.address_space<gm>>
        %wr9 = hivm.hir.pointer_cast(%c2_i64, %c1_i64, %c4_i64) : memref<1xi1, #hivm.address_space<ub>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%rd9 : memref<1xi1, #hivm.address_space<gm>>) outs(%wr9 : memref<1xi1, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]

        // RWOperation [PIPE_MTE3] read:[2, 3, 4] write:[2, 3, 4]
        %rd10 = hivm.hir.pointer_cast(%c2_i64, %c3_i64, %c4_i64) : memref<1xi1, #hivm.address_space<ub>>
        %wr10 = hivm.hir.pointer_cast(%c2_i64, %c3_i64, %c4_i64) : memref<1xi1, #hivm.address_space<gm>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%rd10 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr10 : memref<1xi1, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]

        scf.if %true {
          // RWOperation [PIPE_MTE3] read:[1, 2] write:[2]
          %rd11 = hivm.hir.pointer_cast(%c1_i64, %c2_i64) : memref<1xi1, #hivm.address_space<ub>>
          %wr11 = hivm.hir.pointer_cast(%c2_i64) : memref<1xi1, #hivm.address_space<gm>>
          // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
          // CHECK-NEXT: hivm.hir.store
          hivm.hir.store ins(%rd11 : memref<1xi1, #hivm.address_space<ub>>) outs(%wr11 : memref<1xi1, #hivm.address_space<gm>>)
        }
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
      }
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    // CHECK-NEXT: return
    return
  }
}

