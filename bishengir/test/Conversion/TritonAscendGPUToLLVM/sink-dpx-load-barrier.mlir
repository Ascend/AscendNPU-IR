// RUN: bishengir-opt --sink-dpx-load %s | FileCheck %s
//
// Regression test: the SinkDPXLoad scheduling phase must not reorder
// operations across ascend_dpx.sync_threads barriers.
//
// The transpose-via-shared-memory pattern is:
//   global load → shared mem store → sync_threads → shared mem load → global store
//
// Before the fix, the scheduler only rooted on ascend_dpx.store ops and walked
// their SSA dependency trees. Since sync_threads has no SSA results, it was
// invisible to the dependency walk and got moved to the end as a "remaining" op,
// causing shared memory loads to execute before the barrier — a race condition.

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @test_barrier_not_reordered
  llvm.func @test_barrier_not_reordered(%gm_in: !llvm.ptr<1>, %gm_out: !llvm.ptr<1>, %smem: !llvm.ptr<3>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c8 = llvm.mlir.constant(8 : i64) : i64

    // --- Segment 1: global load → shared mem store ---

    // Load two elements from global memory.
    %gm_addr = llvm.getelementptr %gm_in[%c4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %gm_vec = ascend_dpx.load %gm_addr : (!llvm.ptr<1>) -> vector<2xi32>
    %elem0 = llvm.extractelement %gm_vec[%c0 : i32] : vector<2xi32>
    %elem1 = llvm.extractelement %gm_vec[%c1 : i32] : vector<2xi32>

    // Store elements to shared memory (transposed offsets).
    %sm_addr0 = llvm.getelementptr %smem[%c4] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %sm_addr1 = llvm.getelementptr %smem[%c8] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %sv0 = llvm.mlir.undef : vector<1xi32>
    %sv0a = llvm.insertelement %elem0, %sv0[%c0 : i32] : vector<1xi32>
    ascend_dpx.store %sm_addr0, %sv0a : <3>, vector<1xi32>
    %sv1 = llvm.mlir.undef : vector<1xi32>
    %sv1a = llvm.insertelement %elem1, %sv1[%c0 : i32] : vector<1xi32>
    ascend_dpx.store %sm_addr1, %sv1a : <3>, vector<1xi32>

    // --- Barrier ---
    ascend_dpx.sync_threads

    // --- Segment 2: shared mem load → global store ---

    // Load transposed elements from shared memory.
    %sm_rd0 = llvm.getelementptr %smem[%c8] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %val0 = ascend_dpx.load %sm_rd0 : (!llvm.ptr<3>) -> i32
    %sm_rd1 = llvm.getelementptr %smem[%c4] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    %val1 = ascend_dpx.load %sm_rd1 : (!llvm.ptr<3>) -> i32

    // Pack and store to global memory.
    %out_addr = llvm.getelementptr %gm_out[%c4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %ov = llvm.mlir.undef : vector<2xi32>
    %ov0 = llvm.insertelement %val0, %ov[%c0 : i32] : vector<2xi32>
    %ov1 = llvm.insertelement %val1, %ov0[%c1 : i32] : vector<2xi32>
    ascend_dpx.store %out_addr, %ov1 : <1>, vector<2xi32>

    llvm.return
  }
}

// The critical invariant: sync_threads must remain between the shared memory
// stores and the shared memory loads. The scheduler must NOT move the barrier
// after the loads. We capture SSA names to verify the data-flow is preserved.

// --- Segment 1: global load → shared memory stores ---
// CHECK:       [[GV:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi32>
// CHECK:       [[E0:%[0-9]+]] = llvm.extractelement [[GV]][{{%[0-9]+}} : i32] : vector<2xi32>
// CHECK:       llvm.insertelement [[E0]], {{%[0-9]+}}[{{%[0-9]+}} : i32] : vector<1xi32>
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <3>, vector<1xi32>
// CHECK:       [[E1:%[0-9]+]] = llvm.extractelement [[GV]][{{%[0-9]+}} : i32] : vector<2xi32>
// CHECK:       llvm.insertelement [[E1]], {{%[0-9]+}}[{{%[0-9]+}} : i32] : vector<1xi32>
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <3>, vector<1xi32>

// --- Barrier: must appear between shared memory stores and loads ---
// CHECK:       ascend_dpx.sync_threads

// --- Segment 2: shared memory loads → global store ---
// CHECK:       [[SV0:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<3>) -> i32
// CHECK:       [[P0:%[0-9]+]] = llvm.insertelement [[SV0]], {{%[0-9]+}}[{{%[0-9]+}} : i32] : vector<2xi32>
// CHECK:       [[SV1:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<3>) -> i32
// CHECK:       [[P1:%[0-9]+]] = llvm.insertelement [[SV1]], [[P0]][{{%[0-9]+}} : i32] : vector<2xi32>
// CHECK:       ascend_dpx.store {{%[0-9]+}}, [[P1]] : <1>, vector<2xi32>
