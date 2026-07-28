// RUN: bishengir-opt %s                                                       \
// RUN:   -pass-pipeline="builtin.module(                                      \
// RUN:     hacc-append-device-spec{target=Ascend910B1},                       \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}),               \
// RUN:     hivm-plan-memory,                                                  \
// RUN:     func.func(hivm-graph-sync-solver,hivm-enable-multi-buffer,hivm-lower-multi-buffer-counter))" \
// RUN:   -split-input-file -verify-diagnostics                                \
// RUN:   | FileCheck %s --check-prefix=CHECK
// RUN: bishengir-opt %s                                                       \
// RUN:   -pass-pipeline="builtin.module(                                      \
// RUN:     hacc-append-device-spec{target=Ascend950PR_9589},                  \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true limit-mix-auto-multi-buffer-buffer=no-limit}), \
// RUN:     hivm-plan-memory,                                                  \
// RUN:     func.func(hivm-graph-sync-solver,hivm-enable-multi-buffer,hivm-lower-multi-buffer-counter))" \
// RUN:   -split-input-file -verify-diagnostics                                \
// RUN:   | FileCheck %s --check-prefix=A5

// -----
// 4-pass end-to-end pipeline on a vadd-style scf.while body.
//
// This locks in the contract that all four multi-buffer passes
// (mark, plan-memory, graph-sync-solver, enable-multi-buffer) compose
// transparently for scf.while using the alloca-based counter scheme.
//
// Inspection items (all CHECK below):
//   1. funcOp top: a memref<1xi64> counter alloca, with an initial store of 0.
//   2. The scf.while op result type list (i1) -> i1 stays unchanged.
//   3. Body head: memref.load + arith.remui + arith.select to pick the slot.
//   4. set_flag/wait_flag pairs use a dynamic event id (selected from the
//      counter) - this is what enables N-way buffer rotation.
//   5. Body tail: arith.addi + memref.store back to the alloca.

// CHECK-LABEL: func.func @while_pipeline_vadd(
func.func @while_pipeline_vadd(%arg0: memref<8xf32, #hivm.address_space<gm>>,
                               %arg1: memref<8xf32, #hivm.address_space<gm>>) {
  %true = arith.constant true
  // Counter alloca + init at funcOp top (item 1).
  // CHECK-DAG: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
  // CHECK-DAG: memref.store %{{.*}}, %[[CTR]]

  // Pre-loop set_flag pre-roll for double buffering (one per slot).
  // CHECK-DAG: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  // CHECK-DAG: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]

  // While signature unchanged (item 2).
  // CHECK: scf.while {{.*}} : (i1) -> i1
  %r = scf.while (%cond = %true) : (i1) -> i1 {
    scf.condition(%cond) %cond : i1
  } do {
  // CHECK: ^bb0
  ^bb0(%cin: i1):
    // Body head load + remui (item 3).
    // CHECK: %[[CUR:.*]] = memref.load %[[CTR]]
    // CHECK: arith.remui %[[CUR]], %{{.*}} : i64
    // CHECK: arith.select {{.*}} : i64
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // wait_flag uses dynamic event id (item 4).
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%tmp, %tmp : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%tmp : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    // Body tail increment + store-back (item 5).
    // CHECK: arith.addi %[[CUR]], %{{.*}} : i64
    // CHECK: memref.store %{{.*}}, %[[CTR]]
    // CHECK: scf.yield
    scf.yield %cin : i1
  }
  return
}

// -----
// Ascend950 MixCV vector-side multi-buffer + PlanMemory reuse.
//
// After SplitMixKernel the vector kernel is AIV + hivm.part_of_mix. The A5 RUN
// explicitly passes limit-mix-auto-multi-buffer-buffer=no-limit to mirror the
// bishengir-compile default from Options.td (LimitAutoMultiBufferBuffer).
// Two non-overlapping phases of equal size then share the same physical
// multi-buffer addresses from PlanMemory, while EnableMultiBuffer still emits
// slot selects on the shared counter.
// A5-LABEL: func.func @a5_mix_vector_multibuffer_reuse(
func.func @a5_mix_vector_multibuffer_reuse(
    %src1: memref<9728xf32, #hivm.address_space<gm>>,
    %dst1: memref<9728xf32, #hivm.address_space<gm>>,
    %src2: memref<9728xf32, #hivm.address_space<gm>>,
    %dst2: memref<9728xf32, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.part_of_mix} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // Shared counter alloca for slot rotation.
  // A5-DAG: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
  // A5-DAG: memref.store %{{.*}}, %[[CTR]]

  // Distinct multi-buffer physical addresses for the two slots of each phase.
  // Phase-1 and phase-2 reuse the same bases (each address appears twice).
  // A5-DAG: %[[C77824:.*]] = arith.constant 77824 : i64
  // A5-DAG: %[[C38912:.*]] = arith.constant 38912 : i64
  // A5-DAG: %[[C116736:.*]] = arith.constant 116736 : i64
  // A5: hivm.hir.pointer_cast(%[[C38912]]) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%[[C116736]]) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%[[C38912]]) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%[[C116736]]) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%{{.*}}) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%[[C77824]]) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%{{.*}}) : memref<9728xf32, #hivm.address_space<ub>>
  // A5: hivm.hir.pointer_cast(%[[C77824]]) : memref<9728xf32, #hivm.address_space<ub>>

  // A5-DAG: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  // A5-DAG: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]

  scf.for %i = %c0 to %c4 step %c1 {
    // A5: %[[CUR:.*]] = memref.load %[[CTR]]
    // A5: arith.remui %[[CUR]], %{{.*}} : i64
    // A5: arith.select {{.*}} : memref<9728xf32, #hivm.address_space<ub>>
    // A5: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]

    %src1_ub = memref.alloc() : memref<9728xf32, #hivm.address_space<ub>>
    %dst1_ub = memref.alloc() : memref<9728xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src1 : memref<9728xf32, #hivm.address_space<gm>>)
                  outs(%src1_ub : memref<9728xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%src1_ub, %src1_ub : memref<9728xf32, #hivm.address_space<ub>>,
                                           memref<9728xf32, #hivm.address_space<ub>>)
                  outs(%dst1_ub : memref<9728xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%dst1_ub : memref<9728xf32, #hivm.address_space<ub>>)
                   outs(%dst1 : memref<9728xf32, #hivm.address_space<gm>>)

    %src2_ub = memref.alloc() : memref<9728xf32, #hivm.address_space<ub>>
    %dst2_ub = memref.alloc() : memref<9728xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src2 : memref<9728xf32, #hivm.address_space<gm>>)
                  outs(%src2_ub : memref<9728xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%src2_ub, %src2_ub : memref<9728xf32, #hivm.address_space<ub>>,
                                           memref<9728xf32, #hivm.address_space<ub>>)
                  outs(%dst2_ub : memref<9728xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%dst2_ub : memref<9728xf32, #hivm.address_space<ub>>)
                   outs(%dst2 : memref<9728xf32, #hivm.address_space<gm>>)

    // A5: arith.addi %[[CUR]], %{{.*}} : i64
    // A5: memref.store %{{.*}}, %[[CTR]]
  }
  return
}
