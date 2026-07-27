// RUN: bishengir-opt --convert-scf-to-cf --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Exercises the MembarAnalysis that runs inside ConvertTritonAscendGPUToLLVM.
//
// The kernel has two ttg.local_alloc's:
//   * %a_out lives OUTSIDE a loop, at shared-memory offset 0
//   * %a_in  lives INSIDE  the loop body, at shared-memory offset 8192
// Both buffers are read by a ttg.local_load in the loop body.
//
// Expected barrier placement (as ascend_dpx.sync_threads after LLVM lowering):
//   * %a_out (offset 0): NO dedicated barrier. Its write dominates every
//     loop-body read, and nothing else ever writes offset 0, so no RAW/WAR
//     hazard is unique to it. (The first loop-body barrier, inserted for
//     %a_in's cross-iteration WAR, also covers the %a_out RAW for free.)
//   * %a_in (offset 8192): TWO barriers inside the loop body.
//       1. At the top of the body: cross-iteration WAR — the previous
//          iteration's load at offset 8192 must finish before this
//          iteration's alloc overwrites it.
//       2. Between the %a_in stores and the %a_in load: intra-iteration RAW
//          on offset 8192.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared  = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem    = #ttg.shared_memory

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 16384 : i32,
    "ttg.enable-bishengir-simt-optimization" = 111 : i32
} {
  // CHECK-LABEL: llvm.func @membar_local_alloc_in_loop
  tt.func public @membar_local_alloc_in_loop(
      %data0 : tensor<32x32xf16, #blocked>,
      %data1 : tensor<32x32xf16, #blocked>,
      %lb    : i32,
      %ub    : i32,
      %step  : i32) {

    // Outer alloc at offset 0 — lives for the whole function.
    %a_out = ttg.local_alloc %data0 {allocation.offset = 0 : i32}
      : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<32x32xf16, #shared, #smem>

    scf.for %i = %lb to %ub step %step : i32 {
      // Inner alloc at offset 8192 — rewritten every iteration.
      %a_in = ttg.local_alloc %data1 {allocation.offset = 8192 : i32}
        : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<32x32xf16, #shared, #smem>

      %v_out = ttg.local_load %a_out
        : !ttg.memdesc<32x32xf16, #shared, #smem>
       -> tensor<32x32xf16, #blocked>
      %v_in  = ttg.local_load %a_in
        : !ttg.memdesc<32x32xf16, #shared, #smem>
       -> tensor<32x32xf16, #blocked>
      scf.yield
    }
    tt.return
  }
}

// Entry block: NO barrier before the branch into the loop header.
// (The outer %a_out is lowered into the entry block; nothing about its
// write alone requires a barrier.)
// CHECK-NOT:   ascend_dpx.sync_threads
// CHECK:       llvm.br ^bb{{[0-9]+}}

// Loop header block (takes an i32 induction var): condition check, no barrier.
// CHECK:       ^bb{{[0-9]+}}(%{{.*}}: i32):
// CHECK-NOT:   ascend_dpx.sync_threads
// CHECK:       llvm.cond_br

// Loop body: exactly TWO barriers, both attributable to the inner alloc.
// First barrier — before the inner alloc's stores (cross-iter WAR on 8192).
// CHECK:       ^bb{{[0-9]+}}:
// CHECK:       ascend_dpx.sync_threads
// CHECK-NOT:   ascend_dpx.sync_threads
// CHECK:       ascend_dpx.store
// Second barrier — between the inner alloc's stores and its load (RAW on 8192).
// CHECK:       ascend_dpx.sync_threads
// CHECK-NOT:   ascend_dpx.sync_threads
// CHECK:       ascend_dpx.load
// CHECK:       llvm.br

// Exit block: no further barriers.
// CHECK:       ^bb{{[0-9]+}}:
// CHECK-NOT:   ascend_dpx.sync_threads
// CHECK:       llvm.return
