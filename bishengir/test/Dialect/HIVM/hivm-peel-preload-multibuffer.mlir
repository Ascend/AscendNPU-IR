// RUN: bishengir-opt %s -hivm-peel-preload | FileCheck %s

// Peeled prologue/epilogue must not collapse multi_buffer_counter onto one
// load for the whole then-path. Each unrolled iteration loads and increments
// so GraphSync event-ID rotation stays aligned with the structured loop.
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>} {
  // CHECK-LABEL: func.func @peel_multibuffer_counter(
  func.func @peel_multibuffer_counter(
      %src: memref<16xf16, #hivm.address_space<gm>>,
      %dst: memref<16xf16, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i64
    // CHECK: memref.alloca
    // CHECK: memref.store
    scf.for %i = %c0 to %c8 step %c1 : i32 {
      %ctr = hivm.hir.multi_buffer_counter -> i64
      %slot = arith.remui %ctr, %c2 : i64
      %cond1 = arith.cmpi sge, %i, %c0 : i32
      annotation.mark %cond1 {hivm.preload_num = 1 : i32} : i1
      scf.if %cond1 {
        hivm.hir.load ins(%src : memref<16xf16, #hivm.address_space<gm>>)
          outs(%dst : memref<16xf16, #hivm.address_space<ub>>)
      }
      %c0_2 = arith.constant 0 : i32
      %cond0 = arith.cmpi sge, %i, %c0_2 : i32
      annotation.mark %cond0 {hivm.preload_num = 0 : i32} : i1
      scf.if %cond0 {
        hivm.hir.store ins(%dst : memref<16xf16, #hivm.address_space<ub>>)
          outs(%src : memref<16xf16, #hivm.address_space<gm>>)
      }
    } {hivm.max_preload_num = 2 : i32}
    return
  }
}

// Shared alloca on the peel-then path; prologue slots and the middle body each
// load then store ctr+1 (not a single load spanning the whole then region).
// Constant trip count folds the peel gate; still expect per-iteration updates.
// CHECK: memref.load
// CHECK: memref.store
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
