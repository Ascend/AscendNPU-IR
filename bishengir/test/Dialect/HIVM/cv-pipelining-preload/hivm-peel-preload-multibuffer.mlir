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

// A nested for that yields its multi-buffer view must reuse the outer counter.
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: func.func @peel_nested_reuses_outer_counter(
  func.func @peel_nested_reuses_outer_counter(
      %src: memref<16xf16, #hivm.address_space<gm>>,
      %dst: memref<16xf16, #hivm.address_space<gm>>,
      %n: i32)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    scf.for %i = %c0 to %c8 step %c1 : i32 {
      %ctr = hivm.hir.multi_buffer_counter -> i64
      %buf = hivm.hir.pointer_cast(%c0_i64, %c128_i64)
          : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %buf {hivm.multi_buffer = 2 : i32}
          : memref<16xf16, #hivm.address_space<ub>>
      %cond = arith.cmpi sge, %i, %c0 : i32
      annotation.mark %cond {hivm.preload_num = 0 : i32} : i1
      scf.if %cond {
        hivm.hir.load ins(%src : memref<16xf16, #hivm.address_space<gm>>)
          outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
        %carried = scf.for %j = %c0 to %n step %c1
            iter_args(%acc = %buf) -> (memref<16xf16, #hivm.address_space<ub>>) : i32 {
          %inner = hivm.hir.pointer_cast(%c0_i64, %c128_i64)
              : memref<16xf16, #hivm.address_space<ub>>
          annotation.mark %inner {hivm.multi_buffer = 2 : i32}
              : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%src : memref<16xf16, #hivm.address_space<gm>>)
            outs(%inner : memref<16xf16, #hivm.address_space<ub>>)
          scf.yield %inner : memref<16xf16, #hivm.address_space<ub>>
        }
        hivm.hir.store ins(%carried : memref<16xf16, #hivm.address_space<ub>>)
          outs(%dst : memref<16xf16, #hivm.address_space<gm>>)
      }
    } {hivm.max_preload_num = 2 : i32}
    return
  }
}

// One counter. The inner for reads it and does not increment it.
// CHECK: memref.alloca() : memref<1xi64>
// CHECK-NOT: memref.alloca()
// CHECK: scf.for
// CHECK-NOT: memref.store
// CHECK: scf.yield
// CHECK: return
