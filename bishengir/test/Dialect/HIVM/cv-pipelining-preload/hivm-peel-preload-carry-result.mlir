// RUN: bishengir-opt %s -hivm-peel-preload | FileCheck %s

// Empty peel epilogue iterations must not replace a carried multi-buffer
// result with a fresh unused buffer view. The value stored after the peeled
// loop must share the address selected by the last producer preload_num=0
// iteration (EnableMultiBuffer's iter_arg copy is too late once the loop is
// peeled).
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>} {
  // CHECK-LABEL: func.func @peel_carry_multibuffer_result(
  func.func @peel_carry_multibuffer_result(
      %src: memref<16xf32, #hivm.address_space<gm>>,
      %dst: memref<16xf32, #hivm.address_space<gm>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %c0_index = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 1.250000e-01 : f32
    %init = hivm.hir.pointer_cast(%c0_i64, %c1024_i64)
        : memref<16xf32, #hivm.address_space<ub>>
    %result = scf.for %i = %c0 to %c8 step %c1
        iter_args(%arg = %init) -> (memref<16xf32, #hivm.address_space<ub>>) : i32 {
      %write = hivm.hir.pointer_cast(%c0_i64, %c1024_i64)
          : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %write {hivm.multi_buffer = 2 : i32}
          : memref<16xf32, #hivm.address_space<ub>>
      %yield_view = hivm.hir.pointer_cast(%c0_i64, %c1024_i64)
          : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %yield_view {hivm.multi_buffer = 2 : i32}
          : memref<16xf32, #hivm.address_space<ub>>
      %tmp = hivm.hir.pointer_cast(%c0_i64)
          : memref<16xf32, #hivm.address_space<ub>>
      %cond1 = arith.cmpi sge, %i, %c0 : i32
      annotation.mark %cond1 {hivm.preload_num = 1 : i32} : i1
      scf.if %cond1 {
        hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>)
          outs(%tmp : memref<16xf32, #hivm.address_space<ub>>)
      }
      %cond0 = arith.cmpi sge, %i, %c0 : i32
      annotation.mark %cond0 {hivm.preload_num = 0 : i32} : i1
      scf.if %cond0 {
        memref.store %cst, %write[%c0_index]
            : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.store ins(%write : memref<16xf32, #hivm.address_space<ub>>)
          outs(%src : memref<16xf32, #hivm.address_space<gm>>)
      }
      scf.yield %yield_view : memref<16xf32, #hivm.address_space<ub>>
    } {hivm.max_preload_num = 2 : i32}
    hivm.hir.store ins(%result : memref<16xf32, #hivm.address_space<ub>>)
      outs(%dst : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// After the middle loop, the last producer epilogue selects %[[ADDR]] and
// builds write/yield casts from it back to back. The post-loop store to %dst
// must use the carried yield cast; the trailing empty epilogue may bump the
// counter but must not insert another arith.select before that store.
// CHECK: scf.for
// CHECK: }
// CHECK: %[[ADDR:.*]] = arith.select
// CHECK-NEXT: %[[WRITE:.*]] = hivm.hir.pointer_cast(%[[ADDR]]
// CHECK-NEXT: %[[CARRIED:.*]] = hivm.hir.pointer_cast(%[[ADDR]]
// CHECK-NEXT: memref.store %{{.*}}, %[[WRITE]][
// CHECK: hivm.hir.store ins(%[[WRITE]]
// CHECK-NOT: arith.select
// CHECK: hivm.hir.store ins(%[[CARRIED]]
