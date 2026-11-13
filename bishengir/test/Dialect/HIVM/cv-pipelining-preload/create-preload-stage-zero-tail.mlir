// RUN: bishengir-opt %s -allow-unregistered-dialect -create-preload="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s --check-prefixes=CHECK,ON --implicit-check-not=scope.scope
// RUN: sed 's/chunk_gated_delta_product_fwd_kernel_h_blockdim64/parallel_path_fwd_kernel/g' %s > %t
// RUN: bishengir-opt %t -allow-unregistered-dialect -create-preload="enable-preload=true set-workspace-multibuffer=2" | FileCheck %t --check-prefixes=CHECK,ON --implicit-check-not=scope.scope
// RUN: bishengir-opt %s -allow-unregistered-dialect -create-preload="enable-preload=false set-workspace-multibuffer=2" | FileCheck %s --check-prefix=OFF
// RUN: bishengir-opt %s -allow-unregistered-dialect -create-preload="enable-preload=true set-workspace-multibuffer=0" | FileCheck %s --check-prefix=OFF
// RUN: sed 's/chunk_gated_delta_product_fwd_kernel_h_blockdim64/off_registry/g' %s | bishengir-opt -allow-unregistered-dialect -create-preload="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s --check-prefix=OFF

// RUN: sed 's/hivm.hir.pipe_barrier\[<PIPE_ALL>\]/"test.outside"() : () -> ()/g' %s | bishengir-opt -allow-unregistered-dialect -create-preload="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s --check-prefix=UNSAFE

// Preserve arbitrary lower/upper bounds, positive step and loop-carried state.
// Only the actual program condition remains; no preload range guards or tail.
// ON-LABEL: func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_tail
// ON-SAME: %[[LB:[^:]+]]: i32, %[[UB:[^:]+]]: i32, %[[STEP:[^:]+]]: i32, %[[INIT:[^:]+]]: i32, %[[SAVE:[^:]+]]: i1
// ON: %[[R:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC:.*]] = %[[INIT]])
// ON-NOT: arith.cmpi
// ON: scf.if %[[SAVE]]
// ON: "test.history"(%[[ACC]])
// ON: %[[NEXT:.*]] = arith.addi %[[ACC]], %[[IV]]
// ON-NEXT: scf.yield %[[NEXT]]
// ON: return %[[R]]
// OFF-LABEL: func.func @{{.*}}_tail
// OFF: arith.addi
// OFF: arith.select
// OFF: scf.for
// OFF: arith.cmpi
// OFF: scf.if
func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_tail(%lb: i32, %ub: i32, %step: i32, %init: i32, %save: i1) -> i32 {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> i32 : i32 {
    %next = scope.scope : () -> i32 {
      scf.if %save {
        "test.history"(%acc) : (i32) -> ()
      }
      %sum = arith.addi %acc, %i : i32
      scope.return %sum : i32
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 1 : i32}
    scf.yield %next : i32
  }
  return %r : i32
}

// Empty loops must not execute effects, and must return their initial state.
// ON-LABEL: func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_empty
// ON-SAME: %[[INIT:[^:]+]]: i32
// ON-NOT: scf.for
// ON-NOT: "test.effect"
// ON: return %[[INIT]]
func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_empty(%init: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %r = scf.for %i = %c5 to %c3 step %c1 iter_args(%acc = %init) -> i32 : i32 {
    %next = scope.scope : () -> i32 {
      "test.effect"() : () -> ()
      %sum = arith.addi %acc, %i : i32
      scope.return %sum : i32
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 1 : i32}
    scf.yield %next : i32
  }
  return %r : i32
}

// Do not remove an extra tick containing synchronization outside stage scopes.
// UNSAFE-LABEL: func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_sync
// UNSAFE: arith.select
// UNSAFE: scf.for
// UNSAFE: "test.outside"
// UNSAFE: scf.if
// CHECK-LABEL: func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_sync
// CHECK: arith.addi
// CHECK: arith.select
// CHECK: scf.for
// CHECK: hivm.hir.pipe_barrier
// CHECK: scf.if
// CHECK: "test.effect"
func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_sync(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    scope.scope : () -> () {
      "test.effect"() : () -> ()
      scope.return
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 1 : i32}
  }
  return
}

// Pure operations outside scopes can still update observable loop state.
// CHECK-LABEL: func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_counter
// CHECK: arith.select
// CHECK: scf.for
// CHECK: arith.addi
// CHECK: scf.if
func.func @chunk_gated_delta_product_fwd_kernel_h_blockdim64_counter(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%count = %c0) -> i32 : i32 {
    %next = arith.addi %count, %c1 : i32
    scope.scope : () -> () {
      "test.effect"() : () -> ()
      scope.return
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 1 : i32}
    scf.yield %next : i32
  }
  return %r : i32
}
