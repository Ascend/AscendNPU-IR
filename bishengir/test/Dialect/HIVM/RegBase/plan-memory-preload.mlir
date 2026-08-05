// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase -split-input-file -verify-diagnostics | FileCheck %s

// RegBase variant of the existing PlanMemory regression introduced by
// 5a67b81d9 in ../plan-memory.mlir. The second loop uses depth 4 instead of 2.
module {
  // This test guards preload local buffer liveness extension.
  //
  // The two preload local buffers are allocated inside two sibling scf.for loops.
  // Each alloc is inside a scope.scope.
  //
  // The important property is that each preload buffer must be tracked by its
  // own enclosing loop:
  //
  //   loop0 -> buf0
  //   loop1 -> buf1
  //
  // Instead of being globally extended to every preload loop:
  //
  //   loop0 -> buf0, buf1
  //   loop1 -> buf0, buf1
  //
  // The buffer is intentionally large. If the sibling-loop preload buffers are
  // incorrectly treated as live in the same loop, memory planning may require
  // both multi-buffer carriers at the same time and fail.

  // CHECK-LABEL: func.func @test_preload_local_buffer_lifetime_is_per_enclosing_loop
  func.func @test_preload_local_buffer_lifetime_is_per_enclosing_loop(
      %src0: memref<81920xi8, #hivm.address_space<gm>>,
      %dst0: memref<81920xi8, #hivm.address_space<gm>>,
      %src1: memref<40960xi8, #hivm.address_space<gm>>,
      %dst1: memref<40960xi8, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // First preload loop.
    //
    // CHECK: scf.for
    // CHECK: %[[P0:.*]] = hivm.hir.pointer_cast({{.*}}) : memref<81920xi8, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[P0]] {{.*}}hivm.multi_buffer = 2 : i32{{.*}}hivm.preload_local_buffer = 1 : i32{{.*}}
    // CHECK: scope.scope
    // CHECK: hivm.hir.load ins(%arg0 : memref<81920xi8, #hivm.address_space<gm>>) outs(%[[P0]] : memref<81920xi8, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.store ins(%[[P0]] : memref<81920xi8, #hivm.address_space<ub>>) outs(%arg1 : memref<81920xi8, #hivm.address_space<gm>>)
    // CHECK: scope.return
    scf.for %i = %c0 to %c4 step %c1 {
      scope.scope : () -> () {
        %buf0 = memref.alloc() : memref<81920xi8, #hivm.address_space<ub>>
        annotation.mark %buf0 {
          hivm.multi_buffer = 2 : i32,
          hivm.preload_local_buffer = 1 : i32
        } : memref<81920xi8, #hivm.address_space<ub>>

        hivm.hir.load ins(%src0 : memref<81920xi8, #hivm.address_space<gm>>)
                      outs(%buf0 : memref<81920xi8, #hivm.address_space<ub>>)
        hivm.hir.store ins(%buf0 : memref<81920xi8, #hivm.address_space<ub>>)
                       outs(%dst0 : memref<81920xi8, #hivm.address_space<gm>>)

        scope.return
      }
    }

    // Second sibling preload loop.
    //
    // This alloc is in a different enclosing loop. The liveness extension should
    // start and end at this second loop, not at the first loop.
    //
    // CHECK: scf.for
    // CHECK: %[[P1:.*]] = hivm.hir.pointer_cast({{.*}}) : memref<40960xi8, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[P1]] {{.*}}hivm.multi_buffer = 4 : i32{{.*}}hivm.preload_local_buffer = 1 : i32{{.*}}
    // CHECK: scope.scope
    // CHECK: hivm.hir.load ins(%arg2 : memref<40960xi8, #hivm.address_space<gm>>) outs(%[[P1]] : memref<40960xi8, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.store ins(%[[P1]] : memref<40960xi8, #hivm.address_space<ub>>) outs(%arg3 : memref<40960xi8, #hivm.address_space<gm>>)
    // CHECK: scope.return
    scf.for %j = %c0 to %c4 step %c1 {
      scope.scope : () -> () {
        %buf1 = memref.alloc() : memref<40960xi8, #hivm.address_space<ub>>
        annotation.mark %buf1 {
          hivm.multi_buffer = 4 : i32,
          hivm.preload_local_buffer = 1 : i32
        } : memref<40960xi8, #hivm.address_space<ub>>

        hivm.hir.load ins(%src1 : memref<40960xi8, #hivm.address_space<gm>>)
                      outs(%buf1 : memref<40960xi8, #hivm.address_space<ub>>)
        hivm.hir.store ins(%buf1 : memref<40960xi8, #hivm.address_space<ub>>)
                       outs(%dst1 : memref<40960xi8, #hivm.address_space<gm>>)

        scope.return
      }
    }

    // CHECK: return
    return
  }
}
