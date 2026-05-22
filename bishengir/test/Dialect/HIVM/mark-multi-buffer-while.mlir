// RUN: bishengir-opt -allow-unregistered-dialect %s             \
// RUN:   -pass-pipeline="builtin.module(                        \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}),cse)" \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s

// -----
// Verify that an alloc inside scf.while gets marked for multi-buffer.
// Before this change MarkMultiBuffer hardcoded scf.for ancestors only and
// would silently skip while loops; now scf.for and scf.while are both
// accepted via the relaxed parent-chain check (see MarkMultiBuffer.cpp
// "Allow scf::ForOp and scf::WhileOp ancestors").

// CHECK-LABEL: func.func @while_mark_basic(
func.func @while_mark_basic(%arg0: memref<8xf32, #hivm.address_space<gm>>,
                            %arg1: memref<8xf32, #hivm.address_space<gm>>) {
  %true = arith.constant true
  // CHECK: scf.while
  %r = scf.while (%cond = %true) : (i1) -> i1 {
    scf.condition(%cond) %cond : i1
  } do {
  ^bb0(%cin: i1):
    %tmp = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[ALLOCA]] {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%tmp : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    scf.yield %cin : i1
  }
  return
}

// -----
// Negative: an alloc whose parent is scf.parallel must still bail out (only
// scf.for and scf.while are accepted). This regression-checks that we did
// not over-relax the type filter.

// CHECK-LABEL: func.func @parallel_no_mark(
func.func @parallel_no_mark(%arg0: memref<8xf32, #hivm.address_space<gm>>,
                            %arg1: memref<8xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %tmp = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%tmp : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    scf.reduce
  }
  return
}
