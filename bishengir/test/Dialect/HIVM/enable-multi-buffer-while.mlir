// RUN: bishengir-opt %s -hivm-enable-multi-buffer -hivm-lower-multi-buffer-counter -split-input-file | FileCheck %s

// -----
// Verify the alloca-based counter scheme for scf.while.
//
// EnableMultiBuffer should:
//  1. Hoist single-address pointer_casts out of the loop (one per slot).
//  2. Materialize a memref<1xi64> counter alloca at the top of the func.
//  3. Initialize that alloca to 0 via memref.store before the loop.
//  4. Insert a memref.load + arith.remui at the head of the after region,
//     followed by the arith.select cascade picking among the slot ptrs.
//  5. Insert arith.addi + memref.store back to the alloca right before the
//     scf.yield terminator, all without changing the scf.while signature.
//
// Most importantly, the scf.while op's result type list (i1) -> i1 must
// stay unchanged: this is the key invariant of the alloca-based design.

module {
// CHECK-LABEL: func.func @while_enable_basic(
  func.func @while_enable_basic(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    // Hoisted single-addr casts (one per slot).
    // CHECK-DAG: hivm.hir.pointer_cast(%{{.*}}) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK-DAG: hivm.hir.pointer_cast(%{{.*}}) : memref<16xf16, #hivm.address_space<ub>>
    // Counter alloca at the top of the func body.
    // CHECK-DAG: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
    // CHECK-DAG: memref.store %{{.*}}, %[[CTR]]

    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %true = arith.constant true

    // While signature must remain (i1) -> i1.
    // CHECK: scf.while {{.*}} : (i1) -> i1
    %r = scf.while (%cond = %true) : (i1) -> i1 {
      scf.condition(%cond) %cond : i1
    } do {
    // CHECK: ^bb0
    ^bb0(%cin: i1):
      // CHECK: %[[CUR:.*]] = memref.load %{{.*}}
      // CHECK: %[[REM:.*]] = arith.remui %[[CUR]], %{{.*}} : i64
      // CHECK: arith.index_cast %[[REM]] : i64 to index
      // CHECK: arith.select {{.*}} : memref<16xf16, #hivm.address_space<ub>>
      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>,
                              memref<16xf16, #hivm.address_space<ub>>)
                    outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      // The +1 store-back is inserted right before the scf.yield terminator.
      // CHECK: arith.addi {{.*}}, %{{.*}} : i64
      // CHECK: memref.store {{.*}} : memref<1xi64>
      // CHECK: scf.yield
      scf.yield %cin : i1
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
