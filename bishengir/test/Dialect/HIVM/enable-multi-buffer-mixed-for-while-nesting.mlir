// RUN: bishengir-opt %s -hivm-enable-multi-buffer -hivm-lower-multi-buffer-counter -split-input-file | FileCheck %s

// Coverage for mixed scf.for / scf.while nesting under the unified
// alloca-based MultiBufferLoopAdapter counter. The four cases below pin
// down (a) the slot-select cascade ends up in the body block of the
// LoopLikeOp that owns the multi-buffer pointer_cast - regardless of
// whether the parent is the outer or inner loop, and (b) the
// counter alloca is hoisted to the funcOp entry exactly once per
// multi-buffer parent. As with the rest of the multi-buffer suite, every
// scf.while op's result-type list is left untouched (key invariant of
// the alloca-based design - no iter_args threading).

// -----
// Case A: outer scf.for + inner scf.while, multi-buffer alloc lives in
// inner-while's after-region. The inner scf.while is the multi-buffer
// parent; outer scf.for is just a passive enclosing loop.
//
// CHECK-LABEL: func.func @case_A_for_outer_while_inner_alloc_in_while(
// CHECK-DAG:    %[[CTR_A:.*]] = memref.alloca() : memref<1xi64>
// CHECK-DAG:    memref.store %{{.*}}, %[[CTR_A]]
// CHECK:        scf.for
// CHECK-NEXT:     scf.while {{.*}} : (i1) -> i1
// While signature stays (i1) -> i1; cascade lives in the after-region.
// CHECK:          ^bb0
// CHECK:            memref.load %[[CTR_A]][%{{.*}}] : memref<1xi64>
// CHECK:            arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:            arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:            arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:            arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:            arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:            arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:            arith.addi %{{.*}}, %{{.*}} : i64
// CHECK:            memref.store %{{.*}}, %[[CTR_A]]
// CHECK:            scf.yield
module {
  func.func @case_A_for_outer_while_inner_alloc_in_while(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true

    scf.for %i = %c0 to %c4 step %c1 {
      %r = scf.while (%cond = %true) : (i1) -> i1 {
        scf.condition(%cond) %cond : i1
      } do {
      ^bb0(%cin: i1):
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
        scf.yield %cin : i1
      }
    }
    return
  }
}

// -----
// Case B: outer scf.while + inner scf.for, multi-buffer alloc lives in
// inner-for body. The inner scf.for is the multi-buffer parent; outer
// scf.while is a passive enclosing loop and its signature must stay
// (i1) -> i1 unchanged.
//
// CHECK-LABEL: func.func @case_B_while_outer_for_inner_alloc_in_for(
// CHECK-DAG:    %[[CTR_B:.*]] = memref.alloca() : memref<1xi64>
// CHECK-DAG:    memref.store %{{.*}}, %[[CTR_B]]
// CHECK:        scf.while {{.*}} : (i1) -> i1
// CHECK:          ^bb0
// CHECK:            scf.for
// CHECK:              memref.load %[[CTR_B]][%{{.*}}] : memref<1xi64>
// CHECK:              arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:              arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:              arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:              arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:              arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:              arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:              arith.addi %{{.*}}, %{{.*}} : i64
// CHECK:              memref.store %{{.*}}, %[[CTR_B]]
// CHECK:            }
// CHECK:            scf.yield
module {
  func.func @case_B_while_outer_for_inner_alloc_in_for(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true

    %r = scf.while (%cond = %true) : (i1) -> i1 {
      scf.condition(%cond) %cond : i1
    } do {
    ^bb0(%cin: i1):
      scf.for %i = %c0 to %c4 step %c1 {
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
      }
      scf.yield %cin : i1
    }
    return
  }
}

// -----
// Case C: outer scf.for + inner scf.while, but the multi-buffer alloc
// is materialized in the *outer*-for body (before the inner while). The
// outer scf.for is the multi-buffer parent; inner scf.while just
// consumes the already-rotated buffer. Cascade must land in outer-for
// body; inner scf.while signature stays (i1) -> i1.
//
// CHECK-LABEL: func.func @case_C_for_outer_while_inner_alloc_in_for(
// CHECK-DAG:    %[[CTR_C:.*]] = memref.alloca() : memref<1xi64>
// CHECK-DAG:    memref.store %{{.*}}, %[[CTR_C]]
// CHECK:        scf.for
// Cascade is in outer-for body, BEFORE the inner scf.while.
// CHECK:          memref.load %[[CTR_C]][%{{.*}}] : memref<1xi64>
// CHECK:          arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:          arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:          arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:          arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:          arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:          arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:          scf.while {{.*}} : (i1) -> i1
// CHECK:          arith.addi %{{.*}}, %{{.*}} : i64
// CHECK:          memref.store %{{.*}}, %[[CTR_C]]
// CHECK:        }
module {
  func.func @case_C_for_outer_while_inner_alloc_in_for(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true

    scf.for %i = %c0 to %c4 step %c1 {
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
      %r = scf.while (%cond = %true) : (i1) -> i1 {
        scf.condition(%cond) %cond : i1
      } do {
      ^bb0(%cin: i1):
        hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        scf.yield %cin : i1
      }
    }
    return
  }
}

// -----
// Case D: outer scf.while + inner scf.for, but the multi-buffer alloc
// is materialized in the *outer*-while's after-region (before the inner
// for). The outer scf.while is the multi-buffer parent; inner scf.for
// just consumes the rotated buffer. Cascade lives in the while
// after-region; outer scf.while signature stays (i1) -> i1.
//
// CHECK-LABEL: func.func @case_D_while_outer_for_inner_alloc_in_while(
// CHECK-DAG:    %[[CTR_D:.*]] = memref.alloca() : memref<1xi64>
// CHECK-DAG:    memref.store %{{.*}}, %[[CTR_D]]
// CHECK:        scf.while {{.*}} : (i1) -> i1
// CHECK:          ^bb0
// CHECK:            memref.load %[[CTR_D]][%{{.*}}] : memref<1xi64>
// CHECK:            arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:            arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:            arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:            arith.remui %{{.*}}, %{{.*}} : i64
// CHECK:            arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK:            arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
// CHECK:            scf.for
// CHECK:            arith.addi %{{.*}}, %{{.*}} : i64
// CHECK:            memref.store %{{.*}}, %[[CTR_D]]
// CHECK:            scf.yield
module {
  func.func @case_D_while_outer_for_inner_alloc_in_while(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true

    %r = scf.while (%cond = %true) : (i1) -> i1 {
      scf.condition(%cond) %cond : i1
    } do {
    ^bb0(%cin: i1):
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
      scf.for %i = %c0 to %c4 step %c1 {
        hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      scf.yield %cin : i1
    }
    return
  }
}
