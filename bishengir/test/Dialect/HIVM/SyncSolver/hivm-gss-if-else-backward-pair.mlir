// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" -split-input-file %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" -split-input-file %s | FileCheck %s

module {
  // CHECK: sync_solver_test_if_else_backward_pair
  func.func @sync_solver_test_if_else_backward_pair(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64

    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg1 = %c0 to %arg0 step %c1 {
       %9 = hivm.hir.pointer_cast(%c0_i64, %c1_i64) : memref<1xi1, #hivm.address_space<gm>>
      %10 = hivm.hir.pointer_cast(%c2_i64, %c3_i64) : memref<1xi1, #hivm.address_space<ub>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, %4]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %4]
      // CHECK: scf.if
      scf.if %false {
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%9 : memref<1xi1, #hivm.address_space<gm>>) outs(%10 : memref<1xi1, #hivm.address_space<ub>>)
      } else {
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%10 : memref<1xi1, #hivm.address_space<ub>>) outs(%9 : memref<1xi1, #hivm.address_space<gm>>)
      }
      // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %4]
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, %4]
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]

    scf.for %arg1 = %c0 to %arg0 step %c1 {
      %0 = hivm.hir.pointer_cast(%c4_i64) : memref<1xi1, #hivm.address_space<gm>>
      %1 = hivm.hir.pointer_cast(%c5_i64) : memref<1xi1, #hivm.address_space<ub>>
      scf.if %false {
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%0 : memref<1xi1, #hivm.address_space<gm>>) outs(%1 : memref<1xi1, #hivm.address_space<ub>>)
        // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      } else {
        // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store ins(%1 : memref<1xi1, #hivm.address_space<ub>>) outs(%0 : memref<1xi1, #hivm.address_space<gm>>)
        // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      }
    }
    return
  }
}

// -----

// Skipping an outer iteration's then branch preserves the backward event.
// CHECK-LABEL: func.func @merge_backward_pair_across_implicit_else(
func.func @merge_backward_pair_across_implicit_else(%n: index, %threshold: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %addr = arith.constant 0 : i64
  %gm = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<gm>>
  %ub = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<ub>>
  // CHECK-NOT: scf.for
  // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[EVENT:EVENT_ID[0-9]+]]>]
  // CHECK-NEXT: scf.for
  scf.for %i = %c0 to %n step %c1 {
    %active = arith.cmpi ult, %i, %threshold : index
    scf.if %active {
      scf.for %j = %c0 to %n step %c1 {
        hivm.hir.load ins(%gm : memref<1xi1, #hivm.address_space<gm>>) outs(%ub : memref<1xi1, #hivm.address_space<ub>>)
        hivm.hir.store ins(%ub : memref<1xi1, #hivm.address_space<ub>>) outs(%gm : memref<1xi1, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[EVENT]]>]
      }
    }
  }
  // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[EVENT]]>]
  // CHECK-NEXT: return
  return
}

// -----

// A standalone if keeps the backward event inside its then branch.
// CHECK-LABEL: func.func @keep_backward_pair_inside_standalone_if(
func.func @keep_backward_pair_inside_standalone_if(%n: index, %threshold: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %addr = arith.constant 0 : i64
  %gm = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<gm>>
  %ub = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<ub>>
  %active = arith.cmpi ult, %c0, %threshold : index
  // CHECK-NOT: hivm.hir.set_flag
  // CHECK: scf.if
  scf.if %active {
    // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[INNER_EVENT:EVENT_ID[0-9]+]]>]
    // CHECK-NEXT: scf.for
    scf.for %j = %c0 to %n step %c1 {
      hivm.hir.load ins(%gm : memref<1xi1, #hivm.address_space<gm>>) outs(%ub : memref<1xi1, #hivm.address_space<ub>>)
      hivm.hir.store ins(%ub : memref<1xi1, #hivm.address_space<ub>>) outs(%gm : memref<1xi1, #hivm.address_space<gm>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[INNER_EVENT]]>]
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[INNER_EVENT]]>]
  }
  // CHECK-NOT: hivm.hir.wait_flag
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: func.func @boundary_if_for_if(
func.func @boundary_if_for_if(%n: index, %threshold: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %addr = arith.constant 0 : i64
  %gm = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<gm>>
  %ub = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<ub>>
  %outer = arith.cmpi ult, %c0, %threshold : index
  // CHECK-NOT: hivm.hir.set_flag
  // CHECK: scf.if
  scf.if %outer {
    // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[NESTED_EVENT:EVENT_ID[0-9]+]]>]
    // CHECK-NEXT: scf.for
    scf.for %i = %c0 to %n step %c1 {
      %active = arith.cmpi ult, %i, %threshold : index
      scf.if %active {
        scf.for %j = %c0 to %n step %c1 {
          hivm.hir.load ins(%gm : memref<1xi1, #hivm.address_space<gm>>) outs(%ub : memref<1xi1, #hivm.address_space<ub>>)
          // CHECK: hivm.hir.store
          hivm.hir.store ins(%ub : memref<1xi1, #hivm.address_space<ub>>) outs(%gm : memref<1xi1, #hivm.address_space<gm>>)
          // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[NESTED_EVENT]]>]
        }
      }
    }
  }
  // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <[[NESTED_EVENT]]>]
  // CHECK-NOT: hivm.hir.wait_flag
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: func.func @boundary_for_if_if(
func.func @boundary_for_if_if(%n: index, %threshold: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %addr = arith.constant 0 : i64
  %gm = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<gm>>
  %ub = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<ub>>
  scf.for %i = %c0 to %n step %c1 {
    %active = arith.cmpi ult, %i, %threshold : index
    // CHECK: scf.if
    scf.if %active {
      // CHECK-NEXT: scf.if
      scf.if %active {
        // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: scf.for
        scf.for %j = %c0 to %n step %c1 {
          hivm.hir.load ins(%gm : memref<1xi1, #hivm.address_space<gm>>) outs(%ub : memref<1xi1, #hivm.address_space<ub>>)
          hivm.hir.store ins(%ub : memref<1xi1, #hivm.address_space<ub>>) outs(%gm : memref<1xi1, #hivm.address_space<gm>>)
        }
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @boundary_if_if_for(
func.func @boundary_if_if_for(%n: index, %threshold: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %addr = arith.constant 0 : i64
  %gm = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<gm>>
  %ub = hivm.hir.pointer_cast(%addr) : memref<1xi1, #hivm.address_space<ub>>
  %active = arith.cmpi ult, %c0, %threshold : index
  // CHECK: scf.if
  scf.if %active {
    // CHECK-NEXT: scf.if
    scf.if %active {
      // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      // CHECK-NEXT: scf.for
      scf.for %j = %c0 to %n step %c1 {
        hivm.hir.load ins(%gm : memref<1xi1, #hivm.address_space<gm>>) outs(%ub : memref<1xi1, #hivm.address_space<ub>>)
        hivm.hir.store ins(%ub : memref<1xi1, #hivm.address_space<ub>>) outs(%gm : memref<1xi1, #hivm.address_space<gm>>)
      }
    }
  }
  return
}
