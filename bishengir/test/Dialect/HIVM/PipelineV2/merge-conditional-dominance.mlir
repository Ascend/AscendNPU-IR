// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %s -o %t
// RUN: FileCheck %s < %t
// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %t -o %t.twice
// RUN: diff %t %t.twice

// Matching condition IDs do not imply that every prerequisite can move before
// the first fragment. Keep the two conditionals inside the merged scope when
// the second fragment needs a value computed after the first conditional.
// CHECK-LABEL: func.func @late_prerequisite
// CHECK: scope.scope
// CHECK-NEXT: scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[LATE:.*]] = arith.addf
// CHECK-NEXT: %[[PRE:.*]] = arith.mulf %[[LATE]],
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[PRE]],
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: scope.return
func.func @late_prerequisite(%cond: i1, %x: f32, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %first = scope.scope : () -> f32 {
      scf.if %cond {
        memref.store %x, %out[%c0] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      %late = arith.addf %x, %x : f32
      scope.return %late : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %prerequisite = arith.mulf %first, %x : f32
      scf.if %cond {
        memref.store %prerequisite, %out[%c1] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Preflight the whole prefix. Moving the second prerequisite while leaving the
// first behind would break dominance even before moving the conditional body.
// CHECK-LABEL: func.func @transitive_prerequisite
// CHECK: scope.scope
// CHECK-NEXT: scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[LATE:.*]] = arith.addf
// CHECK-NEXT: %[[A:.*]] = arith.mulf %[[LATE]],
// CHECK-NEXT: %[[B:.*]] = arith.addf %[[A]],
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[B]],
func.func @transitive_prerequisite(%cond: i1, %x: f32, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %first = scope.scope : () -> f32 {
      scf.if %cond {
        memref.store %x, %out[%c0] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      %late = arith.addf %x, %x : f32
      scope.return %late : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = arith.mulf %first, %x : f32
      %b = arith.addf %a, %x : f32
      scf.if %cond {
        memref.store %b, %out[%c1] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A region operation has no direct operands but can capture a late value.
// CHECK-LABEL: func.func @region_captured_prerequisite
// CHECK: scope.scope
// CHECK-NEXT: scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[LATE:.*]] = arith.addf
// CHECK-NEXT: %[[CAPTURE:.*]] = scf.execute_region
// CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[LATE]],
// CHECK-NEXT: scf.yield %[[MUL]]
// CHECK-NEXT: }
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[CAPTURE]],
func.func @region_captured_prerequisite(%cond: i1, %x: f32, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %first = scope.scope : () -> f32 {
      scf.if %cond {
        memref.store %x, %out[%c0] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      %late = arith.addf %x, %x : f32
      scope.return %late : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %captured = scf.execute_region -> f32 {
        %a = arith.mulf %first, %x : f32
        scf.yield %a : f32
      }
      scf.if %cond {
        memref.store %captured, %out[%c1] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Check operands captured directly by the second fragment as well as its prefix.
// CHECK-LABEL: func.func @branch_captured_late_value
// CHECK: scope.scope
// CHECK-NEXT: scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[LATE:.*]] = arith.addf
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[LATE]],
func.func @branch_captured_late_value(%cond: i1, %x: f32, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %first = scope.scope : () -> f32 {
      scf.if %cond {
        memref.store %x, %out[%c0] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      %late = arith.addf %x, %x : f32
      scope.return %late : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scf.if %cond {
        memref.store %first, %out[%c1] : memref<2xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A result used outside the branch cannot be replaced by a branch-local yield.
// CHECK-LABEL: func.func @prefix_uses_if_result
// CHECK: scope.scope
// CHECK-NEXT: %[[IF:.*]] = scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[PRE:.*]] = arith.addf %[[IF]],
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[PRE]],
func.func @prefix_uses_if_result(%cond: i1, %x: f32, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %first = scope.scope : () -> f32 {
      %r = scf.if %cond -> f32 {
        %a = arith.addf %x, %x : f32
        scf.yield %a : f32
      } else {
        scf.yield %x : f32
      } {pipeline.if.cond = 0 : i64}
      scope.return %r : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = arith.addf %first, %x : f32
      scf.if %cond {
        memref.store %a, %out[%c0] : memref<1xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Preserve memory ordering even if no SSA edge connects an intervening store
// to the second conditional's load.
// CHECK-LABEL: func.func @intervening_effect
// CHECK: scope.scope
// CHECK-NEXT: scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: memref.store
// CHECK-NEXT: scf.if
// CHECK-NEXT: %[[LOAD:.*]] = memref.load
// CHECK-NEXT: memref.store %[[LOAD]],
func.func @intervening_effect(%cond: i1, %x: f32, %y: f32, %buf: memref<1xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scf.if %cond {
        memref.store %x, %buf[%c0] : memref<1xf32>
      } {pipeline.if.cond = 0 : i64}
      memref.store %y, %buf[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scf.if %cond {
        %loaded = memref.load %buf[%c0] : memref<1xf32>
        memref.store %loaded, %out[%c0] : memref<1xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A load in the second prefix must not move before the first conditional's
// store, despite the matching condition IDs.
// CHECK-LABEL: func.func @effectful_prefix
// CHECK: scope.scope
// CHECK-NEXT: scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[LOAD:.*]] = memref.load
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[LOAD]],
func.func @effectful_prefix(%cond: i1, %x: f32, %buf: memref<1xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scf.if %cond {
        memref.store %x, %buf[%c0] : memref<1xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %loaded = memref.load %buf[%c0] : memref<1xf32>
      scf.if %cond {
        memref.store %loaded, %out[%c0] : memref<1xf32>
      } {pipeline.if.cond = 0 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}
