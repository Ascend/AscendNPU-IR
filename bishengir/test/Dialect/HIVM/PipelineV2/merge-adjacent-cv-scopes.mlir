// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %s -o %t
// RUN: FileCheck %s < %t
// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %t -o %t.twice
// RUN: diff %t %t.twice

// Independent producers form a maximal run and retain all escaping results.
// CHECK-LABEL: func.func @independent_producers
// CHECK: scf.for
// CHECK-NEXT: %[[RESULTS:.*]]:3 = scope.scope
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: %[[B:.*]] = arith.mulf
// CHECK-NEXT: %[[C:.*]] = arith.subf
// CHECK-NEXT: scope.return %[[A]], %[[B]], %[[C]] : f32, f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: memref.store %[[RESULTS]]#0
// CHECK-NEXT: memref.store %[[RESULTS]]#1
// CHECK-NEXT: memref.store %[[RESULTS]]#2
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
func.func @independent_producers(%x: f32, %y: f32, %out: memref<3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %a = scope.scope : () -> f32 {
      %v = arith.addf %x, %y : f32
      scope.return %v : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %b = scope.scope : () -> f32 {
      %v = arith.mulf %x, %y : f32
      scope.return %v : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %v = arith.subf %x, %y : f32
      scope.return %v : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      memref.store %a, %out[%c0] : memref<3xf32>
      memref.store %b, %out[%c1] : memref<3xf32>
      memref.store %c, %out[%c2] : memref<3xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Resultless stores to the same address must retain their order.
// CHECK-LABEL: func.func @ordered_stores(
// CHECK-SAME: %[[X:.*]]: f32, %[[Y:.*]]: f32,
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: memref.store %[[X]]
// CHECK-NEXT: memref.store %[[Y]]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @ordered_stores(%x: f32, %y: f32, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      memref.store %x, %out[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      memref.store %y, %out[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// CUBE scopes also merge; an internal-only producer result is removed.
// CHECK-LABEL: func.func @cube_dependency
// CHECK: scf.for
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: %[[B:.*]] = arith.mulf %[[A]], %[[A]]
// CHECK-NEXT: memref.store %[[B]]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @cube_dependency(%x: f32, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %a = scope.scope : () -> f32 {
      %v = arith.addf %x, %x : f32
      scope.return %v : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %v = arith.mulf %a, %a : f32
      memref.store %v, %out[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Separate original conditionals have distinct provenance IDs. Preserve their
// order around the load even when their condition SSA values are identical.
// CHECK-LABEL: func.func @preserve_conditional_order
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store
// CHECK-NEXT: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[LOADED:.*]] = memref.load
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[LOADED]]
// CHECK-NEXT: } {pipeline.if.cond = 1 : i64}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @preserve_conditional_order(%cond: i1, %x: f32, %buf: memref<1xf32>, %out: memref<1xf32>) {
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
      %v = memref.load %buf[%c0] : memref<1xf32>
      scf.if %cond {
        memref.store %v, %out[%c0] : memref<1xf32>
      } {pipeline.if.cond = 1 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Every boundary must prevent merging, including an operation between scopes,
// unknown/flexible core types, and a mixed wrapper with nested scopes.
// CHECK-LABEL: func.func @boundaries
// CHECK: scf.for
// CHECK: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: memref.store
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.return
// CHECK-NEXT: }
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_OR_VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @boundaries(%x: f32, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    memref.store %x, %out[%c0] : memref<1xf32>
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () { scope.return }
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE_OR_VECTOR>}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.return
    } {cv_merged_scope}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Matching fragments reconstruct one original conditional. The load preceding
// that original conditional is restored before the fused if without requiring
// it to be pure. Multiple matches in the second scope must use the newly merged
// if, not an erased op.
// All escaping results and branch-local producer/consumer uses are preserved.
// CHECK-LABEL: func.func @fuse_matching_ifs
// CHECK: scf.for
// CHECK-NEXT: %[[SCOPE:.*]]:3 = scope.scope
// CHECK-NEXT: %[[LOADED:.*]] = memref.load
// CHECK-NEXT: %[[BIAS:.*]] = arith.addf
// CHECK-NEXT: %[[IF:.*]]:3 = scf.if {{.*}} -> (f32, f32, f32) {
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: %[[B:.*]] = arith.addf %[[A]], %[[BIAS]]
// CHECK-NEXT: %[[C:.*]] = arith.mulf %[[B]],
// CHECK-NEXT: scf.yield %[[A]], %[[B]], %[[C]]
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[D:.*]] = arith.subf
// CHECK-NEXT: %[[E:.*]] = arith.subf %[[D]], %[[BIAS]]
// CHECK-NEXT: %[[F:.*]] = arith.mulf %[[E]],
// CHECK-NEXT: scf.yield %[[D]], %[[E]], %[[F]]
// CHECK-NEXT: } {pipeline.if.cond = 7 : i64}
// CHECK-NEXT: scope.return %[[IF]]#0, %[[IF]]#1, %[[IF]]#2
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: memref.store %[[SCOPE]]#0
// CHECK-NEXT: memref.store %[[SCOPE]]#1
// CHECK-NEXT: memref.store %[[SCOPE]]#2
func.func @fuse_matching_ifs(%cond: i1, %x: f32, %y: f32, %out: memref<3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %a = scope.scope : () -> f32 {
      %r = scf.if %cond -> (f32) {
        %v = arith.addf %x, %x : f32
        scf.yield %v : f32
      } else {
        %v = arith.subf %x, %x : f32
        scf.yield %v : f32
      } {pipeline.if.cond = 7 : i64}
      scope.return %r : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %b:2 = scope.scope : () -> (f32, f32) {
      %loaded = memref.load %out[%c0] : memref<3xf32>
      %bias = arith.addf %x, %loaded : f32
      %r = scf.if %cond -> (f32) {
        %v = arith.addf %a, %bias : f32
        scf.yield %v : f32
      } else {
        %v = arith.subf %a, %bias : f32
        scf.yield %v : f32
      } {pipeline.if.cond = 7 : i64}
      %s = scf.if %cond -> (f32) {
        %v = arith.mulf %r, %x : f32
        scf.yield %v : f32
      } else {
        %v = arith.mulf %r, %y : f32
        scf.yield %v : f32
      } {pipeline.if.cond = 7 : i64}
      scope.return %r, %s : f32, f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    memref.store %a, %out[%c0] : memref<3xf32>
    memref.store %b#0, %out[%c1] : memref<3xf32>
    memref.store %b#1, %out[%c2] : memref<3xf32>
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Distinct original conditionals remain separate around a trailing store
// in the first scope.
// CHECK-LABEL: func.func @preserve_first_scope_suffix
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store
// CHECK-NEXT: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: memref.store
// CHECK-NEXT: scf.if
// CHECK-NEXT: %[[V:.*]] = memref.load
// CHECK-NEXT: memref.store %[[V]]
// CHECK-NEXT: } {pipeline.if.cond = 1 : i64}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @preserve_first_scope_suffix(%cond: i1, %x: f32, %y: f32, %buf: memref<1xf32>, %out: memref<1xf32>) {
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
        %v = memref.load %buf[%c0] : memref<1xf32>
        memref.store %v, %out[%c0] : memref<1xf32>
      } {pipeline.if.cond = 1 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Distinct original conditionals remain separate when the second depends
// on a computation using the first conditional's result.
// CHECK-LABEL: func.func @preserve_if_result_dependency
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: %[[IF:.*]] = scf.if
// CHECK: } {pipeline.if.cond = 0 : i64}
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[IF]],
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store %[[SUM]]
// CHECK-NEXT: } {pipeline.if.cond = 1 : i64}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @preserve_if_result_dependency(%cond: i1, %x: f32, %y: f32, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %a = scope.scope : () -> f32 {
      %r = scf.if %cond -> (f32) {
        scf.yield %x : f32
      } else {
        scf.yield %y : f32
      } {pipeline.if.cond = 0 : i64}
      scope.return %r : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %v = arith.addf %a, %y : f32
      scf.if %cond {
        memref.store %v, %out[%c0] : memref<1xf32>
      } {pipeline.if.cond = 1 : i64}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}
