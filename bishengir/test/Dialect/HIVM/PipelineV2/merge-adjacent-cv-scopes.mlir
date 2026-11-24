// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %s -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=COUNT < %t
// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %t -o %t.twice
// RUN: diff %t %t.twice

// Same-core predecessors of a common CUBE consumer merge and retain all escaping results.
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

// Disconnected scopes have no graph merge candidate. Unknown/flexible core
// scopes and the mixed wrapper also remain intact.
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
// it to be pure. Each scope contains one fragment of the original conditional,
// and successive adjacent merges reconstruct all three fragments.
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
    %b = scope.scope : () -> f32 {
      %loaded = memref.load %out[%c0] : memref<3xf32>
      %bias = arith.addf %x, %loaded : f32
      %r = scf.if %cond -> (f32) {
        %v = arith.addf %a, %bias : f32
        scf.yield %v : f32
      } else {
        %v = arith.subf %a, %bias : f32
        scf.yield %v : f32
      } {pipeline.if.cond = 7 : i64}
      scope.return %r : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %s = scf.if %cond -> (f32) {
        %v = arith.mulf %b, %x : f32
        scf.yield %v : f32
      } else {
        %v = arith.mulf %b, %y : f32
        scf.yield %v : f32
      } {pipeline.if.cond = 7 : i64}
      scope.return %s : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    memref.store %a, %out[%c0] : memref<3xf32>
    memref.store %b, %out[%c1] : memref<3xf32>
    memref.store %c, %out[%c2] : memref<3xf32>
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

// -----

// V1 -> C -> V3 and V2 -> V3: merge V2/V3 while preserving V1 -> C -> V23.
// CHECK-LABEL: func.func @graph_merges_consumer_with_cross_core_input(
// CHECK: scf.for
// CHECK-COUNT-3: scope.scope
// CHECK-NOT: scope.scope
// CHECK: } {hivm.cv_pipelined_loop}
func.func @graph_merges_consumer_with_cross_core_input(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %value = arith.negf %v1 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v3 = scope.scope : () -> f32 {
      %value = arith.addf %c, %v2 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// C stays outside. The two following VECTOR scopes are fused into one.
// CHECK-LABEL: func.func @adjacent_preserves_incoming_overlap(
// COUNT-LABEL: func.func @adjacent_preserves_incoming_overlap(
// COUNT-COUNT-2: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_incoming_overlap(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %c = scope.scope : () -> f32 {
      %value = arith.negf %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %c, %c : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// V1 -> bridge -> C -> V3 prevents V1/V3 merging, but V2/V3 can merge.
// CHECK-LABEL: func.func @graph_merges_consumer_after_transitive_boundary(
// CHECK: scf.for
// CHECK-COUNT-3: scope.scope
// CHECK-NOT: scope.scope
// CHECK: } {hivm.cv_pipelined_loop}
func.func @graph_merges_consumer_after_transitive_boundary(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %bridge = arith.negf %v1 : f32
    %c = scope.scope : () -> f32 {
      %value = arith.negf %bridge : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v3 = scope.scope : () -> f32 {
      %value = arith.addf %c, %v2 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The bridge keeps C outside the VECTOR run. The two VECTOR scopes are fused.
// CHECK-LABEL: func.func @adjacent_preserves_transitive_incoming_overlap(
// COUNT-LABEL: func.func @adjacent_preserves_transitive_incoming_overlap(
// COUNT-COUNT-2: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_transitive_incoming_overlap(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %c = scope.scope : () -> f32 {
      %value = arith.negf %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %bridge = arith.negf %c : f32
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %bridge, %bridge : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// C0 -> V1 -> V2 -> C1: both VECTOR scopes have the same boundaries.
// CHECK-LABEL: func.func @adjacent_merges_shared_transitive_boundaries(
// CHECK: scf.for
// CHECK-COUNT-3: scope.scope
// CHECK-NOT: scope.scope
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_merges_shared_transitive_boundaries(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %c0val = scope.scope : () -> f32 {
      %value = arith.negf %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %c0val, %c0val : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %v1, %v1 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c1val = scope.scope : () -> f32 {
      %value = arith.negf %v2 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Three disconnected VECTOR scopes are fused into one VECTOR scope.
// CHECK-LABEL: func.func @graph_keeps_disconnected_scopes(
// COUNT-LABEL: func.func @graph_keeps_disconnected_scopes(
// COUNT: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @graph_keeps_disconnected_scopes(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v3 = scope.scope : () -> f32 {
      %value = arith.negf %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// V1/V2/V3 are fused into one VECTOR scope. The CUBE consumer stays outside.
// CHECK-LABEL: func.func @adjacent_rechecks_after_merge(
// COUNT-LABEL: func.func @adjacent_rechecks_after_merge(
// COUNT-COUNT-2: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_rechecks_after_merge(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v3 = scope.scope : () -> f32 {
      %value = arith.negf %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %value = arith.addf %v1, %v2 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The two CUBE scopes are fused. The VECTOR consumer stays outside.
// CHECK-LABEL: func.func @adjacent_preserves_cube_overlap(
// COUNT-LABEL: func.func @adjacent_preserves_cube_overlap(
// COUNT-COUNT-2: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_cube_overlap(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %c1val = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %c2val = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v = scope.scope : () -> f32 {
      %value = arith.negf %c1val : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The two VECTOR scopes are fused. CUBE_AND_VECTOR stays outside.
// CHECK-LABEL: func.func @adjacent_preserves_mixed_scope_overlap(
// COUNT-LABEL: func.func @adjacent_preserves_mixed_scope_overlap(
// COUNT-COUNT-2: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_mixed_scope_overlap(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %mixed = scope.scope : () -> f32 {
      %value = arith.negf %v1 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The two VECTOR scopes are fused. A scope without a core type stays outside.
// CHECK-LABEL: func.func @adjacent_preserves_unknown_scope_overlap(
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope
// CHECK-NOT: hivm.loop_core_type
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_unknown_scope_overlap(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %unknown = scope.scope : () -> f32 {
      %value = arith.negf %v1 : f32
      scope.return %value : f32
    }
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The two VECTOR scopes are fused. The mixed wrapper stays separate.
// CHECK-LABEL: func.func @adjacent_preserves_wrapper_overlap(
// COUNT-LABEL: func.func @adjacent_preserves_wrapper_overlap(
// COUNT-COUNT-4: scope.scope
// COUNT-NOT: scope.scope
// COUNT: } {hivm.cv_pipelined_loop}
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {cv_merged_scope}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_wrapper_overlap(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %value = arith.addf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %value = arith.mulf %a, %a : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %c = scope.scope : () -> f32 {
        %value = arith.negf %v1 : f32
        scope.return %value : f32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %value = arith.negf %c : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The two VECTOR scopes are fused. The CUBE scope that conflicts with the
// first store stays outside.
// CHECK-LABEL: func.func @adjacent_preserves_memory_overlap(
// CHECK: scf.for
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: } {hivm.cv_pipelined_loop}
func.func @adjacent_preserves_memory_overlap(%a: f32, %buffer: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    scope.scope : () -> () {
      memref.store %a, %buffer[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %value = arith.mulf %a, %a : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %value = memref.load %buffer[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}
