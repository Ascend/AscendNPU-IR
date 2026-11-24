// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %s -o %t
// RUN: FileCheck %s < %t
// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %t -o %t.twice
// RUN: diff %t %t.twice

// V1 -> V2 can merge even though V1 also feeds C. The consumer C must move
// after the merged scope, and the external input to V2 must still dominate it.
// CHECK-LABEL: func.func @chain_reduces_overlap
// CHECK: scf.for
// CHECK-NEXT: %[[INPUT:.*]] = arith.negf
// CHECK-NEXT: %[[V:.*]]:2 = scope.scope
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: %[[B:.*]] = arith.mulf %[[A]], %[[INPUT]]
// CHECK-NEXT: scope.return %[[A]], %[[B]] : f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[C:.*]] = scope.scope
// CHECK-NEXT: %[[NEG:.*]] = arith.negf %[[V]]#0
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: memref.store %[[V]]#1
// CHECK-NEXT: memref.store %[[C]]
// CHECK-NOT: cv_merged_scope
func.func @chain_reduces_overlap(%x: f32, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %a = arith.addf %x, %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %a = arith.negf %v1 : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %input = arith.negf %x : f32
    %v2 = scope.scope : () -> f32 {
      %a = arith.mulf %v1, %input : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    memref.store %v2, %out[%c0] : memref<2xf32>
    memref.store %c, %out[%c1] : memref<2xf32>
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Three VECTOR successors of a CUBE scope merge to a fixed point. The scalar
// operation between the successors is retained outside the merged scope.
// CHECK-LABEL: func.func @fan_out
// CHECK: scf.for
// CHECK-NEXT: %[[C:.*]] = scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: %[[INPUT:.*]] = arith.negf
// CHECK-NEXT: %[[V:.*]]:3 = scope.scope
// CHECK-NEXT: %[[A:.*]] = arith.addf %[[C]], %[[C]]
// CHECK-NEXT: %[[B:.*]] = arith.mulf %[[C]], %[[INPUT]]
// CHECK-NEXT: %[[D:.*]] = arith.subf %[[C]], %[[C]]
// CHECK-NEXT: scope.return %[[A]], %[[B]], %[[D]] : f32, f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: memref.store %[[V]]#0
// CHECK-NEXT: memref.store %[[V]]#1
// CHECK-NEXT: memref.store %[[V]]#2
func.func @fan_out(%x: f32, %out: memref<3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %c = scope.scope : () -> f32 {
      %a = arith.negf %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v1 = scope.scope : () -> f32 {
      %a = arith.addf %c, %c : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %input = arith.negf %x : f32
    %v2 = scope.scope : () -> f32 {
      %a = arith.mulf %c, %input : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v3 = scope.scope : () -> f32 {
      %a = arith.subf %c, %c : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    memref.store %v1, %out[%c0] : memref<3xf32>
    memref.store %v2, %out[%c1] : memref<3xf32>
    memref.store %v3, %out[%c2] : memref<3xf32>
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The direct V1 -> V2 edge does not allow crossing the alternate V1 -> C -> V2
// path. This would introduce a cycle after contraction.
// CHECK-LABEL: func.func @cross_core_detour
// CHECK: scf.for
// CHECK-COUNT-3: scope.scope
// CHECK-NOT: scope.scope
// CHECK: } {hivm.cv_pipelined_loop}
func.func @cross_core_detour(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %a = arith.addf %x, %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %a = arith.negf %v1 : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %a = arith.mulf %v1, %c : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Convexity includes non-scope nodes, even when there is also a direct edge.
// CHECK-LABEL: func.func @scalar_detour
// CHECK: scf.for
// CHECK: %[[V:.*]] = scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[BRIDGE:.*]] = arith.negf %[[V]]
// CHECK-NEXT: scope.scope
// CHECK-NEXT: {{.*}} = arith.mulf %[[V]], %[[BRIDGE]]
func.func @scalar_detour(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %v = scope.scope : () -> f32 {
      %a = arith.addf %x, %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %bridge = arith.negf %v : f32
    scope.scope : () -> () {
      %a = arith.mulf %v, %bridge : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// V1 and V2 cannot merge, but the common C0 predecessor still allows V1 and V3
// to merge. The C1 consumer moves after the pair; V2 remains after C1.
// CHECK-LABEL: func.func @partial_fan_out
// CHECK: scf.for
// CHECK-NEXT: %[[C0:.*]] = scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: %[[V:.*]] = scope.scope
// CHECK-NEXT: %[[A:.*]] = arith.addf %[[C0]], %[[C0]]
// CHECK-NEXT: {{.*}} = arith.subf %[[C0]], %[[C0]]
// CHECK-NEXT: scope.return %[[A]] : f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[C1:.*]] = scope.scope
// CHECK-NEXT: {{.*}} = arith.negf %[[V]]
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: {{.*}} = arith.mulf %[[C0]], %[[C1]]
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @partial_fan_out(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %c = scope.scope : () -> f32 {
      %a = arith.negf %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %v1 = scope.scope : () -> f32 {
      %a = arith.addf %c, %c : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %d = scope.scope : () -> f32 {
      %a = arith.negf %v1 : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %a = arith.mulf %c, %d : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = arith.subf %c, %c : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// No SSA edge: two stores followed by a load are connected through a view of
// the same buffer. The CUBE load must move after the merged VECTOR stores.
// CHECK-LABEL: func.func @memory_consumer_moves_after
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.store
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: {{.*}} = memref.load
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
func.func @memory_consumer_moves_after(%x: f32, %buf: memref<1xf32>, %other: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %view = memref.cast %buf : memref<1xf32> to memref<?xf32>
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      memref.store %x, %buf[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = memref.load %view[%c0] : memref<?xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      memref.store %x, %other[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = memref.load %buf[%c0] : memref<1xf32>
      %b = memref.load %other[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// RAW and WAR edges through a CUBE load prevent merging the VECTOR stores.
// CHECK-LABEL: func.func @memory_cross_core_detour
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: memref.store
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: {{.*}} = memref.load
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: memref.store
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @memory_cross_core_detour(%x: f32, %buf: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      memref.store %x, %buf[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = memref.load %buf[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      memref.store %x, %buf[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A mixed loop-carried wrapper anchors fan-in from CUBE predecessors while
// small (1 input, 0 output) VECTOR consumers of its inner VECTOR scope merge
// directly into that inner VECTOR scope.
// CHECK-LABEL: func.func @wrapper_anchors_neighbors
// CHECK: scf.for
// CHECK-NEXT: %[[P:.*]]:2 = scope.scope
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: %[[B:.*]] = arith.mulf
// CHECK-NEXT: scope.return %[[A]], %[[B]] : f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[C:.*]] = scope.scope
// CHECK-NEXT: {{.*}} = arith.addf %[[P]]#0, %[[P]]#1
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[NEG:.*]] = arith.negf %[[C]]
// CHECK-NEXT: {{.*}} = arith.negf %[[NEG]]
// CHECK-NEXT: {{.*}} = arith.mulf %[[NEG]], %[[NEG]]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @wrapper_anchors_neighbors(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %v1 = scope.scope : () -> f32 {
      %a = arith.addf %x, %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v2 = scope.scope : () -> f32 {
      %a = arith.mulf %x, %x : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %wrapper = scope.scope : () -> f32 {
      %c = scope.scope : () -> f32 {
        %a = arith.addf %v1, %v2 : f32
        scope.return %a : f32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %v = scope.scope : () -> f32 {
        %a = arith.negf %c : f32
        scope.return %a : f32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %v : f32
    } {cv_merged_scope}
    scope.scope : () -> () {
      %a = arith.negf %wrapper : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = arith.mulf %wrapper, %wrapper : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// When a mixed loop-carried wrapper exports a CUBE scope result, VECTOR
// consumers do not merge into the CUBE scope and still merge via wrapper
// fan-out in applyGraphSameCoreRule.
// CHECK-LABEL: func.func @wrapper_anchors_cube_fan_out
// CHECK: scf.for
// CHECK-NEXT: %[[W:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[C:.*]] = scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return %[[C]] : f32
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: {{.*}} = arith.negf %[[W]]
// CHECK-NEXT: {{.*}} = arith.mulf %[[W]], %[[W]]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @wrapper_anchors_cube_fan_out(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %wrapper = scope.scope : () -> f32 {
      %c = scope.scope : () -> f32 {
        %a = arith.addf %x, %x : f32
        scope.return %a : f32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %a = arith.negf %c : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %c : f32
    } {cv_merged_scope}
    scope.scope : () -> () {
      %a = arith.negf %wrapper : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %a = arith.mulf %wrapper, %wrapper : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Model the two loop-carried branches of DHU: V0 -> V1 -> C -> tail,
// and gate -> tail. Rule 2 must not absorb the adjacent, unrelated gate
// into V0/V1. Rule 5 merges gate with tail inside the LCD wrapper even
// though tail also has a CUBE predecessor. The alternate path through C
// must still prevent merging V0/V1 with gate/tail.
// CHECK-LABEL: func.func @lcd_gate_merges_with_tail
// CHECK: scf.for {{.*}} iter_args(%[[ARG:.*]] = {{.*}}) -> (f32) {
// CHECK-NEXT: %[[W:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[HEAD:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[A:.*]] = arith.addf %[[ARG]], %[[ARG]]
// CHECK-NEXT: %[[B:.*]] = arith.negf %[[A]]
// CHECK-NEXT: scope.return %[[B]] : f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[C:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[D:.*]] = arith.negf %[[HEAD]]
// CHECK-NEXT: scope.return %[[D]] : f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: %[[TAIL:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[GATE:.*]] = arith.mulf %[[ARG]], %[[ARG]]
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[GATE]], %[[C]]
// CHECK-NEXT: scope.return %[[SUM]] : f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return %[[TAIL]] : f32
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: scf.yield %[[W]] : f32
func.func @lcd_gate_merges_with_tail(%x: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg = %x) -> f32 {
    %v0 = scope.scope : () -> f32 {
      %a = arith.addf %arg, %arg : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %gate = scope.scope : () -> f32 {
      %a = arith.mulf %arg, %arg : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %v1 = scope.scope : () -> f32 {
      %a = arith.negf %v0 : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> f32 {
      %a = arith.negf %v1 : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %tail = scope.scope : () -> f32 {
      %a = arith.addf %gate, %c : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %tail : f32
  } {hivm.cv_pipelined_loop}
  return %result : f32
}

// -----

// Sharing an external function argument is not a shared scope predecessor.
// The single-core wrapper rule fuses these children and collapses both
// wrappers before graph merging, even in isolation.
// CHECK-LABEL: func.func @nested_unconnected_scopes
// CHECK: scf.for
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: {{.*}} = arith.addf
// CHECK-NEXT: {{.*}} = arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @nested_unconnected_scopes(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        scope.scope : () -> () {
          %a = arith.addf %x, %x : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %a = arith.mulf %x, %x : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.return
      } {cv_merged_scope}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Nested Rule 5 must retain memory-order edges: the CUBE load between two
// VECTOR stores prevents their contraction even without any SSA edges.
// CHECK-LABEL: func.func @nested_memory_detour
// CHECK: scf.for
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: memref.store
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: {{.*}} = memref.load
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: memref.store
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
func.func @nested_memory_detour(%x: f32, %buf: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        memref.store %x, %buf[%c0] : memref<1xf32>
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %a = memref.load %buf[%c0] : memref<1xf32>
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        memref.store %x, %buf[%c0] : memref<1xf32>
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Recurse through multiple scope bodies and merge fan-out to a fixed point.
// Each VECTOR scope has an incompatible CUBE predecessor, so Rule 2 cannot
// merge them through its backward walk.
// CHECK-LABEL: func.func @deep_scope_fan_out
// CHECK: scf.for
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[C:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[A:.*]] = arith.negf
// CHECK-NEXT: scope.return %[[A]] : f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: {{.*}} = arith.addf %[[C]], %[[C]]
// CHECK-NEXT: {{.*}} = arith.mulf %[[C]], %[[C]]
// CHECK-NEXT: {{.*}} = arith.subf %[[C]], %[[C]]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
func.func @deep_scope_fan_out(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %c = scope.scope : () -> f32 {
          %a = arith.negf %x : f32
          scope.return %a : f32
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %a = arith.addf %c, %c : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %a = arith.mulf %c, %c : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %a = arith.subf %c, %c : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.return
      } {cv_merged_scope}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Traversing scope bodies must not optimize an unmarked nested loop as part
// of its enclosing pipelined loop.
// CHECK-LABEL: func.func @leave_unmarked_nested_loop
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: %[[V:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: scope.return %[[A]] : f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: {{.*}} = arith.mulf %[[V]], %[[V]]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @leave_unmarked_nested_loop(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scf.for %j = %c0 to %c1 step %c1 {
      %v = scope.scope : () -> f32 {
        %a = arith.addf %x, %x : f32
        scope.return %a : f32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %a = arith.mulf %v, %v : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    }
  } {hivm.cv_pipelined_loop}
  return
}
