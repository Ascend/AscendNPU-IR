// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %s -o %t
// RUN: FileCheck %s < %t
// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %t -o %t.twice
// RUN: diff %t %t.twice

// Normalize VECTOR-only wrappers first, then fuse the exposed top-level run.
// The following CUBE scope remains separate.
// CHECK-LABEL: func.func @fuse_exposed_vector_run
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.subf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: memref.store
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @fuse_exposed_vector_run(%x: f32, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
    scope.scope : () -> () {
      %t = arith.subf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
    scope.scope : () -> () {
      memref.store %x, %out[%c0] : memref<1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Independent plain VECTOR scopes fuse without a pre-existing wrapper.
// CHECK-LABEL: func.func @fuse_independent_vectors
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @fuse_independent_vectors(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      %t = arith.addf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %t = arith.mulf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A CUBE scope splits two VECTOR scopes into separate runs.
// CHECK-LABEL: func.func @cube_splits_vector_run
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.subf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @cube_splits_vector_run(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      %t = arith.addf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %t = arith.mulf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %t = arith.subf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A cv_merged_scope that contains both cores is not a single-core scope, so
// it splits the VECTOR scopes on either side.
// CHECK-LABEL: func.func @mixed_merged_scope_splits_run
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.subf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @mixed_merged_scope_splits_run(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      %t = arith.addf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %t = arith.divf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.return
    } {cv_merged_scope}
    scope.scope : () -> () {
      %t = arith.subf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Normalize an isolated wrapper and preserve its result order even when it
// differs from the order of its children. These results feed the loop yield.
// CHECK-LABEL: func.func @isolated_vector_wrapper_results
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK-NEXT: %[[PAIR:.*]]:2 = scope.scope : () -> (f32, f32) {
// CHECK-NEXT: %[[SUM:.*]] = arith.addf
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf
// CHECK-NEXT: scope.return %[[PRODUCT]], %[[SUM]] : f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scf.yield %[[PAIR]]#0, %[[PAIR]]#1 : f32, f32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
// CHECK-NEXT: return %[[LOOP]]#0, %[[LOOP]]#1 : f32, f32
func.func @isolated_vector_wrapper_results(%x: f32, %y: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop:2 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc0 = %x, %acc1 = %y) -> (f32, f32) {
    %pair:2 = scope.scope : () -> (f32, f32) {
      %a = scope.scope : () -> f32 {
        %sum = arith.addf %x, %x : f32
        scope.return %sum : f32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %b = scope.scope : () -> f32 {
        %product = arith.mulf %y, %y : f32
        scope.return %product : f32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %b, %a : f32, f32
    } {cv_merged_scope}
    scf.yield %pair#0, %pair#1 : f32, f32
  } {hivm.cv_pipelined_loop}
  return %loop#0, %loop#1 : f32, f32
}

// -----

// CUBE-only wrappers receive the same treatment as VECTOR-only wrappers.
// CHECK-LABEL: func.func @isolated_cube_wrapper
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @isolated_cube_wrapper(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Recursively collapse nested single-core wrappers, then fuse the resulting
// top-level scope with its following VECTOR sibling.
// CHECK-LABEL: func.func @nested_vector_wrappers
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.subf
// CHECK-NEXT: arith.divf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @nested_vector_wrappers(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        scope.scope : () -> () {
          %t = arith.addf %x, %x : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %t = arith.mulf %x, %x : f32
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
        scope.return
      } {cv_merged_scope}
      scope.scope : () -> () {
        %t = arith.subf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
    scope.scope : () -> () {
      %t = arith.divf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A non-scope operation splits the children and prevents wrapper collapse.
// CHECK-LABEL: func.func @non_scope_splits_wrapper
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: arith.constant 4.200000e+01 : f32
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @non_scope_splits_wrapper(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %marker = arith.constant 42.0 : f32
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// An unclassified child scope splits the VECTOR children and keeps the wrapper.
// CHECK-LABEL: func.func @unclassified_scope_splits_wrapper
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: }
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.subf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @unclassified_scope_splits_wrapper(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      }
      scope.scope : () -> () {
        %t = arith.subf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A typed VECTOR scope without cv_merged_scope is not a wrapper to normalize.
// CHECK-LABEL: func.func @ordinary_vector_scope_keeps_children
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @ordinary_vector_scope_keeps_children(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %t = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %t = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Wrapper fusion must run before graph merging so the exposed VECTOR scope
// can merge with its dependent sibling. Yielding the wrapper results directly
// also prevents the earlier small-vector rule from merging these scopes.
// CHECK-LABEL: func.func @vector_wrapper_fusion_before_graph_merge
// CHECK: %[[LOOP:.*]]:3 = scf.for
// CHECK-NEXT: %[[MERGED:.*]]:3 = scope.scope : () -> (f32, f32, f32) {
// CHECK-NEXT: %[[SUM:.*]] = arith.addf
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf
// CHECK-NEXT: %[[DIFFERENCE:.*]] = arith.subf %[[SUM]], %[[PRODUCT]] : f32
// CHECK-NEXT: scope.return %[[SUM]], %[[PRODUCT]], %[[DIFFERENCE]] : f32, f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scf.yield %[[MERGED]]#1, %[[MERGED]]#2, %[[MERGED]]#0 : f32, f32, f32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
// CHECK-NEXT: return %[[LOOP]]#0, %[[LOOP]]#1, %[[LOOP]]#2 : f32, f32, f32
func.func @vector_wrapper_fusion_before_graph_merge(%x: f32, %y: f32) -> (f32, f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop:3 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc0 = %x, %acc1 = %y, %acc2 = %x) -> (f32, f32, f32) {
    %pair:2 = scope.scope : () -> (f32, f32) {
      %a = scope.scope : () -> f32 {
        %sum = arith.addf %x, %x : f32
        scope.return %sum : f32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %b = scope.scope : () -> f32 {
        %product = arith.mulf %y, %y : f32
        scope.return %product : f32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %a, %b : f32, f32
    } {cv_merged_scope}
    %difference = scope.scope : () -> f32 {
      %value = arith.subf %pair#0, %pair#1 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %pair#1, %difference, %pair#0 : f32, f32, f32
  } {hivm.cv_pipelined_loop}
  return %loop#0, %loop#1, %loop#2 : f32, f32, f32
}

// -----

// The same ordering exposes CUBE wrappers to the unchanged graph merge rule.
// CHECK-LABEL: func.func @cube_wrapper_fusion_before_graph_merge
// CHECK: %[[LOOP:.*]]:3 = scf.for
// CHECK-NEXT: %[[MERGED:.*]]:3 = scope.scope : () -> (f32, f32, f32) {
// CHECK-NEXT: %[[SUM:.*]] = arith.addf
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf
// CHECK-NEXT: %[[DIFFERENCE:.*]] = arith.subf %[[SUM]], %[[PRODUCT]] : f32
// CHECK-NEXT: scope.return %[[SUM]], %[[PRODUCT]], %[[DIFFERENCE]] : f32, f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.yield %[[MERGED]]#1, %[[MERGED]]#2, %[[MERGED]]#0 : f32, f32, f32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
// CHECK-NEXT: return %[[LOOP]]#0, %[[LOOP]]#1, %[[LOOP]]#2 : f32, f32, f32
func.func @cube_wrapper_fusion_before_graph_merge(%x: f32, %y: f32) -> (f32, f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop:3 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc0 = %x, %acc1 = %y, %acc2 = %x) -> (f32, f32, f32) {
    %pair:2 = scope.scope : () -> (f32, f32) {
      %a = scope.scope : () -> f32 {
        %sum = arith.addf %x, %x : f32
        scope.return %sum : f32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %b = scope.scope : () -> f32 {
        %product = arith.mulf %y, %y : f32
        scope.return %product : f32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.return %a, %b : f32, f32
    } {cv_merged_scope}
    %difference = scope.scope : () -> f32 {
      %value = arith.subf %pair#0, %pair#1 : f32
      scope.return %value : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scf.yield %pair#1, %difference, %pair#0 : f32, f32, f32
  } {hivm.cv_pipelined_loop}
  return %loop#0, %loop#1, %loop#2 : f32, f32, f32
}

// -----

// Independent CUBE results must keep their external mapping after fusion.
// CHECK-LABEL: func.func @independent_cube_results
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK-NEXT: %[[PAIR:.*]]:2 = scope.scope
// CHECK-NEXT: %[[A:.*]] = arith.addf
// CHECK-NEXT: %[[B:.*]] = arith.mulf
// CHECK-NEXT: scope.return %[[A]], %[[B]] : f32, f32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.yield %[[PAIR]]#1, %[[PAIR]]#0 : f32, f32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
// CHECK-NEXT: return %[[LOOP]]#0, %[[LOOP]]#1 : f32, f32
func.func @independent_cube_results(%x: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.for %i = %c0 to %c1 step %c1 iter_args(%a = %x, %b = %x) -> (f32, f32) {
    %first = scope.scope : () -> f32 {
      %sum = arith.addf %x, %x : f32
      scope.return %sum : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %second = scope.scope : () -> f32 {
      %product = arith.mulf %x, %x : f32
      scope.return %product : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scf.yield %second, %first : f32, f32
  } {hivm.cv_pipelined_loop}
  return %r#0, %r#1 : f32, f32
}

// -----

// Unknown and CUBE_AND_VECTOR scopes interrupt a run, even if their bodies
// contain only scalar arithmetic. A non-scope operation also interrupts it.
// CHECK-LABEL: func.func @top_level_run_boundaries
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.negf
// CHECK-NEXT: scope.return
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.subf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.negf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.divf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: arith.constant 4.200000e+01
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @top_level_run_boundaries(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      %v = arith.addf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %v = arith.mulf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %v = arith.negf %x : f32
      scope.return
    }
    scope.scope : () -> () {
      %v = arith.subf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %v = arith.negf %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
    scope.scope : () -> () {
      %v = arith.divf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %marker = arith.constant 42.0 : f32
    scope.scope : () -> () {
      %v = arith.addf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %v = arith.mulf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Do not enter an unmarked nested loop while processing its pipelined parent.
// CHECK-LABEL: func.func @nested_loop_boundary
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.subf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: }
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.divf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @nested_loop_boundary(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      %v = arith.addf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scf.for %j = %c0 to %c1 step %c1 {
      scope.scope : () -> () {
        %v = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %v = arith.subf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    }
    scope.scope : () -> () {
      %v = arith.divf %x, %x : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// The new rule must not fuse disconnected adjacent VECTOR children inside a
// mixed wrapper: only its containing loop body's first level is scanned.
// CHECK-LABEL: func.func @mixed_wrapper_keeps_adjacent_children
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.subf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {cv_merged_scope}
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
func.func @mixed_wrapper_keeps_adjacent_children(%x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scope.scope : () -> () {
      scope.scope : () -> () {
        %v = arith.addf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %v = arith.mulf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %v = arith.subf %x, %x : f32
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      scope.return
    } {cv_merged_scope}
  } {hivm.cv_pipelined_loop}
  return
}
