// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %s -o %t
// RUN: FileCheck %s < %t
// RUN: bishengir-opt -merge-cv-scopes -verify-each -split-input-file %t -o %t.twice
// RUN: diff %t %t.twice

// A VECTOR-only cv_merged_scope, a VECTOR scope, and another VECTOR-only
// cv_merged_scope are one run. They are fused in order into one VECTOR
// scope. The following CUBE scope stays outside.
// CHECK-LABEL: func.func @group_vector_run
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
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
func.func @group_vector_run(%x: f32, %out: memref<1xf32>) {
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

// Two plain VECTOR scopes are fused in order into one VECTOR scope.
// CHECK-LABEL: func.func @wrap_independent_vectors
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
func.func @wrap_independent_vectors(%x: f32) {
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
